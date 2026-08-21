from __future__ import annotations

import asyncio
import csv
import logging
import time
import zlib
from pathlib import Path
from typing import Any

from sqlalchemy import select, text

from src.config.config import Settings
from src.db.database import Database
from src.db.models import Account
from src.graph.client import Neo4jClient
from src.graph.schema import init_neo4j_schema
from src.graph.utils import build_node_id, clean_name_lower

logger = logging.getLogger(__name__)

BATCH_SIZE = 500
GRAPH_BOOTSTRAP_LOCK_ID = zlib.crc32(b"neo4j_graph_bootstrap_lock")

BATCH_UNWIND_CONCEPT_NODES = """
UNWIND $rows AS row
MERGE (c:Concept {id: row.id})
SET c.code = row.code,
    c.name = row.name,
    c.name_lower = row.name_lower,
    c.tier_1 = row.tier_1,
    c.tier_2 = row.tier_2,
    c.tier_3 = row.tier_3,
    c.tier_4 = row.tier_4,
    c.extension = row.extension
"""

BATCH_UNWIND_CONCEPT_RELATIONS = """
UNWIND $rows AS row
MATCH (p:Concept {id: row.parent_id})
MATCH (c:Concept {id: row.id})
MERGE (p)-[r:PARENT_OF]->(c)
SET r.depth = row.depth
"""

BATCH_UNWIND_ACTOR = """
UNWIND $batch AS row
MERGE (a:Actor {id: row.id})
ON CREATE SET
    a.account_id = row.account_id,
    a.name = row.name,
    a.name_lower = row.name_lower,
    a.handle = row.handle,
    a.platform = row.platform,
    a.platform_id = row.platform_id,
    a.location_name = row.location_name
ON MATCH SET
    a.name = row.name,
    a.name_lower = row.name_lower,
    a.handle = row.handle,
    a.location_name = coalesce(row.location_name, a.location_name)
"""


def _chunked(items: list[dict[str, Any]], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


async def ensure_iab_taxonomy(client: Neo4jClient, taxonomy_path: Path | str) -> int:
    taxonomy_path = Path(taxonomy_path)
    rows = await client.execute_read("MATCH (c:Concept) RETURN count(c) AS cnt")
    if rows and rows[0]["cnt"] >= 500:
        return 0

    if not taxonomy_path.exists():
        logger.warning("Taxonomy file not found at %s, skipping IAB import", taxonomy_path)
        return 0

    concepts: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []

    with taxonomy_path.open(encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t")
        all_rows = list(reader)

    for line in all_rows[2:]:
        if not line or not line[0].strip():
            continue
        unique_id = line[0].strip()
        parent = line[1].strip() if len(line) > 1 else ""
        name = line[2].strip() if len(line) > 2 else ""
        tier_1 = line[3].strip() if len(line) > 3 else ""
        tier_2 = line[4].strip() if len(line) > 4 else ""
        tier_3 = line[5].strip() if len(line) > 5 else ""
        tier_4 = line[6].strip() if len(line) > 6 else ""
        extension = line[7].strip() if len(line) > 7 else ""

        if tier_4:
            depth = 4
        elif tier_3:
            depth = 3
        elif tier_2:
            depth = 2
        else:
            depth = 1

        concepts.append({
            "id": f"concept_{unique_id}",
            "parent_id": f"concept_{parent}" if parent else None,
            "code": unique_id,
            "name": name,
            "name_lower": clean_name_lower(name),
            "tier_1": tier_1,
            "tier_2": tier_2 or None,
            "tier_3": tier_3 or None,
            "tier_4": tier_4 or None,
            "extension": extension or None,
        })

        if parent:
            relations.append({
                "id": f"concept_{unique_id}",
                "parent_id": f"concept_{parent}",
                "depth": depth,
            })

    for batch in _chunked(concepts, BATCH_SIZE):
        await client.execute_write(BATCH_UNWIND_CONCEPT_NODES, {"rows": batch})

    for batch in _chunked(relations, BATCH_SIZE):
        await client.execute_write(BATCH_UNWIND_CONCEPT_RELATIONS, {"rows": batch})

    return len(concepts)


async def sync_verified_actors(db: Database, client: Neo4jClient, batch_size: int = 5000) -> int:
    rows = await client.execute_read("MATCH (a:Actor) RETURN count(a) AS cnt LIMIT 1")
    if rows and rows[0]["cnt"] > 0:
        return 0

    total_synced = 0
    last_id = 0

    while True:
        async with db.async_session() as session:
            stmt = (
                select(
                    Account.id,
                    Account.platform,
                    Account.platform_id,
                    Account.username,
                    Account.title,
                    Account.raw_metadata,
                )
                .where(Account.status == "verified", Account.id > last_id)
                .order_by(Account.id.asc())
                .limit(batch_size)
            )
            result = await session.execute(stmt)
            accounts = result.all()

        if not accounts:
            break

        batch: list[dict[str, Any]] = []
        for row in accounts:
            raw = row.raw_metadata or {}
            location_name: str | None = None
            if isinstance(raw, dict):
                location_name = raw.get("location")
                if not location_name:
                    geo = raw.get("geo_data")
                    if isinstance(geo, dict):
                        location_name = geo.get("city")

            name = row.title or row.username or f"Account_{row.id}"

            if not row.platform:
                continue
            platform_str = str(getattr(row.platform, "value", row.platform)).lower()

            batch.append({
                "id": build_node_id("Actor", "", platform=platform_str, account_id=row.id),
                "account_id": row.id,
                "name": name,
                "name_lower": clean_name_lower(name),
                "handle": row.username,
                "platform": platform_str,
                "platform_id": str(row.platform_id) if row.platform_id else "",
                "location_name": location_name,
            })

        await client.execute_write(BATCH_UNWIND_ACTOR, {"batch": batch})
        total_synced += len(batch)
        last_id = accounts[-1].id

    return total_synced


async def bootstrap_graph(settings: Settings, db: Database, client: Neo4jClient) -> dict[str, Any]:
    async with db.engine.connect() as conn:
        result = await conn.execute(
            text("SELECT pg_try_advisory_lock(:lock_id) AS acquired"),
            {"lock_id": GRAPH_BOOTSTRAP_LOCK_ID},
        )
        row = result.one()
        acquired: bool = row.acquired

        if acquired:
            start = time.monotonic()

            try:
                t0 = time.monotonic()
                await init_neo4j_schema(client)
                schema_elapsed = time.monotonic() - t0
                logger.info("Schema and indexes ensured in %.2f s", schema_elapsed)

                cleanup_query = "MATCH (c:Concept) WHERE c.code IS NULL OR c.code = '' DETACH DELETE c"
                await client.execute_write(cleanup_query)

                taxonomy_path = settings.taxonomy_path

                t1 = time.monotonic()
                iab_count = await ensure_iab_taxonomy(client, taxonomy_path)
                iab_elapsed = time.monotonic() - t1
                logger.info("IAB taxonomy loaded %d concepts in %.2f s", iab_count, iab_elapsed)

                t2 = time.monotonic()
                actor_count = await sync_verified_actors(db, client)
                actor_elapsed = time.monotonic() - t2
                logger.info("Synced %d verified actors in %.2f s", actor_count, actor_elapsed)

                total_elapsed = time.monotonic() - start
                logger.info(
                    "Bootstrap complete in %.2f s: schema=%.2f s, iab=%d (%.2f s), actors=%d (%.2f s)",
                    total_elapsed, schema_elapsed, iab_count, iab_elapsed, actor_count, actor_elapsed,
                )

                return {
                    "total_elapsed_s": round(total_elapsed, 2),
                    "schema_elapsed_s": round(schema_elapsed, 2),
                    "iab_concepts_loaded": iab_count,
                    "iab_elapsed_s": round(iab_elapsed, 2),
                    "verified_actors_synced": actor_count,
                    "actor_elapsed_s": round(actor_elapsed, 2),
                }
            finally:
                await conn.execute(
                    text("SELECT pg_advisory_unlock(:lock_id)"),
                    {"lock_id": GRAPH_BOOTSTRAP_LOCK_ID},
                )

    for _ in range(120):
        await asyncio.sleep(1.0)
        async with db.engine.connect() as conn:
            result = await conn.execute(
                text("SELECT pg_try_advisory_lock(:lock_id) AS acquired"),
                {"lock_id": GRAPH_BOOTSTRAP_LOCK_ID},
            )
            row = result.one()
            if row.acquired:
                await conn.execute(
                    text("SELECT pg_advisory_unlock(:lock_id)"),
                    {"lock_id": GRAPH_BOOTSTRAP_LOCK_ID},
                )
                return {"status": "skipped", "message": "Bootstrap completed by leader worker"}

    raise TimeoutError("Graph bootstrap timed out waiting for leader worker")


if __name__ == "__main__":
    from src.config.config import load_settings

    async def _main() -> None:
        settings = load_settings()
        db = Database(settings.db_url)
        client = Neo4jClient(settings)
        try:
            await db.init_db()
            await client.connect()
            stats = await bootstrap_graph(settings, db, client)
            logger.info("Bootstrap stats: %s", stats)
        finally:
            await client.close()
            await db.close()

    asyncio.run(_main())