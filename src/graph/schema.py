from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.graph.client import Neo4jClient


SCHEMA_QUERIES: list[str] = [
    "CREATE CONSTRAINT constraint_account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT constraint_post_id IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT constraint_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT constraint_concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT constraint_concept_code IF NOT EXISTS FOR (c:Concept) REQUIRE c.code IS UNIQUE",
    "CREATE INDEX index_entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX index_entity_label IF NOT EXISTS FOR (e:Entity) ON (e.label)",
    "CREATE INDEX index_post_published IF NOT EXISTS FOR (p:Post) ON (p.published_at)",
    "CREATE INDEX index_account_category IF NOT EXISTS FOR (a:Account) ON (a.category_id)",
]


async def init_neo4j_schema(client: Neo4jClient) -> None:
    for query in SCHEMA_QUERIES:
        await client.execute_write(query)


BATCH_UNWIND_CONCEPT = """
UNWIND $rows AS row
MERGE (c:Concept {id: row.id})
SET c.code = row.code,
    c.name = row.name,
    c.tier_1 = row.tier_1,
    c.tier_2 = row.tier_2,
    c.tier_3 = row.tier_3,
    c.tier_4 = row.tier_4,
    c.extension = row.extension
WITH c, row
WHERE row.parent_id IS NOT NULL AND row.parent_id <> ''
MATCH (p:Concept {id: row.parent_id})
MERGE (p)-[:PARENT_OF]->(c)
"""


async def import_iab_taxonomy(client: Neo4jClient, taxonomy_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    with taxonomy_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for record in reader:
            unique_id = record.get("Unique ID", "").strip()
            if not unique_id:
                continue
            parent = record.get("Parent", "").strip()
            name = record.get("Name", "").strip()
            tier_1 = record.get("Tier 1", "").strip()
            tier_2 = record.get("Tier 2", "").strip()
            tier_3 = record.get("Tier 3", "").strip()
            tier_4 = record.get("Tier 4", "").strip()
            extension = record.get("Extension", "").strip()
            rows.append({
                "id": unique_id,
                "parent_id": parent if parent else None,
                "code": unique_id,
                "name": name,
                "tier_1": tier_1,
                "tier_2": tier_2,
                "tier_3": tier_3,
                "tier_4": tier_4,
                "extension": extension,
            })
    if rows:
        await client.execute_batch_unwind(BATCH_UNWIND_CONCEPT, rows)
