from __future__ import annotations

import asyncio
import csv
import logging
import uuid
from pathlib import Path

from qdrant_client.http import models

from src.config.config import Settings
from src.embeddings.client import CATEGORIES_COLLECTION, QdrantClientManager
from src.embeddings.generator import EmbeddingGenerator

logger = logging.getLogger(__name__)

TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "src" / "config" / "Content Taxonomy 3.1.tsv"
BATCH_SIZE = 64


def _parse_taxonomy(path: Path) -> list[dict[str, str | None]]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        all_rows = list(reader)

    rows: list[dict[str, str | None]] = []
    for line in all_rows[2:]:
        if not line or not line[0].strip():
            continue
        unique_id = line[0].strip()
        name = line[2].strip() if len(line) > 2 else ""
        tier1 = line[3].strip() if len(line) > 3 else ""
        tier2 = line[4].strip() if len(line) > 4 else ""
        tier3 = line[5].strip() if len(line) > 5 else ""
        tier4 = line[6].strip() if len(line) > 6 else ""
        extension = line[7].strip() if len(line) > 7 else ""

        category_path = " > ".join([t for t in (tier1, tier2, tier3, tier4) if t])
        text_for_embedding = f"{category_path}: {name}" if category_path else name

        rows.append({
            "unique_id": unique_id,
            "name": name,
            "tier1": tier1 or None,
            "tier2": tier2 or None,
            "tier3": tier3 or None,
            "tier4": tier4 or None,
            "extension": extension or None,
            "category_path": category_path,
            "text_for_embedding": text_for_embedding,
        })
    return rows


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = Settings()  # type: ignore[call-arg]
    client_manager = QdrantClientManager(settings)
    await client_manager.initialize()

    await client_manager._ensure_categories_collection()

    rows = _parse_taxonomy(TAXONOMY_PATH)
    logger.info("Parsed %d taxonomy rows from %s", len(rows), TAXONOMY_PATH)

    generator = EmbeddingGenerator(settings)
    client = client_manager.client

    total_rows = len(rows)
    total_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
    completed = 0
    for batch_idx in range(0, total_rows, BATCH_SIZE):
        batch_rows = rows[batch_idx : batch_idx + BATCH_SIZE]
        batch_texts: list[str] = [r["text_for_embedding"] or "" for r in batch_rows]
        dense_list, _ = await generator.generate_batch(batch_texts)

        points: list[models.PointStruct] = []
        for row, dense_emb in zip(batch_rows, dense_list):
            unique_id = str(row.get("unique_id") or "").strip()
            if not unique_id:
                continue
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"concept:{unique_id}"))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={"text": dense_emb},
                    payload={
                        "id": f"concept_{unique_id}",
                        "code": unique_id,
                        "name": row["name"],
                        "tier_1": row["tier1"],
                        "tier_2": row["tier2"],
                        "tier_3": row["tier3"],
                        "tier_4": row["tier4"],
                        "extension": row["extension"],
                        "category_path": row["category_path"],
                    },
                )
            )

        await client.upsert(
            collection_name=CATEGORIES_COLLECTION,
            points=points,
            wait=True,
        )
        completed += len(points)
        logger.info("Upserted batch %d/%d (%d points)", batch_idx // BATCH_SIZE + 1, total_batches, len(points))

    logger.info("Successfully upserted %d category points to %s", completed, CATEGORIES_COLLECTION)

    await client_manager.ensure_payload_indexes()

    query_text = "книги и литература"
    dense_list, _ = await generator.generate_batch([query_text])
    dense_emb = dense_list[0]

    response = await client.query_points(
        collection_name=CATEGORIES_COLLECTION,
        query=dense_emb,
        using="text",
        limit=5,
        with_payload=True,
    )

    if response.points:
        logger.info("Validation search results for '%s':", query_text)
        for hit in response.points:
            logger.info("  Score=%.4f  Payload=%s", hit.score, hit.payload)
    else:
        logger.warning("Validation search returned no results for '%s'", query_text)

    await generator.close()
    await client_manager.close()


if __name__ == "__main__":
    asyncio.run(main())