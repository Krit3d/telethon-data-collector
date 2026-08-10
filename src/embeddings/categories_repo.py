from __future__ import annotations

import logging
from typing import Any

from qdrant_client.http import models

from src.embeddings.client import CATEGORIES_COLLECTION, QdrantClientManager
from src.embeddings.generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


class CategoriesVectorRepository:

    def __init__(self, client_manager: QdrantClientManager, generator: EmbeddingGenerator) -> None:
        self._client_manager = client_manager
        self._generator = generator
        self._client = client_manager.client

    async def upsert_categories_batch(self, points: list[models.PointStruct]) -> None:
        if not points:
            return

        for attempt in range(1, 4):
            try:
                await self._client.upsert(
                    collection_name=CATEGORIES_COLLECTION,
                    points=points,
                    wait=True,
                )
                break
            except Exception as e:
                logger.warning(
                    "Categories upsert attempt %d/3 failed: %s", attempt, e,
                    extra={"collection": CATEGORIES_COLLECTION},
                )
                if attempt == 3:
                    raise
                import asyncio
                await asyncio.sleep(1.0)

        logger.info(
            "Upserted %d category points to %s",
            len(points), CATEGORIES_COLLECTION,
        )

    async def search_matching_concept(
        self, query: str, limit: int = 5,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            logger.warning("Empty query provided for category search")
            return []

        try:
            dense_list, _ = await self._generator.generate_batch([query])
            dense_emb = dense_list[0]

            response = await self._client.query_points(
                collection_name=CATEGORIES_COLLECTION,
                query=dense_emb,
                using="text",
                limit=limit,
                with_payload=True,
            )

            results: list[dict[str, Any]] = []
            for hit in response.points:
                results.append({
                    "score": hit.score,
                    "payload": hit.payload,
                })

            logger.debug(
                "Category search successful",
                extra={"query": query, "results": len(results), "limit": limit},
            )

            return results

        except Exception as e:
            logger.error(
                "Error during category search",
                exc_info=e,
                extra={"query": query, "limit": limit},
            )
            raise RuntimeError(f"Category search failed: {e}") from e