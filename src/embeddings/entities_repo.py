from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client.http import models

from src.embeddings.client import ENTITIES_COLLECTION, QdrantClientManager
from src.embeddings.generator import EmbeddingGenerator
from src.graph.ontology import ExtractedEntity

logger = logging.getLogger(__name__)


class EntitiesVectorRepository:

    def __init__(self, client_manager: QdrantClientManager, generator: EmbeddingGenerator) -> None:
        self._client_manager = client_manager
        self._generator = generator
        self._client = client_manager.client

    async def upsert_entities(self, nodes: list[ExtractedEntity]) -> None:
        if not nodes:
            logger.debug("No entities to upsert")
            return

        if not self._client_manager._initialized:
            logger.warning("QdrantClientManager not initialized, skipping entity upsert")
            return

        try:
            texts: list[str] = []
            node_ids: list[ExtractedEntity] = []
            labels: list[str] = []
            original_ids: list[str] = []

            for node in nodes:
                node_id = node.id
                if not node_id:
                    logger.warning(
                        "Skipping entity with missing 'id'",
                        extra={"label": node.label},
                    )
                    continue

                text_parts = [f"{node.label}: {node.name}"]

                for key, value in node.properties.items():
                    text_parts.append(f", {key}: {value}")

                text = "".join(text_parts)

                if not text.strip():
                    logger.warning(
                        "Skipping entity with empty text representation",
                        extra={"id": node_id, "label": node.label},
                    )
                    continue

                texts.append(text)
                node_ids.append(node)
                labels.append(node.label)
                original_ids.append(str(node_id))

            if not texts:
                logger.debug("No valid entities to upsert after filtering")
                return

            dense_list, sparse_list = await self._generator.generate_batch(texts)

            point_structs = []
            for node, dense_emb, sparse_emb, label, orig_id in zip(
                node_ids, dense_list, sparse_list, labels, original_ids,
            ):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, orig_id))

                point_structs.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            "text": dense_emb,
                            "text_sparse": sparse_emb,
                        },
                        payload={
                            "original_id": orig_id,
                            "label": label,
                            "name": node.name,
                            "properties": node.properties,
                        },
                    )
                )

            for attempt in range(1, 4):
                try:
                    await self._client.upsert(
                        collection_name=ENTITIES_COLLECTION,
                        points=point_structs,
                        wait=True,
                    )
                    break
                except Exception as e:
                    logger.warning(
                        "Upsert attempt %d/3 failed: %s", attempt, e,
                        extra={"collection": ENTITIES_COLLECTION},
                    )
                    if attempt == 3:
                        raise
                    import asyncio
                    await asyncio.sleep(1.0)

            logger.info(
                "Upserted %d entity embeddings to Qdrant",
                len(point_structs),
                extra={"collection": ENTITIES_COLLECTION},
            )

        except Exception as e:
            logger.error(
                "Failed to upsert entity embeddings",
                exc_info=e,
                extra={"node_count": len(nodes)},
            )
            raise RuntimeError(f"Entity upsert failed: {e}") from e

    async def search_similar_entities(
        self, query: str, limit: int = 5, score_threshold: float = 0.88,
    ) -> list[dict[str, Any]]:
        if not self._client_manager._initialized:
            await self._client_manager.initialize()

        if not query or not query.strip():
            logger.warning("Empty query provided for entity search")
            return []

        try:
            dense_list, _ = await self._generator.generate_batch([query])
            dense_emb = dense_list[0]

            response = await self._client.query_points(
                collection_name=ENTITIES_COLLECTION,
                query=dense_emb,
                using="text",
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )

            entities: list[dict[str, Any]] = []
            for hit in response.points:
                if hit.payload and "original_id" in hit.payload:
                    entities.append({
                        "entity_id": str(hit.payload["original_id"]),
                        "score": hit.score,
                        "payload": hit.payload,
                    })

            logger.debug(
                "Entity search successful",
                extra={
                    "query": query,
                    "entities_found": len(entities),
                    "limit": limit,
                    "score_threshold": score_threshold,
                },
            )

            return entities

        except Exception as e:
            logger.error(
                "Error during entity search in Qdrant",
                exc_info=e,
                extra={"query": query, "limit": limit},
            )
            raise RuntimeError(f"Entity search failed: {e}") from e