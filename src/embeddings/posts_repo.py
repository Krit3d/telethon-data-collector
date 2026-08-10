from __future__ import annotations

import logging
import re
from typing import Any

from qdrant_client.http import models

from src.embeddings.client import QdrantClientManager
from src.embeddings.generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


class PostsVectorRepository:

    def __init__(self, client_manager: QdrantClientManager, generator: EmbeddingGenerator) -> None:
        self._client_manager = client_manager
        self._generator = generator
        self._client = client_manager.client
        self._collection_name = client_manager.collection_name

    async def upsert_batch(
        self,
        points: list[dict[str, Any]],
        visual_embeddings: list[list[float] | None] | None = None,
    ) -> None:
        if not points:
            return

        if not self._collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME is not configured")

        try:
            filtered_indices = [
                i for i, p in enumerate(points)
                if p.get("text") and p["text"].strip()
            ]
            if not filtered_indices:
                logger.debug("No points with valid text to upsert")
                return

            filtered_points = [points[i] for i in filtered_indices]
            texts = [p["text"] for p in filtered_points]

            dense_list, sparse_list = await self._generator.generate_batch(texts)

            point_structs = []
            for local_idx, payload in enumerate(filtered_points):
                vectors: dict[str, Any] = {
                    "text": dense_list[local_idx],
                    "text_sparse": sparse_list[local_idx],
                }

                if visual_embeddings is not None:
                    orig_idx = filtered_indices[local_idx]
                    vis = visual_embeddings[orig_idx] if orig_idx < len(visual_embeddings) else None
                    if vis is not None:
                        vectors["video_clip"] = vis

                point_structs.append(
                    models.PointStruct(
                        id=payload["post_id"],
                        vector=vectors,
                        payload=payload,
                    )
                )

            for attempt in range(1, 4):
                try:
                    await self._client.upsert(
                        collection_name=self._collection_name,
                        points=point_structs,
                        wait=True,
                    )
                    break
                except Exception as e:
                    logger.warning(
                        "Upsert attempt %d/3 failed: %s", attempt, e,
                        extra={"collection": self._collection_name},
                    )
                    if attempt == 3:
                        raise
                    import asyncio
                    await asyncio.sleep(1.0)

            logger.debug(
                "Batch upserted %d embeddings to collection %s",
                len(point_structs),
                self._collection_name,
            )
        except Exception as e:
            logger.error(
                "Failed to batch upsert embeddings: %s",
                e,
                extra={"batch_size": len(points), "collection": self._collection_name},
            )
            raise

    async def upsert_post(
        self,
        payload: dict[str, Any],
        visual_embedding: list[float] | None = None,
    ) -> None:
        post_id: int = payload["post_id"]
        text: str = payload.get("text", "")

        if not text or not text.strip():
            logger.warning(
                "Empty text provided for embedding", extra={"post_id": post_id},
            )
            return

        if not self._client_manager._initialized:
            raise RuntimeError(
                "QdrantClientManager not initialized. Call initialize() first.",
            )

        if not self._collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME is not configured")

        try:
            dense_list, sparse_list = await self._generator.generate_batch([text])

            vectors: dict[str, Any] = {
                "text": dense_list[0],
                "text_sparse": sparse_list[0],
            }

            if visual_embedding is not None:
                vectors["video_clip"] = visual_embedding

            point = models.PointStruct(
                id=post_id,
                vector=vectors,
                payload=payload,
            )

            for attempt in range(1, 4):
                try:
                    await self._client.upsert(
                        collection_name=self._collection_name, points=[point], wait=True,
                    )
                    break
                except Exception as e:
                    logger.warning(
                        "Upsert attempt %d/3 failed: %s", attempt, e,
                        extra={"collection": self._collection_name},
                    )
                    if attempt == 3:
                        raise
                    import asyncio
                    await asyncio.sleep(1.0)

            logger.debug(
                "Content embedding upserted for post %s to collection %s",
                post_id,
                self._collection_name,
            )
        except Exception as e:
            logger.error(
                "Failed to upsert post embedding for post %s: %s",
                post_id,
                e,
                extra={"post_id": post_id, "collection": self._collection_name},
            )
            raise RuntimeError(
                f"Failed to upsert embedding for post {post_id}: {e}",
            ) from e

    async def search_posts(
        self,
        query: str | None = None,
        dense_query: str | None = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        min_followers: int | None = None,
        min_engagement_rate: float | None = None,
        platform: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self._client_manager._initialized:
            await self._client_manager.initialize()

        if not self._collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME is not configured")

        effective_query = (dense_query or query or "").strip()
        if not effective_query:
            logger.warning("Empty query provided for search")
            return []

        try:
            dense_list, _ = await self._generator.generate_batch([effective_query])
            dense_emb = dense_list[0]

            filter_conditions: list[models.Condition] = []
            if min_followers is not None:
                filter_conditions.append(
                    models.FieldCondition(
                        key="subscribers_count",
                        range=models.Range(gte=min_followers),
                    )
                )
            if min_engagement_rate is not None:
                filter_conditions.append(
                    models.FieldCondition(
                        key="engagement_rate",
                        range=models.Range(gte=min_engagement_rate),
                    )
                )
            if platform is not None:
                filter_conditions.append(
                    models.FieldCondition(
                        key="platform",
                        match=models.MatchValue(value=platform.upper()),
                    )
                )

            query_filter: models.Filter | None = (
                models.Filter(must=filter_conditions)
                if filter_conditions
                else None
            )

            response = await self._client.query_points(
                collection_name=self._collection_name,
                query=dense_emb,
                using="text",
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )

            results: list[dict[str, Any]] = []
            for hit in response.points:
                entry: dict[str, Any] = {
                    "post_id": hit.id,
                    "score": hit.score,
                    "engagement_rate": hit.payload.get("engagement_rate", 0.0) if hit.payload else 0.0,
                    "subscribers_count": hit.payload.get("subscribers_count", 0) if hit.payload else 0,
                    "platform": hit.payload.get("platform", "") if hit.payload else "",
                }
                if hit.payload:
                    entry.update(hit.payload)
                results.append(entry)

            logger.debug(
                "Query successful",
                extra={
                    "query": effective_query,
                    "results_count": len(results),
                    "limit": limit,
                    "score_threshold": score_threshold,
                },
            )

            return results

        except Exception as e:
            logger.error(
                "Error during Qdrant query_points",
                exc_info=e,
                extra={"query": effective_query, "limit": limit},
            )
            raise RuntimeError(f"Search failed: {e}") from e

    async def search_entity_posts(
        self, graph_entities: list[str], limit: int = 300,
    ) -> dict[int, float]:
        if not graph_entities:
            return {}

        if not self._client_manager._initialized:
            await self._client_manager.initialize()

        clean_entities = [e.strip().lower() for e in graph_entities if e.strip()]
        if not clean_entities:
            return {}

        try:
            from src.embeddings.client import ENTITIES_COLLECTION

            filter_ = models.Filter(
                should=[
                    models.FieldCondition(
                        key="name_lower",
                        match=models.MatchAny(any=clean_entities),
                    ),
                    models.FieldCondition(
                        key="name",
                        match=models.MatchAny(any=clean_entities),
                    ),
                    models.FieldCondition(
                        key="id",
                        match=models.MatchAny(any=clean_entities),
                    ),
                ]
            )

            all_points: list[models.Record] = []
            next_offset = None
            while True:
                response = await self._client.scroll(
                    collection_name=ENTITIES_COLLECTION,
                    scroll_filter=filter_,
                    limit=limit,
                    offset=next_offset,
                    with_payload=True,
                )
                points, next_offset = response
                if not points:
                    break
                all_points.extend(points)
                if next_offset is None:
                    break

            post_entity_map: dict[int, set[str]] = {}
            for hit in all_points:
                if hit.payload is None:
                    continue
                payload = hit.payload

                matched_entity: str | None = None
                entity_name_lower = payload.get("name_lower", "")
                if isinstance(entity_name_lower, str) and entity_name_lower in clean_entities:
                    matched_entity = entity_name_lower
                else:
                    entity_name = payload.get("name", "")
                    if isinstance(entity_name, str) and entity_name.lower() in clean_entities:
                        matched_entity = entity_name.lower()
                    else:
                        entity_id = payload.get("id", "")
                        if isinstance(entity_id, str) and entity_id.lower() in clean_entities:
                            matched_entity = entity_id.lower()

                if matched_entity is None:
                    continue

                post_ids = self._extract_post_ids(payload)
                for pid in post_ids:
                    if pid not in post_entity_map:
                        post_entity_map[pid] = set()
                    post_entity_map[pid].add(matched_entity)

            if not post_entity_map:
                logger.info("Qdrant entity search returned 0 candidate posts from %d entities", len(all_points))
                return {}

            total_entities = len(clean_entities)
            scored = {pid: len(entities) / total_entities for pid, entities in post_entity_map.items()}
            sorted_scores = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:limit]

            logger.info(
                "Qdrant entity search returned %d candidate posts from %d matched entities",
                len(sorted_scores),
                len(all_points),
            )
            return dict(sorted_scores)

        except Exception as e:
            logger.error("Qdrant entity search failed: %s", e)
            return {}

    @staticmethod
    def _extract_post_ids(payload: dict) -> list[int]:
        post_ids: list[int] = []

        raw_post_ids = payload.get("post_ids")
        if isinstance(raw_post_ids, list):
            for item in raw_post_ids:
                pid = PostsVectorRepository._to_int(item)
                if pid is not None:
                    post_ids.append(pid)

        if "post_id" in payload:
            pid = PostsVectorRepository._to_int(payload["post_id"])
            if pid is not None:
                post_ids.append(pid)

        if "db_post_id" in payload:
            pid = PostsVectorRepository._to_int(payload["db_post_id"])
            if pid is not None:
                post_ids.append(pid)

        raw_pub = payload.get("pub_node_id")
        if isinstance(raw_pub, str):
            pid = PostsVectorRepository._to_int(raw_pub)
            if pid is not None:
                post_ids.append(pid)

        props = payload.get("properties")
        if isinstance(props, dict):
            for key in ("post_ids", "post_id", "db_post_id", "pub_node_id"):
                if key in props:
                    val = props[key]
                    if key == "post_ids" and isinstance(val, list):
                        for item in val:
                            pid = PostsVectorRepository._to_int(item)
                            if pid is not None:
                                post_ids.append(pid)
                    else:
                        pid = PostsVectorRepository._to_int(val)
                        if pid is not None:
                            post_ids.append(pid)

        return list(set(post_ids))

    @staticmethod
    def _to_int(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            digits = re.findall(r"\d+", value)
            if digits:
                return int("".join(digits))
        return None