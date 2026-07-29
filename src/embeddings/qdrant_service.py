from __future__ import annotations

import base64
import hashlib
import httpx
import logging
import struct
import asyncio
import random
import uuid
import warnings
from typing import Any, Final

from openai import AsyncOpenAI, RateLimitError
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from src.config.config import Settings
from src.graph.schema import ExtractedEntity, PropertyType

warnings.filterwarnings("ignore", message="Api key is used with an insecure connection.")

logger = logging.getLogger(__name__)

EMBEDDING_DIM: Final[int] = 1024
EMBEDDING_METRIC: Final[models.Distance] = models.Distance.COSINE

POSTS_COLLECTION: Final[str] = "social_posts"
ENTITIES_COLLECTION: Final[str] = "social_entities"


class QdrantService:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.collection_name = settings.qdrant_collection_name or POSTS_COLLECTION

        http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
            timeout=httpx.Timeout(60.0, connect=10.0, read=60.0, write=10.0),
        )
        self.openai_client = AsyncOpenAI(
            api_key=settings.cloud_ru_api_key,
            base_url=settings.cloud_ru_base_url,
            http_client=http_client,
        )

        if settings.qdrant_grpc_url is not None:
            self.client = AsyncQdrantClient(
                url=settings.qdrant_grpc_url,
                prefer_grpc=True,
                api_key=settings.qdrant_api_key,
                timeout=settings.qdrant_timeout,
            )
        else:
            self.client = AsyncQdrantClient(
                url=settings.qdrant_url,
                prefer_grpc=False,
                timeout=settings.qdrant_timeout,
                api_key=settings.qdrant_api_key,
            )
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        max_attempts = 15
        for attempt in range(1, max_attempts + 1):
            try:
                await self.client.get_collections()
                logger.info("Qdrant connection established")
                break
            except Exception:
                if attempt == max_attempts:
                    logger.error(
                        "Qdrant initialization failed: All connection attempts failed after retries",
                        extra={"url": self.settings.qdrant_url},
                    )
                    raise RuntimeError("Qdrant initialization failed: All connection attempts failed after retries")
                logger.warning("Qdrant not ready yet (attempt %d/15), retrying...", attempt)
                await asyncio.sleep(5)

        try:
            await self._ensure_collection()
            self._initialized = True
            logger.info(
                "Qdrant service initialized successfully",
                extra={
                    "collection": self.collection_name,
                    "url": self.settings.qdrant_url,
                    "model": self.settings.cloud_ru_embedding_model,
                },
            )

        except Exception as e:
            logger.error(
                "Failed to initialize Qdrant service",
                exc_info=e,
                extra={
                    "url": self.settings.qdrant_url,
                    "collection": self.collection_name,
                },
            )
            raise RuntimeError(f"Qdrant initialization failed: {e}") from e

    async def _ensure_collection(self) -> None:
        if not self.collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME is not configured")

        try:
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            sparse_config = {
                "text_sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            }

            posts_collection = self.collection_name
            if posts_collection not in collection_names:
                logger.info(
                    "Creating Qdrant posts collection",
                    extra={
                        "collection": posts_collection,
                        "dimension": EMBEDDING_DIM,
                    },
                )
                vectors_config: dict[str, models.VectorParams] = {
                    "text": models.VectorParams(
                        size=EMBEDDING_DIM, distance=EMBEDDING_METRIC,
                    ),
                    "video_clip": models.VectorParams(
                        size=self.settings.visual_embedding_dim,
                        distance=EMBEDDING_METRIC,
                    ),
                }
                await self.client.create_collection(
                    collection_name=posts_collection,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_config,
                )
                await self.client.create_payload_index(
                    collection_name=posts_collection,
                    field_name="account_id",
                    field_schema=models.PayloadSchemaType.INTEGER,
                )
                await self.client.create_payload_index(
                    collection_name=posts_collection,
                    field_name="subscribers_count",
                    field_schema=models.PayloadSchemaType.INTEGER,
                )
                await self.client.create_payload_index(
                    collection_name=posts_collection,
                    field_name="engagement_rate",
                    field_schema=models.PayloadSchemaType.FLOAT,
                )
                await self.client.create_payload_index(
                    collection_name=posts_collection,
                    field_name="platform",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                logger.info(
                    "Qdrant posts collection created successfully",
                    extra={"collection": posts_collection},
                )
            else:
                logger.debug(
                    "Qdrant posts collection already exists",
                    extra={"collection": posts_collection},
                )

            entities_collection = ENTITIES_COLLECTION
            if entities_collection not in collection_names:
                logger.info(
                    "Creating Qdrant entities collection",
                    extra={
                        "collection": entities_collection,
                        "dimension": EMBEDDING_DIM,
                    },
                )
                entities_vectors_config: dict[str, models.VectorParams] = {
                    "text": models.VectorParams(
                        size=EMBEDDING_DIM, distance=EMBEDDING_METRIC,
                    ),
                }
                await self.client.create_collection(
                    collection_name=entities_collection,
                    vectors_config=entities_vectors_config,
                    sparse_vectors_config=sparse_config,
                )
                await self.client.create_payload_index(
                    collection_name=entities_collection,
                    field_name="label",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                logger.info(
                    "Qdrant entities collection created successfully",
                    extra={"collection": entities_collection},
                )
            else:
                logger.debug(
                    "Qdrant entities collection already exists",
                    extra={"collection": entities_collection},
                )

        except Exception as e:
            logger.error(
                "Failed to ensure Qdrant collections",
                exc_info=e,
                extra={"collection": self.collection_name},
            )
            raise

    async def _optimize_collections_for_production(self) -> None:
        try:
            await self.client.update_collection(
                collection_name=self.collection_name,
                hnsw_config=models.HnswConfigDiff(on_disk=True),
                vectors_config={
                    "text": models.VectorParamsDiff(
                        on_disk=True,
                        quantization_config=models.ScalarQuantization(
                            scalar=models.ScalarQuantizationConfig(
                                type=models.ScalarType.INT8,
                                quantile=0.99,
                                always_ram=False,
                            ),
                        ),
                    ),
                    "video_clip": models.VectorParamsDiff(
                        on_disk=True,
                        quantization_config=models.ScalarQuantization(
                            scalar=models.ScalarQuantizationConfig(
                                type=models.ScalarType.INT8,
                                quantile=0.99,
                                always_ram=False,
                            ),
                        ),
                    ),
                },
            )
            logger.info(
                "Optimized posts collection for production",
                extra={"collection": self.collection_name},
            )
        except Exception as e:
            logger.warning(
                "Failed to optimize posts collection for production: %s",
                e,
                extra={"collection": self.collection_name},
            )

        try:
            await self.client.update_collection(
                collection_name=ENTITIES_COLLECTION,
                hnsw_config=models.HnswConfigDiff(on_disk=True),
                vectors_config={
                    "text": models.VectorParamsDiff(
                        on_disk=True,
                        quantization_config=models.ScalarQuantization(
                            scalar=models.ScalarQuantizationConfig(
                                type=models.ScalarType.INT8,
                                quantile=0.99,
                                always_ram=False,
                            ),
                        ),
                    ),
                },
            )
            logger.info(
                "Optimized entities collection for production",
                extra={"collection": ENTITIES_COLLECTION},
            )
        except Exception as e:
            logger.warning(
                "Failed to optimize entities collection for production: %s",
                e,
                extra={"collection": ENTITIES_COLLECTION},
            )

    @staticmethod
    def _make_fallback_sparse(text: str) -> models.SparseVector:
        tokens = text.lower().split()
        index_map: dict[int, float] = {}
        for token in tokens:
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = token_hash % 30_000
            weight = index_map.get(idx, 0.0) + 1.0
            index_map[idx] = weight
        sorted_items = sorted(index_map.items())
        if not sorted_items:
            return models.SparseVector(indices=[0], values=[0.0])
        return models.SparseVector(
            indices=[k for k, _ in sorted_items],
            values=[v for _, v in sorted_items],
        )

    async def _generate_cloud_embeddings_batch(
        self,
        texts: list[str],
    ) -> tuple[list[list[float]], list[models.SparseVector]]:
        if not texts:
            return [], []

        valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        if not valid_indices:
            return [[] for _ in texts], [self._make_fallback_sparse("") for _ in texts]

        valid_texts = [texts[i] for i in valid_indices]

        max_retries = 6
        base_delay = 1.0
        max_delay = 10.0
        for attempt in range(1, max_retries + 1):
            try:
                raw_response = await self.openai_client.embeddings.with_raw_response.create(
                    model=self.settings.cloud_ru_embedding_model,
                    input=valid_texts,
                )
                payload = raw_response.http_response.json()
                break
            except RateLimitError:
                if attempt == max_retries:
                    logger.error(
                        "Cloud.ru embedding API rate limits exceeded after %d attempts",
                        attempt,
                    )
                    raise RuntimeError("Cloud.ru embedding API rate limits were exceeded")
                sleep_duration = min(base_delay * (2 ** (attempt - 1)), max_delay)
                sleep_duration *= random.uniform(0.5, 1.5)
                logger.warning(
                    "Rate limited on attempt %d/%d, sleeping for %.2fs",
                    attempt,
                    max_retries,
                    sleep_duration,
                )
                await asyncio.sleep(sleep_duration)

        dense_map: dict[int, list[float]] = {}
        sparse_map: dict[int, models.SparseVector] = {}

        for local_idx, item in enumerate(payload.get("data", [])):
            global_idx = valid_indices[local_idx]

            dense_emb = item.get("embedding", [])
            if isinstance(dense_emb, str):
                try:
                    decoded = base64.b64decode(dense_emb)
                    dense_emb = list(struct.unpack(f"{len(decoded) // 4}f", decoded))
                except Exception:
                    dense_emb = [0.0] * EMBEDDING_DIM
            dense_map[global_idx] = dense_emb

            sparse_vector: models.SparseVector | None = None

            for key in ("sparse", "sparse_embedding"):
                raw_sparse = item.get(key)
                if raw_sparse is not None:
                    if isinstance(raw_sparse, dict):
                        indices = raw_sparse.get("indices")
                        values = raw_sparse.get("values")
                        if (
                            isinstance(indices, list)
                            and isinstance(values, list)
                            and indices
                            and values
                        ):
                            sparse_vector = models.SparseVector(
                                indices=[int(i) for i in indices],
                                values=[float(v) for v in values],
                            )
                            break
                        else:
                            sorted_items = sorted(
                                (int(k), float(v)) for k, v in raw_sparse.items()
                            )
                            if sorted_items:
                                sparse_vector = models.SparseVector(
                                    indices=[k for k, _ in sorted_items],
                                    values=[v for _, v in sorted_items],
                                )
                                break

            if sparse_vector is None:
                sparse_vector = self._make_fallback_sparse(valid_texts[local_idx])

            sparse_map[global_idx] = sparse_vector

        result_dense: list[list[float]] = []
        result_sparse: list[models.SparseVector] = []
        for i in range(len(texts)):
            result_dense.append(dense_map.get(i, [0.0] * EMBEDDING_DIM))
            result_sparse.append(
                sparse_map.get(i, self._make_fallback_sparse(texts[i]))
            )

        return result_dense, result_sparse

    async def upsert_batch(
        self,
        points: list[dict[str, Any]],
        visual_embeddings: list[list[float] | None] | None = None,
    ) -> None:
        if not points:
            return

        if not self.collection_name:
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

            dense_list, sparse_list = await self._generate_cloud_embeddings_batch(texts)

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
                    await self.client.upsert(
                        collection_name=self.collection_name,
                        points=point_structs,
                        wait=True,
                    )
                    break
                except Exception as e:
                    logger.warning(
                        "Upsert attempt %d/3 failed: %s", attempt, e,
                        extra={"collection": self.collection_name},
                    )
                    if attempt == 3:
                        raise
                    await asyncio.sleep(1.0)

            logger.debug(
                "Batch upserted %d embeddings to collection %s",
                len(point_structs),
                self.collection_name,
            )
        except Exception as e:
            logger.error(
                "Failed to batch upsert embeddings: %s",
                e,
                extra={"batch_size": len(points), "collection": self.collection_name},
            )
            raise

    async def upsert_entities(self, nodes: list[ExtractedEntity]) -> None:
        if not nodes:
            logger.debug("No entities to upsert")
            return

        if not self._initialized:
            logger.warning("QdrantService not initialized, skipping entity upsert")
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

                for prop in node.properties:
                    if prop.type in (PropertyType.TEXT, PropertyType.LOCATION):
                        text_parts.append(f", {prop.key}: {prop.value}")

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

            dense_list, sparse_list = await self._generate_cloud_embeddings_batch(texts)

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
                            "properties": node.get_property_dict(),
                        },
                    )
                )

            for attempt in range(1, 4):
                try:
                    await self.client.upsert(
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

    async def upsert_post_embedding(
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

        if not self._initialized:
            raise RuntimeError(
                "QdrantService not initialized. Call initialize() first.",
            )

        if not self.collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME is not configured")

        try:
            dense_list, sparse_list = await self._generate_cloud_embeddings_batch([text])

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
                    await self.client.upsert(
                        collection_name=self.collection_name, points=[point], wait=True,
                    )
                    break
                except Exception as e:
                    logger.warning(
                        "Upsert attempt %d/3 failed: %s", attempt, e,
                        extra={"collection": self.collection_name},
                    )
                    if attempt == 3:
                        raise
                    await asyncio.sleep(1.0)

            logger.debug(
                "Content embedding upserted for post %s to collection %s",
                post_id,
                self.collection_name,
            )

        except Exception as e:
            logger.error(
                "Failed to upsert post embedding for post %s: %s",
                post_id,
                e,
                extra={"post_id": post_id, "collection": self.collection_name},
            )
            raise RuntimeError(
                f"Failed to upsert embedding for post {post_id}: {e}",
            ) from e

    async def search_entities(
        self, query: str, limit: int = 5, score_threshold: float = 0.35,
    ) -> list[dict]:
        if not self._initialized:
            await self.initialize()

        collection_name = ENTITIES_COLLECTION

        if not query or not query.strip():
            logger.warning("Empty query provided for entity search")
            return []

        try:
            dense_list, _ = await self._generate_cloud_embeddings_batch([query])

            dense_emb = dense_list[0]

            response = await self.client.query_points(
                collection_name=collection_name,
                query=dense_emb,
                using="text",
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )

            entities: list[dict] = []
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

    async def search_posts(
        self,
        query: str | None = None,
        dense_query: str | None = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        min_followers: int | None = None,
        min_engagement_rate: float | None = None,
        platform: str | None = None,
    ) -> list[dict]:
        if not self._initialized:
            await self.initialize()

        if not self.collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME is not configured")

        effective_query = (dense_query or query or "").strip()
        if not effective_query:
            logger.warning("Empty query provided for search")
            return []

        try:
            dense_list, _ = await self._generate_cloud_embeddings_batch([effective_query])

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

            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=dense_emb,
                using="text",
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )

            results = []
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

    async def close(self) -> None:
        try:
            await self.client.close()
            await self.openai_client.close()
            logger.debug("Qdrant client closed")
        except Exception as e:
            logger.warning("Error closing Qdrant client", exc_info=e)

    async def __aenter__(self) -> QdrantService:
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.close()
