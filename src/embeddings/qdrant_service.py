from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Final

import numpy as np
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from src.config.config import Settings
from src.graph.schema import ExtractedEntity, PropertyType

logger = logging.getLogger(__name__)

EMBEDDING_DIM: Final[int] = 1024
EMBEDDING_METRIC: Final[models.Distance] = models.Distance.COSINE

POSTS_COLLECTION: Final[str] = "social_posts"
ENTITIES_COLLECTION: Final[str] = "social_entities"


class QdrantService:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.collection_name = settings.qdrant_collection_name or POSTS_COLLECTION

        if settings.onnxruntime_provider == "CUDAExecutionProvider":
            self.dense_model = TextEmbedding(
                model_name=settings.embedding_model_dense,
                threads=settings.embedding_threads,
                providers=["CUDAExecutionProvider"],
            )
            self.sparse_model = SparseTextEmbedding(
                model_name=settings.embedding_model_sparse,
                threads=settings.embedding_threads,
                providers=["CUDAExecutionProvider"],
            )
        else:
            self.dense_model = TextEmbedding(
                model_name=settings.embedding_model_dense,
                threads=settings.embedding_threads,
            )
            self.sparse_model = SparseTextEmbedding(
                model_name=settings.embedding_model_sparse,
                threads=settings.embedding_threads,
            )

        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            timeout=settings.qdrant_timeout,
            api_key=settings.qdrant_api_key,
        )
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        try:
            await self._ensure_collection()
            self._initialized = True
            logger.info(
                "Qdrant service initialized successfully",
                extra={
                    "collection": self.collection_name,
                    "url": self.settings.qdrant_url,
                    "model": self.settings.embedding_model_dense,
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

    async def _generate_dense_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        try:
            embeddings = await asyncio.to_thread(
                lambda: list(self.dense_model.embed(valid_texts)),
            )
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(
                "Failed to generate dense embeddings",
                exc_info=e,
                extra={"text_count": len(valid_texts)},
            )
            raise RuntimeError(f"Dense embedding generation failed: {e}") from e

    async def _generate_sparse_batch(self, texts: list[str]) -> list[Any]:
        if not texts:
            return []

        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return []

        try:
            sparse_results = await asyncio.to_thread(
                lambda: list(self.sparse_model.embed(valid_texts)),
            )
            return sparse_results
        except Exception as e:
            logger.error(
                "Failed to generate sparse embeddings",
                exc_info=e,
                extra={"text_count": len(valid_texts)},
            )
            raise RuntimeError(f"Sparse embedding generation failed: {e}") from e

    async def upsert_batch(
        self,
        points: list[tuple[int, str, int]],
        visual_embeddings: list[list[float] | None] | None = None,
    ) -> None:
        if not points:
            return

        if not self.collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME is not configured")

        try:
            filtered_indices = [
                i for i, (_, text, _) in enumerate(points)
                if text and text.strip()
            ]
            if not filtered_indices:
                logger.debug("No points with valid text to upsert")
                return

            filtered_points = [points[i] for i in filtered_indices]
            texts = [p[1] for p in filtered_points]

            dense_embeddings = await self._generate_dense_batch(texts)
            sparse_embeddings = await self._generate_sparse_batch(texts)

            point_structs = []
            for local_idx, (post_id, text, channel_id) in enumerate(filtered_points):
                vectors: dict[str, Any] = {
                    "text": dense_embeddings[local_idx].tolist(),
                    "text_sparse": models.SparseVector(
                        indices=sparse_embeddings[local_idx].indices.tolist(),
                        values=sparse_embeddings[local_idx].values.tolist(),
                    ),
                }

                if visual_embeddings is not None:
                    orig_idx = filtered_indices[local_idx]
                    vis = visual_embeddings[orig_idx] if orig_idx < len(visual_embeddings) else None
                    if vis is not None:
                        vectors["video_clip"] = vis

                point_structs.append(
                    models.PointStruct(
                        id=post_id,
                        vector=vectors,
                        payload={"account_id": channel_id, "text": text},
                    )
                )

            await self.client.upsert(
                collection_name=self.collection_name,
                points=point_structs,
                wait=True,
            )

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

            dense_embeddings = await self._generate_dense_batch(texts)
            sparse_embeddings = await self._generate_sparse_batch(texts)

            point_structs = []
            for node, dense_emb, sparse_emb, label, orig_id in zip(
                node_ids, dense_embeddings, sparse_embeddings, labels, original_ids,
            ):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, orig_id))

                point_structs.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            "text": dense_emb.tolist(),
                            "text_sparse": models.SparseVector(
                                indices=sparse_emb.indices.tolist(),
                                values=sparse_emb.values.tolist(),
                            ),
                        },
                        payload={
                            "original_id": orig_id,
                            "label": label,
                            "name": node.name,
                            "properties": node.get_property_dict(),
                        },
                    )
                )

            await self.client.upsert(
                collection_name=ENTITIES_COLLECTION,
                points=point_structs,
                wait=True,
            )

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
        post_id: int,
        text: str,
        account_id: int,
        visual_embedding: list[float] | None = None,
    ) -> None:
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
            dense_embeddings = await self._generate_dense_batch([text])
            sparse_embeddings = await self._generate_sparse_batch([text])

            dense_emb = dense_embeddings[0]
            sparse_emb = sparse_embeddings[0]

            vectors: dict[str, Any] = {
                "text": dense_emb.tolist(),
                "text_sparse": models.SparseVector(
                    indices=sparse_emb.indices.tolist(),
                    values=sparse_emb.values.tolist(),
                ),
            }

            if visual_embedding is not None:
                vectors["video_clip"] = visual_embedding

            point = models.PointStruct(
                id=post_id,
                vector=vectors,
                payload={"account_id": account_id, "text": text},
            )

            await self.client.upsert(
                collection_name=self.collection_name, points=[point],
            )

            logger.debug(
                "Content embedding upserted for post %d to collection %s",
                post_id,
                self.collection_name,
            )

        except Exception as e:
            logger.error(
                "Failed to upsert post embedding for post %d: %s",
                post_id,
                e,
                extra={"post_id": post_id, "account_id": account_id, "collection": self.collection_name},
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
            dense_embeddings = await self._generate_dense_batch([query])
            sparse_embeddings = await self._generate_sparse_batch([query])

            dense_emb = dense_embeddings[0]
            sparse_emb = sparse_embeddings[0]

            response = await self.client.query_points(
                collection_name=collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_emb.tolist(),
                        using="text",
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_emb.indices.tolist(),
                            values=sparse_emb.values.tolist(),
                        ),
                        using="text_sparse",
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
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
        self, query: str, limit: int = 10, score_threshold: float = 0.35,
    ) -> list[dict]:
        if not self._initialized:
            await self.initialize()

        if not self.collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME is not configured")

        if not query or not query.strip():
            logger.warning("Empty query provided for search")
            return []

        try:
            dense_embeddings = await self._generate_dense_batch([query])
            sparse_embeddings = await self._generate_sparse_batch([query])

            dense_emb = dense_embeddings[0]
            sparse_emb = sparse_embeddings[0]

            response = await self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_emb.tolist(),
                        using="text",
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_emb.indices.tolist(),
                            values=sparse_emb.values.tolist(),
                        ),
                        using="text_sparse",
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )

            results = [
                {
                    "post_id": hit.id,
                    "score": hit.score,
                    "text": hit.payload.get("text", "") if hit.payload else "",
                    "account_id": hit.payload.get("account_id", 0) if hit.payload else 0,
                }
                for hit in response.points
            ]

            logger.debug(
                "Query successful",
                extra={
                    "query": query,
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
                extra={"query": query, "limit": limit},
            )
            raise RuntimeError(f"Search failed: {e}") from e

    async def close(self) -> None:
        try:
            await self.client.close()
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
