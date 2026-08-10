from __future__ import annotations

import asyncio
import logging
from typing import Any, Final

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from src.config.config import Settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM: Final[int] = 1024
EMBEDDING_METRIC: Final[models.Distance] = models.Distance.COSINE

quantization_config = models.ScalarQuantization(
    scalar=models.ScalarQuantizationConfig(
        type=models.ScalarType.INT8,
        quantile=0.99,
        always_ram=True,
    ),
)

POSTS_COLLECTION: Final[str] = "social_posts"
ENTITIES_COLLECTION: Final[str] = "social_entities"
CATEGORIES_COLLECTION: Final[str] = "iab_categories"


class QdrantClientManager:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.collection_name = settings.qdrant_collection_name or POSTS_COLLECTION

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
            await self._ensure_collections()
            self._initialized = True
            logger.info(
                "Qdrant client manager initialized successfully",
                extra={
                    "collection": self.collection_name,
                    "url": self.settings.qdrant_url,
                },
            )
        except Exception as e:
            logger.error(
                "Failed to initialize Qdrant client manager",
                exc_info=e,
                extra={"url": self.settings.qdrant_url},
            )
            raise RuntimeError(f"Qdrant initialization failed: {e}") from e

    async def _ensure_collections(self) -> None:
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
                    extra={"collection": posts_collection, "dimension": EMBEDDING_DIM},
                )
                vectors_config: dict[str, models.VectorParams] = {
                    "text": models.VectorParams(
                        size=EMBEDDING_DIM, distance=EMBEDDING_METRIC,
                        on_disk=True,
                        quantization_config=quantization_config,
                    ),
                    "video_clip": models.VectorParams(
                        size=self.settings.visual_embedding_dim,
                        distance=EMBEDDING_METRIC,
                        on_disk=True,
                        quantization_config=quantization_config,
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
                    extra={"collection": entities_collection, "dimension": EMBEDDING_DIM},
                )
                entities_vectors_config: dict[str, models.VectorParams] = {
                    "text": models.VectorParams(
                        size=EMBEDDING_DIM, distance=EMBEDDING_METRIC,
                        on_disk=True,
                        quantization_config=quantization_config,
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

            await self._ensure_categories_collection()
            await self.ensure_payload_indexes()

        except Exception as e:
            logger.error(
                "Failed to ensure Qdrant collections",
                exc_info=e,
                extra={"collection": self.collection_name},
            )
            raise

    async def _ensure_categories_collection(self) -> None:
        collection_name = CATEGORIES_COLLECTION
        try:
            collections = await self.client.get_collections()
            existing = {c.name for c in collections.collections}

            if collection_name not in existing:
                logger.info(
                    "Creating Qdrant categories collection",
                    extra={"collection": collection_name, "dimension": EMBEDDING_DIM},
                )
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "text": models.VectorParams(
                            size=EMBEDDING_DIM, distance=EMBEDDING_METRIC,
                        ),
                    },
                )
                logger.info(
                    "Qdrant categories collection created successfully",
                    extra={"collection": collection_name},
                )
            else:
                logger.debug(
                    "Qdrant categories collection already exists",
                    extra={"collection": collection_name},
                )
        except Exception as e:
            logger.error(
                "Failed to init categories collection",
                exc_info=e,
                extra={"collection": collection_name},
            )
            raise

    async def ensure_payload_indexes(self) -> None:
        categories_indexes = ["code", "tier_1", "tier_2", "tier_3", "tier_4"]
        if await self.client.collection_exists(CATEGORIES_COLLECTION):
            for field in categories_indexes:
                try:
                    await self.client.create_payload_index(
                        collection_name=CATEGORIES_COLLECTION,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception as e:
                    logger.warning(
                        "Payload index %s on %s may already exist: %s",
                        field, CATEGORIES_COLLECTION, e,
                    )

        entities_indexes = ["label", "original_id"]
        if await self.client.collection_exists(ENTITIES_COLLECTION):
            for field in entities_indexes:
                try:
                    await self.client.create_payload_index(
                        collection_name=ENTITIES_COLLECTION,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception as e:
                    logger.warning(
                        "Payload index %s on %s may already exist: %s",
                        field, ENTITIES_COLLECTION, e,
                    )

        logger.info("Payload indexes registered on categories and social_entities collections")

    async def close(self) -> None:
        try:
            await self.client.close()
            logger.debug("Qdrant client closed")
        except Exception as e:
            logger.warning("Error closing Qdrant client", exc_info=e)