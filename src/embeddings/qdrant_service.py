"""Async Qdrant service for storing and searching post embeddings."""

from __future__ import annotations

import logging
from typing import Final

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    # FieldCondition,
    # Filter,
    # MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from src.config.config import Settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM: Final[int] = 384
EMBEDDING_METRIC: Final[Distance] = Distance.COSINE


class QdrantService:
    """Service for managing post embeddings in Qdrant."""

    def __init__(self, settings: Settings) -> None:
        """Initialize Qdrant client and ensure collection exists.

        Args:
            settings: Application settings containing Qdrant configuration.

        Raises:
            RuntimeError: If Qdrant connection or collection creation fails.
        """

        self.settings = settings
        self.client = AsyncQdrantClient(url=settings.qdrant_url)
        self.collection_name = settings.qdrant_collection_name
        self.embedding_model = SentenceTransformer(
            settings.embedding_model_name
        )
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the service and ensure collection exists.

        This method should be called after creating the service instance.
        """

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
        """Check if collection exists and create it if missing."""

        try:
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(
                    "Creating Qdrant collection",
                    extra={
                        "collection": self.collection_name,
                        "dimension": EMBEDDING_DIM,
                    },
                )
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIM, distance=EMBEDDING_METRIC
                    ),
                )
                logger.info(
                    "Qdrant collection created successfully",
                    extra={"collection": self.collection_name},
                )
            else:
                logger.debug(
                    "Qdrant collection already exists",
                    extra={"collection": self.collection_name},
                )

        except Exception as e:
            logger.error(
                "Failed to ensure Qdrant collection",
                exc_info=e,
                extra={"collection": self.collection_name},
            )
            raise

    async def upsert_post_embedding(
        self, post_id: int, text: str, channel_id: int
    ) -> None:
        """Generate embedding for post text and upsert it into Qdrant.

        Args:
            post_id: PostgreSQL post ID (used as Qdrant point ID).
            text: Post text to generate embedding for.
            channel_id: Telegram channel ID stored in payload.

        Raises:
            ValueError: If text is empty or invalid.
            RuntimeError: If embedding generation or Qdrant operation fails.
        """

        if not text or not text.strip():
            logger.warning(
                "Empty text provided for embedding", extra={"post_id": post_id}
            )
            return

        if not self._initialized:
            raise RuntimeError(
                "QdrantService not initialized. Call initialize() first."
            )

        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)

            point = PointStruct(
                id=post_id,
                vector=embedding.tolist(),
                payload={"channel_id": channel_id, "text": text},
            )

            await self.client.upsert(
                collection_name=self.collection_name, points=[point]
            )

            logger.debug(
                "Post embedding upserted successfully",
                extra={"post_id": post_id, "channel_id": channel_id},
            )

        except Exception as e:
            logger.error(
                "Failed to upsert post embedding",
                exc_info=e,
                extra={"post_id": post_id, "channel_id": channel_id},
            )

            raise RuntimeError(
                f"Failed to upsert embedding for post {post_id}: {e}"
            ) from e

    async def close(self) -> None:
        """Close the Qdrant client connection."""

        try:
            await self.client.close()

            logger.debug("Qdrant client closed")

        except Exception as e:
            logger.warning("Error closing Qdrant client", exc_info=e)

    async def __aenter__(self) -> QdrantService:
        """Async context manager entry."""

        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""

        await self.close()
