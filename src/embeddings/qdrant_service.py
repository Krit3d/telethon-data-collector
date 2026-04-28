"""Async Qdrant service for storing and searching post embeddings."""

from __future__ import annotations

import asyncio
import logging
from typing import Final

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    PayloadSchemaType,
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
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            timeout=settings.qdrant_timeout,
            # For gRPC (faster):
            # grpc_port=settings.qdrant_grpc_port if settings.qdrant_grpc_url else None,
        )
        self.collection_name = settings.qdrant_collection_name
        self.embedding_model = SentenceTransformer(
            settings.embedding_model_name, device="cpu"  # Explicitly set device
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
                # Create payload indexes for faster filtering
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="channel_id",
                    field_schema=PayloadSchemaType.INTEGER,
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

    async def _generate_embeddings_batch(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for a list of texts using a separate thread.

        Args:
            texts: List of text strings to generate embeddings for.
            batch_size: Optional batch size for processing large text lists.
                    If None, processes all texts in a single batch.
            show_progress: Whether to show a progress bar (requires tqdm).

        Returns:
            numpy.ndarray: Array of embeddings with shape (len(texts), EMBEDDING_DIM).

        Raises:
            ValueError: If texts list is empty or contains invalid entries.
            RuntimeError: If embedding generation fails.
        """

        if not texts:
            logger.warning("Empty texts list provided for embedding generation")
            
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        # Validate texts
        valid_texts = [text for text in texts if text and text.strip()]
        if len(valid_texts) != len(texts):
            logger.warning(
                "Filtered out empty texts from embedding batch",
                extra={"total": len(texts), "valid": len(valid_texts)},
            )

        if not valid_texts:
            logger.warning("No valid texts to generate embeddings for")

            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        try:
            # Process in chunks if batch_size is specified
            if batch_size and batch_size > 0:
                all_embeddings = []

                for i in range(0, len(valid_texts), batch_size):
                    batch = valid_texts[i : i + batch_size]
                    batch_embeddings = await asyncio.to_thread(
                        self.embedding_model.encode,
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=show_progress and (i == 0),
                    )

                    all_embeddings.append(batch_embeddings)

                embeddings = np.vstack(all_embeddings)
            else:
                embeddings = await asyncio.to_thread(
                    self.embedding_model.encode,
                    valid_texts,
                    convert_to_numpy=True,
                    show_progress_bar=show_progress,
                )

            logger.debug(
                "Generated embeddings batch",
                extra={
                    "requested": len(texts),
                    "valid": len(valid_texts),
                    "embedding_shape": embeddings.shape,
                },
            )

            return embeddings

        except Exception as e:
            logger.error(
                "Failed to generate embeddings batch",
                exc_info=e,
                extra={
                    "text_count": len(texts),
                    "valid_count": (
                        len(valid_texts) if "valid_texts" in locals() else 0
                    ),
                },
            )
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    async def upsert_batch(self, points: list[tuple[int, str, int]]) -> None:
        """Upsert multiple post embeddings in a single batch.

        Args:
            points: List of (post_id, text, channel_id) tuples.
        """

        if not points:
            return

        try:
            # Generate embeddings in parallel
            texts = [p[1] for p in points]
            embeddings = await self._generate_embeddings_batch(texts)

            point_structs = [
                PointStruct(
                    id=post_id,
                    vector=embedding.tolist(),
                    payload={"channel_id": channel_id, "text": text},
                )

                for (post_id, text, channel_id), embedding in zip(
                    points, embeddings
                )
            ]

            await self.client.upsert(
                collection_name=self.collection_name,
                points=point_structs,
                wait=True,  # Wait for indexing
            )

            logger.debug(
                "Batch upserted %d embeddings",
                len(points),
                extra={"collection": self.collection_name},
            )
        except Exception as e:
            logger.error(
                "Failed to batch upsert embeddings",
                exc_info=e,
                extra={"batch_size": len(points)},
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
            embedding = await asyncio.to_thread(
                self.embedding_model.encode, text, convert_to_numpy=True
            )

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
