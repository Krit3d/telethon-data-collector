"""Async Qdrant service for storing and searching post embeddings."""

from __future__ import annotations

import logging
from typing import Final

import httpx
import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from src.config.config import Settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM: Final[int] = 1024
EMBEDDING_METRIC: Final[Distance] = Distance.COSINE


class QdrantService:
    """Service for managing post embeddings in Qdrant."""

    def __init__(self, settings: Settings) -> None:
        """Initialize Qdrant client and ensure collection exists.

        Args:
            settings: Application settings containing Qdrant and embedding API configuration.

        Raises:
            ValueError: If required settings are missing.
            RuntimeError: If Qdrant connection or collection creation fails.
        """

        self.settings = settings

        if not settings.embedding_api_url:
            raise ValueError(
                "EMBEDDING_API_URL must be set in settings/environment"
            )

        if not settings.embedding_api_key:
            raise ValueError(
                "EMBEDDING_API_KEY must be set in settings/environment"
            )

        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            timeout=settings.qdrant_timeout,
        )
        self.collection_name = settings.qdrant_collection_name
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            headers={"Authorization": f"Bearer {settings.embedding_api_key}"}
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

        if not self.collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME is not configured")

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
    ) -> np.ndarray:
        """Generate embeddings for a list of texts using external API.

        Args:
            texts: List of text strings to generate embeddings for.

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
            # Make API request
            payload = {
                "input": valid_texts,
                "model": self.settings.embedding_model,
            }
            response = await self.http_client.post(
                self.settings.embedding_api_url,
                json=payload
            )
            response.raise_for_status()

            # Parse response
            response_data = response.json()
            embeddings = [
                item["embedding"] for item in response_data["data"]
            ]

            result = np.array(embeddings, dtype=np.float32)

            logger.debug(
                "Generated embeddings batch",
                extra={
                    "requested": len(texts),
                    "valid": len(valid_texts),
                    "embedding_shape": result.shape,
                },
            )

            return result

        except httpx.HTTPError as e:
            logger.error(
                "HTTP error during embedding generation",
                exc_info=e,
                extra={
                    "text_count": len(texts),
                    "valid_count": len(valid_texts),
                    "api_url": self.settings.embedding_api_url,
                },
            )
            raise RuntimeError(f"Embedding API request failed: {e}") from e
        except KeyError as e:
            logger.error(
                "Invalid response format from embedding API",
                exc_info=e,
                extra={"response_keys": list(response_data.keys()) if 'response_data' in locals() else None},
            )
            raise RuntimeError(f"Invalid embedding API response format: {e}") from e
        except Exception as e:
            logger.error(
                "Failed to generate embeddings batch",
                exc_info=e,
                extra={
                    "text_count": len(texts),
                    "valid_count": len(valid_texts),
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

        if not self.collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME is not configured")

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

            await self.client.upsert(  # type: ignore[attr-defined]
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

        if not self.collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME is not configured")

        try:
            embedding_array = await self._generate_embeddings_batch([text])
            embedding = embedding_array[0]

            point = PointStruct(
                id=post_id,
                vector=embedding.tolist(),
                payload={"channel_id": channel_id, "text": text},
            )

            await self.client.upsert(  # type: ignore[attr-defined]
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

    async def search_posts(
        self, query: str, limit: int = 10, score_threshold: float = 0.35
    ) -> list[dict]:
        """Search for posts using the unified query API.

        Args:
            query: Search query text.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score threshold (0-1).

        Returns:
            List of dictionaries containing post_id, score, text, and channel_id.

        Raises:
            RuntimeError: If service is not initialized or search fails.
        """

        if not self._initialized:
            await self.initialize()

        if not self.collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME is not configured")

        if not query or not query.strip():
            logger.warning("Empty query provided for search")
            return []

        try:
            # Generate embedding for the query
            embedding_array = await self._generate_embeddings_batch([query])
            query_embedding = embedding_array[0]

            # Use modern query_points instead of deprecated search
            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )

            # Transform results
            results = [
                {
                    "post_id": hit.id,
                    "score": hit.score,
                    "text": hit.payload.get("text", "") if hit.payload else "",
                    "channel_id": hit.payload.get("channel_id", 0) if hit.payload else 0,
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
        """Close the Qdrant client connection and HTTP client."""

        try:
            await self.client.close()
            await self.http_client.aclose()
            logger.debug("Qdrant client and HTTP client closed")
        except Exception as e:
            logger.warning("Error closing clients", exc_info=e)

    async def __aenter__(self) -> QdrantService:
        """Async context manager entry."""

        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""

        await self.close()
