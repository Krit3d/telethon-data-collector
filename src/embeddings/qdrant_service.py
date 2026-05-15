"""Async Qdrant service for storing and searching post embeddings."""

from __future__ import annotations

import logging
import uuid
from typing import Final

import numpy as np
from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from src.config.config import Settings
from src.graph.schema import ExtractedEntity, PropertyType

logger = logging.getLogger(__name__)

EMBEDDING_DIM: Final[int] = 1024
EMBEDDING_METRIC: Final[Distance] = Distance.COSINE

# Collection names
POSTS_COLLECTION: Final[str] = "posts"
ENTITIES_COLLECTION: Final[str] = "telegram_entities"


class QdrantService:
    """Service for managing post and entity embeddings in Qdrant using local fastembed."""

    def __init__(self, settings: Settings) -> None:
        """Initialize Qdrant client and ensure collection exists.

        Args:
            settings: Application settings containing Qdrant and embedding configuration.

        Raises:
            ValueError: If required settings are missing.
            RuntimeError: If Qdrant connection or collection creation fails.
        """

        self.settings = settings
        self.collection_name = settings.qdrant_collection_name or POSTS_COLLECTION

        # Initialize fastembed model with limited threads to control CPU usage
        self.model = TextEmbedding(
            model_name=settings.embedding_model_name,
            threads=settings.embedding_threads,
        )

        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            timeout=settings.qdrant_timeout,
            api_key=settings.qdrant_api_key,
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
                    "model": self.settings.embedding_model_name,
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
        """Check if collections exist and create them if missing."""

        if not self.collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME is not configured")

        try:
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            # Ensure posts collection (default collection for post embeddings)
            posts_collection = self.collection_name
            if posts_collection not in collection_names:
                logger.info(
                    "Creating Qdrant posts collection",
                    extra={
                        "collection": posts_collection,
                        "dimension": EMBEDDING_DIM,
                    },
                )
                await self.client.create_collection(
                    collection_name=posts_collection,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIM, distance=EMBEDDING_METRIC
                    ),
                )
                await self.client.create_payload_index(
                    collection_name=posts_collection,
                    field_name="channel_id",
                    field_schema=PayloadSchemaType.INTEGER,
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

            # Ensure telegram_entities collection (for graph nodes)
            entities_collection = ENTITIES_COLLECTION
            if entities_collection not in collection_names:
                logger.info(
                    "Creating Qdrant entities collection",
                    extra={
                        "collection": entities_collection,
                        "dimension": EMBEDDING_DIM,
                    },
                )
                await self.client.create_collection(
                    collection_name=entities_collection,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIM, distance=EMBEDDING_METRIC
                    ),
                )
                # Create payload indexes for entities collection
                await self.client.create_payload_index(
                    collection_name=entities_collection,
                    field_name="label",
                    field_schema=PayloadSchemaType.KEYWORD,
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

    async def _generate_embeddings_batch(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Generate embeddings for a list of texts using local fastembed model.

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
            # Generate embeddings using fastembed
            embeddings = self.model.embed(valid_texts)

            # Convert generator to numpy array
            result = np.array(list(embeddings), dtype=np.float32)

            logger.debug(
                "Generated embeddings batch",
                extra={
                    "requested": len(texts),
                    "valid": len(valid_texts),
                    "embedding_shape": result.shape,
                },
            )

            return result

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

    async def upsert_entities(self, nodes: list[ExtractedEntity]) -> None:
        """Upsert graph entity nodes into the telegram_entities collection.

        Args:
            nodes: List of ExtractedEntity objects to upsert.

        Raises:
            RuntimeError: If service is not initialized or upsert fails.
        """

        if not nodes:
            logger.debug("No entities to upsert")
            return

        if not self._initialized:
            logger.warning("QdrantService not initialized, skipping entity upsert")
            return

        try:
            # Construct textual representations for embedding generation
            texts = []
            node_ids = []
            labels = []
            original_ids = []

            for node in nodes:
                # node.id is directly available
                node_id = node.id
                if not node_id:
                    logger.warning(
                        "Skipping entity with missing 'id'",
                        extra={"label": node.label},
                    )
                    continue

                # Build text representation: "Label: name"
                # Optionally append properties of type TEXT or LOCATION
                text_parts = [f"{node.label}: {node.name}"]
                
                # Append relevant properties
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

            # Generate embeddings
            embeddings = await self._generate_embeddings_batch(texts)

            # Create point structs with deterministic UUIDs
            point_structs = []
            for node, embedding, label, orig_id in zip(
                node_ids, embeddings, labels, original_ids
            ):
                # Convert string ID to deterministic UUID
                point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, orig_id))

                point_structs.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            "original_id": orig_id,
                            "label": label,
                            "name": node.name,
                            "properties": node.get_property_dict(),
                        },
                    )
                )

            # Upsert to telegram_entities collection
            await self.client.upsert(  # type: ignore[attr-defined]
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

    async def search_entities(
        self, query: str, limit: int = 5, score_threshold: float = 0.35
    ) -> list[dict]:
        """Search for semantic entities in the telegram_entities collection.

        Args:
            query: Search query text.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score threshold (0-1).

        Returns:
            List of dictionaries containing entity_id, score, and payload data.
            Each dict has keys: 'entity_id' (original_id), 'score', 'payload' (dict).

        Raises:
            RuntimeError: If service is not initialized or search fails.
        """

        if not self._initialized:
            await self.initialize()

        collection_name = ENTITIES_COLLECTION

        if not query or not query.strip():
            logger.warning("Empty query provided for entity search")
            return []

        try:
            # Generate embedding for the query
            embedding_array = await self._generate_embeddings_batch([query])
            query_embedding = embedding_array[0]

            # Query the entities collection
            response = await self.client.query_points(
                collection_name=collection_name,
                query=query_embedding.tolist(),
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )

            # Extract entity data with scores and full payload
            entities = []
            for hit in response.points:
                if hit.payload and "original_id" in hit.payload:
                    entities.append({
                        "entity_id": str(hit.payload["original_id"]),
                        "score": hit.score,
                        "payload": hit.payload
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
