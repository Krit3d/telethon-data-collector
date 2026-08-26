from __future__ import annotations

import logging
import warnings
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from src.config.config import Settings
from src.embeddings.categories_repo import CategoriesVectorRepository
from src.embeddings.client import (
    POSTS_COLLECTION,
    QdrantClientManager,
)
from src.embeddings.entities_repo import EntitiesVectorRepository
from src.embeddings.generator import EmbeddingGenerator
from src.embeddings.posts_repo import PostsVectorRepository
from src.graph.ontology import ExtractedEntity

warnings.filterwarnings("ignore", message="Api key is used with an insecure connection.")

logger = logging.getLogger(__name__)


class QdrantService:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.collection_name = settings.qdrant_collection_name or POSTS_COLLECTION

        self.client_manager = QdrantClientManager(settings)
        self.generator = EmbeddingGenerator(settings)

        self.categories_repo = CategoriesVectorRepository(self.client_manager, self.generator)
        self.entities_repo = EntitiesVectorRepository(self.client_manager, self.generator)
        self.posts_repo = PostsVectorRepository(self.client_manager, self.generator)

        self.client: AsyncQdrantClient = self.client_manager.client
        self.openai_client = self.generator.openai_client
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self.client_manager.initialize()
        self._initialized = self.client_manager._initialized

    async def _ensure_collection(self) -> None:
        await self.client_manager.initialize()

    async def init_categories_collection(self) -> None:
        await self.client_manager._ensure_categories_collection()

    async def create_payload_indexes(self) -> None:
        await self.client_manager.ensure_payload_indexes()

    async def ensure_payload_indexes(self) -> None:
        await self.client_manager.ensure_payload_indexes()

    @staticmethod
    def _make_fallback_sparse(text: str) -> models.SparseVector:
        return EmbeddingGenerator._make_fallback_sparse(text)

    async def _generate_cloud_embeddings_batch(
        self,
        texts: list[str],
    ) -> tuple[list[list[float]], list[models.SparseVector]]:
        return await self.generator.generate_batch(texts)

    async def generate_dense_embedding(self, text: str) -> list[float]:
        dense_list, _ = await self.generator.generate_batch([text])
        return dense_list[0]

    async def upsert_batch(
        self,
        points: list[dict[str, Any]],
        visual_embeddings: list[list[float] | None] | None = None,
    ) -> None:
        await self.posts_repo.upsert_batch(points, visual_embeddings)

    async def upsert_post_embedding(
        self,
        payload: dict[str, Any],
        visual_embedding: list[float] | None = None,
    ) -> None:
        await self.posts_repo.upsert_post(payload, visual_embedding)

    async def upsert_entities(self, nodes: list[ExtractedEntity]) -> None:
        await self.entities_repo.upsert_entities(nodes)

    async def search_entities(
        self, query: str, limit: int = 5, score_threshold: float = 0.35,
    ) -> list[dict]:
        return await self.entities_repo.search_similar_entities(
            query, limit=limit, score_threshold=score_threshold,
        )

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
        return await self.posts_repo.search_posts(
            query=query,
            dense_query=dense_query,
            limit=limit,
            score_threshold=score_threshold,
            min_followers=min_followers,
            min_engagement_rate=min_engagement_rate,
            platform=platform,
        )

    async def search_entity_posts(self, graph_entities: list[str], limit: int = 300) -> dict[int, float]:
        return await self.posts_repo.search_entity_posts(graph_entities, limit=limit)

    @staticmethod
    def _extract_post_ids(payload: dict) -> list[int]:
        return PostsVectorRepository._extract_post_ids(payload)

    @staticmethod
    def _to_int(value: Any) -> int | None:
        return PostsVectorRepository._to_int(value)

    async def close(self) -> None:
        try:
            await self.client_manager.close()
            await self.generator.close()
            logger.debug("Qdrant clients closed")
        except Exception as e:
            logger.warning("Error closing Qdrant clients", exc_info=e)

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
