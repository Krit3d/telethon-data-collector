import logging
import time

from qdrant_client.http import models

from src.api.schemas import AuthorVectorAggregate
from src.api.services.search.dbsf_engine import DbsfRankingEngine
from src.embeddings.qdrant_service import QdrantService

logger = logging.getLogger(__name__)


class VectorRetriever:

    def __init__(self, qdrant_service: QdrantService) -> None:
        self._qdrant_service = qdrant_service

    async def retrieve_vector_candidates(
        self, dense_query: str, limit: int = 600, max_authors_limit: int = 150, score_threshold: float = 0.45, is_author_blog: bool | None = None,
    ) -> tuple[dict[int, AuthorVectorAggregate], dict[str, float]]:
        if not dense_query:
            return {}, {"embedding_ms": 0.0, "qdrant_posts_ms": 0.0}

        timings: dict[str, float] = {}

        embed_start = time.perf_counter()
        dense_query_emb = await self._qdrant_service.generate_dense_embedding(dense_query)
        timings["embedding_ms"] = (time.perf_counter() - embed_start) * 1000.0

        if not self._qdrant_service.initialized:
            await self._qdrant_service.initialize()

        qdrant_filter: models.Filter | None = None
        if is_author_blog is not None:
            qdrant_filter = models.Filter(
                must=[models.FieldCondition(key="is_author_blog", match=models.MatchValue(value=is_author_blog))]
            )

        qdrant_start = time.perf_counter()
        response = await self._qdrant_service.client.query_points(
            collection_name=self._qdrant_service.collection_name,
            query=dense_query_emb,
            using="text",
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            query_filter=qdrant_filter,
        )
        timings["qdrant_posts_ms"] = (time.perf_counter() - qdrant_start) * 1000.0

        aggregates: dict[int, AuthorVectorAggregate] = {}

        for hit in response.points:
            try:
                post_id = int(hit.id)
            except (ValueError, TypeError):
                continue

            if hit.payload is None:
                continue

            try:
                account_id_raw = hit.payload.get("account_id")
                if account_id_raw is None:
                    continue
                account_id = int(account_id_raw)
            except (ValueError, TypeError):
                continue

            published_at_raw = hit.payload.get("published_at")
            published_at: int | None = int(published_at_raw) if published_at_raw is not None else None

            score = float(hit.score)

            if score < score_threshold:
                continue

            if account_id not in aggregates:
                aggregates[account_id] = AuthorVectorAggregate(
                    account_id=account_id,
                    post_scores=[],
                    max_vector_score=score,
                    matched_posts_count=0,
                )

            aggregate = aggregates[account_id]
            aggregate.post_scores.append(score)
            aggregate.max_vector_score = max(aggregate.max_vector_score, score)
            aggregate.matched_posts_count += 1

        for aggregate in aggregates.values():
            aggregate.decay_vector_score = DbsfRankingEngine.aggregate_author_vector_score(aggregate.post_scores, base_threshold=score_threshold)

        sorted_authors = sorted(aggregates.values(), key=lambda a: a.decay_vector_score, reverse=True)
        top_authors = sorted_authors[:max_authors_limit]

        filtered: dict[int, AuthorVectorAggregate] = {}
        for agg in top_authors:
            filtered[agg.account_id] = agg

        return filtered, timings