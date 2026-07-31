import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import ReformulatedQuery, SearchRequest
from src.db.models import Account, Content

logger = logging.getLogger(__name__)


@dataclass
class CandidateAuthor:
    account_id: int
    platform: str
    username: str | None
    title: str
    url: str | None
    category_id: int | str | None
    category_path: str | None
    static_avg_er: float | None
    raw_metadata: dict | None
    explanation: str | None
    subscribers_count: int | None
    max_vector_score: float
    max_graph_score: float
    most_recent_post_date: datetime | None
    has_contacts: bool


class SearchRetriever:

    def __init__(self, session: AsyncSession, qdrant_service: Any, graph_search_repo: Any) -> None:
        self._session = session
        self._qdrant_service = qdrant_service
        self._graph_search_repo = graph_search_repo

    async def retrieve_candidates(
        self, request: SearchRequest, reformulated: ReformulatedQuery
    ) -> tuple[list[CandidateAuthor], dict[str, float]]:
        timings: dict[str, float] = {}

        qdrant_limit = max(1500, request.limit * 10)
        age_limit = min(600, max(200, int(request.limit * 2.5)))

        async def _qdrant_task() -> tuple[dict[int, float], float, float]:
            embedding_start = time.perf_counter()
            dense_list, _ = await self._qdrant_service._generate_cloud_embeddings_batch([reformulated.dense_query])
            dense_emb = dense_list[0]
            embedding_ms = (time.perf_counter() - embedding_start) * 1000.0

            if not self._qdrant_service._initialized:
                await self._qdrant_service.initialize()

            qdrant_start = time.perf_counter()
            qdrant_response = await self._qdrant_service.client.query_points(
                collection_name=self._qdrant_service.collection_name,
                query=dense_emb,
                using="text",
                limit=qdrant_limit,
                score_threshold=0.30,
                with_payload=True,
            )
            qdrant_ms = (time.perf_counter() - qdrant_start) * 1000.0

            qdrant_map: dict[int, float] = {}
            for hit in qdrant_response.points:
                try:
                    pid = int(hit.id)
                    qdrant_map[pid] = float(hit.score)
                except (ValueError, TypeError):
                    continue

            return qdrant_map, embedding_ms, qdrant_ms

        async def _graph_task() -> tuple[dict[int, float], float]:
            graph_start = time.perf_counter()
            age_map = await self._graph_search_repo.search_posts_by_entities(
                entities=reformulated.graph_entities,
                author_type=request.author_type,
                limit=age_limit,
            )
            graph_ms = (time.perf_counter() - graph_start) * 1000.0
            return age_map, graph_ms

        (qdrant_map, embedding_ms, qdrant_ms), (age_map, graph_ms) = await asyncio.gather(
            _qdrant_task(),
            _graph_task(),
        )

        timings["embedding_ms"] = embedding_ms
        timings["qdrant_posts_ms"] = qdrant_ms
        timings["graph_index_ms"] = graph_ms

        all_post_ids = set(qdrant_map.keys()) | set(age_map.keys())

        logger.info("Qdrant post search returned %d candidates", len(qdrant_map))
        logger.info("Graph index search returned %d candidates", len(age_map))
        logger.info("Total unique post IDs after merge: %d", len(all_post_ids))

        if not all_post_ids:
            return [], timings

        query = (
            select(Content, Account)
            .join(Account, Content.account_id == Account.id)
            .where(Content.id.in_(all_post_ids))
            .where(Content.is_enriched == True)
        )

        if request.author_type == "expert":
            query = query.where(Account.is_author_blog == True)
        elif request.author_type == "business":
            query = query.where(Account.is_author_blog == False)

        if request.min_followers is not None:
            query = query.where(Account.subscribers_count >= request.min_followers)

        postgres_start = time.perf_counter()
        result = await self._session.execute(query)
        rows = result.fetchall()
        timings["postgres_ms"] = (time.perf_counter() - postgres_start) * 1000.0

        account_groups: dict[int, dict[str, Any]] = {}

        for content, account in rows:
            aid = account.id

            if aid not in account_groups:
                account_groups[aid] = {
                    "account": account,
                    "max_vector_score": 0.0,
                    "max_graph_score": 0.0,
                    "most_recent_post_date": None,
                }

            group = account_groups[aid]

            vector_score = qdrant_map.get(content.id, 0.0)
            if vector_score > group["max_vector_score"]:
                group["max_vector_score"] = vector_score

            graph_score = age_map.get(content.id, 0.0)
            if graph_score > group["max_graph_score"]:
                group["max_graph_score"] = graph_score

            if content.published_at is not None:
                if group["most_recent_post_date"] is None or content.published_at > group["most_recent_post_date"]:
                    group["most_recent_post_date"] = content.published_at

        candidates: list[CandidateAuthor] = []

        for aid, group in account_groups.items():
            account: Account = group["account"]
            raw_meta = account.raw_metadata or {}
            url = raw_meta.get("url") or raw_meta.get("link")

            contacts_nested = raw_meta.get("contacts") or {}
            raw_payload = raw_meta.get("raw_profile_payload") or {}
            has_contacts = any(
                bool(contacts_nested.get(k)) for k in [
                    "emails", "phones", "telegram_handles", "telegram_channels",
                    "telegram_personal", "advertising_emails", "advertising_telegrams",
                ]
            ) or bool(raw_payload.get("business_email")) or any(
                bool(raw_meta.get(k)) for k in [
                    "emails", "phones", "telegram_handles", "business_email",
                ]
            )

            candidates.append(CandidateAuthor(
                account_id=aid,
                platform=account.platform,
                username=account.username,
                title=account.title,
                url=url,
                category_id=account.category_id,
                category_path=account.category_path,
                static_avg_er=account.static_avg_er,
                raw_metadata=account.raw_metadata,
                explanation=account.explanation,
                subscribers_count=account.subscribers_count,
                max_vector_score=group["max_vector_score"],
                max_graph_score=group["max_graph_score"],
                most_recent_post_date=group["most_recent_post_date"],
                has_contacts=has_contacts,
            ))

        return candidates, timings