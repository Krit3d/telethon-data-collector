import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import ReformulatedQuery, SearchRequest
from src.db.models import Account, Content
from src.graph.db.search_repo import GraphSearchRepository

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

    def __init__(self, session: AsyncSession, qdrant_service: Any, graph_repo: GraphSearchRepository) -> None:
        self._session = session
        self._qdrant_service = qdrant_service
        self._graph_repo = graph_repo

    async def retrieve_candidates(self, request: SearchRequest, reformulated: ReformulatedQuery) -> list[CandidateAuthor]:
        qdrant_limit = max(1500, request.limit * 10)
        age_limit = min(600, max(200, int(request.limit * 2.5)))

        qdrant_task = self._qdrant_service.search_posts(
            dense_query=reformulated.dense_query, limit=qdrant_limit, score_threshold=0.35
        )
        graph_task = self._graph_repo.search_posts_by_entities(
            entities=reformulated.graph_entities,
            author_type=request.author_type,
            limit=age_limit,
        )

        qdrant_map: dict[int, float] = {}
        age_map: dict[int, float] = {}

        try:
            qdrant_results, graph_results = await asyncio.gather(qdrant_task, graph_task, return_exceptions=True)
        except Exception:
            logger.error("asyncio.gather failed for candidate retrieval, falling back to empty results")
            qdrant_results = []
            graph_results = {}

        if isinstance(qdrant_results, BaseException):
            logger.error("Qdrant search failed: %s", qdrant_results)
            qdrant_results = []
        if isinstance(graph_results, BaseException):
            logger.error("Graph search failed: %s", graph_results)
            await self._session.rollback()
            graph_results = {}

        if isinstance(qdrant_results, list):
            for r in qdrant_results:
                try:
                    pid = int(r["post_id"])
                    qdrant_map[pid] = float(r["score"])
                except (KeyError, ValueError, TypeError):
                    continue

        if isinstance(graph_results, dict):
            age_map = graph_results

        all_post_ids = set(qdrant_map.keys()) | set(age_map.keys())

        logger.info("Qdrant returned %d candidates", len(qdrant_map))
        logger.info("Apache AGE returned %d candidates", len(age_map))
        logger.info("Total unique post IDs after merge: %d", len(all_post_ids))

        if not all_post_ids:
            return []

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

        result = await self._session.execute(query)
        rows = result.fetchall()

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

        return candidates