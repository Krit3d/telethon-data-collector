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
from src.embeddings.client import ENTITIES_COLLECTION

logger = logging.getLogger(__name__)


@dataclass
class CandidateAuthor:
    account_id: int
    platform: str
    username: str | None
    title: str
    url: str | None
    category_id: str | None
    category_path: str | None
    static_avg_er: float | None
    raw_metadata: dict[str, Any] | None
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

    @staticmethod
    def _extract_profile_url(raw_meta: dict[str, Any] | None) -> str | None:
        if not raw_meta:
            return None
        for key in ("profile_url", "url", "link"):
            val = raw_meta.get(key)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
        return None

    async def _disambiguate_entities(self, entity_embeddings: list[list[float]]) -> list[str]:
        if not entity_embeddings:
            return []

        if not self._qdrant_service._initialized:
            await self._qdrant_service.initialize()

        async def _resolve(emb: list[float]) -> str | None:
            response = await self._qdrant_service.client.query_points(
                collection_name=ENTITIES_COLLECTION,
                query=emb,
                using="text",
                limit=1,
                score_threshold=0.75,
                with_payload=True,
            )
            if not response.points:
                return None
            hit = response.points[0]
            payload = hit.payload or {}
            canonical = payload.get("id") or str(hit.id)
            return str(canonical).strip() or None

        resolved = await asyncio.gather(*(_resolve(emb) for emb in entity_embeddings))
        return [cid for cid in resolved if cid]

    async def retrieve_candidates(
        self, request: SearchRequest, reformulated: ReformulatedQuery
    ) -> tuple[list[CandidateAuthor], dict[str, float], dict[str, int]]:
        timings: dict[str, float] = {}

        texts = [reformulated.dense_query] + reformulated.graph_entities

        embed_start = time.perf_counter()
        dense_list, _ = await self._qdrant_service._generate_cloud_embeddings_batch(texts)
        timings["embedding_ms"] = (time.perf_counter() - embed_start) * 1000.0

        dense_query_emb = dense_list[0]
        entity_embs = dense_list[1:]

        disamb_start = time.perf_counter()
        canonical_entity_ids = await self._disambiguate_entities(entity_embs)
        timings["entity_disambiguation_ms"] = (time.perf_counter() - disamb_start) * 1000.0

        if not self._qdrant_service._initialized:
            await self._qdrant_service.initialize()

        async def _qdrant_task() -> dict[int, float]:
            qdrant_start = time.perf_counter()
            qdrant_response = await self._qdrant_service.client.query_points(
                collection_name=self._qdrant_service.collection_name,
                query=dense_query_emb,
                using="text",
                limit=1500,
                score_threshold=0.30,
                with_payload=True,
            )
            timings["qdrant_posts_ms"] = (time.perf_counter() - qdrant_start) * 1000.0

            qdrant_map: dict[int, float] = {}
            for hit in qdrant_response.points:
                try:
                    pid = int(hit.id)
                    qdrant_map[pid] = float(hit.score)
                except (ValueError, TypeError):
                    continue
            return qdrant_map

        async def _graph_task() -> dict[int, float]:
            graph_start = time.perf_counter()
            graph_map = await self._graph_search_repo.find_candidate_posts_with_weights(
                canonical_entity_ids,
                reformulated.target_topics,
                limit=600,
            )
            timings["graph_index_ms"] = (time.perf_counter() - graph_start) * 1000.0
            return graph_map

        qdrant_map, graph_map = await asyncio.gather(_qdrant_task(), _graph_task())

        all_post_ids = set(qdrant_map.keys()) | set(graph_map.keys())

        counts = {
            "qdrant_candidates_count": len(qdrant_map),
            "graph_candidates_count": len(graph_map),
            "total_unique_candidates_count": len(all_post_ids),
        }

        logger.info("Qdrant post search returned %d candidates", len(qdrant_map))
        logger.info("Graph index search returned %d candidates", len(graph_map))
        logger.info("Total unique post IDs after merge: %d", len(all_post_ids))

        if not all_post_ids:
            return [], timings, counts

        query = (
            select(Content, Account)
            .join(Account, Content.account_id == Account.id)
            .where(Content.id.in_(all_post_ids))
            .where(Content.is_enriched == True)
            .where(Content.graph_status == 2)
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

        for row in rows:
            content = row.Content
            account = row.Account
            aid = account.id
            post_id = content.id
            published_at = content.published_at

            if aid not in account_groups:
                raw_cat_id = account.category_id
                account_groups[aid] = {
                    "platform": account.platform,
                    "username": account.username,
                    "title": account.title,
                    "category_id": str(raw_cat_id) if raw_cat_id is not None else None,
                    "category_path": account.category_path,
                    "static_avg_er": account.static_avg_er,
                    "raw_metadata": account.raw_metadata,
                    "explanation": account.explanation,
                    "subscribers_count": account.subscribers_count,
                    "max_vector_score": 0.0,
                    "max_graph_score": 0.0,
                    "most_recent_post_date": None,
                }

            group = account_groups[aid]

            vector_score = qdrant_map.get(post_id, 0.0)
            if vector_score > group["max_vector_score"]:
                group["max_vector_score"] = vector_score

            graph_score = graph_map.get(post_id, 0.0)
            if graph_score > group["max_graph_score"]:
                group["max_graph_score"] = graph_score

            if published_at is not None:
                if group["most_recent_post_date"] is None or published_at > group["most_recent_post_date"]:
                    group["most_recent_post_date"] = published_at

        candidates: list[CandidateAuthor] = []

        for aid, group in account_groups.items():
            raw_meta = group["raw_metadata"] or {}
            url = self._extract_profile_url(group["raw_metadata"])

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
                platform=group["platform"],
                username=group["username"],
                title=group["title"],
                url=url,
                category_id=group["category_id"],
                category_path=group["category_path"],
                static_avg_er=group["static_avg_er"],
                raw_metadata=group["raw_metadata"],
                explanation=group["explanation"],
                subscribers_count=group["subscribers_count"],
                max_vector_score=group["max_vector_score"],
                max_graph_score=group["max_graph_score"],
                most_recent_post_date=group["most_recent_post_date"],
                has_contacts=has_contacts,
            ))

        return candidates, timings, counts