import csv
import logging
import re
from typing import Any

from src.api.schemas import AuthorSearchResultItem, QueryMetadata, ReformulatedQuery, SearchRequest, SearchResponse
from src.api.services.search.retriever import CandidateAuthor

logger = logging.getLogger(__name__)


class TaxonomyLoader:

    def load_ancestors_map(self, tsv_path: str) -> dict[str, list[str]]:
        try:
            with open(tsv_path, encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                rows = list(reader)
        except FileNotFoundError:
            logger.warning("Taxonomy TSV not found at %s, returning empty ancestors map", tsv_path)
            return {}
        except Exception:
            logger.warning("Failed to read taxonomy TSV at %s, returning empty ancestors map", tsv_path)
            return {}

        if len(rows) < 3:
            logger.warning("Taxonomy TSV has insufficient rows (%d), returning empty ancestors map", len(rows))
            return {}

        data_rows = rows[2:]

        parent_of: dict[str, str] = {}
        all_ids: set[str] = set()

        for row in data_rows:
            if len(row) < 2:
                continue
            raw_id = row[0].strip()
            raw_parent = row[1].strip()
            if not raw_id:
                continue
            all_ids.add(raw_id)
            if raw_parent:
                parent_of[raw_id] = raw_parent

        ancestors_map: dict[str, list[str]] = {}

        for cid in all_ids:
            chain: list[str] = []
            current = cid
            while current in parent_of:
                parent = parent_of[current]
                chain.append(parent)
                current = parent
            if chain:
                ancestors_map[cid] = chain

        return ancestors_map

    def load_name_to_id_map(self, tsv_path: str) -> dict[str, str]:
        try:
            with open(tsv_path, encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                rows = list(reader)
        except FileNotFoundError:
            logger.warning("Taxonomy TSV not found at %s, returning empty name-to-id map", tsv_path)
            return {}
        except Exception:
            logger.warning("Failed to read taxonomy TSV at %s, returning empty name-to-id map", tsv_path)
            return {}

        if len(rows) < 3:
            logger.warning("Taxonomy TSV has insufficient rows (%d), returning empty name-to-id map", len(rows))
            return {}

        data_rows = rows[2:]
        name_to_id: dict[str, str] = {}

        for row in data_rows:
            if len(row) < 3:
                continue
            raw_id = row[0].strip()
            raw_name = row[2].strip()
            if not raw_id or not raw_name:
                continue
            name_to_id[raw_name.lower()] = raw_id

        return name_to_id


class SearchRanker:

    _SAFETY_PATTERN = re.compile(
        r"\b(1xbet|1win|casino|казино|вулкан|ставки\s+на\с+спорт|adult|18\+|порно|slots|слоты|scam|скам|криптосигналы|crypto\s*signals|betting|bet|online\s*casino|gambling|азартные\s+игры)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _scale_vector_score(raw_score: float) -> float:
        if raw_score <= 0.0:
            return 0.0
        import math
        scaled = 1.0 / (1.0 + math.exp(-10.0 * (raw_score - 0.45)))
        return max(0.0, min(1.0, scaled))

    @staticmethod
    def _normalize_graph_score(weight: float) -> float:
        if weight <= 0.0:
            return 0.0
        import math
        return 1.0 / (1.0 + math.exp(-1.5 * (weight - 1.5)))

    def __init__(
        self,
        ancestors_map: dict[str, list[str]] | None = None,
        name_to_id_map: dict[str, str] | None = None,
    ) -> None:
        self._ancestors_map = ancestors_map or {}
        self._name_to_id_map = name_to_id_map or {}

    def resolve_target_iab_ids(self, target_topics: list[str]) -> list[str]:
        if not target_topics or not self._name_to_id_map:
            return []

        resolved: set[str] = set()

        for topic in target_topics:
            topic_lower = topic.lower().strip()
            if not topic_lower:
                continue

            exact_id = self._name_to_id_map.get(topic_lower)
            if exact_id is not None:
                resolved.add(exact_id)
                continue

            for name, cid in self._name_to_id_map.items():
                if topic_lower in name or name in topic_lower:
                    resolved.add(cid)

        return list(resolved)

    def calculate_tms(self, category_id: str | None, target_iab_ids: list[str]) -> float:
        if category_id is None:
            return 0.10

        if not target_iab_ids:
            return 0.10

        if category_id in target_iab_ids:
            return 1.0

        cand_anc = self._ancestors_map.get(category_id, [])

        if len(cand_anc) > 0:
            cand_parent = cand_anc[0]
            for t in target_iab_ids:
                t_anc = self._ancestors_map.get(t, [])
                if len(t_anc) > 0:
                    t_parent = t_anc[0]
                    if cand_parent == t_parent:
                        return 0.75

        cand_root = cand_anc[-1] if cand_anc else category_id
        for t in target_iab_ids:
            anc = self._ancestors_map.get(t, [])
            target_root = anc[-1] if anc else t
            if cand_root == target_root:
                return 0.50

        return 0.0

    def _calculate_engagement_score(self, static_avg_er: float | None) -> float:
        er = static_avg_er if static_avg_er is not None else 0.0
        return er / (20.0 + er)

    def _calculate_topical_score(self, max_vector_score: float, max_graph_score: float, tms_score: float) -> float:
        scaled_vector = self._scale_vector_score(max_vector_score)
        normalized_graph = self._normalize_graph_score(max_graph_score)

        if tms_score > 0.0:
            effective_graph = normalized_graph
        else:
            effective_graph = normalized_graph * 0.15

        if effective_graph > 0.0:
            topical_score = (0.50 * scaled_vector) + (0.30 * effective_graph) + (0.20 * tms_score)
        else:
            topical_score = (0.75 * scaled_vector) + (0.25 * tms_score)

        return min(1.0, topical_score)

    def _calculate_final_score(self, topical_score: float, engagement_score: float) -> float:
        return (0.88 * topical_score) + (0.12 * engagement_score)

    def _is_safe(self, candidate: CandidateAuthor) -> bool:
        fields_to_check: list[str] = []
        if candidate.title:
            fields_to_check.append(candidate.title)
        if candidate.explanation:
            fields_to_check.append(candidate.explanation)
        for field in fields_to_check:
            if self._SAFETY_PATTERN.search(field):
                return False
        return True

    def rank_and_format(
        self,
        candidates: list[CandidateAuthor],
        request: SearchRequest,
        reformulated: ReformulatedQuery,
        execution_time_ms: float = 0.0,
        timings: dict[str, float] | None = None,
        counts: dict[str, int] | None = None,
    ) -> SearchResponse:
        if not reformulated.target_iab_ids:
            reformulated.target_iab_ids = self.resolve_target_iab_ids(reformulated.target_topics)

        safe_candidates: list[CandidateAuthor] = []
        for candidate in candidates:
            if self._is_safe(candidate):
                safe_candidates.append(candidate)
            else:
                logger.info("Discarded candidate %d due to safety filter", candidate.account_id)

        scored: list[tuple[float, CandidateAuthor, float, dict[str, Any] | None, str | None, str | None]] = []

        for candidate in safe_candidates:
            tms = self.calculate_tms(candidate.category_id, reformulated.target_iab_ids)

            engagement_score = self._calculate_engagement_score(candidate.static_avg_er)
            topical_score = self._calculate_topical_score(candidate.max_vector_score, candidate.max_graph_score, tms)

            final_score = self._calculate_final_score(topical_score, engagement_score)

            if request.include_contacts and candidate.raw_metadata:
                contacts = candidate.raw_metadata.get("contacts") or {
                    "emails": candidate.raw_metadata.get("emails"),
                    "phones": candidate.raw_metadata.get("phones"),
                    "telegram_handles": candidate.raw_metadata.get("telegram_handles"),
                    "telegram_channels": candidate.raw_metadata.get("telegram_channels"),
                    "telegram_personal": candidate.raw_metadata.get("telegram_personal"),
                    "advertising_emails": candidate.raw_metadata.get("advertising_emails"),
                    "advertising_telegrams": candidate.raw_metadata.get("advertising_telegrams"),
                }
            else:
                contacts = None

            url = candidate.url.strip() if candidate.url and candidate.url.strip() else None

            if candidate.category_path:
                category_path = candidate.category_path
            elif candidate.category_id is not None:
                cat_str = str(candidate.category_id)
                anc = self._ancestors_map.get(cat_str, [])
                if anc:
                    category_path = " > ".join(reversed(anc))
                else:
                    category_path = None
            else:
                category_path = None

            scored.append((final_score, candidate, tms, contacts, url, category_path))

        scored.sort(key=lambda x: x[0], reverse=True)

        truncated = scored[: request.limit]

        top_max_vector = max((cand.max_vector_score for _, cand, _, _, _, _ in truncated), default=0.0)
        top_max_graph = max((cand.max_graph_score for _, cand, _, _, _, _ in truncated), default=0.0)

        items: list[AuthorSearchResultItem] = []
        for final_score, candidate, tms, contacts, url, category_path in truncated:
            if request.include_analytics:
                vector_score = candidate.max_vector_score
                graph_score = candidate.max_graph_score
                tms_score = tms
                static_avg_er = candidate.static_avg_er
            else:
                vector_score = None
                graph_score = None
                tms_score = None
                static_avg_er = None

            items.append(
                AuthorSearchResultItem(
                    account_id=candidate.account_id,
                    platform=candidate.platform,
                    username=candidate.username,
                    title=candidate.title,
                    url=url,
                    final_score=final_score,
                    vector_score=vector_score,
                    graph_score=graph_score,
                    tms_score=tms_score,
                    static_avg_er=static_avg_er,
                    category_path=category_path,
                    explanation=candidate.explanation,
                    contacts=contacts,
                    has_contacts=candidate.has_contacts,
                    subscribers_count=candidate.subscribers_count,
                )
            )

        if request.include_analytics and counts:
            qdrant_candidates_count = counts.get("qdrant_candidates_count")
            graph_candidates_count = counts.get("graph_candidates_count")
            total_unique_candidates_count = counts.get("total_unique_candidates_count")
        else:
            qdrant_candidates_count = None
            graph_candidates_count = None
            total_unique_candidates_count = None

        query_metadata = QueryMetadata(
            original_query=request.query,
            dense_query=reformulated.dense_query,
            graph_entities=reformulated.graph_entities,
            target_iab_ids=reformulated.target_iab_ids,
            resolved_profile_type=request.author_type,
            execution_time_ms=execution_time_ms,
            timings=timings or {},
            qdrant_candidates_count=qdrant_candidates_count,
            graph_candidates_count=graph_candidates_count,
            total_unique_candidates_count=total_unique_candidates_count,
        )

        if not truncated or (top_max_graph == 0.0 and top_max_vector < 0.62):
            return SearchResponse(
                items=items,
                total=len(items),
                query_metadata=query_metadata,
                message=None,
                confidence_level="NONE",
                warning_message="No relevant authors found matching your query topic in the current database. Showing closest available profiles.",
            )

        return SearchResponse(
            items=items,
            total=len(items),
            query_metadata=query_metadata,
            message=None,
            confidence_level="HIGH",
            warning_message=None,
        )
