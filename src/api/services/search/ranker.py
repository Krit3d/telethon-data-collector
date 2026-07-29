import csv
import logging

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


class SearchRanker:

    def __init__(self, ancestors_map: dict[str, list[str]] | None = None) -> None:
        self._ancestors_map = ancestors_map or {}

    def calculate_tms(self, category_id: int | str | None, target_iab_ids: list[int]) -> float:
        if category_id is None or not target_iab_ids:
            return 0.0

        cat_str = str(category_id)
        targets = [str(t) for t in target_iab_ids]

        if cat_str in targets:
            return 1.0

        cand_anc = self._ancestors_map.get(cat_str, [])

        if len(cand_anc) > 0:
            cand_parent = cand_anc[0]
            for t in targets:
                t_anc = self._ancestors_map.get(t, [])
                if len(t_anc) > 0:
                    t_parent = t_anc[0]
                    if cand_parent == t_parent:
                        return 0.75

        cand_root = cand_anc[-1] if cand_anc else cat_str
        for t in targets:
            anc = self._ancestors_map.get(t, [])
            target_root = anc[-1] if anc else t
            if cand_root == target_root:
                return 0.4

        return 0.0

    def rank_and_format(
        self,
        candidates: list[CandidateAuthor],
        request: SearchRequest,
        reformulated: ReformulatedQuery,
        execution_time_ms: float,
    ) -> SearchResponse:
        scored: list[tuple[float, CandidateAuthor, float, dict | None, str | None, str | None]] = []

        for candidate in candidates:
            tms = self.calculate_tms(candidate.category_id, reformulated.target_iab_ids)
            final_score = (0.5 * candidate.max_vector_score) + (0.3 * candidate.max_graph_score) + (0.2 * tms)

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

            if candidate.platform.upper() == "TELEGRAM" and candidate.username:
                url = f"https://t.me/{candidate.username}"
            else:
                url = candidate.url

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

        items: list[AuthorSearchResultItem] = []
        for final_score, candidate, tms, contacts, url, category_path in truncated:
            items.append(
                AuthorSearchResultItem(
                    account_id=candidate.account_id,
                    platform=candidate.platform,
                    username=candidate.username,
                    title=candidate.title,
                    url=url,
                    final_score=final_score,
                    vector_score=candidate.max_vector_score,
                    graph_score=candidate.max_graph_score,
                    tms_score=tms,
                    static_avg_er=candidate.static_avg_er,
                    category_path=category_path,
                    explanation=candidate.explanation,
                    contacts=contacts,
                    has_contacts=candidate.has_contacts,
                    subscribers_count=candidate.subscribers_count,
                )
            )

        query_metadata = QueryMetadata(
            original_query=request.query,
            dense_query=reformulated.dense_query,
            graph_entities=reformulated.graph_entities,
            target_iab_ids=reformulated.target_iab_ids,
            resolved_profile_type=request.author_type,
            execution_time_ms=execution_time_ms,
        )

        return SearchResponse(items=items, total=len(items), query_metadata=query_metadata)