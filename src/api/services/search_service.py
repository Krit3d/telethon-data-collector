import asyncio
import csv
import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import (
    APIError,
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from pydantic import ValidationError

from src.api.schemas import (
    AuthorSearchResultItem,
    ReformulatedQuery,
    SearchRequest,
    SearchResponse,
)
from src.config.config import Settings
from src.db.database import Database
from src.embeddings.qdrant_service import QdrantService
from src.graph.db.search_repo import GraphSearchRepository

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "with", "by", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "from", "up", "down", "of", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now"
}

_RU_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к", "ко", "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от", "о", "из", "ему", "им", "уже", "когда", "быть", "был", "него", "до", "вас", "тоже", "себя", "под", "жизнь", "надо", "без", "если", "хочешь", "будет", "свое"
}

_UNINFORMATIVE_WORDS_RU = {
    "бизнес", "дело", "проект", "компания", "работа", "услуги", "продукт", "бренд", "блог", "канал", "автор", "человек", "нужен", "ищу", "что-то"
}

_SPECIFICITY_EXCLUDED = {
    "адвокат", "юрист", "помощь", "услуги", "бизнес",
}

logger = logging.getLogger(__name__)

_DEFAULT_EXPLANATION = (
    "Этот автор публикует контент, полностью соответствующий тематике вашего проекта, "
    "и имеет активную аудиторию."
)
_FALLBACK_GRAPH_ENTITIES: list[str] = []

_NEGATIVE_SIGNALS_PATTERN = re.compile(
    r"\b(1xbet|1win|casino|казино|вулкан|ставки\s+на\с+спорт|adult|18\+|порно|slots|слоты|scam|скам|криптосигналы|crypto\s*signals|betting|bet|online\s*casino|gambling|азартные\s+игры)\b",
    re.IGNORECASE,
)

_INSTAGRAM_MEDIA_ID_PATTERN = re.compile(r'^\d+_\d+$')

_DORMANT_WARNING_RU = (
    "Внимание: автор не публиковал новый контент более 180 дней, но включён в выдачу как редкий эксперт в данной нише."
)

def _has_negative_signals(text: str | None) -> bool:
    if not text:
        return False
    return _NEGATIVE_SIGNALS_PATTERN.search(text) is not None

def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    intersection = set_a & set_b
    return len(intersection) / len(union)

def _calibrate_bge_m3_score(raw_score: float) -> float:
    if raw_score <= 0.20:
        return 0.0
    if raw_score >= 0.75:
        return 1.0
    return (raw_score - 0.20) / (0.75 - 0.20)


def _tokenize_entity_name(name: str) -> list[str]:
    tokens: list[str] = []
    parts = re.split(r'[|,\-]', name)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        words = part.split()
        for w in words:
            w_clean = w.strip().lower()
            if len(w_clean) >= 3:
                tokens.append(w_clean)
        if len(words) >= 2:
            for i in range(len(words) - 1):
                w1 = words[i].strip().lower()
                w2 = words[i + 1].strip().lower()
                if len(w1) >= 3 and len(w2) >= 3:
                    tokens.append(f"{w1} {w2}")
    return tokens


class SearchService:
    def __init__(
        self,
        settings: Settings,
        qdrant: QdrantService,
        db: Database,
        graph_search_repo: GraphSearchRepository,
    ) -> None:
        self._settings = settings
        self._qdrant = qdrant
        self._db = db
        self._graph_search_repo = graph_search_repo
        if not settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY must be configured for search service")
        self._llm_client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=30.0,
        )
        self._llm_model = settings.deepseek_llm_model

        taxonomy_path = Path(__file__).resolve().parents[2] / "config" / "Content Taxonomy 3.1.tsv"
        self._cached_yaml_taxonomy, self._taxonomy_dict = self._parse_taxonomy(taxonomy_path)

    def _parse_taxonomy(self, path: Path) -> tuple[str, dict[str, dict[str, str | None]]]:
        if not path.exists():
            logger.warning("Taxonomy file not found at %s", path)
            return ("", {})

        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            all_rows = list(reader)

        for row in all_rows:
            if not row or not row[0].strip() or not row[0].strip().isdigit():
                continue
            unique_id = row[0].strip()
            parent_id = row[1].strip() if len(row) > 1 and row[1].strip() else ""
            name = row[2].strip() if len(row) > 2 else ""
            extension = row[7].strip() if len(row) > 7 else ""
            if not name:
                continue
            rows.append({
                "unique_id": unique_id,
                "parent_id": parent_id,
                "name": name,
                "extension": extension,
            })

        logger.info("Parsed %d taxonomy entries from %s", len(rows), path)

        tree_dict = self._build_tree(rows)
        yaml_str = self._dict_to_yaml(tree_dict)
        lookup_dict = self._build_lookup(rows)

        logger.info(
            "Built taxonomy YAML (%d chars) and lookup dict with %d entries",
            len(yaml_str),
            len(lookup_dict),
        )

        return yaml_str, lookup_dict

    def _build_tree(self, rows: list[dict[str, str]]) -> dict[str, dict | str]:
        node_map: dict[str, dict] = {}
        children_map: dict[str, list[str]] = {}

        for row in rows:
            uid = row["unique_id"]
            node_map[uid] = {"name": row["name"], "parent_id": row["parent_id"]}
            children_map[uid] = []

        for uid, node in node_map.items():
            parent_id = node["parent_id"]
            if parent_id and parent_id in children_map:
                children_map[parent_id].append(uid)

        def _to_dict(uid: str) -> dict | str:
            children = children_map.get(uid, [])
            if not children:
                return node_map[uid]["name"]
            result: dict[str, dict | str] = {}
            for child_uid in sorted(children, key=lambda x: int(x) if x.isdigit() else x):
                result[child_uid] = _to_dict(child_uid)
            return result

        tree: dict[str, dict | str] = {}
        for uid, node in node_map.items():
            if not node["parent_id"]:
                tree[uid] = _to_dict(uid)

        return tree

    def _dict_to_yaml(self, d: dict[str, dict | str], indent: int = 0) -> str:
        lines: list[str] = []
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, str):
                lines.append(f'{prefix}{key}: {value}')
            elif isinstance(value, dict):
                lines.append(f'{prefix}{key}:')
                lines.append(self._dict_to_yaml(value, indent + 1))
        return "\n".join(lines)

    def _build_lookup(self, rows: list[dict[str, str]]) -> dict[str, dict[str, str | None]]:
        node_map: dict[str, dict[str, str]] = {}
        for row in rows:
            node_map[row["unique_id"]] = {
                "name": row["name"],
                "parent_id": row["parent_id"],
                "extension": row.get("extension", ""),
            }

        def _build_path(uid: str) -> str:
            parts: list[str] = []
            current = uid
            while current and current in node_map:
                parts.insert(0, node_map[current]["name"])
                current = node_map[current]["parent_id"]
            return " > ".join(parts)

        lookup: dict[str, dict[str, str | None]] = {}
        for uid in node_map:
            ext = node_map[uid]["extension"]
            lookup[uid] = {
                "path": _build_path(uid),
                "extension": None if ext.strip() == "" else ext,
            }

        return lookup

    def _calculate_taxonomy_match_score(self, target_iab_ids: list[str], author_category_id: str | int | None) -> float:
        if author_category_id is None or author_category_id == "" or not target_iab_ids:
            return 0.0
        author_cat_str = str(author_category_id).strip()
        normalized_targets = [tid.strip() for tid in target_iab_ids]
        if author_cat_str in normalized_targets:
            return 1.0
        author_entry = self._taxonomy_dict.get(author_cat_str)
        if author_entry is None:
            return 0.0
        author_path = author_entry["path"]
        if not author_path:
            return 0.0
        author_parts = author_path.split(" > ")
        for target_id in normalized_targets:
            target_entry = self._taxonomy_dict.get(target_id)
            if target_entry is None:
                continue
            target_path = target_entry["path"]
            if not target_path:
                continue
            target_parts = target_path.split(" > ")
            min_len = min(len(author_parts), len(target_parts))
            common_prefix_len = 0
            for i in range(min_len):
                if author_parts[i] == target_parts[i]:
                    common_prefix_len += 1
                else:
                    break
            if common_prefix_len >= 1:
                if common_prefix_len == min_len or common_prefix_len >= 2:
                    return 0.5
        return 0.0

    async def _call_llm(
        self, messages: list[dict[str, Any]], temperature: float = 0.2,
    ) -> str:
        response = await self._llm_client.chat.completions.create(
            model=self._llm_model,
            messages=messages,  # type: ignore
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def _reformulate_query(self, raw_query: str) -> ReformulatedQuery:
        taxonomy_block = self._cached_yaml_taxonomy
        system_prompt = (
            "You are an expert search query reformulator and IAB taxonomy classifier. "
            "Analyze the user project description and output JSON with:\n"
            "- dense_query: Expanded semantic search string in Russian.\n"
            "- lexical_queries: 3-5 specific domain keywords.\n"
            "- graph_entities: 3-5 highly specific niche terms/entities in Russian. "
            "NEVER include generic uninformative words (e.g. 'адвокат', 'юрист', 'бизнес', 'компания', 'услуги', 'помощь').\n"
            "- target_iab_ids: list of 1 to 5 exact IAB Category UIDs from the taxonomy below that best match the project intent.\n"
            "- profile_type_intent: 'expert', 'business', or 'both'.\n\n"
            "IAB Taxonomy:\n"
            f"{taxonomy_block}\n\n"
            '{"dense_query": "string", "lexical_queries": ["string"], "graph_entities": ["string"], "target_iab_ids": ["string"], "profile_type_intent": "expert|business|both"}'
        )

        try:
            response = await self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Project description: {raw_query}"},
                ],
                extra_body={"thinking": {"type": "disabled"}},
                response_format={"type": "json_object"},
                max_tokens=300,
                temperature=0.1,
            )
            content = (response.choices[0].message.content or "").strip()
            parsed = ReformulatedQuery.model_validate_json(content)
            if not parsed.dense_query.strip():
                parsed.dense_query = raw_query.strip()
            if parsed.profile_type_intent not in ("expert", "business", "both"):
                parsed.profile_type_intent = "both"
            validated_ids: list[str] = []
            for uid in parsed.target_iab_ids:
                stripped = uid.strip()
                if stripped in self._taxonomy_dict:
                    validated_ids.append(stripped)
            parsed.target_iab_ids = validated_ids
            logger.info(
                "LLM reformulated query: dense_query=%r target_iab_ids=%s profile_type_intent=%s",
                parsed.dense_query,
                parsed.target_iab_ids,
                parsed.profile_type_intent,
            )
            return parsed
        except (APIConnectionError, APITimeoutError, RateLimitError, APIError):
            logger.warning("LLM query reformulation failed due to API error, using original query", exc_info=True)
            return ReformulatedQuery(
                dense_query=raw_query.strip(),
                lexical_queries=[],
                graph_entities=[],
                target_iab_ids=[],
                profile_type_intent="both",
            )
        except (json.JSONDecodeError, ValidationError):
            logger.warning("LLM query reformulation returned invalid JSON, using original query", exc_info=True)
            return ReformulatedQuery(
                dense_query=raw_query.strip(),
                lexical_queries=[],
                graph_entities=[],
                target_iab_ids=[],
                profile_type_intent="both",
            )

    def _apply_flat_mmr(self, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        candidates = [c for c in candidates if c.get("final_score", 0.0) > 0.0]
        candidates.sort(key=lambda x: x["final_score"], reverse=True)

        selected: list[dict[str, Any]] = []
        remaining = list(candidates)

        for _ in range(min(limit, len(remaining))):
            if len(selected) >= limit:
                break
            best_candidate = None
            best_mmr_score = -float("inf")
            for candidate in remaining:
                candidate_entities = candidate.get("matched_entities", set())
                max_similarity = 0.0
                for s in selected:
                    selected_entities = s.get("matched_entities", set())
                    sim = _jaccard_similarity(candidate_entities, selected_entities)
                    max_similarity = max(max_similarity, sim)
                mmr_score = 0.7 * candidate["final_score"] - 0.3 * max_similarity
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = candidate
            if best_candidate is None:
                break
            selected.append(best_candidate)
            remaining.remove(best_candidate)

        return selected

    async def execute_search(self, payload: SearchRequest) -> SearchResponse:
        query = payload.query.strip()
        query_lower = query.lower()
        words = query_lower.split()
        cleaned_words = [re.sub(r'[^\w\s]', '', w) for w in words if re.sub(r'[^\w\s]', '', w)]
        all_stopwords = _STOPWORDS | _RU_STOPWORDS | _UNINFORMATIVE_WORDS_RU
        meaningful_words = [w for w in cleaned_words if w not in all_stopwords]

        if not meaningful_words or (len(cleaned_words) <= 2 and not meaningful_words):
            return SearchResponse(
                results=[],
                message=(
                    "Запрос слишком короткий или не содержит конкретных ключевых слов (например, 'бизнес' или 'дело'). "
                    "Пожалуйста, опишите ваш проект подробнее, указав конкретную нишу или сферу деятельности."
                ),
            )

        if len(cleaned_words) == 1:
            if len(cleaned_words[0]) < 3:
                return SearchResponse(
                    results=[],
                    message=(
                        "Запрос слишком короткий. "
                        "Пожалуйста, введите слово из 3 или более символов."
                    ),
                )
        elif len(meaningful_words) == 0:
            return SearchResponse(
                results=[],
                message=(
                    "Запрос слишком короткий или не содержит значимых слов. "
                    "Пожалуйста, опишите ваш проект подробнее (например, сферу деятельности, цели или целевую аудиторию)."
                ),
            )

        reformulated = await self._reformulate_query(query)
        vector_query = reformulated.dense_query
        combined_query = f"{reformulated.dense_query} {' '.join(reformulated.lexical_queries)}".strip()
        graph_entities = reformulated.graph_entities
        logger.info(
            "Query reformulation complete. vector_query=%r target_iab_ids=%s profile_type_intent=%s graph_entities=%s",
            vector_query,
            reformulated.target_iab_ids,
            reformulated.profile_type_intent,
            graph_entities[:10],
        )

        posts_fetch_limit = 150
        entities_fetch_limit = 100

        topic_post_weights: dict[int, float] = {}
        posts_data: list[dict[str, Any]] = []
        entities_data: list[dict[str, Any]] = []

        if graph_entities:
            qdrant_task = asyncio.create_task(
                self._fetch_qdrant_data(
                    combined_query, posts_fetch_limit, entities_fetch_limit, payload,
                )
            )
            expanded_topics = list(graph_entities)
            for phrase in graph_entities:
                expanded_topics.extend(_tokenize_entity_name(phrase))
            topic_task = asyncio.create_task(
                self._graph_search_repo.search_posts_by_topics(expanded_topics)
            )
            qdrant_result, topic_result = await asyncio.gather(
                qdrant_task, topic_task, return_exceptions=True,
            )
            if isinstance(qdrant_result, BaseException):
                logger.warning("Qdrant fetch failed", exc_info=True)
                posts_data = []
                entities_data = []
            else:
                posts_data, entities_data = qdrant_result
            if isinstance(topic_result, BaseException):
                logger.warning("Graph topic search failed", exc_info=True)
                topic_post_weights = {}
            else:
                topic_post_weights = topic_result if isinstance(topic_result, dict) else {}
        else:
            posts_data, entities_data = await self._fetch_qdrant_data(
                combined_query, posts_fetch_limit, entities_fetch_limit, payload,
            )

        logger.info(
            "Qdrant fetch complete. posts_data length=%d entities_data length=%d topic_post_weights=%d",
            len(posts_data),
            len(entities_data),
            len(topic_post_weights),
        )

        logger.info("Qdrant entities count: %d", len(entities_data))
        for i, e in enumerate(entities_data[:5]):
            ent_payload = e.get("payload", {}) or {}
            logger.info(
                "Qdrant entity %d: id=%s entity_id=%s name=%s name_lower=%s",
                i,
                e.get("id"),
                ent_payload.get("entity_id"),
                ent_payload.get("name"),
                ent_payload.get("name_lower"),
            )

        entity_id_to_score: dict[str, float] = {}
        entity_ids_with_scores: list[tuple[str, float]] = []
        all_tokens: set[str] = set()
        for e in entities_data:
            raw_score = float(e.get("score", 0.0))
            calibrated_score = _calibrate_bge_m3_score(raw_score)
            ent_payload = e.get("payload", {}) or {}
            item_id = str(e.get("id", "") or "")
            payload_id = str(ent_payload.get("id", "") or "")
            payload_entity_id = str(ent_payload.get("entity_id", "") or "")
            name = str(ent_payload.get("name", "") or "")
            name_lower = str(ent_payload.get("name_lower", "") or "")
            seen: set[str] = set()
            for identifier in (item_id, payload_id, payload_entity_id, name, name_lower):
                normalized = identifier.strip().lower()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                if _INSTAGRAM_MEDIA_ID_PATTERN.match(normalized):
                    continue
                if normalized not in entity_id_to_score:
                    entity_id_to_score[normalized] = calibrated_score
                    entity_ids_with_scores.append((normalized, calibrated_score))
                all_tokens.add(normalized)
            if name_lower:
                for token in _tokenize_entity_name(name_lower):
                    if token not in all_tokens:
                        all_tokens.add(token)
                        if token not in entity_id_to_score:
                            entity_id_to_score[token] = calibrated_score
                            entity_ids_with_scores.append((token, calibrated_score))

        entity_ids_with_scores.sort(key=lambda x: x[1], reverse=True)
        top_entity_ids = [eid for eid, _ in entity_ids_with_scores[:150]]
        for ge in graph_entities:
            ge_lower = ge.strip().lower()
            if ge_lower and ge_lower not in all_tokens:
                all_tokens.add(ge_lower)
                top_entity_ids.append(ge_lower)
        seen_top: set[str] = set()
        deduped_top: list[str] = []
        for eid in top_entity_ids:
            if eid not in seen_top:
                seen_top.add(eid)
                deduped_top.append(eid)
        top_entity_ids = deduped_top[:150]

        logger.info("top_entity_ids passed to graph search: %s", top_entity_ids)

        graph_post_entity_weights: dict[int, list[tuple[str, str, float]]] = {}

        if top_entity_ids:
            try:
                graph_post_entity_weights = (
                    await self._graph_search_repo.search_posts_by_entities(
                        top_entity_ids
                    )
                )
                logger.info(
                    "Graph search returned %d matched posts from %d entities",
                    len(graph_post_entity_weights),
                    len(top_entity_ids),
                )
            except Exception:
                logger.warning(
                    "Graph search failed, continuing without graph data", exc_info=True,
                )

        vector_scores: dict[int, float] = {}
        for item in posts_data:
            try:
                post_id = int(item["post_id"])
                vector_scores[post_id] = _calibrate_bge_m3_score(float(item.get("score", 0.0)))
            except (ValueError, KeyError, TypeError):
                continue

        graph_scores: dict[int, float] = {}
        graph_post_entities: dict[int, list[str]] = {}

        doc_freq: dict[str, int] = {}
        for pid, entity_matches in graph_post_entity_weights.items():
            seen_entities: set[str] = set()
            for eid, ename_lower, _ in entity_matches:
                key = str(eid).strip().lower()
                if key:
                    seen_entities.add(key)
                key2 = str(ename_lower).strip().lower()
                if key2:
                    seen_entities.add(key2)
            for entity_key in seen_entities:
                doc_freq[entity_key] = doc_freq.get(entity_key, 0) + 1

        for pid, entity_matches in graph_post_entity_weights.items():
            try:
                pid_int = int(pid)
            except (ValueError, TypeError):
                continue
            entity_ids_for_post = [eid for eid, _, _ in entity_matches]
            graph_post_entities[pid_int] = entity_ids_for_post
            post_entity_scores: list[float] = []
            for eid, ename_lower, rel_count in entity_matches:
                eid_norm = str(eid).strip().lower()
                ename_norm = str(ename_lower).strip().lower()
                score = entity_id_to_score.get(eid_norm) or entity_id_to_score.get(ename_norm) or 0.0
                if score > 0:
                    base_score = score
                else:
                    base_score = 0.30
                idf = 1.0 / math.log2(doc_freq.get(eid_norm, 1) + 1.0)
                edge_factor = min(1.0, 0.70 + 0.30 * (rel_count / 2.0))
                s_i = base_score * idf * edge_factor
                post_entity_scores.append(s_i)
            if post_entity_scores:
                noisy_or = 1.0 - math.prod(1.0 - min(0.90, s) for s in post_entity_scores)
                graph_scores[pid_int] = max(0.0, min(1.0, noisy_or))
            else:
                graph_scores[pid_int] = 0.0

        graph_matched_post_ids = list(graph_scores.keys())
        logger.info(
            "Graph search matched post IDs: %s", graph_matched_post_ids[:50]
        )
        logger.info(
            "Graph scores for matched posts: %s",
            {pid: round(gs, 4) for pid, gs in list(graph_scores.items())[:20]},
        )

        all_post_ids: set[int] = set(vector_scores.keys()) | set(graph_scores.keys()) | set(topic_post_weights.keys())
        has_specific_terms = any(
            term.lower() not in _SPECIFICITY_EXCLUDED and term.lower() not in _UNINFORMATIVE_WORDS_RU
            for term in graph_entities
        ) if graph_entities else False
        for pid, rel_count in topic_post_weights.items():
            topic_score = min(0.50, (rel_count / (rel_count + 3.0)) * 0.50)
            graph_scores[pid] = 1.0 - (1.0 - graph_scores.get(pid, 0.0)) * (1.0 - topic_score)
        safe_post_ids: list[int] = sorted(int(x) for x in all_post_ids)
        logger.info(
            "Unique post IDs in safe_post_ids before DB query: %d",
            len(safe_post_ids),
        )

        if not safe_post_ids:
            return SearchResponse(
                results=[],
                message=(
                    "По вашему запросу не найдено подходящих авторов. "
                    "Попробуйте переформулировать запрос или расширить описание проекта."
                ),
            )

        if payload.author_type == "expert":
            target_is_author_blog = True
        elif payload.author_type == "business":
            target_is_author_blog = False
        else:
            target_is_author_blog = None

        pre_filter_graph_positive = sum(1 for gs in graph_scores.values() if gs > 0)
        logger.info(
            "Posts with graph_score > 0 before DB filtering: %d out of %d graph-scored posts, %d total post IDs",
            pre_filter_graph_positive,
            len(graph_scores),
            len(safe_post_ids),
        )
        logger.info(
            "DB candidate query params: location=%r min_followers=%r is_author_blog=%r",
            payload.location,
            payload.min_followers,
            target_is_author_blog,
        )
        candidates_rows = await self._db.get_search_candidates(
            content_ids=safe_post_ids,
            location=payload.location,
            min_followers=payload.min_followers,
            is_author_blog=target_is_author_blog,
        )
        candidate_ids = {row["id"] for row in candidates_rows}
        post_filter_graph_positive = sum(
            1 for pid, gs in graph_scores.items() if pid in candidate_ids and gs > 0
        )
        logger.info(
            "Raw candidates fetched from PostgreSQL: %d (posts with graph_score > 0 after filtering: %d out of %d)",
            len(candidates_rows),
            post_filter_graph_positive,
            pre_filter_graph_positive,
        )

        graph_matched_ids_in_candidates = set(graph_scores.keys()) & candidate_ids
        missing_graph_ids = set(graph_scores.keys()) - graph_matched_ids_in_candidates
        logger.info(
            "Graph-matched post IDs present in candidates_rows: %d out of %d",
            len(graph_matched_ids_in_candidates),
            len(graph_scores),
        )
        if missing_graph_ids:
            logger.warning(
                "Graph-matched post IDs missing from candidates_rows: %s",
                sorted(missing_graph_ids),
            )

        if not candidates_rows:
            return SearchResponse(
                results=[],
                message=(
                    "По вашему запросу не найдено подходящих авторов. "
                    "Попробуйте переформулировать запрос или расширить описание проекта."
                ),
            )

        author_map: dict[int, dict[str, Any]] = {}
        current_utc = datetime.now(timezone.utc)

        for row in candidates_rows:
            if row.get("is_enriched") is False:
                logger.warning(
                    "Skipping candidate post_id=%s account_id=%s due to is_enriched is False",
                    row.get("id"),
                    row.get("account_id"),
                )
                continue
            post_id = row["id"]
            account_id = row["account_id"]

            if account_id not in author_map:
                author_map[account_id] = {
                    "author_id": account_id,
                    "username": row.get("username"),
                    "title": row.get("account_title", ""),
                    "description": row.get("description"),
                    "subscribers_count": row.get("subscribers_count"),
                    "platform": row.get("platform", "TELEGRAM"),
                    "category_path": row.get("category_path"),
                    "category_id": row.get("category_id"),
                    "category_extension": row.get("category_extension"),
                    "is_author_blog": row.get("is_author_blog"),
                    "contacts": None,
                    "posts": [],
                    "vector_scores": [],
                    "graph_scores": [],
                    "static_avg_er": max(0.0, float(row.get("static_avg_er") or 0.0)),
                    "explanation": row.get("explanation") or _DEFAULT_EXPLANATION,
                    "has_contacts": False,
                    "most_recent_post": None,
                    "matched_entities": set(),
                    "is_dormant": False,
                }

            author = author_map[account_id]
            vs = vector_scores.get(post_id, 0.0)
            gs = graph_scores.get(post_id, 0.0)

            published_at = row.get("published_at") or row.get("created_at")
            if published_at is not None:
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
                if author["most_recent_post"] is None or published_at > author["most_recent_post"]:
                    author["most_recent_post"] = published_at

            author["vector_scores"].append(vs)
            author["graph_scores"].append(gs)

            post_entity_ids = graph_post_entities.get(post_id, [])
            author["matched_entities"].update(post_entity_ids)

            author["posts"].append({
                "post_id": post_id,
                "text": row.get("content") or row.get("transcription") or "",
                "published_at": published_at,
            })

            raw_metadata = row.get("raw_metadata")
            if isinstance(raw_metadata, str):
                try:
                    parsed_metadata = json.loads(raw_metadata)
                except (json.JSONDecodeError, TypeError):
                    parsed_metadata = None
            else:
                parsed_metadata = raw_metadata
            if isinstance(parsed_metadata, dict):
                contacts = parsed_metadata.get("contacts")
                if isinstance(contacts, dict):
                    author["contacts"] = contacts
                    if not author["has_contacts"]:
                        for key in ["email", "telegram", "phone"]:
                            if contacts.get(key):
                                author["has_contacts"] = True
                                break

        safe_author_map: dict[int, dict[str, Any]] = {}
        for account_id, author in author_map.items():
            if not author["posts"]:
                continue
            title = author.get("title", "")
            description = author.get("description") or ""
            post_texts = [p["text"] for p in author["posts"]]
            all_text = f"{title} {description} {' '.join(post_texts)}"
            if _has_negative_signals(all_text):
                has_graph_matched = any(gs > 0 for gs in author["graph_scores"])
                logger.info(
                    "Author %d discarded due to negative signals (had_graph_matched_posts=%s)",
                    account_id,
                    has_graph_matched,
                )
                continue
            safe_author_map[account_id] = author
        logger.info(
            "Authors in safe_author_map after negative signals check: %d",
            len(safe_author_map),
        )

        if not safe_author_map:
            return SearchResponse(
                results=[],
                message=(
                    "По вашему запросу не найдено подходящих авторов. "
                    "Попробуйте переформулировать запрос или расширить описание проекта."
                ),
            )

        all_excluded = all_stopwords | _UNINFORMATIVE_WORDS_RU | _SPECIFICITY_EXCLUDED
        key_specific_terms = {w for w in cleaned_words if len(w) >= 4 and w not in all_excluded}
        logger.info(
            "Key specific terms for niche specificity boosting: %s",
            key_specific_terms,
        )

        max_matched_posts = max(
            sum(
                1
                for vs, gs in zip(
                    a["vector_scores"], a["graph_scores"], strict=False
                )
                if vs >= payload.score_threshold or gs >= payload.score_threshold
            )
            for a in safe_author_map.values()
        ) if safe_author_map else 1
        if max_matched_posts < 1:
            max_matched_posts = 1

        ranked_authors: list[dict[str, Any]] = []
        for account_id, author in safe_author_map.items():
            if not author["posts"]:
                continue
            max_vs = max(author["vector_scores"]) if author["vector_scores"] else 0.0
            max_gs = max(author["graph_scores"]) if author["graph_scores"] else 0.0

            relevant_posts_count = sum(
                1
                for vs, gs in zip(
                    author["vector_scores"], author["graph_scores"], strict=False
                )
                if vs >= payload.score_threshold or gs >= payload.score_threshold
            )
            expertise_ratio = relevant_posts_count / max_matched_posts

            tax_score = self._calculate_taxonomy_match_score(reformulated.target_iab_ids, author.get("category_id"))

            base_t_raw = 0.45 * max_vs + 0.35 * max_gs + 0.20 * tax_score
            if max_gs > 0.0:
                graph_boosted_t_raw = 0.70 * max_gs + 0.15 * max_vs + 0.15 * tax_score
                T_raw = max(base_t_raw, graph_boosted_t_raw)
            else:
                T_raw = base_t_raw

            initial_topical_score = T_raw

            if target_is_author_blog is True and author.get("is_author_blog") is False:
                initial_topical_score *= 0.85
            if target_is_author_blog is False and author.get("is_author_blog") is True:
                initial_topical_score *= 0.85

            if initial_topical_score < payload.score_threshold:
                author["final_score"] = 0.0
                author["vector_score"] = max_vs
                author["graph_score"] = max_gs
                author["avg_engagement_rate"] = author["static_avg_er"]
                author["expertise_ratio"] = expertise_ratio
                author["initial_topical_score"] = initial_topical_score
                author["topical_score"] = initial_topical_score
                author["top_post_texts"] = [
                    s["text"][:200] for s in author["posts"][:3]
                ]
                ranked_authors.append(author)
                continue

            decay_factor = 1.0
            is_dormant = False
            if author["most_recent_post"] is not None:
                days_since_last_post = (current_utc - author["most_recent_post"]).days
                if days_since_last_post <= 90:
                    decay_factor = 1.0
                elif days_since_last_post <= 180:
                    decay_factor = math.exp(-0.015 * (days_since_last_post - 90))
                else:
                    is_dormant = True
                    decay_factor = 0.1

            specificity_multiplier = 1.0
            if key_specific_terms:
                title_lower = (author.get("title") or "").lower()
                desc_lower = (author.get("description") or "").lower()
                post_texts_lower = " ".join(
                    p.get("text", "") for p in author["posts"]
                ).lower()
                match_count = 0
                for term in key_specific_terms:
                    if term in title_lower or term in desc_lower or term in post_texts_lower:
                        match_count += 1
                if match_count == 0:
                    specificity_multiplier = 0.50
                elif match_count == 1:
                    specificity_multiplier = 0.90
                elif match_count == 2:
                    specificity_multiplier = 1.05
                else:
                    specificity_multiplier = 1.15

            if max_gs > 0.0 or author["matched_entities"]:
                specificity_multiplier = max(1.0, specificity_multiplier)

            T = min(1.0, max(0.0, T_raw * specificity_multiplier * decay_factor))

            E = author["static_avg_er"] / (20.0 + author["static_avg_er"])

            B = 0.70 * T + 0.15 * E + 0.15 * expertise_ratio

            B_contact = B + (0.02 if author["has_contacts"] else 0.0)

            final_score = 0.92 * min(1.0, math.pow(max(0.0, B_contact), 0.90))

            if max_gs > 0.0:
                logger.info(
                    "Score breakdown for author %d: vector_score=%.4f graph_score=%.4f tax_score=%.4f initial_topical_score=%.4f final_score=%.4f",
                    account_id,
                    max_vs,
                    max_gs,
                    tax_score,
                    initial_topical_score,
                    final_score,
                )

            author["is_dormant"] = is_dormant

            if is_dormant:
                if _DORMANT_WARNING_RU not in author["explanation"]:
                    author["explanation"] += " " + _DORMANT_WARNING_RU

            author["vector_score"] = max_vs
            author["graph_score"] = max_gs
            author["avg_engagement_rate"] = author["static_avg_er"]
            author["expertise_ratio"] = expertise_ratio
            author["initial_topical_score"] = initial_topical_score
            author["final_score"] = final_score
            author["topical_score"] = T
            author["top_post_texts"] = [
                s["text"][:200] for s in author["posts"][:3]
            ]
            ranked_authors.append(author)

        logger.info(
            "Authors in ranked_authors after temporal decay and dormancy rules: %d",
            len(ranked_authors),
        )

        if not ranked_authors:
            return SearchResponse(
                results=[],
                message=(
                    "По вашему запросу не найдено подходящих авторов. "
                    "Попробуйте переформулировать запрос или расширить описание проекта."
                ),
            )

        ranked_authors.sort(key=lambda x: x["final_score"], reverse=True)

        rerank_pool_size = min(60, max(30, payload.limit * 3))
        top_k_candidates = ranked_authors[:min(rerank_pool_size, len(ranked_authors))]

        graph_pool_count = sum(1 for c in top_k_candidates if c.get("graph_score", 0.0) > 0.0)
        logger.info(
            "Candidates before flat MMR: %d (pool size cap: %d)", len(top_k_candidates), rerank_pool_size,
        )
        logger.info("Top pool candidates with graph_score > 0: %d out of %d", graph_pool_count, len(top_k_candidates))

        final_candidates = self._apply_flat_mmr(top_k_candidates, payload.limit)

        logger.info(
            "Candidates after flat MMR: %d",
            len(final_candidates),
        )

        if not final_candidates:
            return SearchResponse(
                results=[],
                message=(
                    "По вашему запросу не найдено подходящих авторов. "
                    "Попробуйте переформулировать запрос или расширить описание проекта."
                ),
            )

        results: list[AuthorSearchResultItem] = []
        for author in final_candidates:
            results.append(
                AuthorSearchResultItem(
                    author_id=str(author["author_id"]),
                    username=author.get("username"),
                    title=author.get("title", ""),
                    description=author.get("description"),
                    subscribers_count=author.get("subscribers_count"),
                    platform=author.get("platform", "TELEGRAM"),
                    final_score=author["final_score"],
                    vector_score=author["vector_score"],
                    graph_score=author["graph_score"],
                    avg_engagement_rate=author["avg_engagement_rate"],
                    explanation=author.get("explanation", _DEFAULT_EXPLANATION),
                    category_path=author.get("category_path"),
                    is_author_blog=author.get("is_author_blog"),
                    contacts=author.get("contacts") if payload.include_contacts else None,
                    category_extension=author.get("category_extension") if payload.include_analytics else None,
                    has_contacts=author.get("has_contacts", False),
                    is_dormant=author.get("is_dormant", False),
                    most_recent_post_at=author["most_recent_post"].isoformat() if author.get("most_recent_post") else None,
                )
            )

        return SearchResponse(results=results)

    async def _fetch_qdrant_data(
        self,
        combined_query: str,
        posts_limit: int,
        entities_limit: int,
        payload: SearchRequest,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        posts_task = self._qdrant.search_posts(
            query=combined_query,
            limit=posts_limit,
            score_threshold=payload.score_threshold,
            min_followers=payload.min_followers,
            min_engagement_rate=None,
            platform=None,
        )
        entities_task = self._qdrant.search_entities(
            query=combined_query,
            limit=entities_limit,
            score_threshold=payload.score_threshold,
        )

        results = await asyncio.gather(
            posts_task, entities_task, return_exceptions=True,
        )

        posts_data: list[dict[str, Any]] = []
        entities_data: list[dict[str, Any]] = []

        if isinstance(results[0], Exception):
            logger.warning("Qdrant posts search failed", exc_info=results[0])
        elif isinstance(results[0], list):
            posts_data = results[0]

        if isinstance(results[1], Exception):
            logger.warning("Qdrant entities search failed", exc_info=results[1])
        elif isinstance(results[1], list):
            entities_data = results[1]

        return posts_data, entities_data
