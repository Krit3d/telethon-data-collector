import asyncio
import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any

import openai
from openai import AsyncOpenAI

from src.api.schemas import (
    AuthorSearchResultItem,
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

_DORMANT_WARNING_RU = (
    "Внимание: автор не публиковал новый контент более 180 дней, но включён в выдачу как редкий эксперт в данной нише."
)

def _safe_json_loads(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        result: object = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass
    return None

def _safe_json_loads_array(text: str) -> list[dict[str, Any]] | None:
    if not text:
        return None
    try:
        result: object = json.loads(text)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    json_match = re.search(r"\[.*\]", text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, TypeError):
            pass
    return None

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
    if raw_score <= 0.35:
        return 0.0
    if raw_score >= 0.75:
        return 1.0
    return (raw_score - 0.35) / (0.75 - 0.35)


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
        self._llm_client = AsyncOpenAI(
            api_key=settings.cloud_ru_api_key,
            base_url=settings.cloud_ru_base_url,
            timeout=60.0,
        )
        self._llm_model = settings.cloud_ru_llm_model

    async def _call_llm(
        self, messages: list[dict[str, Any]], temperature: float = 0.2,
    ) -> str:
        response = await self._llm_client.chat.completions.create(
            model=self._llm_model,
            messages=messages,  # type: ignore
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def _reformulate_query(self, raw_query: str) -> tuple[str, list[str]]:
        system_prompt = (
            "You are a search query optimization assistant for Russian/CIS content. "
            "The input query may be in Russian, English, or a mix of both. "
            "Generate a JSON object with exactly three keys: "
            "'dense_query' - a clean, semantically expanded Russian sentence representing the core meaning (for dense vector search); "
            "'lexical_queries' - an array of strings containing key terms, Russian synonyms, and domain-specific acronyms "
            "(e.g., for 'автострахование', include 'КАСКО', 'ОСАГО', 'страховой полис') to increase hit rates; "
            "'graph_entities' - an array of key topics, brands, or entities in Russian to match Apache AGE graph nodes. "
            "CRITICAL: Any explanatory text or reasoning MUST be in English only. "
            "Output strictly valid JSON, no markdown, no explanation."
        )
        user_prompt = f"Project description: {raw_query}"

        try:
            content = await self._call_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            parsed = _safe_json_loads(content)
            if parsed is not None:
                dense_query = parsed.get("dense_query", "")
                lexical_queries = parsed.get("lexical_queries", [])
                graph_entities = parsed.get("graph_entities", [])

                if not isinstance(dense_query, str) or not dense_query.strip():
                    dense_query = raw_query
                if not isinstance(lexical_queries, list):
                    lexical_queries = []
                else:
                    lexical_queries = [str(q) for q in lexical_queries if q][:5]
                if not isinstance(graph_entities, list):
                    graph_entities = _FALLBACK_GRAPH_ENTITIES
                else:
                    graph_entities = [str(e) for e in graph_entities if e]

                combined_query = dense_query
                if lexical_queries:
                    combined_query = f"{dense_query} {' '.join(lexical_queries[:3])}"

                return combined_query, graph_entities
        except openai.APIError:
            logger.warning("LLM query reformulation failed, using original query", exc_info=True)
        except Exception:
            logger.warning("Unexpected error in query reformulation", exc_info=True)

        return raw_query, _FALLBACK_GRAPH_ENTITIES

    async def _grade_candidates(self, project_description: str, candidates: list[dict[str, Any]]) -> dict[str, int]:
        if not candidates:
            return {}

        authors_json: list[dict[str, Any]] = []
        for c in candidates:
            authors_json.append({
                "author_id": str(c["author_id"]),
                "bio": c.get("description", ""),
                "subscriber_count": c.get("subscribers_count", 0),
                "platform": c.get("platform", ""),
                "top_posts": c.get("top_post_texts", [])[:3],
                "is_dormant": c.get("is_dormant", False),
            })

        system_prompt = (
            "You are an expert talent matching assistant for Russian/CIS content. "
            "Analyze the Russian project description and the Russian creator metadata (bios and post snippets). "
            "Output a JSON array where each element has exactly two keys: "
            '"author_id" (string, must be the exact ID as provided without rounding or mathematical alteration) and '
            '"relevance_grade" (integer: 0, 1, or 2). '
            "Do NOT include any explanation, reasoning, or other fields in the output. "
            "Relevance grading: "
            "2 = highly relevant domain expert (dedicated auto-lawyers, professional insurance brokers, specialized accident assessment); "
            "1 = partially relevant (general-profile lawyers, generic insurance comparison portals); "
            "0 = completely irrelevant (spam/scam, OR corporate dealership/car brand accounts with no active domain expertise). "
            "Output ONLY valid JSON array, no markdown, no extra text, no explanation outside JSON."
        )
        user_prompt = (
            f"Project description: {project_description}\n\n"
            f"Authors: {json.dumps(authors_json, ensure_ascii=False)}"
        )

        try:
            content = await self._call_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            parsed = _safe_json_loads_array(content)
            if parsed is not None:
                grade_map: dict[str, int] = {}
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    aid = item.get("author_id")
                    grade = item.get("relevance_grade")
                    if aid is None or grade is None:
                        continue
                    try:
                        aid_str = str(aid)
                        grade_int = int(grade)
                        if grade_int not in (0, 1, 2):
                            continue
                        grade_map[aid_str] = grade_int
                    except (ValueError, TypeError):
                        continue
                return grade_map
        except openai.APIError:
            logger.warning("LLM grading failed, returning empty grade map", exc_info=True)
        except Exception:
            logger.warning("Unexpected error in LLM grading", exc_info=True)

        return {}

    async def _generate_explanations(self, project_description: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return candidates

        authors_json: list[dict[str, Any]] = []
        for c in candidates:
            authors_json.append({
                "author_id": str(c["author_id"]),
                "bio": c.get("description", ""),
                "subscriber_count": c.get("subscribers_count", 0),
                "platform": c.get("platform", ""),
                "top_posts": c.get("top_post_texts", [])[:3],
                "is_dormant": c.get("is_dormant", False),
            })

        system_prompt = (
            "You are an expert talent matching assistant for Russian/CIS content. "
            "Analyze the Russian project description and the Russian creator metadata (bios and post snippets). "
            "Output a JSON array where each element has exactly two keys: "
            '"author_id" (string, must be the exact ID as provided without rounding or mathematical alteration) and '
            '"explanation" (string, strictly in Russian, 1-2 short sentences). '
            "Analyze the creator's metadata (bio, top posts) relative to the project description "
            "to write a polished Russian explanation. "
            "Output ONLY valid JSON array, no markdown, no extra text, no explanation outside JSON."
        )
        user_prompt = (
            f"Project description: {project_description}\n\n"
            f"Authors: {json.dumps(authors_json, ensure_ascii=False)}"
        )

        try:
            content = await self._call_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            parsed = _safe_json_loads_array(content)
            if parsed is not None:
                id_to_explanation: dict[str, str] = {}
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    aid = item.get("author_id")
                    explanation = item.get("explanation")
                    if aid is None or explanation is None:
                        continue
                    id_to_explanation[str(aid)] = str(explanation)

                for c in candidates:
                    aid = str(c.get("author_id", ""))
                    explanation = id_to_explanation.get(aid, _DEFAULT_EXPLANATION)
                    if c.get("is_dormant", False):
                        if not explanation.endswith(_DORMANT_WARNING_RU):
                            explanation += " " + _DORMANT_WARNING_RU
                    c["explanation"] = explanation

                return candidates
        except openai.APIError:
            logger.warning("LLM explanation generation failed, using default explanation", exc_info=True)
        except Exception:
            logger.warning("Unexpected error in LLM explanation generation", exc_info=True)

        for c in candidates:
            c.setdefault("explanation", _DEFAULT_EXPLANATION)
        return candidates

    def _apply_bucketed_mmr(
        self,
        candidates: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        grade_2_candidates = [c for c in candidates if c.get("relevance_grade") == 2 and c.get("final_score", 0.0) > 0.0]
        grade_1_candidates = [c for c in candidates if c.get("relevance_grade") == 1 and c.get("final_score", 0.0) > 0.0]

        grade_2_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        grade_1_candidates.sort(key=lambda x: x["final_score"], reverse=True)

        selected: list[dict[str, Any]] = []
        remaining_grade_2 = list(grade_2_candidates)
        remaining_grade_1 = list(grade_1_candidates)

        for _ in range(min(limit, len(remaining_grade_2))):
            if len(selected) >= limit:
                break
            best_candidate = None
            best_mmr_score = -float("inf")
            for candidate in remaining_grade_2:
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
            remaining_grade_2.remove(best_candidate)

        if len(selected) < limit:
            for _ in range(min(limit - len(selected), len(remaining_grade_1))):
                if len(selected) >= limit:
                    break
                best_candidate = None
                best_mmr_score = -float("inf")
                for candidate in remaining_grade_1:
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
                remaining_grade_1.remove(best_candidate)

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

        vector_query, graph_entities = await self._reformulate_query(query)
        logger.info(
            "Query reformulation complete. vector_query=%r graph_entities=%s",
            vector_query,
            graph_entities[:10],
        )

        posts_fetch_limit = 1500
        entities_fetch_limit = 500

        topic_post_ids: list[int] = []
        posts_data: list[dict[str, Any]] = []
        entities_data: list[dict[str, Any]] = []

        if graph_entities:
            qdrant_task = asyncio.create_task(
                self._fetch_qdrant_data(
                    vector_query, posts_fetch_limit, entities_fetch_limit, payload,
                )
            )
            topic_task = asyncio.create_task(
                self._graph_search_repo.search_posts_by_topics(graph_entities)
            )
            try:
                posts_data, entities_data = await qdrant_task
            except Exception:
                logger.warning("Qdrant fetch failed", exc_info=True)
                posts_data = []
                entities_data = []
            try:
                result = await topic_task
                topic_post_ids = result if isinstance(result, list) else []
            except Exception:
                logger.warning("Graph topic search failed", exc_info=True)
        else:
            posts_data, entities_data = await self._fetch_qdrant_data(
                vector_query, posts_fetch_limit, entities_fetch_limit, payload,
            )

        logger.info(
            "Qdrant fetch complete. posts_data length=%d entities_data length=%d topic_post_ids=%d",
            len(posts_data),
            len(entities_data),
            len(topic_post_ids),
        )

        entity_id_to_score: dict[str, float] = {}
        entity_ids_with_scores: list[tuple[str, float]] = []
        for e in entities_data:
            eid = e.get("entity_id", "")
            score = e.get("score", 0.0)
            if eid:
                entity_id_to_score[eid] = score
                entity_ids_with_scores.append((eid, score))

        entity_ids_with_scores.sort(key=lambda x: x[1], reverse=True)
        top_entity_ids = [eid for eid, _ in entity_ids_with_scores[:200]]

        graph_post_entities: dict[int, list[str]] = {}
        graph_post_ers: dict[int, float] = {}

        if top_entity_ids:
            try:
                graph_post_entities, graph_post_ers = (
                    await self._graph_search_repo.search_posts_by_entities(
                        top_entity_ids
                    )
                )
                logger.info(
                    "Graph search returned %d matched posts from %d entities",
                    len(graph_post_entities),
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
        for pid, connected_entities in graph_post_entities.items():
            try:
                pid_int = int(pid)
            except (ValueError, TypeError):
                continue
            entity_scores = [entity_id_to_score.get(eid, 0.0) for eid in connected_entities]
            graph_scores[pid_int] = max(entity_scores) if entity_scores else 0.0

        all_post_ids: set[int] = set(vector_scores.keys()) | set(graph_scores.keys()) | set(topic_post_ids)
        for pid in topic_post_ids:
            graph_scores[pid] = max(graph_scores.get(pid, 0.0), 0.65)
        safe_post_ids: list[int] = sorted(all_post_ids)
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

        logger.info(
            "DB candidate query params: location=%r min_followers=%r",
            payload.location,
            payload.min_followers,
        )
        candidates_rows = await self._db.get_search_candidates(
            content_ids=safe_post_ids,
            location=payload.location,
            min_followers=payload.min_followers,
        )
        logger.info(
            "Raw candidates fetched from PostgreSQL: %d",
            len(candidates_rows),
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
            post_id = row["id"]
            account_id = row["account_id"]

            if account_id not in author_map:
                raw_static_avg_er = row.get("static_avg_er")
                author_map[account_id] = {
                    "author_id": account_id,
                    "username": row.get("username"),
                    "title": row.get("account_title", ""),
                    "description": row.get("description"),
                    "subscribers_count": row.get("subscribers_count"),
                    "platform": row.get("platform", "TELEGRAM"),
                    "posts": [],
                    "vector_scores": [],
                    "graph_scores": [],
                    "static_avg_er": float(raw_static_avg_er) if raw_static_avg_er is not None else 0.0,
                    "explanation": _DEFAULT_EXPLANATION,
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
            parsed_metadata = _safe_json_loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
            if isinstance(parsed_metadata, dict) and not author["has_contacts"]:
                contacts = parsed_metadata.get("contacts")
                if isinstance(contacts, dict):
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
                logger.info("Author %d discarded due to negative signals", account_id)
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

            normalized_er = min(1.0, max(0.0, author["static_avg_er"] / 15.0))
            relevant_posts_count = sum(
                1
                for vs, gs in zip(
                    author["vector_scores"], author["graph_scores"], strict=False
                )
                if vs >= payload.score_threshold or gs >= payload.score_threshold
            )
            expertise_ratio = relevant_posts_count / max_matched_posts

            initial_topical_score = max(max_vs, max_gs) + 0.15 * min(max_vs, max_gs)
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

            decayed_topical_score = (max(max_vs, max_gs) + 0.15 * min(max_vs, max_gs)) * decay_factor

            base_score = 0.7 * decayed_topical_score + 0.15 * normalized_er + 0.15 * expertise_ratio
            contact_multiplier = 1.15 if author["has_contacts"] else 1.0
            final_raw_score = min(1.0, base_score * contact_multiplier)

            if is_dormant:
                author["is_dormant"] = True
            else:
                author["is_dormant"] = False

            author["vector_score"] = max_vs
            author["graph_score"] = max_gs
            author["avg_engagement_rate"] = author["static_avg_er"]
            author["expertise_ratio"] = expertise_ratio
            author["initial_topical_score"] = initial_topical_score
            author["final_score"] = final_raw_score
            author["topical_score"] = decayed_topical_score
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

        ranked_authors.sort(key=lambda x: x["initial_topical_score"], reverse=True)

        rerank_pool_size = min(40, max(20, payload.limit * 2))
        top_k_candidates = ranked_authors[:min(rerank_pool_size, len(ranked_authors))]

        logger.info(
            "Candidates before LLM grading: %d (pool size cap: %d)", len(top_k_candidates), rerank_pool_size,
        )

        grade_map: dict[str, int] = {}
        if len(top_k_candidates) >= 1:
            try:
                grade_map = await self._grade_candidates(payload.query, top_k_candidates)
            except Exception:
                logger.warning(
                    "Stage 1 grading failed, using default grades", exc_info=True,
                )

        for c in top_k_candidates:
            aid = str(c.get("author_id", ""))
            grade = grade_map.get(aid, 0)
            base_final_score = c.get("final_score", 0.0)
            if grade == 2:
                c["final_score"] = 0.75 + 0.25 * base_final_score
            elif grade == 1:
                c["final_score"] = payload.score_threshold + (0.74 - payload.score_threshold) * base_final_score
            else:
                c["final_score"] = 0.0
            c["relevance_grade"] = grade

        filtered_candidates = [
            c for c in top_k_candidates
            if c.get("final_score", 0.0) > 0.0
        ]

        final_candidates = self._apply_bucketed_mmr(filtered_candidates, payload.limit)

        if final_candidates:
            try:
                final_candidates = await self._generate_explanations(payload.query, final_candidates)
            except Exception:
                logger.warning(
                    "Stage 2 explanation generation failed, using default explanation", exc_info=True,
                )

        logger.info(
            "Candidates after bucketed MMR: %d",
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
                )
            )

        return SearchResponse(results=results)

    async def _fetch_qdrant_data(
        self,
        vector_query: str,
        posts_limit: int,
        entities_limit: int,
        payload: SearchRequest,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        posts_task = self._qdrant.search_posts(
            query=vector_query,
            limit=posts_limit,
            score_threshold=max(payload.score_threshold, 0.35),
            min_followers=payload.min_followers,
            min_engagement_rate=None,
            platform=None,
        )
        entities_task = self._qdrant.search_entities(
            query=vector_query,
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
