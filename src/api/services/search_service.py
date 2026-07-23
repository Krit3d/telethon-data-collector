import asyncio
import csv
import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openai
from openai import AsyncOpenAI

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

        taxonomy_path = Path(__file__).resolve().parents[2] / "config" / "Content Taxonomy 3.1.tsv"
        taxonomy_lines: list[str] = []
        parent_map: dict[str, str] = {}
        name_map: dict[str, str] = {}
        with open(taxonomy_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 3 and row[0].strip().isdigit():
                    uid = row[0].strip()
                    name = row[2].strip()
                    taxonomy_lines.append(f"{uid}: {name}")
                    name_map[uid] = name
                    parent_raw = row[1].strip() if len(row) > 1 else ""
                    if parent_raw and parent_raw.isdigit():
                        parent_map[uid] = parent_raw
        self._taxonomy_str = "\n".join(taxonomy_lines)

        self._taxonomy_dict: dict[str, dict[str, str]] = {}
        for uid in name_map:
            path_parts: list[str] = []
            current = uid
            visited = {current}
            while current in parent_map:
                current = parent_map[current]
                if current in visited:
                    break
                visited.add(current)
                if current in name_map:
                    path_parts.insert(0, name_map[current])
            path_parts.append(name_map[uid])
            self._taxonomy_dict[uid] = {
                "name": name_map[uid],
                "parent_id": parent_map.get(uid, ""),
                "path": " > ".join(path_parts),
            }

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
        if " > " in author_path:
            author_parent_path = " > ".join(author_path.split(" > ")[:-1])
        else:
            author_parent_path = author_path
        for target_id in normalized_targets:
            target_entry = self._taxonomy_dict.get(target_id)
            if target_entry is not None:
                target_path = target_entry["path"]
                if " > " in target_path:
                    target_parent_path = " > ".join(target_path.split(" > ")[:-1])
                else:
                    target_parent_path = target_path
                if author_parent_path and target_parent_path and author_parent_path == target_parent_path:
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
        system_prompt = (
            "You are an expert search query reformulator for a semantic creator discovery platform. "
            "Analyze the user project description (Russian/English) and produce a structured ReformulatedQuery:\n"
            "1. dense_query: Clean, expanded semantic sentence in Russian for dense vector search.\n"
            "2. lexical_queries: Array of 3-7 keywords, Russian synonyms, and domain acronyms.\n"
            "3. graph_entities: Array of key topics or entities in Russian for Apache AGE graph nodes.\n"
            "4. target_iab_ids: Array of up to 3 exact numeric category IDs matching the query intent from the provided IAB Taxonomy.\n"
            "5. profile_type_intent: Automatically classify intent as 'expert' (individual specialists/creators), "
            "'business' (clinics, agencies, commercial companies), or 'both'.\n\n"
            f"Available IAB Taxonomy categories:\n{self._taxonomy_str}"
        )

        try:
            response = await self._llm_client.beta.chat.completions.parse(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Project description: {raw_query}"},
                ],
                response_format=ReformulatedQuery,
                temperature=0.1,
            )
            parsed = response.choices[0].message.parsed
            if parsed is not None:
                return parsed
        except openai.APIError:
            logger.warning("LLM query reformulation failed, using original query", exc_info=True)
        except Exception:
            logger.warning("Unexpected error in query reformulation", exc_info=True)

        return ReformulatedQuery(
            dense_query=raw_query,
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
                    combined_query, posts_fetch_limit, entities_fetch_limit, payload,
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
                combined_query, posts_fetch_limit, entities_fetch_limit, payload,
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

        if payload.author_type == "expert":
            target_is_author_blog = True
        elif payload.author_type == "business":
            target_is_author_blog = False
        elif payload.author_type in ("all", None):
            if reformulated.profile_type_intent == "expert":
                target_is_author_blog = True
            elif reformulated.profile_type_intent == "business":
                target_is_author_blog = False
            else:
                target_is_author_blog = None
        else:
            target_is_author_blog = None

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
            if row.get("is_enriched") is False:
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
                    "static_avg_er": float(row.get("static_avg_er") or 0.0),
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

            base_topical_score = max(max_vs, max_gs) + 0.15 * min(max_vs, max_gs)

            tax_score = self._calculate_taxonomy_match_score(reformulated.target_iab_ids, author.get("category_id"))
            if tax_score > 0.0:
                base_topical_score *= (1.0 + 0.2 * tax_score)

            initial_topical_score = base_topical_score

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

            decayed_topical_score = base_topical_score * decay_factor

            base_score = 0.7 * decayed_topical_score + 0.15 * normalized_er + 0.15 * expertise_ratio
            contact_multiplier = 1.15 if author["has_contacts"] else 1.0
            final_raw_score = min(1.0, base_score * contact_multiplier)

            author["is_dormant"] = is_dormant

            if is_dormant:
                if _DORMANT_WARNING_RU not in author["explanation"]:
                    author["explanation"] += " " + _DORMANT_WARNING_RU

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

        ranked_authors.sort(key=lambda x: x["final_score"], reverse=True)

        rerank_pool_size = min(40, max(20, payload.limit * 2))
        top_k_candidates = ranked_authors[:min(rerank_pool_size, len(ranked_authors))]

        logger.info(
            "Candidates before flat MMR: %d (pool size cap: %d)", len(top_k_candidates), rerank_pool_size,
        )

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
                    contacts=author.get("contacts"),
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
            score_threshold=max(payload.score_threshold, 0.35),
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
