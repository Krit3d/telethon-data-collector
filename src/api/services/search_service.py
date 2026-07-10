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
    AuthorPostSnippet,
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

logger = logging.getLogger(__name__)

_DEFAULT_EXPLANATION = (
    "This author creates content relevant to the project description "
    "and has engaged audience interaction."
)
_FALLBACK_GRAPH_ENTITIES: list[str] = []

_NEGATIVE_SIGNALS_PATTERN = re.compile(
    r"(1xbet|1win|casino|казино|вулкан|ставки\s+на\s+спорт|adult|18\+|порно|slots|слоты|scam|скам|криптосигналы|crypto\s*signals|betting|bet|online\s*casino|gambling|азартные\s+игры|ставки)",
    re.IGNORECASE,
)

def _build_content_url(
    platform: str | None,
    username: str | None,
    message_id: int | None,
    account_id: int | None,
    platform_content_id: str | None,
) -> str | None:
    if platform and platform.upper() == "TELEGRAM" and message_id is not None:
        if username:
            return f"https://t.me/{username}/{message_id}"
        if account_id is not None:
            return f"https://t.me/c/{account_id}/{message_id}"
    if platform and platform_content_id:
        return f"https://platform/{platform.lower()}/content/{platform_content_id}"
    return None

def _normalize_er(er: float) -> float:
    if er > 1.0:
        er = er / 100.0
    return min(1.0, max(0.0, er))

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
        )
        self._llm_model = settings.cloud_ru_llm_model

    async def _call_llm(
        self, messages: list[dict[str, str | list[dict[str, str]]]], temperature: float = 0.2,
    ) -> str:
        response = await self._llm_client.chat.completions.create(
            model=self._llm_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def _reformulate_query(self, raw_query: str) -> tuple[str, list[str]]:
        system_prompt = (
            "You are a search query optimization assistant for Russian/CIS content. "
            "The input query may be in Russian, English, or a mix of both. "
            "Generate 'vector_query' as a clean, expanded Russian search query optimized for BGE-M3 semantic search "
            "(e.g., expand terms, add synonyms like 'КАСКО' for 'автострахование'). "
            "Generate 'graph_entities' as a list of key topics, brands, or entities strictly in the Russian language "
            "to match Russian entity nodes in the Apache AGE graph. "
            "Output strictly in the defined JSON format with English keys ('vector_query' and 'graph_entities'). "
            "Output ONLY valid JSON, no markdown, no explanation."
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
                vector_query = parsed.get("vector_query", raw_query)
                graph_entities = parsed.get("graph_entities", [])
                if not isinstance(vector_query, str) or not vector_query.strip():
                    vector_query = raw_query
                if not isinstance(graph_entities, list):
                    graph_entities = _FALLBACK_GRAPH_ENTITIES
                else:
                    graph_entities = [str(e) for e in graph_entities if e]
                return vector_query, graph_entities
        except openai.APIError:
            logger.warning("LLM query reformulation failed, using original query", exc_info=True)
        except Exception:
            logger.warning("Unexpected error in query reformulation", exc_info=True)

        return raw_query, _FALLBACK_GRAPH_ENTITIES

    async def _rerank_and_explain(
        self,
        project_description: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates:
            return candidates

        authors_json: list[dict[str, Any]] = []
        for c in candidates:
            authors_json.append(
                {
                    "author_id": c["author_id"],
                    "bio": c.get("description", ""),
                    "subscriber_count": c.get("subscribers_count", 0),
                    "platform": c.get("platform", ""),
                    "top_posts": c.get("top_post_texts", [])[:3],
                }
            )

        system_prompt = (
            "You are an expert talent matching assistant for Russian/CIS content. "
            "Analyze the Russian project description and the Russian creator metadata (bios and post snippets). "
            "Detect and penalize negative CIS-specific signals "
            "(e.g., Russian/CIS scam schemes, online casinos like 1win/1xbet, adult content, fraudulent financial services). "
            "If any negative signals are detected, set the final_score to 0.0. "
            "The 'explanation' field MUST be strictly in Russian, written in a professional, concise tone (2-3 sentences), "
            "explaining why this creator matches the project based strictly on their provided Russian posts and bio. "
            "Calibrate final_score between 0.0 and 1.0. "
            "Output a JSON array where each element has: "
            '"author_id" (integer), "final_score" (float between 0 and 1), "explanation" (string). '
            "Output ONLY valid JSON, no markdown, no extra text."
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
                id_to_update: dict[int, dict[str, Any]] = {}
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    aid = item.get("author_id")
                    score = item.get("final_score")
                    explanation = item.get("explanation")
                    if aid is None or score is None:
                        continue
                    try:
                        aid_int = int(aid)
                        score_float = float(score)
                        if score_float < 0.0 or score_float > 1.0:
                            continue
                        id_to_update[aid_int] = {
                            "final_score": score_float,
                            "explanation": (
                                str(explanation) if explanation else _DEFAULT_EXPLANATION
                            ),
                        }
                    except (ValueError, TypeError):
                        continue

                if id_to_update:
                    for c in candidates:
                        aid = c.get("author_id")
                        if aid in id_to_update:
                            c["final_score"] = id_to_update[aid]["final_score"]
                            c["explanation"] = id_to_update[aid]["explanation"]
                    return candidates
        except openai.APIError:
            logger.warning("LLM reranking failed, keeping initial scores", exc_info=True)
        except Exception:
            logger.warning("Unexpected error in LLM reranking", exc_info=True)

        for c in candidates:
            c.setdefault("explanation", _DEFAULT_EXPLANATION)
        return candidates

    async def execute_search(self, payload: SearchRequest) -> SearchResponse:
        query = payload.query.strip()
        query_words = query.split()
        if len(query_words) < 2:
            return SearchResponse(results=[])
        if all(word.lower() in _STOPWORDS for word in query_words):
            return SearchResponse(results=[])

        vector_query, graph_entities = await self._reformulate_query(query)
        logger.info(
            "Query reformulation complete. vector_query=%r graph_entities=%s",
            vector_query,
            graph_entities[:10],
        )

        posts_fetch_limit = max(1000, payload.limit * 20)
        entities_fetch_limit = 500

        posts_data, entities_data = await self._fetch_qdrant_data(
            vector_query, posts_fetch_limit, entities_fetch_limit, payload,
        )

        entity_id_to_score: dict[str, float] = {}
        label_to_entity_ids: dict[str, list[str]] = {}
        for e in entities_data:
            eid = e.get("entity_id", "")
            score = e.get("score", 0.0)
            if eid:
                entity_id_to_score[eid] = score
                label = e.get("label") or e.get("entity_label") or "Entity"
                if label not in label_to_entity_ids:
                    label_to_entity_ids[label] = []
                label_to_entity_ids[label].append(eid)

        graph_post_entities: dict[int, list[str]] = {}
        graph_post_ers: dict[int, float] = {}

        if label_to_entity_ids:
            try:
                graph_post_entities, graph_post_ers = (
                    await self._graph_search_repo.search_posts_by_entities(
                        label_to_entity_ids
                    )
                )
                logger.info(
                    "Graph search returned %d matched posts from %d entity labels",
                    len(graph_post_entities),
                    len(label_to_entity_ids),
                )
            except Exception:
                logger.warning(
                    "Graph search failed, continuing without graph data", exc_info=True,
                )

        vector_scores: dict[int, float] = {}
        post_id_to_er: dict[int, float] = {}
        for item in posts_data:
            try:
                post_id = int(item["post_id"])
                vector_scores[post_id] = float(item.get("score", 0.0))
                raw_er = item.get("engagement_rate", 0.0)
                post_id_to_er[post_id] = float(raw_er) if raw_er is not None else 0.0
            except (ValueError, KeyError, TypeError):
                continue

        for pid, er in graph_post_ers.items():
            post_id_to_er[pid] = er

        graph_scores: dict[int, float] = {}
        for pid, connected_entities in graph_post_entities.items():
            try:
                pid_int = int(pid)
            except (ValueError, TypeError):
                continue
            entity_scores = [entity_id_to_score.get(eid, 0.0) for eid in connected_entities]
            graph_scores[pid_int] = max(entity_scores) if entity_scores else 0.0

        all_post_ids: set[int] = set(vector_scores.keys()) | set(graph_scores.keys())
        safe_post_ids: list[int] = sorted(all_post_ids)

        if not safe_post_ids:
            return SearchResponse(results=[])

        candidates_rows = await self._db.get_search_candidates(
            content_ids=safe_post_ids,
            location=payload.location,
            min_followers=payload.min_followers,
        )

        if not candidates_rows:
            return SearchResponse(results=[])

        author_map: dict[int, dict[str, Any]] = {}
        current_utc = datetime.now(timezone.utc)

        for row in candidates_rows:
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
                    "posts": [],
                    "vector_scores": [],
                    "graph_scores": [],
                    "engagement_rates": [],
                    "explanation": _DEFAULT_EXPLANATION,
                    "has_contacts": False,
                    "most_recent_post": None,
                    "matched_entities": set(),
                }

            author = author_map[account_id]
            vs = vector_scores.get(post_id, 0.0)
            gs = graph_scores.get(post_id, 0.0)
            er = _normalize_er(post_id_to_er.get(post_id, 0.0))

            published_at = row.get("published_at") or row.get("created_at")
            decay_factor = 1.0
            if published_at is not None:
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
                days = (current_utc - published_at).days
                decay_factor = math.exp(-0.005 * days)
                vs *= decay_factor
                gs *= decay_factor
                if author["most_recent_post"] is None or published_at > author["most_recent_post"]:
                    author["most_recent_post"] = published_at

            platform = row.get("platform")
            username = row.get("username")
            message_id = row.get("message_id")
            platform_content_id = row.get("platform_content_id")

            if published_at is not None:
                snippet = AuthorPostSnippet(
                    post_id=post_id,
                    text=(row.get("content") or row.get("transcription") or "")[:500],
                    published_at=published_at,
                    url=_build_content_url(
                        platform, username, message_id, account_id, platform_content_id,
                    ),
                    engagement_rate=er,
                )

                author["posts"].append(
                    {
                        "snippet": snippet,
                        "vector_score": vs,
                        "graph_score": gs,
                        "engagement_rate": er,
                    }
                )
                author["vector_scores"].append(vs)
                author["graph_scores"].append(gs)
                author["engagement_rates"].append(er)

                post_entity_ids = graph_post_entities.get(post_id, [])
                author["matched_entities"].update(post_entity_ids)

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
            post_texts = [p["snippet"].text for p in author["posts"]]
            all_text = f"{title} {description} {' '.join(post_texts)}"
            if _has_negative_signals(all_text):
                logger.info("Author %d discarded due to negative signals", account_id)
                continue
            safe_author_map[account_id] = author

        if not safe_author_map:
            return SearchResponse(results=[])

        max_matched_posts = max(len(a["posts"]) for a in safe_author_map.values()) if safe_author_map else 1

        active_authors_count = 0
        for a in safe_author_map.values():
            if a["most_recent_post"] is not None:
                days_since_last_post = (current_utc - a["most_recent_post"]).days
                if days_since_last_post <= 180:
                    active_authors_count += 1

        ranked_authors: list[dict[str, Any]] = []
        for account_id, author in safe_author_map.items():
            if not author["posts"]:
                continue
            max_vs = max(author["vector_scores"]) if author["vector_scores"] else 0.0
            max_gs = max(author["graph_scores"]) if author["graph_scores"] else 0.0
            avg_er = (
                sum(author["engagement_rates"]) / len(author["engagement_rates"])
                if author["engagement_rates"] else 0.0
            )
            expertise_ratio = len(author["posts"]) / max_matched_posts if max_matched_posts > 0 else 0.0
            base_score = 0.4 * max_vs + 0.3 * max_gs + 0.15 * avg_er + 0.15 * expertise_ratio
            final_raw_score = base_score + (0.15 if author["has_contacts"] else 0.0)

            if author["most_recent_post"] is not None:
                days_since_last_post = (current_utc - author["most_recent_post"]).days
                if days_since_last_post > 180:
                    if active_authors_count >= payload.limit:
                        continue
                    else:
                        final_raw_score *= 0.1

            author["vector_score"] = max_vs
            author["graph_score"] = max_gs
            author["avg_engagement_rate"] = avg_er
            author["expertise_ratio"] = expertise_ratio
            author["final_score"] = final_raw_score
            author["top_post_texts"] = [
                s["snippet"].text[:200] for s in author["posts"][:3]
            ]
            ranked_authors.append(author)

        if not ranked_authors:
            return SearchResponse(results=[])

        mmr_selection_size = min(15, payload.limit * 2)
        mmr_selected: list[dict[str, Any]] = []

        remaining_candidates = list(ranked_authors)

        for _ in range(min(mmr_selection_size, len(remaining_candidates))):
            best_candidate = None
            best_mmr_score = -float("inf")

            for candidate in remaining_candidates:
                if candidate in mmr_selected:
                    continue
                candidate_entities = candidate.get("matched_entities", set())
                max_similarity = 0.0
                for selected in mmr_selected:
                    selected_entities = selected.get("matched_entities", set())
                    sim = _jaccard_similarity(candidate_entities, selected_entities)
                    max_similarity = max(max_similarity, sim)
                mmr_score = 0.7 * candidate["final_score"] - 0.3 * max_similarity
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = candidate

            if best_candidate is None:
                break
            mmr_selected.append(best_candidate)
            remaining_candidates.remove(best_candidate)

        top_candidates = mmr_selected if mmr_selected else ranked_authors[:15]

        if len(top_candidates) > 1:
            try:
                top_candidates = await self._rerank_and_explain(
                    payload.query, top_candidates,
                )
            except Exception:
                logger.warning(
                    "Reranking step failed, using initial scores", exc_info=True,
                )

        top_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        final_candidates = top_candidates[: payload.limit]

        results: list[AuthorSearchResultItem] = []
        for author in final_candidates:
            sorted_posts = sorted(
                author["posts"],
                key=lambda x: (x["vector_score"] + x["graph_score"]) / 2,
                reverse=True,
            )
            relevant_posts = [p["snippet"] for p in sorted_posts]

            explanation = author.get("explanation", _DEFAULT_EXPLANATION)
            if author.get("has_contacts") is True:
                explanation = explanation + " (В профиле автора найдены контактные данные)."

            results.append(
                AuthorSearchResultItem(
                    author_id=author["author_id"],
                    username=author.get("username"),
                    title=author.get("title", ""),
                    description=author.get("description"),
                    subscribers_count=author.get("subscribers_count"),
                    platform=author.get("platform", "TELEGRAM"),
                    final_score=author["final_score"],
                    vector_score=author["vector_score"],
                    graph_score=author["graph_score"],
                    avg_engagement_rate=author["avg_engagement_rate"],
                    explanation=explanation,
                    relevant_posts=relevant_posts,
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
            score_threshold=payload.score_threshold,
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
