import asyncio
import logging
import re
from typing import Any

from src.api.schemas import (
    GraphEdge,
    GraphEntity,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from src.db.database import Database
from src.embeddings.qdrant_service import QdrantService
from src.graph.db.search_repo import GraphSearchRepository

logger = logging.getLogger(__name__)

_MAX_INT32: int = 2147483647
_MIN_INT32: int = -2147483648
_CONTENT_LABELS: frozenset[str] = frozenset({"content", "event", "publication"})
_POISONED_LABELS: frozenset[str] = frozenset({"language", "category"})


def _normalize_er(er: float) -> float:
    if er > 1.0:
        er = er / 100.0
    return min(1.0, max(0.0, er))


def _build_content_url(row: dict[str, Any]) -> str:
    try:
        platform = row.get("platform", "TELEGRAM")
        username = row.get("username")
        message_id = row.get("message_id")
        if platform == "TELEGRAM" and message_id is not None:
            if username:
                return f"https://t.me/{username}/{message_id}"
            return f"https://t.me/c/{row.get('account_id', '')}/{message_id}"
        return (
            f"https://platform/{platform.lower()}/content/"
            f"{row.get('platform_content_id', 'unknown')}"
        )
    except Exception:
        return ""


def _build_node_id_to_post_id(
    nodes_data: list[dict[str, Any]],
) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for node in nodes_data:
            label = (node.get("label") or "").lower()
            if label not in _CONTENT_LABELS:
                continue
            node_id = node.get("id", "")
            props = node.get("properties", {}) or {}
            raw_db_post_id = props.get("db_post_id")
            post_id: int | None = None
            if raw_db_post_id is not None:
                try:
                    post_id = int(raw_db_post_id)
                except (ValueError, TypeError):
                    post_id = None
            if post_id is None:
                match = re.search(r"(\d+)$", str(node_id))
                if match:
                    try:
                        post_id = int(match.group(1))
                    except (ValueError, TypeError):
                        post_id = None
            if post_id is not None:
                result[node_id] = post_id
    except (TypeError, ValueError):
        pass
    return result


def _build_post_id_to_er(
    nodes_data: list[dict[str, Any]],
    node_id_to_post_id: dict[str, int],
) -> dict[int, float]:
    result: dict[int, float] = {}
    try:
        for node in nodes_data:
            node_id = node.get("id", "")
            if node_id not in node_id_to_post_id:
                continue
            post_id = node_id_to_post_id[node_id]
            props = node.get("properties", {}) or {}
            raw_er = props.get("engagement_rate", 0.0)
            try:
                result[post_id] = float(raw_er)
            except (ValueError, TypeError):
                result[post_id] = 0.0
    except (TypeError, ValueError):
        pass
    return result


def _compute_kag_scores(
    vector_scores: dict[int, float],
    graph_scores: dict[int, float],
    post_id_to_er: dict[int, float],
) -> dict[int, tuple[float, float, float, float]]:
    w_er = 0.10

    all_ids: set[int] = set(vector_scores.keys()) | set(graph_scores.keys())
    results: dict[int, tuple[float, float, float, float]] = {}

    for cid in all_ids:
        vector_score = vector_scores.get(cid, 0.0)
        g_score = graph_scores.get(cid, 0.0)
        er_score = _normalize_er(post_id_to_er.get(cid, 0.0))

        relevance = 1.0 - (1.0 - vector_score) * (1.0 - g_score)
        final_score = min(1.0, relevance + (w_er * er_score))

        results[cid] = (final_score, vector_score, g_score, er_score)

    return results


class SearchService:

    def __init__(
        self,
        qdrant: QdrantService,
        db: Database,
        graph_search_repo: GraphSearchRepository,
    ) -> None:
        self._qdrant = qdrant
        self._db = db
        self._graph_search_repo = graph_search_repo

    async def execute_search(self, payload: SearchRequest) -> SearchResponse:
        posts_fetch_limit = max(1000, payload.limit * 2)
        entities_fetch_limit = 80

        posts_data, entities_data = await asyncio.gather(
            self._qdrant.search_posts(
                query=payload.query,
                limit=posts_fetch_limit,
                score_threshold=payload.score_threshold,
                min_followers=payload.min_followers,
                min_engagement_rate=payload.min_engagement_rate,
                platform=getattr(payload, "platform", None),
            ),
            self._qdrant.search_entities(
                query=payload.query,
                limit=entities_fetch_limit,
                score_threshold=payload.score_threshold,
            ),
        )

        entities_data = [
            e for e in entities_data
            if (e.get("label") or e.get("entity_label") or "").lower()
            not in _POISONED_LABELS
        ]

        entity_id_to_score: dict[str, float] = {
            e["entity_id"]: e["score"] for e in entities_data
        }
        entity_ids: set[str] = set(entity_id_to_score.keys())

        if entity_ids:
            logger.info(
                "Found %d entities. Score range: %.4f-%.4f",
                len(entity_ids),
                min(entity_id_to_score.values()),
                max(entity_id_to_score.values()),
            )
        else:
            logger.info("No entities found; graph traversal will be skipped.")

        vector_scores: dict[int, float] = {}
        for item in posts_data:
            try:
                vector_scores[int(item["post_id"])] = float(item["score"])
            except (ValueError, TypeError):
                continue

        post_id_to_er: dict[int, float] = {}
        for item in posts_data:
            try:
                post_id = int(item["post_id"])
                raw_er = item.get("engagement_rate", 0.0)
                post_id_to_er[post_id] = float(raw_er)
            except (ValueError, TypeError):
                continue

        graph_post_scores: dict[int, float] = {}

        if entity_ids:
            try:
                label_to_entity_ids: dict[str, list[str]] = {}
                for e in entities_data:
                    label = e.get("label") or e.get("entity_label") or "Entity"
                    if label not in label_to_entity_ids:
                        label_to_entity_ids[label] = []
                    label_to_entity_ids[label].append(e["entity_id"])

                graph_post_entities, graph_post_ers = (
                    await self._graph_search_repo.search_posts_by_entities(
                        label_to_entity_ids
                    )
                )
                logger.info(
                    "Graph Cypher search returned %d matched posts",
                    len(graph_post_entities),
                )

                for pid, er in graph_post_ers.items():
                    post_id_to_er[pid] = er

                for pid, connected_entities in graph_post_entities.items():
                    try:
                        pid_int = int(pid)
                    except (ValueError, TypeError):
                        continue
                    entity_scores = [
                        entity_id_to_score.get(e_id, 0.0)
                        for e_id in connected_entities
                    ]
                    graph_post_scores[pid_int] = max(entity_scores) if entity_scores else 0.0

            except Exception:
                logger.warning("Graph search failed", exc_info=True)

        logger.info(
            "Vector scored: %d items, Graph scored: %d items",
            len(vector_scores),
            len(graph_post_scores),
        )

        unique_content_ids: set[int] = set(vector_scores.keys()) | set(graph_post_scores.keys())

        safe_content_ids: list[int] = [
            cid for cid in unique_content_ids
            if _MIN_INT32 <= cid <= _MAX_INT32
        ]

        filtered_count = len(unique_content_ids) - len(safe_content_ids)
        if filtered_count > 0:
            logger.warning(
                "Filtered out %d content IDs exceeding PostgreSQL int32 range "
                "(min=%d, max=%d). Dropped IDs: %s",
                filtered_count,
                _MIN_INT32,
                _MAX_INT32,
                sorted(set(unique_content_ids) - set(safe_content_ids)),
            )

        candidates = await self._db.get_search_candidates(
            content_ids=safe_content_ids,
            location=payload.location,
            min_followers=payload.min_followers,
        )

        candidate_id_set: set[int] = {row["id"] for row in candidates}

        kag_scores = _compute_kag_scores(vector_scores, graph_post_scores, post_id_to_er)

        merged: list[dict[str, Any]] = []
        for row in candidates:
            cid = row["id"]
            if cid not in kag_scores:
                continue
            final_score, vec_score, norm_graph_score, er_score = kag_scores[cid]
            merged.append(
                {
                    "row": row,
                    "final_score": final_score,
                    "vector_score": vec_score,
                    "graph_score": norm_graph_score,
                    "er_score": er_score,
                    "url": _build_content_url(row),
                    "in_graph": cid in graph_post_scores,
                    "in_vector": cid in vector_scores,
                }
            )

        merged.sort(key=lambda x: x["final_score"], reverse=True)
        merged = merged[: payload.limit]

        final_post_ids: set[int] = {item["row"]["id"] for item in merged}
        graph_entities: list[GraphEntity] = []

        if final_post_ids and entity_ids and graph_post_entities:
            try:
                active_entity_ids: set[str] = set()
                top_posts_for_graph = merged[:15]
                for item in top_posts_for_graph:
                    pid = item["row"]["id"]
                    if pid in graph_post_entities:
                        for eid in graph_post_entities[pid]:
                            active_entity_ids.add(eid)

                if not active_entity_ids:
                    top_entity_ids = [
                        e["entity_id"] for e in entities_data[:10]
                    ]
                    active_entity_ids = set(top_entity_ids)

                if active_entity_ids:
                    lazy_label_to_ids: dict[str, list[str]] = {}
                    for e in entities_data:
                        eid = e["entity_id"]
                        if eid in active_entity_ids:
                            label = e.get("label") or e.get("entity_label") or "Entity"
                            if label not in lazy_label_to_ids:
                                lazy_label_to_ids[label] = []
                            lazy_label_to_ids[label].append(eid)

                    lazy_edges_data = await self._graph_search_repo.fetch_subgraph_edges(
                        lazy_label_to_ids
                    )
                    logger.info(
                        "Optimized lazy hydration: fetched %d edges for %d active entities",
                        len(lazy_edges_data),
                        len(active_entity_ids),
                    )

                    if lazy_edges_data:
                        lazy_node_ids: set[str] = set()
                        for edge in lazy_edges_data:
                            try:
                                lazy_node_ids.add(edge["source_id"])
                                lazy_node_ids.add(edge["target_id"])
                            except (KeyError, TypeError):
                                continue

                        lazy_node_id_to_label: dict[str, str] = {}
                        for nid in lazy_node_ids:
                            if nid in active_entity_ids:
                                for e in entities_data:
                                    if e["entity_id"] == nid:
                                        lazy_node_id_to_label[nid] = (
                                            e.get("label")
                                            or e.get("entity_label")
                                            or "Entity"
                                        )
                                        break

                        label_to_node_ids_lazy: dict[str, list[str]] = {}
                        for nid, lbl in lazy_node_id_to_label.items():
                            if lbl not in label_to_node_ids_lazy:
                                label_to_node_ids_lazy[lbl] = []
                            label_to_node_ids_lazy[lbl].append(nid)

                        if label_to_node_ids_lazy:
                            nodes_data_lazy = await self._graph_search_repo.fetch_nodes_by_ids(
                                label_to_node_ids_lazy
                            )
                            logger.info(
                                "Optimized lazy hydration: fetched %d nodes for graph entity details",
                                len(nodes_data_lazy),
                            )
                            node_lookup: dict[str, dict[str, Any]] = {
                                n["id"]: n for n in nodes_data_lazy
                            }
                            entity_lookup: dict[str, GraphEntity] = {}

                            for edge in lazy_edges_data:
                                try:
                                    source_id = edge["source_id"]
                                    target_id = edge["target_id"]
                                except (KeyError, TypeError):
                                    continue

                                for nid in (source_id, target_id):
                                    if nid in active_entity_ids:
                                        if nid not in entity_lookup:
                                            nd = node_lookup.get(nid, {})
                                            entity_lookup[nid] = GraphEntity(
                                                entity_id=nid,
                                                entity_label=nd.get("label", "Unknown"),
                                                entity_name=nd.get("name"),
                                                properties=nd.get("properties", {}),
                                                relationships=[],
                                            )
                                        ge = entity_lookup[nid]
                                        rel = GraphEdge(
                                            source_id=edge.get("source_id", ""),
                                            source_label=edge.get("source_label", ""),
                                            source_name=edge.get("source_name"),
                                            relation_type=edge.get("relation_type", ""),
                                            target_id=edge.get("target_id", ""),
                                            target_label=edge.get("target_label", ""),
                                            target_name=edge.get("target_name"),
                                        )
                                        if rel not in ge.relationships:
                                            ge.relationships.append(rel)

                            graph_entities = list(entity_lookup.values())

            except Exception:
                logger.warning("Failed to build graph entities with lazy hydration", exc_info=True)
                graph_entities = []

        results: list[SearchResultItem] = []
        seen_texts: set[str] = set()
        for item in merged:
            row = item["row"]
            text = (row.get("content") or row.get("transcription") or "").strip().lower()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)

            author_id: int | None = None
            author_name: str | None = None
            if payload.include_author_info:
                author_id = row.get("account_id")
                author_name = row.get("account_title")

            results.append(
                SearchResultItem(
                    post_id=row["id"],
                    account_id=row["account_id"],
                    text=row.get("content") or row.get("transcription") or "",
                    score=item["final_score"],
                    vector_score=item["vector_score"],
                    graph_score=item["graph_score"],
                    er_score=item["er_score"],
                    created_at=row.get("created_at"),
                    url=item["url"],
                    author_id=author_id,
                    author_name=author_name,
                    boosted=item["in_graph"],
                )
            )

        return SearchResponse(results=results, graph_entities=graph_entities)
