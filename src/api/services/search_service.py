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


def _build_content_url(content: Any) -> str:
    try:
        account = getattr(content, "account", None)
        platform = getattr(account, "platform", "TELEGRAM") if account else "TELEGRAM"
        username = getattr(account, "username", None) if account else None
        if platform == "TELEGRAM" and getattr(content, "message_id", None) is not None:
            if username:
                return f"https://t.me/{username}/{content.message_id}"
            return f"https://t.me/c/{content.account_id}/{content.message_id}"
        return (
            f"https://platform/{platform.lower()}/content/"
            f"{getattr(content, 'platform_content_id', 'unknown')}"
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


def _author_matches_location(
    content: Any,
    edges_data: list[dict[str, Any]],
    location: str,
) -> bool:
    try:
        account = getattr(content, "account", None)
        if account is not None:
            raw_meta = getattr(account, "raw_metadata", None) or {}
            if isinstance(raw_meta, dict):
                for key in ("geo_data", "location"):
                    geo_obj = raw_meta.get(key)
                    if isinstance(geo_obj, dict):
                        for field in ("country", "city", "name", "region"):
                            val = geo_obj.get(field)
                            if isinstance(val, str) and location.lower() in val.lower():
                                return True
        if account is not None:
            platform = getattr(account, "platform", None) or "TELEGRAM"
            author_node_id = f"actor_{platform.lower()}_{content.account_id}"
            loc_lower = location.lower()
            for edge in edges_data:
                try:
                    if edge.get("relation_type") != "BASED_IN":
                        continue
                    if edge.get("source_id") != author_node_id:
                        continue
                    target_name = edge.get("target_name") or ""
                    target_id = edge.get("target_id") or ""
                    if loc_lower in target_name.lower() or loc_lower in target_id.lower():
                        return True
                except (TypeError, AttributeError):
                    continue
    except (TypeError, AttributeError):
        pass
    return False


def _compute_kag_scores(
    vector_scores: dict[int, float],
    graph_scores: dict[int, float],
    post_id_to_er: dict[int, float],
) -> dict[int, tuple[float, float, float, float]]:
    w_graph = 0.4
    w_vector = 0.4
    w_er = 0.2

    all_ids: set[int] = set(vector_scores.keys()) | set(graph_scores.keys())
    results: dict[int, tuple[float, float, float, float]] = {}

    for cid in all_ids:
        vector_score = vector_scores.get(cid, 0.0)
        g_score = graph_scores.get(cid, 0.0)
        er_score = post_id_to_er.get(cid, 0.0)

        final_score = (
            w_graph * g_score
            + w_vector * vector_score
            + w_er * min(1.0, er_score)
        )

        results[cid] = (final_score, vector_score, g_score, min(1.0, er_score))

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
        fetch_limit = payload.limit * 3

        posts_data, entities_data = await asyncio.gather(
            self._qdrant.search_posts(
                query=payload.query,
                limit=fetch_limit,
                score_threshold=payload.score_threshold,
                min_followers=payload.min_followers,
                min_engagement_rate=payload.min_engagement_rate,
                platform=getattr(payload, "platform", None),
            ),
            self._qdrant.search_entities(
                query=payload.query,
                limit=fetch_limit,
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

        vector_scores: dict[int, float] = {
            item["post_id"]: item["score"] for item in posts_data
        }
        vector_set: set[int] = set(vector_scores.keys())

        post_id_to_er: dict[int, float] = {}
        for item in posts_data:
            raw_er = item.get("engagement_rate", 0.0)
            try:
                post_id_to_er[item["post_id"]] = float(raw_er)
            except (ValueError, TypeError):
                post_id_to_er[item["post_id"]] = 0.0

        graph_post_scores: dict[int, float] = {}
        edges_data: list[dict[str, Any]] = []
        node_id_to_post_id: dict[str, int] = {}

        if entity_ids:
            try:
                graph_post_entities = await self._graph_search_repo.search_posts_by_entities(
                    {"Entity": list(entity_ids)}
                )
                logger.info(
                    "Graph Cypher search returned %d matched posts",
                    len(graph_post_entities),
                )

                for pid, connected_entities in graph_post_entities.items():
                    entity_scores = [entity_id_to_score.get(e_id, 0.0) for e_id in connected_entities]
                    graph_post_scores[pid] = max(entity_scores) if entity_scores else 0.0

                edges_data = await self._graph_search_repo.fetch_subgraph_edges(
                    list(entity_ids)
                )
                logger.info("Graph subgraph returned %d edges for entity display", len(edges_data))

                node_ids: set[str] = set()
                for edge in edges_data:
                    try:
                        node_ids.add(edge["source_id"])
                        node_ids.add(edge["target_id"])
                    except (KeyError, TypeError):
                        continue

                if node_ids:
                    nodes_data = await self._graph_search_repo.fetch_nodes_by_ids(
                        {"Entity": list(node_ids), "Event": list(node_ids)}
                    )
                    node_id_to_post_id = _build_node_id_to_post_id(nodes_data)
                    post_id_to_er.update(_build_post_id_to_er(
                        nodes_data, node_id_to_post_id
                    ))
                    logger.info(
                        "Built ID mapping for %d content nodes, "
                        "ER data for %d posts",
                        len(node_id_to_post_id),
                        len(post_id_to_er),
                    )

            except Exception:
                logger.warning("Graph search failed", exc_info=True)

        logger.info(
            "Vector scored: %d items, Graph scored: %d items",
            len(vector_scores),
            len(graph_post_scores),
        )

        kag_scores = _compute_kag_scores(vector_scores, graph_post_scores, post_id_to_er)

        all_content_ids: set[int] = set(kag_scores.keys())

        safe_content_ids: set[int] = {
            cid for cid in all_content_ids
            if _MIN_INT32 <= cid <= _MAX_INT32
        }

        filtered_count = len(all_content_ids) - len(safe_content_ids)
        if filtered_count > 0:
            logger.warning(
                "Filtered out %d content IDs exceeding PostgreSQL int32 range "
                "(min=%d, max=%d). Dropped IDs: %s",
                filtered_count,
                _MIN_INT32,
                _MAX_INT32,
                sorted(all_content_ids - safe_content_ids),
            )

        posts_dict: dict[int, Any] = (
            await self._db.get_content_by_ids(list(safe_content_ids))
            if safe_content_ids
            else {}
        )

        merged: list[dict[str, Any]] = []
        for cid in kag_scores:
            content = posts_dict.get(cid)
            if content is None:
                logger.warning(
                    "Content ID %d not found in PostgreSQL", cid
                )
                continue

            account = getattr(content, "account", None)
            if account is None or getattr(account, "status", None) != "verified":
                continue

            if payload.location and payload.location.strip():
                if not _author_matches_location(content, edges_data, payload.location):
                    continue

            final_score, vec_score, norm_graph_score, er_score = kag_scores[cid]
            merged.append(
                {
                    "content": content,
                    "final_score": final_score,
                    "vector_score": vec_score,
                    "graph_score": norm_graph_score,
                    "er_score": er_score,
                    "url": _build_content_url(content),
                    "in_graph": cid in graph_post_scores,
                    "in_vector": cid in vector_set,
                }
            )

        merged.sort(key=lambda x: x["final_score"], reverse=True)
        merged = merged[: payload.limit]

        results: list[SearchResultItem] = []
        for item in merged:
            content = item["content"]
            account = getattr(content, "account", None)
            author_id: int | None = None
            author_name: str | None = None
            if payload.include_author_info:
                if account:
                    author_id = account.id
                    author_name = account.title
                else:
                    author_id = content.account_id
                    author_name = "Unknown"

            results.append(
                SearchResultItem(
                    post_id=content.id,
                    account_id=content.account_id,
                    text=content.content or content.transcription or "",
                    score=item["final_score"],
                    vector_score=item["vector_score"],
                    graph_score=item["graph_score"],
                    er_score=item["er_score"],
                    created_at=content.created_at,
                    url=item["url"],
                    author_id=author_id,
                    author_name=author_name,
                    boosted=item["in_graph"] and not item["in_vector"],
                )
            )

        graph_entities: list[GraphEntity] = []
        if entity_ids and edges_data:
            try:
                all_node_ids: set[str] = set()
                for edge in edges_data:
                    try:
                        all_node_ids.add(edge["source_id"])
                        all_node_ids.add(edge["target_id"])
                    except (KeyError, TypeError):
                        continue

                if all_node_ids:
                    nodes_data_full = await self._graph_search_repo.fetch_nodes_by_ids(
                        {"Entity": list(all_node_ids), "Event": list(all_node_ids)}
                    )
                    node_lookup: dict[str, dict[str, Any]] = {
                        n["id"]: n for n in nodes_data_full
                    }
                    entity_lookup: dict[str, GraphEntity] = {}

                    for edge in edges_data:
                        try:
                            source_id = edge["source_id"]
                            target_id = edge["target_id"]
                        except (KeyError, TypeError):
                            continue
                        for nid in (source_id, target_id):
                            if nid in entity_ids:
                                if nid not in entity_lookup:
                                    nd = node_lookup.get(nid, {})
                                    entity_lookup[nid] = GraphEntity(
                                        entity_id=nid,
                                        entity_label=nd.get("label", ""),
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
                logger.warning("Failed to build graph entities", exc_info=True)

        return SearchResponse(results=results, graph_entities=graph_entities)
