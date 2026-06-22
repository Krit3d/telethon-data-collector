import asyncio
import logging
import re
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas import (
    GraphEdge,
    GraphEntity,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from src.api.dependencies import get_db, get_graph_repo, get_qdrant
from src.db.database import Database
from src.graph.db.graph_repo import GraphRepository
from src.embeddings.qdrant_service import QdrantService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["Search"])

RRF_K: int = 60
_CONTENT_LABELS: frozenset[str] = frozenset({"content", "event", "publication"})
_SEMANTIC_RELATIONS: frozenset[str] = frozenset(
    {"COVERS_TOPIC", "COMPETES_WITH", "SIMILAR_TO"}
)
_DIRECT_CONNECT_WEIGHT: float = 1.0
_SEMANTIC_TRAVERSAL_WEIGHT: float = 0.6
_BRAND_TRAVERSAL_WEIGHT: float = 0.5
_ER_BOOST_WEIGHT: float = 0.1


def _strip_surrounding_quotes(val: Any) -> Any:
    if isinstance(val, str) and len(val) >= 2 and val[0] == '"' and val[-1] == '"':
        return val[1:-1]
    return val


def _build_content_url(content: Any) -> str:
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


def _build_node_id_to_post_id(
    nodes_data: list[dict[str, Any]],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for node in nodes_data:
        label = (node.get("label") or "").lower()
        if label not in _CONTENT_LABELS:
            continue
        node_id = node.get("id", "")
        props = node.get("properties", {})
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
    return result


def _build_post_id_to_er(
    nodes_data: list[dict[str, Any]],
    node_id_to_post_id: dict[str, int],
) -> dict[int, float]:
    result: dict[int, float] = {}
    for node in nodes_data:
        node_id = node.get("id", "")
        if node_id not in node_id_to_post_id:
            continue
        post_id = node_id_to_post_id[node_id]
        props = node.get("properties", {})
        raw_er = props.get("engagement_rate", 0.0)
        try:
            result[post_id] = float(raw_er)
        except (ValueError, TypeError):
            result[post_id] = 0.0
    return result


def _traverse_graph_for_posts(
    edges_data: list[dict[str, Any]],
    entity_ids: set[str],
    entity_id_to_score: dict[str, float],
    node_id_to_post_id: dict[str, int],
) -> tuple[dict[int, float], dict[str, set[int]]]:
    graph_post_scores: dict[int, float] = {}
    entity_connected: dict[str, set[int]] = defaultdict(set)

    adj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges_data:
        adj[edge["source_id"]].append(edge)
        adj[edge["target_id"]].append(edge)

    semantic_nb: dict[str, set[str]] = defaultdict(set)
    brand_nodes: set[str] = set()
    for edge in edges_data:
        src_id, tgt_id = edge["source_id"], edge["target_id"]
        rel_type = edge["relation_type"]
        if rel_type in _SEMANTIC_RELATIONS:
            semantic_nb[src_id].add(tgt_id)
            semantic_nb[tgt_id].add(src_id)
        if str(src_id).startswith("brand_") or str(tgt_id).startswith("brand_"):
            brand_nodes.add(src_id)
            brand_nodes.add(tgt_id)

    def _update(cid: int | None, score: float) -> None:
        if cid is not None and score > graph_post_scores.get(cid, 0.0):
            graph_post_scores[cid] = score

    for edge in edges_data:
        src_id, tgt_id = edge["source_id"], edge["target_id"]
        src_label = (edge["source_label"] or "").lower()
        tgt_label = (edge["target_label"] or "").lower()

        if src_label in _CONTENT_LABELS and tgt_id in entity_ids:
            cid = node_id_to_post_id.get(src_id)
            _update(cid, entity_id_to_score.get(tgt_id, 0.0) * _DIRECT_CONNECT_WEIGHT)
            if cid is not None:
                entity_connected[tgt_id].add(cid)
        elif tgt_label in _CONTENT_LABELS and src_id in entity_ids:
            cid = node_id_to_post_id.get(tgt_id)
            _update(cid, entity_id_to_score.get(src_id, 0.0) * _DIRECT_CONNECT_WEIGHT)
            if cid is not None:
                entity_connected[src_id].add(cid)

    visited_pairs: set[tuple[str, str]] = set()
    for eid in entity_ids:
        escore = entity_id_to_score.get(eid, 0.0)
        if escore <= 0:
            continue
        for nb in semantic_nb.get(eid, set()):
            pair = (min(eid, nb), max(eid, nb))
            if pair in visited_pairs:
                continue
            visited_pairs.add(pair)
            weighted = escore * _SEMANTIC_TRAVERSAL_WEIGHT
            for edge in adj.get(nb, []):
                ns, nt = edge["source_id"], edge["target_id"]
                nsl = (edge["source_label"] or "").lower()
                ntl = (edge["target_label"] or "").lower()
                if nsl in _CONTENT_LABELS and nt == nb:
                    _update(node_id_to_post_id.get(ns), weighted)
                elif ntl in _CONTENT_LABELS and ns == nb:
                    _update(node_id_to_post_id.get(nt), weighted)

    avg_entity_score = (
        sum(entity_id_to_score.values()) / len(entity_id_to_score)
        if entity_id_to_score
        else 0.0
    )
    brand_weighted = avg_entity_score * _BRAND_TRAVERSAL_WEIGHT
    for brand_id in brand_nodes:
        for edge in adj.get(brand_id, []):
            bs, bt = edge["source_id"], edge["target_id"]
            bsl = (edge["source_label"] or "").lower()
            btl = (edge["target_label"] or "").lower()
            cid: int | None = None
            if bsl in _CONTENT_LABELS and bt == brand_id:
                cid = node_id_to_post_id.get(bs)
            elif btl in _CONTENT_LABELS and bs == brand_id:
                cid = node_id_to_post_id.get(bt)
            if cid is not None and cid not in graph_post_scores:
                graph_post_scores[cid] = brand_weighted

    return graph_post_scores, dict(entity_connected)


def _compute_rrf_scores(
    vector_ranked: list[int],
    graph_ranked: list[int],
) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    for rank, cid in enumerate(vector_ranked, start=1):
        scores[cid] += 1.0 / (RRF_K + rank)
    for rank, cid in enumerate(graph_ranked, start=1):
        scores[cid] += 1.0 / (RRF_K + rank)
    return dict(scores)


@router.post("", response_model=SearchResponse)
async def search_content(
    payload: SearchRequest,
    qdrant: QdrantService = Depends(get_qdrant),
    db: Database = Depends(get_db),
    graph_repo: GraphRepository = Depends(get_graph_repo),
) -> SearchResponse:
    try:
        fetch_limit = payload.limit * 3

        posts_data, entities_data = await asyncio.gather(
            qdrant.search_posts(
                query=payload.query,
                limit=fetch_limit,
                score_threshold=payload.score_threshold,
            ),
            qdrant.search_entities(
                query=payload.query,
                limit=fetch_limit,
                score_threshold=payload.score_threshold,
            ),
        )

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

        vector_ranked: list[int] = [item["post_id"] for item in posts_data]
        vector_set: set[int] = set(vector_ranked)

        graph_post_scores: dict[int, float] = {}
        edges_data: list[dict[str, Any]] = []
        node_id_to_post_id: dict[str, int] = {}
        post_id_to_er: dict[int, float] = {}

        if entity_ids:
            try:
                raw_edges = await graph_repo.get_subgraph_for_entities(
                    list(entity_ids)
                )
                edges_data = [
                    {
                        **edge,
                        "source_id": _strip_surrounding_quotes(edge["source_id"]),
                        "target_id": _strip_surrounding_quotes(edge["target_id"]),
                    }
                    for edge in raw_edges
                ]
                logger.info("Graph subgraph returned %d edges", len(edges_data))

                node_ids: set[str] = set()
                for edge in edges_data:
                    node_ids.add(edge["source_id"])
                    node_ids.add(edge["target_id"])

                if node_ids:
                    nodes_data = await graph_repo.get_nodes_by_ids(list(node_ids))
                    node_id_to_post_id = _build_node_id_to_post_id(nodes_data)
                    post_id_to_er = _build_post_id_to_er(
                        nodes_data, node_id_to_post_id
                    )
                    logger.info(
                        "Built ID mapping for %d content nodes, "
                        "ER data for %d posts",
                        len(node_id_to_post_id),
                        len(post_id_to_er),
                    )

                graph_post_scores, _ = _traverse_graph_for_posts(
                    edges_data, entity_ids, entity_id_to_score, node_id_to_post_id
                )
            except Exception:
                logger.warning("Graph traversal failed", exc_info=True)

        graph_ranked: list[int] = [
            cid
            for cid, _ in sorted(
                graph_post_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]

        logger.info(
            "Vector ranked: %d items, Graph ranked: %d items",
            len(vector_ranked),
            len(graph_ranked),
        )

        rrf_scores = _compute_rrf_scores(vector_ranked, graph_ranked)

        all_content_ids: set[int] = set(rrf_scores.keys())
        posts_dict: dict[int, Any] = (
            await db.get_content_by_ids(list(all_content_ids))
            if all_content_ids
            else {}
        )

        merged: list[dict[str, Any]] = []
        for cid in rrf_scores:
            content = posts_dict.get(cid)
            if content is None:
                logger.warning(
                    "Content ID %d not found in PostgreSQL", cid
                )
                continue
            er_boost = post_id_to_er.get(cid, 0.0)
            final_score = rrf_scores[cid] * (1.0 + (er_boost * _ER_BOOST_WEIGHT))
            merged.append(
                {
                    "content": content,
                    "final_score": final_score,
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
                    text=content.content or "",
                    score=item["final_score"],
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
                    all_node_ids.add(edge["source_id"])
                    all_node_ids.add(edge["target_id"])

                if all_node_ids:
                    nodes_data_full = await graph_repo.get_nodes_by_ids(
                        list(all_node_ids)
                    )
                    node_lookup: dict[str, dict[str, Any]] = {
                        n["id"]: n for n in nodes_data_full
                    }
                    entity_lookup: dict[str, GraphEntity] = {}

                    for edge in edges_data:
                        for nid in (edge["source_id"], edge["target_id"]):
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
                                    source_id=edge["source_id"],
                                    source_label=edge["source_label"],
                                    source_name=edge["source_name"],
                                    relation_type=edge["relation_type"],
                                    target_id=edge["target_id"],
                                    target_label=edge["target_label"],
                                    target_name=edge["target_name"],
                                )
                                if rel not in ge.relationships:
                                    ge.relationships.append(rel)

                    graph_entities = list(entity_lookup.values())
            except Exception:
                logger.warning(
                    "Failed to build graph entities", exc_info=True
                )

        return SearchResponse(results=results, graph_entities=graph_entities)

    except Exception:
        logger.exception("Search endpoint failed")
        raise HTTPException(status_code=500, detail="Search operation failed.")
