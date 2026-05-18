"""Search router for semantic search operations."""

import asyncio
from fastapi import APIRouter, Depends, HTTPException
import logging
from typing import Any

from src.api.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    GraphEdge,
    GraphEntity,
)
from src.embeddings.qdrant_service import QdrantService
from src.api.dependencies import get_qdrant, get_db, get_graph_repo
from src.db.database import Database
from src.db.graph_repo import GraphRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["Search"])


def _clean_post_id(node_id: Any) -> int | None:
    """
    Clean and convert a graph node ID to a PostgreSQL-compatible integer post ID.

    Handles cases where the node ID is:
    - An integer (returned as-is)
    - A string with "post_" prefix (e.g., "post_12345" -> 12345)
    - A plain numeric string (e.g., "12345" -> 12345)

    Args:
        node_id: The node ID from graph edge data (source_id or target_id).

    Returns:
        Integer post ID if conversion succeeds, None otherwise.

    Raises:
        No exceptions; all conversion errors are caught and logged.
    """
    if node_id is None:
        return None

    # Convert to string for prefix checking
    node_id_str = str(node_id)

    # Strip "post_" prefix if present
    if node_id_str.startswith("post_"):
        node_id_str = node_id_str[5:]  # Remove "post_" prefix

    # Try to convert to integer
    try:
        return int(node_id_str)
    except (ValueError, TypeError) as e:
        logger.debug(
            "Failed to convert node_id to integer: %r (original: %r)",
            node_id_str,
            node_id,
            exc_info=e,
        )
        return None


@router.post("", response_model=SearchResponse)
async def search_posts(
    payload: SearchRequest,
    qdrant: QdrantService = Depends(get_qdrant),
    db: Database = Depends(get_db),
    graph_repo: GraphRepository = Depends(get_graph_repo),
) -> SearchResponse:
    """Hybrid search for posts with graph context and intelligent ranking.

    Performs concurrent semantic search for posts and entities, then:
    - Applies Reciprocal Rank Fusion to combine post and entity relevance scores
    - Boosts posts that have associated highly relevant entities
    - Fetches graph context and groups relationships by entity
    - Optionally includes author (Actor) node details for found posts

    Args:
        payload: Search request with query, limit, score_threshold, and include_author_info.
        qdrant: Qdrant service instance (injected).
        db: Database instance (injected) for ORM operations.
        graph_repo: GraphRepository instance (injected) for Apache AGE queries.

    Returns:
        SearchResponse with intelligently ranked posts, grouped graph entities, and optional author info.

    Raises:
        HTTPException: If search operation fails.
    """

    try:
        # Concurrently search for posts and entities with scores
        posts_task = qdrant.search_posts(
            query=payload.query,
            limit=payload.limit * 2,  # Fetch more to allow for entity overlap
            score_threshold=payload.score_threshold,
        )
        entities_task = qdrant.search_entities(
            query=payload.query,
            limit=payload.limit,
            score_threshold=payload.score_threshold,
        )

        posts_data, entities_data = await asyncio.gather(
            posts_task, entities_task
        )

        # Extract entity info with scores
        entity_results: list[dict[str, Any]] = []
        entity_id_to_score: dict[str, float] = {}
        for entity in entities_data:
            entity_id: str = entity["entity_id"]
            entity_results.append(entity)
            entity_id_to_score[entity_id] = entity["score"]

        entity_ids: list[str] = [e["entity_id"] for e in entity_results]

        # Log detailed entity matching information for debugging
        if entity_ids:
            scores = list(entity_id_to_score.values())
            max_score = max(scores) if scores else 0.0
            min_score = min(scores) if scores else 0.0
            logger.info(
                "Found %d entities in Qdrant for query. Score range: min=%.4f, max=%.4f",
                len(entity_ids),
                min_score,
                max_score,
            )
        else:
            logger.warning(
                "No entities found in Qdrant for query. Graph context will be empty."
            )

        # Fetch full post records from PostgreSQL (with channels eagerly loaded)
        post_ids: list[int] = [item["post_id"] for item in posts_data]
        posts_dict: dict[int, Any] = (
            await db.get_posts_by_ids(post_ids) if post_ids else {}
        )

        # Build initial results with Qdrant scores
        merged_results: list[dict[str, Any]] = []
        for item in posts_data:
            post = posts_dict.get(item["post_id"])
            if post is None:
                logger.warning(
                    "Post ID %d from Qdrant results not found in PostgreSQL",
                    item["post_id"],
                )
                continue

            # Build URL: use channel username if available, otherwise fall back to channel_id
            channel_username = (
                getattr(post.channel, "username", None)
                if post.channel
                else None
            )
            if channel_username:
                url = f"https://t.me/{channel_username}/{post.message_id}"
            else:
                url = f"https://t.me/c/{post.channel_id}/{post.message_id}"

            merged_results.append(
                {
                    "post_obj": post,
                    "score": item["score"],
                    "url": url,
                    "channel_id": post.channel_id,
                    "text": post.content or "",
                    "created_at": post.created_at,
                    "post_id": post.id,
                    "channel": post.channel,  # Include channel for author info
                }
            )

        # Fetch graph relationships to identify posts connected to matched entities
        connected_post_ids: set[int] = set()
        entity_to_connected_posts: dict[str, list[int]] = (
            {}
        )  # entity_id -> list of cleaned post IDs

        edges_data: list[dict[str, Any]] = []
        if entity_ids:
            try:
                edges_data = await graph_repo.get_subgraph_for_entities(
                    entity_ids
                )

                if edges_data:
                    # Strip extra quotes from source_id and target_id if they are string-represented agtypes
                    for edge in edges_data:
                        source_id = edge["source_id"]
                        target_id = edge["target_id"]

                        # Clean source_id: strip surrounding quotes if present (e.g., '"entity_123"' -> 'entity_123')
                        if (
                            isinstance(source_id, str)
                            and source_id.startswith('"')
                            and source_id.endswith('"')
                        ):
                            edge["source_id"] = source_id[1:-1]

                        # Clean target_id: strip surrounding quotes if present
                        if (
                            isinstance(target_id, str)
                            and target_id.startswith('"')
                            and target_id.endswith('"')
                        ):
                            edge["target_id"] = target_id[1:-1]

                    # Log first 3 edges for debugging graph structure
                    logger.info(
                        "First 3 edges from graph subgraph: %s",
                        edges_data[:3],
                    )
                else:
                    logger.warning(
                        "Graph subgraph returned 0 edges for %d entity IDs. "
                        "This may indicate: (a) graph has no edges yet, "
                        "(b) entity IDs do not match any nodes in the graph, "
                        "or (c) the graph query timed out over the VPN link. "
                        "graph_context and graph_entities will be empty.",
                        len(entity_ids),
                    )

                for edge in edges_data:
                    source_id: Any = edge["source_id"]
                    target_id: Any = edge["target_id"]
                    source_label: str = edge["source_label"]
                    target_label: str = edge["target_label"]

                    # Identify Post nodes (label should be 'Post')
                    # The graph connects entities to posts via relationships
                    # Clean post IDs by stripping "post_" prefix and converting to int
                    # Use case-insensitive comparison to handle variations like "POST", "post", etc.
                    if (
                        source_label.lower() == "post"
                        and target_id in entity_ids
                    ):
                        cleaned_source_id = _clean_post_id(source_id)
                        if cleaned_source_id is not None:
                            connected_post_ids.add(cleaned_source_id)
                            if target_id not in entity_to_connected_posts:
                                entity_to_connected_posts[target_id] = []
                            entity_to_connected_posts[target_id].append(
                                cleaned_source_id
                            )

                    if (
                        target_label.lower() == "post"
                        and source_id in entity_ids
                    ):
                        cleaned_target_id = _clean_post_id(target_id)
                        if cleaned_target_id is not None:
                            connected_post_ids.add(cleaned_target_id)
                            if source_id not in entity_to_connected_posts:
                                entity_to_connected_posts[source_id] = []
                            entity_to_connected_posts[source_id].append(
                                cleaned_target_id
                            )
            except Exception as e:
                logger.warning(
                    "Failed to fetch graph for post-entity connection: %s",
                    str(e),
                    exc_info=e,
                )

        # Apply score boost for posts connected to high-scoring entities
        for result in merged_results:
            post_id_int: int = result[
                "post_id"
            ]  # Already an integer from PostgreSQL
            base_score: float = result["score"]
            boost: float = 0.0

            if post_id_int in connected_post_ids:
                # Find which entities this post is connected to
                connected_entity_scores: list[float] = []
                for (
                    entity_id,
                    connected_posts,
                ) in entity_to_connected_posts.items():
                    if post_id_int in connected_posts:
                        entity_score: float = entity_id_to_score.get(
                            entity_id, 0.0
                        )
                        connected_entity_scores.append(entity_score)

                if connected_entity_scores:
                    # Boost by the maximum entity score (could also use average)
                    max_entity_score = max(connected_entity_scores)
                    WEIGHT_FACTOR = (
                        0.5  # Adjust how much entity presence matters
                    )
                    boost = max_entity_score * WEIGHT_FACTOR
                    # Asymptotically approach 1.0 without exceeding it
                    result["score"] = base_score + (1.0 - base_score) * boost
                    result["boosted"] = True
                else:
                    result["boosted"] = False
            else:
                result["boosted"] = False

        # Re-sort by final score
        merged_results.sort(key=lambda x: x["score"], reverse=True)

        # Limit to requested number
        merged_results = merged_results[: payload.limit]

        # Build SearchResultItem objects with optional author info
        final_results: list[SearchResultItem] = []
        for result in merged_results:
            post = result["post_obj"]
            channel = post.channel  # Eager-loaded channel

            # Determine author info if requested
            author_id = None
            author_name = None
            if payload.include_author_info:
                if channel:
                    author_id = channel.id
                    author_name = channel.title
                else:
                    # Gracefully handle missing channel
                    author_id = post.channel_id
                    author_name = "Unknown"

            item = SearchResultItem(
                post_id=result["post_id"],
                channel_id=result["channel_id"],
                text=result["text"],
                score=result["score"],
                created_at=result["created_at"],
                url=result["url"],
                media_url=post.media_url,
                author_id=author_id,
                author_name=author_name,
            )
            final_results.append(item)

        # Fetch graph entities and group by entity
        graph_entities: list[GraphEntity] = []

        if entity_ids and edges_data:
            try:
                # Group edges by entity to create GraphEntity objects
                # Collect all unique node IDs from edges
                node_ids: set[Any] = set()
                for edge in edges_data:
                    node_ids.add(edge["source_id"])
                    node_ids.add(edge["target_id"])

                # Fetch full node details for all nodes in the subgraph
                if node_ids:
                    nodes_data: list[dict[str, Any]] = (
                        await graph_repo.get_nodes_by_ids(list(node_ids))
                    )

                    # Build node lookup: node_id -> node data
                    node_lookup: dict[Any, dict[str, Any]] = {}
                    for node in nodes_data:
                        node_lookup[node["id"]] = node

                    # Build entity lookup: entity_id -> GraphEntity
                    entity_lookup = {}

                    # First, identify which nodes are entities (from our entity search results)
                    matched_entity_ids = set(entity_ids)

                    # Process each edge to build relationships per entity
                    for edge in edges_data:
                        # For both source and target, if they are in our matched entities,
                        # we want to include the relationship in that entity's relationships

                        for node_id in [edge["source_id"], edge["target_id"]]:
                            if node_id in matched_entity_ids:
                                # This is a matched entity
                                if node_id not in entity_lookup:
                                    # Get node details
                                    node_data: dict[str, Any] = node_lookup.get(
                                        node_id, {}
                                    )
                                    entity_lookup[node_id] = GraphEntity(
                                        entity_id=node_id,
                                        entity_label=node_data.get("label", ""),
                                        entity_name=node_data.get("name"),
                                        properties=node_data.get(
                                            "properties", {}
                                        ),
                                        relationships=[],
                                    )

                                # Add the edge as a relationship (from entity's perspective)
                                entity = entity_lookup[node_id]

                                # Create GraphEdge for this relationship
                                # The edge already contains source and target info
                                rel = GraphEdge(
                                    source_id=edge["source_id"],
                                    source_label=edge["source_label"],
                                    source_name=edge["source_name"],
                                    relation_type=edge["relation_type"],
                                    target_id=edge["target_id"],
                                    target_label=edge["target_label"],
                                    target_name=edge["target_name"],
                                )

                                # Avoid duplicate relationships
                                if rel not in entity.relationships:
                                    entity.relationships.append(rel)

                    # Convert entity lookup values to list
                    graph_entities = list(entity_lookup.values())

            except Exception as e:
                logger.warning(
                    "Failed to fetch graph context: %s",
                    str(e),
                    exc_info=e,
                    extra={"entity_count": len(entity_ids)},
                )
                # Continue without graph context

        return SearchResponse(
            results=final_results,
            graph_entities=graph_entities,
        )

    except Exception as e:
        logger.exception("Search endpoint failed")
        raise HTTPException(status_code=500, detail="Search operation failed.")
