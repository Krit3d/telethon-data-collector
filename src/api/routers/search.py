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


def _clean_content_id(node_id: Any) -> int | None:
    """
    Clean and convert a graph node ID to a PostgreSQL-compatible integer content ID.

    Handles cases where the node ID is:
    - An integer (returned as-is)
    - A string with "content_" prefix (e.g., "content_12345" -> 12345)
    - A plain numeric string (e.g., "12345" -> 12345)

    Args:
        node_id: The node ID from graph edge data (source_id or target_id).

    Returns:
        Integer content ID if conversion succeeds, None otherwise.

    Raises:
        No exceptions; all conversion errors are caught and logged.
    """
    if node_id is None:
        return None

    # Convert to string for prefix checking
    node_id_str = str(node_id)

    # Strip "content_" prefix if present
    if node_id_str.startswith("content_"):
        node_id_str = node_id_str[8:]  # Remove "content_" prefix

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
async def search_content(
    payload: SearchRequest,
    qdrant: QdrantService = Depends(get_qdrant),
    db: Database = Depends(get_db),
    graph_repo: GraphRepository = Depends(get_graph_repo),
) -> SearchResponse:
    """Hybrid search for content with graph entities and intelligent ranking.

    Performs concurrent semantic search for content and entities, then:
    - Applies Reciprocal Rank Fusion to combine content and entity relevance scores
    - Boosts content that has associated highly relevant entities
    - Fetches graph relationships and groups them by entity
    - Optionally includes author (Actor) node details for found content

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

        # Extract vector post IDs to scores mapping
        vector_scores: dict[int, float] = {
            item["post_id"]: item["score"] for item in posts_data
        }

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
                        "graph_entities will be empty.",
                        len(entity_ids),
                    )

                for edge in edges_data:
                    source_id: Any = edge["source_id"]
                    target_id: Any = edge["target_id"]
                    source_label: str = edge["source_label"]
                    target_label: str = edge["target_label"]

                    # Identify Content nodes (label should be 'Content')
                    # The graph connects entities to content via relationships
                    # Clean content IDs by stripping "content_" prefix and converting to int
                    # Use case-insensitive comparison to handle variations like "CONTENT", "content", etc.
                    if (
                        source_label.lower() == "content"
                        and target_id in entity_ids
                    ):
                        cleaned_source_id = _clean_content_id(source_id)
                        if cleaned_source_id is not None:
                            connected_post_ids.add(cleaned_source_id)
                            if target_id not in entity_to_connected_posts:
                                entity_to_connected_posts[target_id] = []
                            entity_to_connected_posts[target_id].append(
                                cleaned_source_id
                            )

                    if (
                        target_label.lower() == "content"
                        and source_id in entity_ids
                    ):
                        cleaned_target_id = _clean_content_id(target_id)
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

        # Combine vector content IDs and graph-connected content IDs
        all_content_ids: set[int] = set(vector_scores.keys()) | connected_post_ids

        # Log ID sets for debugging
        logger.info("Vector search content IDs: %s", list(vector_scores.keys()))
        logger.info("Graph connected content IDs: %s", connected_post_ids)
        logger.info("Union of all content IDs: %s", all_content_ids)

        # Fetch ALL these content from the database
        posts_dict: dict[int, Any] = (
            await db.get_content_by_ids(list(all_content_ids)) if all_content_ids else {}
        )

        # Build merged results from all_content_ids
        merged_results: list[dict[str, Any]] = []
        for content_id in all_content_ids:
            content = posts_dict.get(content_id)
            if content is None:
                logger.warning(
                    "Content ID %d from combined results not found in PostgreSQL",
                    content_id,
                )
                continue

            # Get base score from vector search (0.0 if not in vector results)
            base_score: float = vector_scores.get(content_id, 0.0)

            # Calculate boost if content is connected to high-scoring entities
            boost: float = 0.0
            if content_id in connected_post_ids:
                connected_entity_scores: list[float] = []
                for (
                    entity_id,
                    connected_posts,
                ) in entity_to_connected_posts.items():
                    if content_id in connected_posts:
                        entity_score: float = entity_id_to_score.get(
                            entity_id, 0.0
                        )
                        connected_entity_scores.append(entity_score)

                if connected_entity_scores:
                    # Boost by the maximum entity score
                    max_entity_score = max(connected_entity_scores)
                    WEIGHT_FACTOR = 0.5  # Adjust how much entity presence matters
                    boost = max_entity_score * WEIGHT_FACTOR

            # Apply RRF/Boost formula: asymptotically approach 1.0 without exceeding it
            final_score: float = base_score + (1.0 - base_score) * boost

            # Build URL: use account username if available, otherwise fall back to account_id
            account_username = (
                getattr(content.account, "username", None)
                if content.account
                else None
            )
            if account_username:
                url = f"https://t.me/{account_username}/{content.message_id}"
            else:
                url = f"https://t.me/c/{content.account_id}/{content.message_id}"

            merged_results.append(
                {
                    "content_obj": content,
                    "score": final_score,
                    "url": url,
                    "account_id": content.account_id,
                    "text": content.content or "",
                    "created_at": content.created_at,
                    "content_id": content.id,
                    "account": content.account,  # Include account for author info
                    "boosted": boost > 0,
                }
            )

            if boost > 0:
                logger.info(
                    "Content %d: base score %.4f, final score %.4f, boost amount %.4f",
                    content_id, base_score, final_score, boost
                )

        # Re-sort by final score
        merged_results.sort(key=lambda x: x["score"], reverse=True)

        # Limit to requested number
        merged_results = merged_results[: payload.limit]

        # Build SearchResultItem objects with optional author info
        final_results: list[SearchResultItem] = []
        for result in merged_results:
            content = result["content_obj"]
            account = content.account  # Eager-loaded account (Account model)

            # Determine author info if requested
            author_id = None
            author_name = None
            if payload.include_author_info:
                if account:
                    author_id = account.id
                    author_name = account.title
                else:
                    # Gracefully handle missing account
                    author_id = content.account_id
                    author_name = "Unknown"

            item = SearchResultItem(
                post_id=result["content_id"],
                account_id=result["account_id"],
                text=result["text"],
                score=result["score"],
                created_at=result["created_at"],
                url=result["url"],
                author_id=author_id,
                author_name=author_name,
                boosted=result.get("boosted", False),
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
