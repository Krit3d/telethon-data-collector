"""Search router for semantic search operations."""

import asyncio
from fastapi import APIRouter, Depends, HTTPException
import logging

from src.api.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    GraphEdge,
    GraphEntity,
)
from src.embeddings.qdrant_service import QdrantService
from src.api.dependencies import get_qdrant, get_db
from src.db.database import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["Search"])


@router.post("", response_model=SearchResponse)
async def search_posts(
    payload: SearchRequest,
    qdrant: QdrantService = Depends(get_qdrant),
    db: Database = Depends(get_db),
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
        db: Database instance (injected).

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
        entity_results = []
        entity_id_to_score = {}
        for entity in entities_data:
            entity_id = entity["entity_id"]
            entity_results.append(entity)
            entity_id_to_score[entity_id] = entity["score"]

        entity_ids = [e["entity_id"] for e in entity_results]

        # Fetch full post records from PostgreSQL
        post_ids = [item["post_id"] for item in posts_data]
        posts_dict = await db.get_posts_by_ids(post_ids) if post_ids else {}

        # Build initial results with Qdrant scores
        merged_results = []
        for item in posts_data:
            post = posts_dict.get(item["post_id"])
            if post is None:
                logger.warning(
                    "Post ID %d from Qdrant results not found in PostgreSQL",
                    item["post_id"],
                )
                continue

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
                }
            )

        # Fetch graph relationships to identify posts connected to matched entities
        connected_post_ids = set()
        entity_to_connected_posts = {}  # entity_id -> list of post IDs in graph

        if entity_ids:
            try:
                edges_data = await db.get_subgraph_for_entities(entity_ids)
                for edge in edges_data:
                    source_id = edge["source_id"]
                    target_id = edge["target_id"]
                    source_label = edge["source_label"]
                    target_label = edge["target_label"]

                    # Identify Post nodes (label should be 'Post')
                    # The graph connects entities to posts via relationships
                    if source_label == "Post" and target_id in entity_ids:
                        connected_post_ids.add(source_id)
                        if target_id not in entity_to_connected_posts:
                            entity_to_connected_posts[target_id] = []
                        entity_to_connected_posts[target_id].append(source_id)

                    if target_label == "Post" and source_id in entity_ids:
                        connected_post_ids.add(target_id)
                        if source_id not in entity_to_connected_posts:
                            entity_to_connected_posts[source_id] = []
                        entity_to_connected_posts[source_id].append(target_id)
            except Exception as e:
                logger.warning(
                    "Failed to fetch graph for post-entity connection",
                    exc_info=e,
                )

        # Apply score boost for posts connected to high-scoring entities
        for result in merged_results:
            post_id_str = str(result["post_id"])
            base_score = result["score"]
            boost = 0.0

            if post_id_str in connected_post_ids:
                # Find which entities this post is connected to
                connected_entity_scores = []
                for (
                    entity_id,
                    connected_posts,
                ) in entity_to_connected_posts.items():
                    if post_id_str in connected_posts:
                        entity_score = entity_id_to_score.get(entity_id, 0.0)
                        connected_entity_scores.append(entity_score)

                if connected_entity_scores:
                    # Boost by the maximum entity score (could also use average)
                    max_entity_score = max(connected_entity_scores)
                    # Weighted combination: 70% post score, 30% best entity score
                    boost = max_entity_score * 0.3
                    result["score"] = base_score + boost
                    result["boosted"] = True
                else:
                    result["boosted"] = False
            else:
                result["boosted"] = False

        # Re-sort by final score
        merged_results.sort(key=lambda x: x["score"], reverse=True)

        # Limit to requested number
        merged_results = merged_results[: payload.limit]

        # Build SearchResultItem objects
        final_results = []
        for result in merged_results:
            item = SearchResultItem(
                post_id=result["post_id"],
                channel_id=result["channel_id"],
                text=result["text"],
                score=result["score"],
                created_at=result["created_at"],
                url=result["url"],
            )
            final_results.append(item)

        # Fetch and include author info if requested
        if payload.include_author_info:
            # Collect unique channel IDs from results
            channel_ids = list(set(r.channel_id for r in final_results))
            # Fetch channel details from database
            channels_dict = (
                await db.get_channels_batch(channel_ids) if channel_ids else {}
            )

            # Attach author info to each result
            for result in final_results:
                channel = channels_dict.get(result.channel_id)
                if channel:
                    result.author_id = channel.id
                    result.author_name = channel.title

        # Fetch graph context and group by entity
        graph_entities = []
        graph_context = []  # Keep for backward compatibility

        if entity_ids:
            try:
                edges_data = await db.get_subgraph_for_entities(entity_ids)
                graph_context = [
                    GraphEdge(
                        source_id=edge["source_id"],
                        source_label=edge["source_label"],
                        source_name=edge["source_name"],
                        relation_type=edge["relation_type"],
                        target_id=edge["target_id"],
                        target_label=edge["target_label"],
                        target_name=edge["target_name"],
                    )
                    for edge in edges_data
                ]

                # Group edges by entity to create GraphEntity objects
                # Collect all unique node IDs from edges
                node_ids = set()
                for edge in edges_data:
                    node_ids.add(edge["source_id"])
                    node_ids.add(edge["target_id"])

                # Fetch full node details for all nodes in the subgraph
                if node_ids:
                    nodes_data = await db.get_nodes_by_ids(list(node_ids))

                    # Build node lookup: node_id -> node data
                    node_lookup = {}
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
                                    node_data = node_lookup.get(node_id, {})
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
                    "Failed to fetch graph context",
                    exc_info=e,
                    extra={"entity_count": len(entity_ids)},
                )
                # Continue without graph context

        return SearchResponse(
            results=final_results,
            graph_context=graph_context,
            graph_entities=graph_entities,
        )

    except Exception as e:
        logger.exception("Search endpoint failed")
        raise HTTPException(status_code=500, detail="Search operation failed.")
