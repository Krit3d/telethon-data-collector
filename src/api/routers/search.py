"""Search router for semantic search operations."""

import asyncio
from fastapi import APIRouter, Depends, HTTPException
import logging

from src.api.schemas import SearchRequest, SearchResponse, SearchResultItem, GraphEdge
from src.embeddings.qdrant_service import QdrantService
from src.api.dependencies import get_qdrant, get_db
from src.db.database import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["Search"])


@router.post("", response_model=SearchResponse)
async def search_posts(
    payload: SearchRequest,
    qdrant: QdrantService = Depends(get_qdrant),
    db: Database = Depends(get_db)
) -> SearchResponse:
    """Hybrid search for posts with graph context.

    Performs concurrent semantic search for posts and entities, then fetches
    the graph subgraph for the matched entities.

    Args:
        payload: Search request with query, limit, and score threshold.
        qdrant: Qdrant service instance (injected).
        db: Database instance (injected).

    Returns:
        SearchResponse with matching posts and graph context.

    Raises:
        HTTPException: If search operation fails.
    """
    
    try:
        # Concurrently search for posts and entities
        posts_task = qdrant.search_posts(
            query=payload.query,
            limit=payload.limit,
            score_threshold=payload.score_threshold
        )
        entities_task = qdrant.search_entities(
            query=payload.query,
            limit=payload.limit,
            score_threshold=payload.score_threshold
        )
        
        posts_data, entity_ids = await asyncio.gather(posts_task, entities_task)
        
        # Fetch full post records from PostgreSQL
        post_ids = [item["post_id"] for item in posts_data]
        posts_dict = await db.get_posts_by_ids(post_ids) if post_ids else {}
        
        # Merge Qdrant scores with PostgreSQL records
        merged_results = []
        for item in posts_data:
            post = posts_dict.get(item["post_id"])
            if post is None:
                logger.warning(
                    "Post ID %d from Qdrant results not found in PostgreSQL",
                    item["post_id"]
                )
                continue
            
            url = f"https://t.me/c/{post.channel_id}/{post.message_id}"
            
            merged_results.append(
                SearchResultItem(
                    post_id=post.id,
                    channel_id=post.channel_id,
                    text=post.content or "",
                    score=item["score"],
                    created_at=post.created_at,
                    url=url
                )
            )
        
        # Ensure ordering by score
        merged_results.sort(key=lambda x: x.score, reverse=True)
        
        # Fetch graph context if entities were found
        graph_context = []
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
                        target_name=edge["target_name"]
                    )
                    for edge in edges_data
                ]
            except Exception as e:
                logger.warning(
                    "Failed to fetch graph context",
                    exc_info=e,
                    extra={"entity_count": len(entity_ids)}
                )
                # Continue without graph context - don't fail the whole search
        
        return SearchResponse(
            results=merged_results,
            graph_context=graph_context
        )
        
    except Exception as e:
        logger.exception("Search endpoint failed")
        raise HTTPException(status_code=500, detail="Search operation failed.")
