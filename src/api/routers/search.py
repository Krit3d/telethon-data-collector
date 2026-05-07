"""Search router for semantic search operations."""

from fastapi import APIRouter, Depends, HTTPException
import logging

from src.api.schemas import SearchRequest, SearchResponse, SearchResultItem
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
    """Search for posts using semantic similarity.

    Args:
        payload: Search request with query, limit, and score threshold.
        qdrant: Qdrant service instance (injected).
        db: Database instance (injected).

    Returns:
        SearchResponse with list of matching posts.

    Raises:
        HTTPException: If search operation fails.
    """
    
    try:
        results_data = await qdrant.search_posts(
            query=payload.query,
            limit=payload.limit,
            score_threshold=payload.score_threshold
        )
        
        if not results_data:
            return SearchResponse(results=[])
        
        # Extract post IDs from Qdrant results
        post_ids = [item["post_id"] for item in results_data]
        
        # Fetch full post records from PostgreSQL
        posts_dict = await db.get_posts_by_ids(post_ids)
        
        # Merge Qdrant scores with PostgreSQL records
        merged_results = []
        for item in results_data:
            post = posts_dict.get(item["post_id"])
            if post is None:
                # Post not found in DB, skip it
                logger.warning(
                    "Post ID %d from Qdrant results not found in PostgreSQL",
                    item["post_id"]
                )
                continue
            
            # Construct URL using channel_id and message_id
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
        
        # Results are already sorted by score from Qdrant, but ensure ordering
        merged_results.sort(key=lambda x: x.score, reverse=True)
        
        return SearchResponse(results=merged_results)
        
    except Exception as e:
        logger.exception("Search endpoint failed")
        raise HTTPException(status_code=500, detail="Search operation failed.")
