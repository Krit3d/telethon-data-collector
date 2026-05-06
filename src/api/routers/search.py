"""Search router for semantic search operations."""

from fastapi import APIRouter, Depends, HTTPException
import logging

from src.api.schemas import SearchRequest, SearchResponse, SearchResultItem
from src.embeddings.qdrant_service import QdrantService
from src.api.dependencies import get_qdrant

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["Search"])


@router.post("", response_model=SearchResponse)
async def search_posts(
    payload: SearchRequest,
    qdrant: QdrantService = Depends(get_qdrant)
) -> SearchResponse:
    """Search for posts using semantic similarity.

    Args:
        payload: Search request with query, limit, and score threshold.
        qdrant: Qdrant service instance (injected).

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
        # Convert dicts to SearchResultItem models
        results = [SearchResultItem(**item) for item in results_data]

        return SearchResponse(results=results)
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search operation failed.")
