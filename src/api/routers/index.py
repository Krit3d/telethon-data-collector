"""Indexing router for batch indexing operations."""

from fastapi import APIRouter, Depends
import logging

from src.api.schemas import IndexRequest, IndexResponse
from src.db.database import Database
from src.embeddings.qdrant_service import QdrantService
from src.api.dependencies import get_db, get_qdrant

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/index", tags=["Indexing"])


@router.post("", response_model=IndexResponse)
async def index_recent_posts(
    payload: IndexRequest,
    db: Database = Depends(get_db),
    qdrant: QdrantService = Depends(get_qdrant),
) -> IndexResponse:
    """Index recent posts from the database into Qdrant.

    Args:
        payload: Index request with limit of posts to index.
        db: Database instance (injected).
        qdrant: Qdrant service instance (injected).

    Returns:
        IndexResponse with count of indexed posts and message.
    """

    posts = await db.get_recent_posts(limit=payload.limit)

    if not posts:
        return IndexResponse(
            indexed_count=0, message="No posts found in database."
        )

    points = []

    for p in posts:
        if p.content and p.content.strip():
            points.append((p.id, p.content, p.channel_id))

    if not points:
        return IndexResponse(
            indexed_count=0, message="No valid text content found in posts."
        )

    await qdrant.upsert_batch(points)

    return IndexResponse(
        indexed_count=len(points), message="Successfully indexed."
    )


@router.get("/stats")
async def get_index_stats(qdrant: QdrantService = Depends(get_qdrant)):
    """Get exact point count from Qdrant collection."""
    try:
        if not qdrant.collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME is not configured")

        collection_info = await qdrant.client.get_collection(
            qdrant.collection_name
        )
        return {
            "status": collection_info.status,
            "points_count": collection_info.points_count,
            "indexed_vectors_count": collection_info.indexed_vectors_count,
            "segments_count": collection_info.segments_count,
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=e)
        return {"error": str(e)}
