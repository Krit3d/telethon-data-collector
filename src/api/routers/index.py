"""Indexing router for batch indexing operations."""

from __future__ import annotations

import asyncio
from fastapi import APIRouter, Depends, BackgroundTasks
import logging

from src.api.schemas import IndexRequest, IndexResponse
from src.db.database import Database
from src.embeddings.qdrant_service import QdrantService
from src.api.dependencies import get_db, get_qdrant

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/index", tags=["Indexing"])

# Batch size for processing posts to avoid memory spikes
BATCH_SIZE = 50


def _chunk_list(items: list, chunk_size: int) -> list[list]:
    """Split a list into chunks of specified size."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


async def _run_indexing(limit: int, db: Database, qdrant: QdrantService) -> None:
    """Background task to index posts in batches.

    Args:
        limit: Maximum number of posts to index.
        db: Database instance.
        qdrant: Qdrant service instance.
    """
    logger.info(f"Starting background indexing task for up to {limit} posts")

    try:
        # Fetch all recent posts
        logger.info(f"Fetching {limit} recent posts from database")
        posts = await db.get_recent_posts(limit=limit)

        if not posts:
            logger.warning("No posts found in database for indexing")
            return

        # Prepare valid points
        points: list[tuple[int, str, int]] = []
        for p in posts:
            if p.content and p.content.strip():
                points.append((p.id, p.content, p.channel_id))

        if not points:
            logger.warning("No valid text content found in posts for indexing")
            return

        logger.info(f"Prepared {len(points)} valid points for indexing")

        # Process in batches
        chunks = _chunk_list(points, BATCH_SIZE)
        total_indexed = 0

        logger.info(f"Processing {len(chunks)} batches (batch size: {BATCH_SIZE})")

        for batch_idx, batch in enumerate(chunks, 1):
            try:
                logger.info(
                    f"Processing batch {batch_idx}/{len(chunks)} with {len(batch)} points"
                )
                await qdrant.upsert_batch(batch)
                total_indexed += len(batch)
                logger.info(
                    f"Completed batch {batch_idx}/{len(chunks)}. Total indexed so far: {total_indexed}"
                )
                # Yield control to event loop to prevent CPU hogging
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(
                    f"Failed to index batch {batch_idx}/{len(chunks)}: {e}",
                    exc_info=e,
                )
                # Continue with next batch instead of crashing
                continue

        logger.info(
            f"Indexing completed. Successfully indexed {total_indexed}/{len(points)} points."
        )

    except Exception as e:
        logger.error(f"Background indexing task failed with fatal error: {e}", exc_info=e)


@router.post("", response_model=IndexResponse, status_code=202)
async def index_recent_posts(
    payload: IndexRequest,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_db),
    qdrant: QdrantService = Depends(get_qdrant),
) -> IndexResponse:
    """Index recent posts from the database into Qdrant.

    This endpoint triggers a background task to index posts, returning immediately
    with a 202 Accepted status to avoid timeouts on large datasets.

    Args:
        payload: Index request with limit of posts to index.
        background_tasks: FastAPI background tasks manager.
        db: Database instance (injected).
        qdrant: Qdrant service instance (injected).

    Returns:
        IndexResponse with confirmation message and estimated count.
    """
    
    # Add the indexing task to background
    background_tasks.add_task(_run_indexing, payload.limit, db, qdrant)

    logger.info(f"Indexing task queued for {payload.limit} posts")

    return IndexResponse(
        indexed_count=0,
        message=f"Indexing of {payload.limit} posts started in the background.",
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
