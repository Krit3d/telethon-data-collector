"""Background worker for knowledge graph extraction from ungraphed content.

This worker runs independently from the scraper service and embedding worker.
It polls the database for content where is_graph_extracted=False, extracts
knowledge triples using LLM, stores them in Apache AGE graph, and syncs
extracted entities to Qdrant for vector search.
"""

import asyncio
import logging
import signal
from datetime import datetime, timezone

from sqlalchemy.exc import OperationalError
from sqlalchemy import text

try:
    import asyncpg.exceptions

    PostgresError = asyncpg.exceptions.PostgresError
except ImportError:
    PostgresError = Exception  # Fallback to base Exception

from src.config.config import Settings, load_settings
from src.db.database import Database
from src.db.graph_repo import GraphRepository
from src.db.models import Content
from src.embeddings.qdrant_service import QdrantService
from src.graph.extractor import KnowledgeExtractor

logger = logging.getLogger(__name__)

# Rate limit cooldown duration (seconds)
RATE_LIMIT_COOLDOWN = 60


class GraphExtractionWorker:
    """Background worker that processes ungraphed content for knowledge graph extraction."""

    def __init__(
        self,
        db: Database,
        graph_repo: GraphRepository,
        qdrant: QdrantService | None,
        extractor: KnowledgeExtractor,
        settings: Settings,
        batch_size: int = 20,
        poll_interval: int = 5,
    ) -> None:
        """Initialize the graph extraction worker.

        Args:
            db: Database instance for data access.
            graph_repo: GraphRepository for Apache AGE graph operations.
            qdrant: Optional QdrantService for entity embedding sync.
            extractor: KnowledgeExtractor for LLM-based extraction.
            settings: Application settings containing concurrency configuration.
            batch_size: Number of content items to fetch per poll.
            poll_interval: Sleep interval in seconds when no content is found.
        """
        self.db = db
        self.graph_repo = graph_repo
        self.qdrant = qdrant
        self.extractor = extractor
        self.settings = settings
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.priority_mode: bool = False
        self._shutdown_event = asyncio.Event()

    def handle_shutdown(self, *args: object) -> None:
        """Signal handler for graceful shutdown."""
        logger.info("Shutdown signal received, stopping graph extraction worker...")
        self._shutdown_event.set()

    async def run(self) -> None:
        """Main worker loop: continuously poll and process ungraphed content."""
        logger.info(
            "Graph extraction worker started (batch_size=%d, poll_interval=%ds, priority_mode=%s)",
            self.batch_size,
            self.poll_interval,
            self.priority_mode,
        )

        while not self._shutdown_event.is_set():
            try:
                # Fetch batch of ungraphed content
                try:
                    posts = await self.db.get_ungraphed_content(
                        limit=self.batch_size, priority_mode=self.priority_mode
                    )
                except (OperationalError, PostgresError) as e:
                    # Database connection error - log warning, back off, and retry
                    logger.warning(
                        "Database connection error while fetching content: %s. "
                        "Retrying after backoff (%.1fs)...",
                        e,
                        self.poll_interval * 2,
                    )
                    # Use wait_for to allow shutdown during sleep
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(),
                            timeout=self.poll_interval * 2,
                        )
                    except asyncio.TimeoutError:
                        # Timeout is expected - continue to retry
                        pass
                    continue

                if not posts:
                    logger.debug(
                        "No ungraphed content found, sleeping %ds",
                        self.poll_interval,
                    )
                    await asyncio.sleep(self.poll_interval)
                    continue

                logger.info(
                    "Processing batch of %d ungraphed content items", len(posts)
                )

                # Configure concurrency semaphore with extractor-specific limit
                semaphore = asyncio.Semaphore(self.settings.extractor_concurrency)

                async def process_with_semaphore(post: Content) -> None:
                    """Process a single content item with semaphore-based rate limiting."""
                    async with semaphore:
                        await self._process_single_post(post)

                # Process content concurrently, return exceptions instead of raising
                tasks = [process_with_semaphore(post) for post in posts]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Log any unhandled exceptions that escaped _process_single_post
                for idx, result in enumerate(results):
                    if isinstance(result, Exception):
                        post = posts[idx]
                        logger.error(
                            "Unhandled exception processing content id=%s: %s",
                            post.id,
                            result,
                            exc_info=result,
                        )

                # Small delay between batches to avoid overwhelming the system
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info("Graph extraction worker task cancelled")
                break
            except Exception as e:
                logger.error(
                    "Unexpected error in graph extraction worker loop: %s", e, exc_info=True
                )
                await asyncio.sleep(5)  # Back off on unexpected errors

        logger.info("Graph extraction worker stopped")

    async def _process_single_post(self, post: Content) -> None:
        """Process a single content item: extract knowledge graph and mark as graphed.

        Args:
            post: Content object from database.
        """
        if not post.content or not post.content.strip():
            logger.debug(
                "Content id=%s has no content, skipping graph extraction", post.id
            )
            await self.db.mark_content_graphed(post.id)
            return

        logger.debug("Extracting knowledge graph from content id=%s", post.id)

        # Extract content metrics directly from SQL model
        post_metrics: dict[str, int | None] = {
            "views": post.views,
            "reactions_count": post.reactions_count,
            "comments_count": post.comments_count,
            "shares_count": post.shares_count,
        }

        # Extract raw metadata directly from SQL model
        raw_metadata: dict = (
            post.raw_metadata if post.raw_metadata is not None else {}
        )

        try:
            await self.extractor.process_post(
                post_id=post.id,
                text=post.content,
                author_id=post.account_id,
                post_metrics=post_metrics,
                raw_metadata=raw_metadata,
                graph_repo=self.graph_repo,
                qdrant=self.qdrant,
            )
        except Exception as e:
            # Log the full traceback for unrecoverable errors (validation, JSON parsing, API failures)
            logger.error(
                "Unrecoverable error extracting knowledge graph for content id=%s: %s",
                post.id,
                e,
                exc_info=True,
            )

            # Append error information to raw_metadata to preserve failure context
            # Uses PostgreSQL JSONB || operator and jsonb_build_object() to merge error info
            try:
                async with self.db.async_session() as session:
                    await session.execute(
                        text("""
                            UPDATE content
                            SET raw_metadata = COALESCE(raw_metadata, '{}'::jsonb) ||
                                jsonb_build_object('graph_extraction_error', :error_msg, 'graph_failed_at', :failed_at)
                            WHERE id = :post_id
                        """),
                        {
                            "post_id": post.id,
                            "error_msg": str(e),
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    await session.commit()
            except Exception as db_err:
                # Log but don't crash - we still need to mark as graphed to prevent infinite loops
                logger.error(
                    "Failed to update raw_metadata for content id=%s: %s",
                    post.id,
                    db_err,
                    exc_info=True,
                )

            # Always mark as graphed to remove from ungraphed queue and prevent infinite loops
            await self.db.mark_content_graphed(post.id)
            return

        # Mark content as graphed only after successful processing
        await self.db.mark_content_graphed(post.id)

        logger.info("Completed graph extraction for content id=%s", post.id)


async def run_graph_extractor() -> None:
    """Entry point for the graph extraction worker service.

    Initializes all dependencies and starts the worker loop.
    This function is intended to be called from the main() block.
    """
    settings = load_settings()

    logger.info("Starting knowledge graph extraction worker")

    # Initialize database
    db = Database(settings.db_url)
    await db.init_db()

    # Initialize Qdrant service (optional for graph extraction)
    qdrant = None
    try:
        qdrant = QdrantService(settings)
        await qdrant.initialize()
        logger.info("Qdrant service initialized")
    except Exception as e:
        logger.error("Failed to initialize Qdrant: %s", e)
        logger.warning(
            "Continuing without Qdrant - entity embeddings will be disabled"
        )
        # Qdrant is optional for graph extraction; we can still extract to AGE graph

    # Initialize knowledge extractor
    extractor = KnowledgeExtractor(settings)

    # Create GraphRepository sharing the same async sessionmaker as Database
    graph_repo = GraphRepository(db.async_session)

    # Create and run worker with optimized settings
    worker = GraphExtractionWorker(
        db=db,
        graph_repo=graph_repo,
        qdrant=qdrant,
        extractor=extractor,
        settings=settings,
        batch_size=settings.extractor_batch_size,
        poll_interval=5,
    )
    # Set priority mode from configuration (default: True for recent content first)
    worker.priority_mode = settings.extraction_priority_mode

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, worker.handle_shutdown)
    signal.signal(signal.SIGTERM, worker.handle_shutdown)

    try:
        await worker.run()
    finally:
        # Cleanup
        await extractor.close()
        if qdrant:
            await qdrant.close()
        await db.close()
        logger.info("Resources cleaned up")


if __name__ == "__main__":
    asyncio.run(run_graph_extractor())
