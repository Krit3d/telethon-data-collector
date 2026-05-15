"""Background worker for knowledge extraction from unextracted posts.

This worker runs independently from the scraper service. It polls the database
for posts where is_extracted=False, extracts knowledge triples using LLM,
stores them in Apache AGE graph, and syncs entities to Qdrant for vector search.
"""

import asyncio
import logging
import signal

from sqlalchemy.exc import OperationalError

try:
    import asyncpg.exceptions
    PostgresError = asyncpg.exceptions.PostgresError
except ImportError:
    # asyncpg may not be installed in all environments
    PostgresError = Exception  # Fallback to base Exception

from src.config.config import load_settings
from src.db.database import Database
from src.db.models import Post
from src.embeddings.qdrant_service import QdrantService
from src.graph.extractor import KnowledgeExtractor

logger = logging.getLogger(__name__)

# Rate limit cooldown duration (seconds)
RATE_LIMIT_COOLDOWN = 60


class ExtractionWorker:
    """Background worker that processes unextracted posts."""

    def __init__(
        self,
        db: Database,
        qdrant: QdrantService,
        extractor: KnowledgeExtractor,
        batch_size: int = 20,
        poll_interval: int = 5,
    ) -> None:
        """Initialize the extraction worker.

        Args:
            db: Database instance for data access.
            qdrant: QdrantService for entity embedding sync.
            extractor: KnowledgeExtractor for LLM-based extraction.
            batch_size: Number of posts to fetch per poll.
            poll_interval: Sleep interval in seconds when no posts are found.
        """
        self.db = db
        self.qdrant = qdrant
        self.extractor = extractor
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.priority_mode: bool = False
        self._shutdown_event = asyncio.Event()

    def handle_shutdown(self, *args: object) -> None:
        """Signal handler for graceful shutdown."""
        logger.info("Shutdown signal received, stopping worker...")
        self._shutdown_event.set()

    async def run(self) -> None:
        """Main worker loop: continuously poll and process unextracted posts."""
        logger.info(
            "Extraction worker started (batch_size=%d, poll_interval=%ds, priority_mode=%s)",
            self.batch_size,
            self.poll_interval,
            self.priority_mode,
        )

        while not self._shutdown_event.is_set():
            try:
                # Fetch batch of unextracted posts
                try:
                    posts = await self.db.get_unextracted_posts(
                        limit=self.batch_size,
                        priority_mode=self.priority_mode
                    )
                except (OperationalError, PostgresError) as e:
                    # Database connection error - log warning, back off, and retry
                    logger.warning(
                        "Database connection error while fetching posts: %s. "
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
                    logger.debug("No unextracted posts found, sleeping %ds", self.poll_interval)
                    await asyncio.sleep(self.poll_interval)
                    continue

                logger.info("Processing batch of %d unextracted posts", len(posts))

                # Process each post sequentially (could be parallelized with semaphore)
                for post in posts:
                    try:
                        await self._process_single_post(post)
                    except Exception as e:
                        logger.error(
                            "Failed to process post id=%s: %s",
                            post.id,
                            e,
                            exc_info=True,
                        )
                        # Continue with next post - do not crash the loop

                # Small delay between batches to avoid overwhelming the system
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info("Worker task cancelled")
                break
            except Exception as e:
                logger.error("Unexpected error in worker loop: %s", e, exc_info=True)
                await asyncio.sleep(5)  # Back off on unexpected errors

        logger.info("Extraction worker stopped")

    async def _process_single_post(self, post: Post) -> None:
        """Process a single post: extract knowledge and mark as extracted.

        Args:
            post: Post object from database.
        """
        if not post.content or not post.content.strip():
            logger.debug("Post id=%s has no content, skipping extraction", post.id)
            await self.db.mark_post_extracted(post.id)
            return

        logger.debug("Extracting knowledge from post id=%s", post.id)

        try:
            await self.extractor.process_post(
                post_id=post.id,
                text=post.content,
                author_id=post.channel_id,
                db=self.db,
                qdrant=self.qdrant,
            )
        except Exception as e:
            logger.error(
                "Failed to extract knowledge for post id=%s: %s",
                post.id,
                e,
                exc_info=True,
            )
            raise  # Re-raise to skip marking as extracted, will be retried later

        # Mark post as extracted only after successful processing
        await self.db.mark_post_extracted(post.id)

        logger.info("Completed post id=%s", post.id)


async def run_extractor() -> None:
    """Entry point for the extraction worker service.

    Initializes all dependencies and starts the worker loop.
    This function is intended to be called from the main() block.
    """
    settings = load_settings()

    # Initialize logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Starting knowledge extraction worker")

    # Initialize database
    db = Database(settings.db_url)
    await db.init_db()

    # Initialize Qdrant service
    qdrant = QdrantService(settings)
    try:
        await qdrant.initialize()
        logger.info("Qdrant service initialized")
    except Exception as e:
        logger.error("Failed to initialize Qdrant: %s", e)
        logger.warning("Continuing without Qdrant - entity embeddings will be disabled")
        # Qdrant is optional for extraction; we can still extract to AGE graph

    # Initialize knowledge extractor
    extractor = KnowledgeExtractor(settings)

    # Create and run worker with optimized settings
    worker = ExtractionWorker(
        db=db,
        qdrant=qdrant,
        extractor=extractor,
        batch_size=20,
        poll_interval=5,
    )
    # Set priority mode from configuration (default: True for recent posts first)
    worker.priority_mode = settings.extraction_priority_mode

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, worker.handle_shutdown)
    signal.signal(signal.SIGTERM, worker.handle_shutdown)

    try:
        await worker.run()
    finally:
        # Cleanup
        await extractor.close()
        await qdrant.close()
        await db.close()
        logger.info("Resources cleaned up")


if __name__ == "__main__":
    asyncio.run(run_extractor())
