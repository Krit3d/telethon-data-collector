"""Background worker for generating and storing content embeddings.

This worker runs independently from the scraper and extractor services. It polls
the database for content where is_embedded=False, generates vector embeddings
using fastembed, stores them in Qdrant for semantic search, and marks the
content as embedded in PostgreSQL.
"""

import asyncio
import logging
import signal

from sqlalchemy.exc import OperationalError
from sqlalchemy import text

try:
    import asyncpg.exceptions

    PostgresError = asyncpg.exceptions.PostgresError
except ImportError:
    PostgresError = Exception  # Fallback to base Exception

from src.config.config import Settings, load_settings
from src.db.database import Database
from src.db.models import Content
from src.embeddings.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

# Backoff settings for error recovery
BASE_BACKOFF = 1.0
MAX_BACKOFF = 60.0


class EmbeddingWorker:
    """Background worker that processes unembedded content."""

    def __init__(
        self,
        db: Database,
        qdrant: QdrantService,
        settings: Settings,
        batch_size: int = 64,
        poll_interval: int = 5,
    ) -> None:
        """Initialize the embedding worker.

        Args:
            db: Database instance for data access.
            qdrant: QdrantService for storing embeddings.
            settings: Application settings containing embedding configuration.
            batch_size: Number of content items to fetch per poll.
            poll_interval: Sleep interval in seconds when no content is found.
        """
        self.db = db
        self.qdrant = qdrant
        self.settings = settings
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.priority_mode: bool = False
        self._shutdown_event = asyncio.Event()

    def handle_shutdown(self, *args: object) -> None:
        """Signal handler for graceful shutdown."""
        logger.info("Shutdown signal received, stopping embedding worker...")
        self._shutdown_event.set()

    async def run(self) -> None:
        """Main worker loop: continuously poll and process unembedded content."""
        logger.info(
            "Embedding worker started (batch_size=%d, poll_interval=%ds, priority_mode=%s)",
            self.batch_size,
            self.poll_interval,
            self.priority_mode,
        )

        consecutive_errors = 0

        while not self._shutdown_event.is_set():
            try:
                # Fetch batch of unembedded content
                try:
                    posts = await self.db.get_unembedded_content(
                        limit=self.batch_size, priority_mode=self.priority_mode
                    )
                except (OperationalError, PostgresError) as e:
                    # Database connection error - log warning, back off, and retry
                    consecutive_errors += 1
                    backoff_time = min(
                        BASE_BACKOFF * (2 ** (consecutive_errors - 1)), MAX_BACKOFF
                    )
                    logger.warning(
                        "Database connection error while fetching content: %s. "
                        "Retrying after backoff (%.1fs, attempt %d)...",
                        e,
                        backoff_time,
                        consecutive_errors,
                    )
                    # Use wait_for to allow shutdown during sleep
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(),
                            timeout=backoff_time,
                        )
                    except asyncio.TimeoutError:
                        # Timeout is expected - continue to retry
                        pass
                    continue

                if not posts:
                    logger.debug(
                        "No unembedded content found, sleeping %ds",
                        self.poll_interval,
                    )
                    consecutive_errors = 0  # Reset error counter on success
                    await asyncio.sleep(self.poll_interval)
                    continue

                logger.info(
                    "Processing batch of %d unembedded content items", len(posts)
                )
                consecutive_errors = 0  # Reset error counter on success

                # Process the batch sequentially to avoid thread/memory contention on ONNX Runtime
                # upsert_batch is already internally parallelized by FastEmbed
                await self._process_batch(posts)

                # Small delay between batches to avoid overwhelming the system
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info("Embedding worker task cancelled")
                break
            except Exception as e:
                consecutive_errors += 1
                backoff_time = min(
                    BASE_BACKOFF * (2 ** (consecutive_errors - 1)), MAX_BACKOFF
                )
                logger.error(
                    "Unexpected error in embedding worker loop: %s. "
                    "Backing off for %.1fs (attempt %d)",
                    e,
                    backoff_time,
                    consecutive_errors,
                    exc_info=True,
                )
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=backoff_time,
                    )
                except asyncio.TimeoutError:
                    pass

        logger.info("Embedding worker stopped")

    async def _process_batch(self, posts: list[Content]) -> None:
        """Process a batch of content items: generate embeddings and store in Qdrant.

        Args:
            posts: List of Content objects to process.
        """

        # Filter out posts with empty content and build points list
        points: list[tuple[int, str, int]] = []
        failed_ids: list[int] = []

        for post in posts:
            # Build text to embed from content and/or transcription
            text_to_embed = ""

            # Get stripped values, handling None cases
            content_text = post.content.strip() if post.content else ""
            transcription_text = post.transcription.strip() if post.transcription else ""

            has_content = bool(content_text)
            has_transcription = bool(transcription_text)

            if has_content and has_transcription:
                text_to_embed = f"Description: {content_text}\nTranscription: {transcription_text}"
            elif has_content:
                text_to_embed = content_text
            elif has_transcription:
                text_to_embed = transcription_text

            if not text_to_embed:
                logger.debug(
                    "Content id=%s has no valid text content or transcription, skipping embedding",
                    post.id,
                )
                failed_ids.append(post.id)
                continue

            points.append((post.id, text_to_embed, post.account_id))

        if not points:
            logger.debug("No valid content to embed in this batch")
            # Mark failed posts as embedded to avoid reprocessing empty content
            if failed_ids:
                await self.db.mark_content_embedded(failed_ids)
            return

        # Generate embeddings and upsert to Qdrant
        try:
            await self.qdrant.upsert_batch(points)

            # Mark all successfully embedded content as embedded in PostgreSQL
            embedded_ids = [p[0] for p in points]
            await self.db.mark_content_embedded(embedded_ids)

            logger.info(
                "Successfully embedded %d content items",
                len(embedded_ids),
            )

        except Exception as e:
            logger.error(
                "Failed to process embedding batch: %s",
                e,
                exc_info=True,
            )

            # Log individual failed items for debugging
            for post_id, text, account_id in points:
                logger.debug(
                    "Failed to embed content id=%s (account_id=%s)",
                    post_id,
                    account_id,
                )

            # Still mark failed items to prevent infinite retry loops
            # In production, you might want to implement a separate error queue
            embedded_ids = [p[0] for p in points]
            await self.db.mark_content_embedded(embedded_ids)

            # Re-raise to trigger backoff in the main loop
            raise

        # Mark posts with empty content as embedded (to skip them in future)
        if failed_ids:
            await self.db.mark_content_embedded(failed_ids)
            logger.debug(
                "Marked %d posts with empty content as embedded (skipped)",
                len(failed_ids),
            )


async def run_embedding_worker() -> None:
    """Entry point for the embedding worker service.

    Initializes all dependencies and starts the worker loop.
    This function is intended to be called from the main() block.
    """
    settings = load_settings()

    logger.info("Starting embedding worker")

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
        logger.warning(
            "Embedding worker cannot function without Qdrant - exiting"
        )
        await db.close()
        return

    # Create and run worker with configured settings
    worker = EmbeddingWorker(
        db=db,
        qdrant=qdrant,
        settings=settings,
        batch_size=settings.embedding_batch_size,
        poll_interval=5,
    )
    # Set priority mode from configuration (default: True for recent content first)
    worker.priority_mode = settings.embedding_priority_mode

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, worker.handle_shutdown)
    signal.signal(signal.SIGTERM, worker.handle_shutdown)

    try:
        await worker.run()
    finally:
        # Cleanup
        await qdrant.close()
        await db.close()
        logger.info("Resources cleaned up")


if __name__ == "__main__":
    asyncio.run(run_embedding_worker())
