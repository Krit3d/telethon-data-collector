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
    PostgresError = Exception

from src.config.config import Settings, load_settings
from src.db.database import Database
from src.db.graph_repo import GraphRepository
from src.db.models import Content
from src.embeddings.qdrant_service import QdrantService
from src.graph.extractor import KnowledgeExtractor

logger = logging.getLogger(__name__)


class GraphExtractionWorker:

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
        logger.info("Shutdown signal received, stopping graph extraction worker...")
        self._shutdown_event.set()

    async def run(self) -> None:
        logger.info(
            "Graph extraction worker started (batch_size=%d, poll_interval=%ds, priority_mode=%s)",
            self.batch_size,
            self.poll_interval,
            self.priority_mode,
        )

        while not self._shutdown_event.is_set():
            try:
                try:
                    posts = await self.db.get_ungraphed_content(
                        limit=self.batch_size, priority_mode=self.priority_mode
                    )
                except (OperationalError, PostgresError) as e:
                    logger.warning(
                        "Database connection error while fetching content: %s. "
                        "Retrying after backoff (%.1fs)...",
                        e,
                        self.poll_interval * 2,
                    )
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(),
                            timeout=self.poll_interval * 2,
                        )
                    except asyncio.TimeoutError:
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

                semaphore = asyncio.Semaphore(self.settings.extractor_concurrency)

                async def process_with_semaphore(post: Content) -> None:
                    async with semaphore:
                        await self._process_single_post(post)

                tasks = [process_with_semaphore(post) for post in posts]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for idx, result in enumerate(results):
                    if isinstance(result, Exception):
                        post = posts[idx]
                        logger.error(
                            "Unhandled exception processing content id=%s: %s",
                            post.id,
                            result,
                            exc_info=result,
                        )

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info("Graph extraction worker task cancelled")
                break
            except Exception as e:
                logger.error(
                    "Unexpected error in graph extraction worker loop: %s", e, exc_info=True
                )
                await asyncio.sleep(5)

        logger.info("Graph extraction worker stopped")

    async def _process_single_post(self, post: Content) -> None:
        text_for_processing: str | None = None

        if post.content and post.content.strip():
            text_for_processing = post.content
        elif post.transcription and post.transcription.strip():
            text_for_processing = post.transcription

        if text_for_processing is None:
            logger.debug(
                "Content id=%s has no content or transcription, skipping graph extraction",
                post.id,
            )
            await self.db.mark_content_graphed(post.id)
            return

        logger.debug("Extracting knowledge graph from content id=%s", post.id)

        post_metrics: dict[str, int | None] = {
            "views": post.views,
            "reactions_count": post.reactions_count,
            "comments_count": post.comments_count,
            "shares_count": post.shares_count,
        }

        raw_metadata: dict = (
            post.raw_metadata if post.raw_metadata is not None else {}
        )
        if post.transcription:
            raw_metadata = {**raw_metadata, "transcription": post.transcription}

        account_metadata: dict | None = None
        if post.account:
            account_metadata = (
                post.account.raw_metadata.copy() if post.account.raw_metadata else {}
            )
            account_metadata["username"] = post.account.username
            account_metadata["title"] = post.account.title
            account_metadata["subscribers_count"] = post.account.subscribers_count

        try:
            await self.extractor.process_post(
                post_id=post.id,
                text=text_for_processing,
                author_id=post.account_id,
                post_metrics=post_metrics,
                raw_metadata=raw_metadata,
                graph_repo=self.graph_repo,
                qdrant=self.qdrant,
                platform=post.account.platform if post.account else None,
                account_metadata=account_metadata,
                platform_content_id=post.platform_content_id,
            )
        except Exception as e:
            logger.error(
                "Unrecoverable error extracting knowledge graph for content id=%s: %s",
                post.id,
                e,
                exc_info=True,
            )

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
                logger.error(
                    "Failed to update raw_metadata for content id=%s: %s",
                    post.id,
                    db_err,
                    exc_info=True,
                )

            await self.db.mark_content_graphed(post.id)
            return

        await self.db.mark_content_graphed(post.id)

        logger.info("Completed graph extraction for content id=%s", post.id)


async def run_graph_extractor() -> None:
    settings = load_settings()

    logger.info("Starting knowledge graph extraction worker")

    db = Database(settings.db_url)
    await db.init_db(graph_name=settings.graph_name)

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

    extractor = KnowledgeExtractor(settings)

    graph_repo = GraphRepository(db.async_session, settings)

    worker = GraphExtractionWorker(
        db=db,
        graph_repo=graph_repo,
        qdrant=qdrant,
        extractor=extractor,
        settings=settings,
        batch_size=settings.extractor_batch_size,
        poll_interval=5,
    )
    worker.priority_mode = settings.extraction_priority_mode

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, worker.handle_shutdown)
        except NotImplementedError:
            signal.signal(sig, worker.handle_shutdown)

    try:
        await worker.run()
    finally:
        await extractor.close()
        if qdrant:
            await qdrant.close()
        await db.close()
        logger.info("Resources cleaned up")


if __name__ == "__main__":
    asyncio.run(run_graph_extractor())
