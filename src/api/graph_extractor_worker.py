import asyncio
import json
import logging
import random
import signal
from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError, InternalError, OperationalError
from sqlalchemy import text

try:
    import asyncpg.exceptions
    PostgresError = asyncpg.exceptions.PostgresError
except ImportError:
    PostgresError = Exception

from openai import APIConnectionError, APIStatusError, RateLimitError

from src.config.config import Settings, load_settings
from src.db.database import Database
from src.db.models import Content
from src.embeddings.qdrant_service import QdrantService
from src.graph.db.extractor_repo import ExtractorRepository
from src.graph.db.graph_repo import GraphRepository
from src.graph.extractor import KnowledgeExtractor

logger = logging.getLogger(__name__)

RECOVERABLE_ERRORS: tuple[type[Exception], ...] = (
    APIConnectionError,
    RateLimitError,
    APIStatusError,
    OperationalError,
    InternalError,
    PostgresError,
    TimeoutError,
    ConnectionError,
    OSError,
    IntegrityError,
)

UNRECOVERABLE_ERRORS: tuple[type[Exception], ...] = (
    json.JSONDecodeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
)

_DB_BACKOFF_BASE = 2.0
_DB_BACKOFF_MAX = 60.0
_DB_MAX_BACKOFF_RETRIES = 5
_DB_JITTER_MAX = 1.0

_RETRY_MAX_ATTEMPTS = 2
_RETRY_JITTER_MIN = 0.5
_RETRY_JITTER_MAX = 2.0

_STRIP_KEYS: tuple[str, ...] = (
    "category",
    "category_name",
    "category_enum",
    "overall_category_name",
    "business_category_name",
    "language",
    "lang",
    "language_code",
    "primary_locale",
)

_HEAVY_KEYS: tuple[str, ...] = (
    "chaining_results",
    "facebook_pages",
    "linked_facebook_page",
    "mutual_followers_data",
    "eligible_promotions",
    "ad_metadata",
    "hd_profile_pic_versions",
    "hd_profile_pic_url_info",
    "bio_links",
    "about_your_account_blurb",
    "edge_owner_to_timeline_media",
    "edge_felix_video_timeline",
    "edge_saved_media",
    "edge_media_collections",
    "edge_mutual_followed_by",
    "edge_related_profiles",
    "biography_with_entities",
    "fb_profile_biolink",
    "profile_pic_url",
    "profile_pic_url_hd",
    "video_dash_manifest",
    "image_versions2",
    "user",
    "owner",
    "clips_metadata",
    "scrubber_spritesheet_info_candidates",
    "organic_tracking_token",
    "candidate_metadata",
)

_OMIT_KEYS: frozenset[str] = frozenset(_STRIP_KEYS) | frozenset(_HEAVY_KEYS)


def _sanitize_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value)
    stripped = value.strip()
    return stripped if stripped else None


def _sanitize_metadata(raw: dict | None) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        return {}
    sanitized: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if key in _OMIT_KEYS:
            continue
        if value is None:
            sanitized[key] = None
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_metadata(value)
        elif isinstance(value, list):
            sanitized[key] = [
                _sanitize_metadata(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = str(value)
    return sanitized


def _db_backoff_delay(attempt: int) -> float:
    exponential = min(_DB_BACKOFF_BASE * (2 ** attempt), _DB_BACKOFF_MAX)
    jitter = random.uniform(0.0, _DB_JITTER_MAX)
    return exponential + jitter


class GraphExtractionWorker:

    def __init__(
        self,
        db: Database,
        extractor_repo: ExtractorRepository,
        graph_repo: GraphRepository,
        qdrant: QdrantService | None,
        extractor: KnowledgeExtractor,
        settings: Settings,
        batch_size: int = 20,
        poll_interval: int = 5,
    ) -> None:
        self.db = db
        self.extractor_repo = extractor_repo
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

    async def _sleep_with_shutdown_check(self, delay: float) -> bool:
        try:
            await asyncio.wait_for(
                self._shutdown_event.wait(),
                timeout=delay,
            )
            return True
        except asyncio.TimeoutError:
            return False

    async def _fetch_batch_with_backoff(self) -> list[Content]:
        last_db_error: Exception | None = None

        for attempt in range(_DB_MAX_BACKOFF_RETRIES):
            if self._shutdown_event.is_set():
                return []

            try:
                posts = await self.extractor_repo.get_ungraphed_content(
                    limit=self.batch_size, priority_mode=self.priority_mode
                )
                return posts
            except (OperationalError, InternalError, PostgresError, ConnectionError, OSError) as e:
                last_db_error = e
                delay = _db_backoff_delay(attempt)
                logger.warning(
                    "Database error while fetching content (attempt %d/%d): %s. "
                    "Retrying after %.1fs backoff...",
                    attempt + 1,
                    _DB_MAX_BACKOFF_RETRIES,
                    e,
                    delay,
                )
                if await self._sleep_with_shutdown_check(delay):
                    return []

        logger.error(
            "Database unavailable after %d backoff attempts: %s. "
            "Sleeping default poll interval.",
            _DB_MAX_BACKOFF_RETRIES,
            last_db_error,
        )
        if await self._sleep_with_shutdown_check(float(self.poll_interval)):
            return []
        return []

    async def run(self) -> None:
        logger.info(
            "Graph extraction worker started (batch_size=%d, poll_interval=%ds, priority_mode=%s)",
            self.batch_size,
            self.poll_interval,
            self.priority_mode,
        )

        while not self._shutdown_event.is_set():
            try:
                posts = await self._fetch_batch_with_backoff()

                if self._shutdown_event.is_set():
                    break

                if not posts:
                    logger.debug(
                        "No ungraphed content found, sleeping %ds",
                        self.poll_interval,
                    )
                    if await self._sleep_with_shutdown_check(float(self.poll_interval)):
                        break
                    continue

                logger.info(
                    "Processing batch of %d ungraphed content items", len(posts)
                )

                sem = asyncio.Semaphore(self.settings.extractor_concurrency)
                marked_count: int = 0
                mark_lock = asyncio.Lock()

                async def process_one(post: Content) -> None:
                    nonlocal marked_count
                    async with sem:
                        should_mark = await self._process_single_post(post)
                        if should_mark:
                            await self.extractor_repo.mark_content_graphed(post.id)
                            async with mark_lock:
                                marked_count += 1

                tasks = [asyncio.create_task(process_one(p)) for i, p in enumerate(posts)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Log any per-post exceptions without crashing the batch
                for i, res in enumerate(results):
                    if isinstance(res, Exception):
                        logger.warning(
                            "Post id=%s failed with %s — will be retried in next batch",
                            posts[i].id,
                            res,
                        )

                if marked_count:
                    logger.info(
                        "Batch completed: %d posts marked as graphed", marked_count
                )

            except asyncio.CancelledError:
                logger.info("Graph extraction worker task cancelled")
                break
            except Exception as e:
                logger.error(
                    "Unexpected error in graph extraction worker loop: %s",
                    e,
                    exc_info=True,
                )
                if await self._sleep_with_shutdown_check(5):
                    break

        logger.info("Graph extraction worker stopped")

    async def _process_single_post(self, post: Content) -> bool:
        text_for_processing: str | None = None

        has_content = post.content is not None and post.content.strip()
        has_transcription = (
            post.transcription is not None and post.transcription.strip()
        )

        if has_content and has_transcription:
            text_for_processing = (
                f"{post.content}\n\nTranscription:\n{post.transcription}"
            )
        elif has_content:
            text_for_processing = post.content
        elif has_transcription:
            text_for_processing = post.transcription

        if text_for_processing is None:
            logger.debug(
                "Content id=%s has no content or transcription, skipping graph extraction",
                post.id,
            )
            return True

        logger.debug("Extracting knowledge graph from content id=%s", post.id)

        post_metrics: dict[str, int | None] = {
            "views": post.views,
            "reactions_count": post.reactions_count,
            "comments_count": post.comments_count,
            "shares_count": post.shares_count,
        }

        raw_metadata = _sanitize_metadata(
            post.raw_metadata if post.raw_metadata is not None else None
        )

        if post.transcription:
            safe_transcription = _sanitize_string(post.transcription)
            if safe_transcription:
                raw_metadata["transcription"] = safe_transcription

        account_metadata: dict | None = None
        if post.account:
            account_metadata = _sanitize_metadata(
                post.account.raw_metadata if post.account.raw_metadata else None
            )
            account_metadata["username"] = _sanitize_string(post.account.username) or "unknown"
            account_metadata["title"] = _sanitize_string(post.account.title) or "Unknown"
            raw_subscribers = post.account.subscribers_count
            account_metadata["subscribers_count"] = (
                int(raw_subscribers)
                if isinstance(raw_subscribers, (int, float))
                else None
            )

            if not account_metadata.get("biography"):
                fallback_bio = _sanitize_string(post.account.description)
                if fallback_bio:
                    account_metadata["biography"] = fallback_bio

        last_recoverable_error: Exception | None = None

        for attempt in range(_RETRY_MAX_ATTEMPTS):
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
                return True
            except UNRECOVERABLE_ERRORS as e:
                logger.error(
                    "Unrecoverable error extracting knowledge graph for content id=%s: %s",
                    post.id,
                    e,
                )

                try:
                    async with self.db.async_session() as session:
                        await session.execute(
                            text("""
                                UPDATE content
                                SET raw_metadata = COALESCE(raw_metadata, '{}'::jsonb) ||
                                    jsonb_build_object(
                                        'graph_extraction_error', CAST(:error_msg AS text),
                                        'graph_failed_at', CAST(:failed_at AS text)
                                    )
                                WHERE id = :post_id
                            """),
                            {
                                "post_id": post.id,
                                "error_msg": str(e)[:1000],
                                "failed_at": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                        await session.commit()
                except Exception as db_err:
                    logger.error(
                        "Failed to update raw_metadata for content id=%s: %s",
                        post.id,
                        db_err,
                        exc_info=db_err,
                    )

                return True
            except RECOVERABLE_ERRORS as e:
                last_recoverable_error = e
                if attempt < _RETRY_MAX_ATTEMPTS - 1:
                    jitter = random.uniform(_RETRY_JITTER_MIN, _RETRY_JITTER_MAX)
                    logger.warning(
                        "Recoverable error extracting knowledge graph for content id=%s "
                        "(attempt %d/%d): %s. Retrying after %.2fs jitter...",
                        post.id,
                        attempt + 1,
                        _RETRY_MAX_ATTEMPTS,
                        e,
                        jitter,
                    )
                    await asyncio.sleep(jitter)
                continue
        else:
            logger.warning(
                "Recoverable error extracting knowledge graph for content id=%s "
                "after %d attempts: %s. Post will be retried in a subsequent batch.",
                post.id,
                _RETRY_MAX_ATTEMPTS,
                last_recoverable_error,
            )
            return False

        return True


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
    extractor_repo = ExtractorRepository(db.async_session, settings)

    worker = GraphExtractionWorker(
        db=db,
        extractor_repo=extractor_repo,
        graph_repo=graph_repo,
        qdrant=qdrant,
        extractor=extractor,
        settings=settings,
        batch_size=settings.extractor_batch_size,
        poll_interval=5,
    )
    worker.priority_mode = getattr(settings, "extraction_priority_mode", False)

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
