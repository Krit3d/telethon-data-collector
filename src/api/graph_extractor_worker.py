import asyncio
import json
import logging
import random
import time
import signal
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

from sqlalchemy.exc import IntegrityError, InternalError, OperationalError
from sqlalchemy import text

try:
    import asyncpg.exceptions
    PostgresError = asyncpg.exceptions.PostgresError
except ImportError:
    PostgresError = Exception

from openai import APIConnectionError, APIStatusError, RateLimitError, APITimeoutError

from src.config.config import Settings, load_settings
from src.db.database import Database
from src.db.models import Content
from src.embeddings.qdrant_service import QdrantService
from src.graph.db.extractor_repo import ExtractorRepository
from src.graph.db.graph_repo import GraphRepository
from src.graph.extractor import KnowledgeExtractor
from src.graph.schema import OpenSPGExtractionResult

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
    json.JSONDecodeError,
    ValueError,
    APITimeoutError,
)

UNRECOVERABLE_ERRORS: tuple[type[Exception], ...] = (
    TypeError,
    KeyError,
    AttributeError,
)

_DB_BACKOFF_BASE = 2.0
_DB_BACKOFF_MAX = 60.0
_DB_MAX_BACKOFF_RETRIES = 5
_DB_JITTER_MAX = 1.0

_RETRY_MAX_ATTEMPTS = 2
_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 30.0

_DEAD_LETTER_THRESHOLD = 5
_LLM_CACHE_MAX_SIZE = 200

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


def _retry_backoff_delay(attempt: int) -> float:
    exponential = min(_RETRY_BASE_DELAY * (2 ** attempt), _RETRY_MAX_DELAY)
    jitter = random.uniform(0.5, 1.5)
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
        self._llm_cache: OrderedDict[int, OpenSPGExtractionResult] = OrderedDict()
        self._retry_counts: dict[int, int] = {}
        self._dead_letters: set[int] = set()
        self._queue: asyncio.Queue[Content] = asyncio.Queue(
            maxsize=settings.graph_concurrency * 2
        )

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

    async def _producer_loop(self) -> None:
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

                filtered_posts = [p for p in posts if p.id not in self._dead_letters]
                if len(filtered_posts) < len(posts):
                    logger.info(
                        "Filtered out %d dead-letter posts from batch",
                        len(posts) - len(filtered_posts),
                    )
                posts = filtered_posts

                if not posts:
                    if await self._sleep_with_shutdown_check(float(self.poll_interval)):
                        break
                    continue

                logger.info(
                    "Feeding %d ungraphed content items into the processing queue",
                    len(posts),
                )

                for post in posts:
                    await self._queue.put(post)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Unexpected error in producer loop: %s",
                    e,
                    exc_info=True,
                )
                if await self._sleep_with_shutdown_check(5):
                    break

    async def _consumer_loop(self, worker_id: int) -> None:
        while not self._shutdown_event.is_set():
            try:
                post = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                logger.info("Consumer %d processing post id=%d", worker_id, post.id)
                success = await self._process_single_post(post)
                if success:
                    await self.extractor_repo.mark_content_graphed(post.id)
                    logger.info("Consumer %d marked post id=%d as graphed", worker_id, post.id)
                else:
                    await self.extractor_repo.release_content_claim(post.id)
            except Exception as e:
                logger.error("Consumer %d error processing post id=%d: %s", worker_id, post.id, e)
                await self.extractor_repo.release_content_claim(post.id)
            finally:
                self._queue.task_done()

    async def _recover_stale_claims(self) -> None:
        threshold = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        async with self.db.async_session() as session:
            async with session.begin():
                await session.execute(
                    text("""
                        UPDATE content
                        SET raw_metadata = raw_metadata - 'graph_status' - 'claimed_at'
                        WHERE is_graph_extracted = false
                          AND raw_metadata->>'graph_status' = 'processing'
                          AND raw_metadata->>'claimed_at' < :threshold
                    """),
                    {"threshold": threshold}
                )

    async def run(self) -> None:
        logger.info(
            "Graph extraction worker started (queue_maxsize=%d, concurrency=%d)",
            self._queue.maxsize,
            self.settings.graph_concurrency,
        )

        logger.info("Recovering stale graph extraction claims...")
        await self._recover_stale_claims()

        producer = asyncio.create_task(self._producer_loop())
        consumers = [
            asyncio.create_task(self._consumer_loop(i + 1))
            for i in range(self.settings.graph_concurrency)
        ]

        try:
            await asyncio.gather(producer, *consumers)
        except asyncio.CancelledError:
            logger.info("Graph extraction worker task cancelled")
        finally:
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
        cached = self._llm_cache.get(post.id)

        for attempt in range(_RETRY_MAX_ATTEMPTS):
            try:
                result = await self.extractor.process_post(
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
                    cached_result=cached,
                )
                if result is None:
                    return True
                if post.id not in self._llm_cache:
                    self._llm_cache[post.id] = result
                    if len(self._llm_cache) > _LLM_CACHE_MAX_SIZE:
                        self._llm_cache.popitem(last=False)
                self._retry_counts.pop(post.id, None)
                self._dead_letters.discard(post.id)
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
                    delay = _retry_backoff_delay(attempt)
                    logger.warning(
                        "Recoverable error extracting knowledge graph for content id=%s "
                        "(attempt %d/%d): %s. Retrying after %.2fs...",
                        post.id,
                        attempt + 1,
                        _RETRY_MAX_ATTEMPTS,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                continue

        current_fails = self._retry_counts.get(post.id, 0) + 1
        self._retry_counts[post.id] = current_fails

        if current_fails >= _DEAD_LETTER_THRESHOLD:
            logger.error(
                "Post id=%s has failed %d consecutive times. "
                "Moving to dead-letter state.",
                post.id,
                current_fails,
            )
            self._dead_letters.add(post.id)
            try:
                async with self.db.async_session() as session:
                    await session.execute(
                        text("""
                            UPDATE content
                            SET raw_metadata = COALESCE(raw_metadata, '{}'::jsonb) ||
                                jsonb_build_object(
                                    'graph_extraction_error', CAST(:error_msg AS text),
                                    'graph_retries_exhausted', CAST(:retries AS text),
                                    'graph_failed_at', CAST(:failed_at AS text)
                                )
                            WHERE id = :post_id
                        """),
                        {
                            "post_id": post.id,
                            "error_msg": str(last_recoverable_error)[:1000],
                            "retries": str(current_fails),
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    await session.commit()
                return False
            except Exception as db_err:
                logger.error(
                    "Failed to record dead-letter for content id=%s: %s",
                    post.id,
                    db_err,
                )
                return False

        logger.warning(
            "Recoverable error extracting knowledge graph for content id=%s "
            "after %d attempts (total consecutive failures: %d): %s. "
            "Post will be retried in a subsequent batch.",
            post.id,
            _RETRY_MAX_ATTEMPTS,
            current_fails,
            last_recoverable_error,
        )
        return False


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
        batch_size=settings.graph_batch_size,
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
