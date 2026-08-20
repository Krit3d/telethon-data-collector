from __future__ import annotations

import asyncio
import logging
import re
import signal
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import selectinload

try:
    import asyncpg.exceptions

    PostgresError = asyncpg.exceptions.PostgresError
except ImportError:
    PostgresError = Exception

from src.config.config import Settings, load_settings
from src.db.database import Database
from src.db.models import Account, Content
from src.embeddings.qdrant_service import QdrantService
from src.embeddings.visual_service import VisualEmbeddingService
from src.parser.creators.core.schemas import ContentMetadata

logger = logging.getLogger(__name__)

_BASE_BACKOFF: float = 1.0
_MAX_BACKOFF: float = 60.0
_MIN_TEXT_LENGTH: int = 15
_MAX_SUB_BATCH: int = 32
_AVG_LEN_TIER_1: int = 2000
_TIER_1_BATCH: int = 16
_AVG_LEN_TIER_2: int = 4000
_TIER_2_BATCH: int = 8
_AVG_LEN_TIER_3: int = 8000
_TIER_3_BATCH: int = 4

_NOISE_WORDS: frozenset[str] = frozenset({
    "music", "музыка", "смех", "шум", "аплодисменты",
    "laughter", "applause", "inaudible", "нрзб",
    "background noise", "crowd", "cheering", "silence",
    "static", "cough", "кашель", "звуковые эффекты",
    "звук", "тишина", "шепот", "whisper", "breathing",
    "дыхание", "грохот", "шипение", "свист", "scribble",
})

_BRACKET_NOISE_RE = re.compile(
    r"\[([^\]]*)\]|\(([^)]*)\)",
    re.IGNORECASE,
)

_HALLUCINATION_RE = re.compile(
    r"(?:"
    r"[Tt]hank(?:s)?\s+(?:you\s+)?for\s+(?:watching|listening|viewing|subscribing)"
    r"|"
    r"Спасибо\s+за\s+(?:просмотр|прослушивание|подписку|поддержку)"
    r"|"
    r"[Ss]ubtitles?\s+(?:by|created|made|translated)"
    r"|"
    r"Субтитры\s+(?:созданы|сделаны|переведены|созданы)"
    r"|"
    r"[Cc]ommunity|Сообщество"
    r"|"
    r"Пожалуйста,\s+(?:подпишитесь|ставьте\s+лайк|нажмите\s+колокольчик)"
    r"|"
    r"[Pp]lease\s+(?:subscribe|like|hit\s+the\s+bell)"
    r")",
)

_WORD_TOKEN_RE = re.compile(r"\b(\w+(?:\s+\w+){0,4})\b", re.UNICODE)

_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r" {2,}")

_SENTENCE_REPEAT_RE = re.compile(r"\s+")

_MAX_CONSECUTIVE_REPEATS: int = 3


def _strip_consecutive_repeats(text: str) -> str:
    tokens = _WORD_TOKEN_RE.findall(text)
    if len(tokens) < _MAX_CONSECUTIVE_REPEATS:
        return text
    result: list[str] = []
    i = 0
    while i < len(tokens):
        run_token = tokens[i].strip().lower()
        run_len = 1
        while (
            i + run_len < len(tokens)
            and tokens[i + run_len].strip().lower() == run_token
        ):
            run_len += 1
        if run_len >= _MAX_CONSECUTIVE_REPEATS:
            result.append(tokens[i])
        else:
            result.extend(tokens[i : i + run_len])
        i += run_len
    if len(result) == len(tokens):
        return text
    return " ".join(result)


def _strip_sentence_repeats(text: str) -> str:
    tokens = _SENTENCE_REPEAT_RE.split(text.strip())
    if len(tokens) < 2:
        return text
    changed = True
    while changed:
        changed = False
        n = len(tokens)
        for w in range(n // 2, 0, -1):
            i = 0
            while i + 2 * w <= n:
                if tokens[i : i + w] == tokens[i + w : i + 2 * w]:
                    tokens = tokens[: i + w] + tokens[i + 2 * w :]
                    changed = True
                    break
                i += 1
            if changed:
                break
    return " ".join(tokens)


class EmbeddingWorker:

    def __init__(
        self,
        db: Database,
        qdrant: QdrantService,
        visual_service: VisualEmbeddingService,
        settings: Settings,
        batch_size: int = 64,
        poll_interval: int = 5,
    ) -> None:
        self.db = db
        self.qdrant = qdrant
        self.visual_service = visual_service
        self.settings = settings
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.priority_mode: bool = False
        self._shutdown_event = asyncio.Event()

    async def _fetch_unembedded_content(self, limit: int) -> list[Content]:
        async with self.db.async_session() as session:
            result = await session.execute(
                select(Content)
                .options(selectinload(Content.account))
                .join(Content.account)
                .where(
                    Content.is_embedded.is_(False),
                    Account.status == "verified",
                )
                .order_by(Content.id.desc() if self.priority_mode else Content.id.asc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def _mark_content_embedded(self, content_ids: list[int]) -> None:
        async with self.db.async_session() as session:
            await session.execute(
                update(Content).where(Content.id.in_(content_ids)).values(is_embedded=True)
            )
            await session.commit()

    def handle_shutdown(self, *args: object) -> None:
        logger.info("Shutdown signal received, finishing current sub-batch before exit")
        self._shutdown_event.set()

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'(?<=[\w])([#@])', r' \1', text)

        def _strip_bracket_noise(m: re.Match[str]) -> str:
            inner = (m.group(1) or m.group(2) or "").strip()
            inner_lower = inner.lower()
            if any(w in inner_lower for w in _NOISE_WORDS):
                return ""
            return m.group(0)

        cleaned = _BRACKET_NOISE_RE.sub(_strip_bracket_noise, text)
        cleaned = _HALLUCINATION_RE.sub("", cleaned)
        cleaned = _strip_consecutive_repeats(cleaned)
        cleaned = _strip_sentence_repeats(cleaned)
        cleaned = _WHITESPACE_RE.sub(" ", cleaned)
        cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
        return cleaned.strip()

    def _safe_parse_content_metadata(
        self, raw: dict[str, Any] | None
    ) -> ContentMetadata | None:
        if not raw:
            return None
        try:
            return ContentMetadata(**raw)
        except Exception:
            return None

    def _assemble_embedding_text(self, post: Content) -> str:
        parts: list[str] = []

        content_meta = self._safe_parse_content_metadata(post.raw_metadata)

        cleaned_content = ""
        if post.content:
            cleaned_content = self._clean_text(post.content)

        cleaned_transcription = ""
        if post.transcription:
            cleaned_transcription = self._clean_text(post.transcription)

        if cleaned_content and cleaned_transcription:
            if (
                cleaned_content == cleaned_transcription
                or cleaned_content in cleaned_transcription
            ):
                parts.append(cleaned_transcription)
            elif cleaned_transcription in cleaned_content:
                parts.append(cleaned_content)
            else:
                parts.append(cleaned_content)
                parts.append(cleaned_transcription)
        elif cleaned_content:
            parts.append(cleaned_content)
        elif cleaned_transcription:
            parts.append(cleaned_transcription)

        accessibility: str | None = None
        if content_meta and content_meta.accessibility_caption:
            accessibility = content_meta.accessibility_caption
        elif post.raw_metadata:
            cap = post.raw_metadata.get("accessibility_caption")
            if isinstance(cap, str) and cap:
                accessibility = cap
        if accessibility:
            cleaned_cap = self._clean_text(accessibility)
            if cleaned_cap:
                parts.append(cleaned_cap)

        hashtags: list[str] | None = None
        if content_meta and content_meta.hashtags:
            hashtags = content_meta.hashtags
        elif post.raw_metadata:
            h = post.raw_metadata.get("hashtags")
            if isinstance(h, list) and h:
                hashtags = [str(x) for x in h]
            elif isinstance(h, str) and h:
                hashtags = [h]
        if hashtags:
            parts.append(", ".join(hashtags))

        return "\n".join(parts)

    def _compute_sub_batch_size(self, texts: list[str]) -> int:
        if not texts:
            return _MAX_SUB_BATCH
        avg_len = sum(len(t) for t in texts) / len(texts)
        if avg_len > _AVG_LEN_TIER_3:
            return _TIER_3_BATCH
        if avg_len > _AVG_LEN_TIER_2:
            return _TIER_2_BATCH
        if avg_len > _AVG_LEN_TIER_1:
            return _TIER_1_BATCH
        return _MAX_SUB_BATCH

    async def _process_batch(self, posts: list[Content]) -> None:
        points: list[dict[str, Any]] = []
        valid_posts: list[Content] = []
        skip_ids: list[int] = []

        for post in posts:
            text_to_embed = self._assemble_embedding_text(post)
            stripped_text = text_to_embed.strip()

            if len(stripped_text) < _MIN_TEXT_LENGTH:
                logger.debug(
                    "Content id=%s produced text shorter than %d chars after cleaning, skipping embedding",
                    post.id,
                    _MIN_TEXT_LENGTH,
                )
                skip_ids.append(post.id)
                continue

            account = post.account
            subscribers_count: int = account.subscribers_count if account and account.subscribers_count else 0
            is_author_blog: bool | None = account.is_author_blog if (account and account.is_author_blog is not None) else None
            views: int = post.views or 0
            comments_count: int = post.comments_count or 0
            shares_count: int = post.shares_count or 0
            reactions_count: int = post.reactions_count or 0

            if subscribers_count > 0:
                engagement_rate = round((reactions_count + comments_count) / subscribers_count, 4)
            else:
                engagement_rate = 0.0

            payload: dict[str, Any] = {
                "post_id": post.id,
                "account_id": post.account_id,
                "platform": account.platform if account else "UNKNOWN",
                "subscribers_count": subscribers_count,
                "is_author_blog": is_author_blog,
                "views": views,
                "comments_count": comments_count,
                "shares_count": shares_count,
                "reactions_count": reactions_count,
                "engagement_rate": engagement_rate,
                "text": stripped_text,
            }

            points.append(payload)
            valid_posts.append(post)

        if skip_ids:
            await self._mark_content_embedded(skip_ids)
            logger.debug(
                "Marked %d posts as embedded (skipped due to insufficient text)",
                len(skip_ids),
            )

        if not points:
            return

        texts = [p["text"] for p in points]
        sub_batch_size = self._compute_sub_batch_size(texts)
        total_embedded = 0

        for i in range(0, len(points), sub_batch_size):
            if self._shutdown_event.is_set():
                logger.info(
                    "Shutdown requested, stopping batch processing after %d/%d items",
                    total_embedded,
                    len(points),
                )
                break

            sub_batch = points[i : i + sub_batch_size]
            sub_posts = valid_posts[i : i + sub_batch_size]
            try:
                visual_tasks: list[asyncio.Task[list[float] | None]] = []
                for post in sub_posts:
                    content_meta = self._safe_parse_content_metadata(post.raw_metadata)
                    video_url = content_meta.video_url if content_meta else None
                    visual_tasks.append(
                        asyncio.ensure_future(
                            self.visual_service.extract_visual_embedding(video_url)
                        )
                    )
                visual_embeddings = await asyncio.gather(*visual_tasks)
                await self.qdrant.upsert_batch(sub_batch, list(visual_embeddings))
                sub_ids = [p["post_id"] for p in sub_batch]
                await self._mark_content_embedded(sub_ids)
                total_embedded += len(sub_ids)
                logger.debug(
                    "Embedded sub-batch of %d items (%d/%d total)",
                    len(sub_ids),
                    total_embedded,
                    len(points),
                )
            except Exception as e:
                logger.error(
                    "Failed to embed sub-batch of %d items: %s",
                    len(sub_batch),
                    e,
                    exc_info=True,
                )
                raise

        logger.info("Successfully embedded %d content items", total_embedded)

    async def _backoff_wait(self, attempt: int) -> None:
        backoff_time = min(
            _BASE_BACKOFF * (2 ** (attempt - 1)),
            _MAX_BACKOFF,
        )
        try:
            await asyncio.wait_for(
                self._shutdown_event.wait(),
                timeout=backoff_time,
            )
        except asyncio.TimeoutError:
            pass

    async def run(self) -> None:
        logger.info(
            "Embedding worker started (batch_size=%d, poll_interval=%ds, priority_mode=%s)",
            self.batch_size,
            self.poll_interval,
            self.priority_mode,
        )

        consecutive_errors = 0

        while not self._shutdown_event.is_set():
            try:
                try:
                    posts = await self._fetch_unembedded_content(
                        limit=self.batch_size
                    )
                except (OperationalError, PostgresError) as e:
                    consecutive_errors += 1
                    logger.warning(
                        "Database error while fetching content: %s. Backing off for %.1fs (attempt %d)",
                        e,
                        min(
                            _BASE_BACKOFF * (2 ** (consecutive_errors - 1)),
                            _MAX_BACKOFF,
                        ),
                        consecutive_errors,
                    )
                    await self._backoff_wait(consecutive_errors)
                    continue

                if not posts:
                    logger.debug(
                        "No unembedded content found, sleeping %ds", self.poll_interval
                    )
                    consecutive_errors = 0
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(),
                            timeout=self.poll_interval,
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue

                logger.info(
                    "Processing batch of %d unembedded content items", len(posts)
                )
                consecutive_errors = 0

                await self._process_batch(posts)

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info("Embedding worker task cancelled")
                break
            except Exception as e:
                consecutive_errors += 1
                backoff_time = min(
                    _BASE_BACKOFF * (2 ** (consecutive_errors - 1)),
                    _MAX_BACKOFF,
                )
                logger.error(
                    "Unexpected error in embedding worker loop: %s. Backing off for %.1fs (attempt %d)",
                    e,
                    backoff_time,
                    consecutive_errors,
                    exc_info=True,
                )
                await self._backoff_wait(consecutive_errors)

        logger.info("Embedding worker stopped")


async def run_embedding_worker() -> None:
    settings = load_settings()

    logger.info("Starting embedding worker")

    db = Database(settings.db_url)
    await db.init_db()

    qdrant = QdrantService(settings)
    try:
        await qdrant.initialize()
        logger.info("Qdrant service initialized")
    except Exception as e:
        logger.error("Failed to initialize Qdrant: %s", e)
        logger.warning("Embedding worker cannot function without Qdrant - exiting")
        await db.close()
        return

    visual_service = VisualEmbeddingService(settings)

    worker = EmbeddingWorker(
        db=db,
        qdrant=qdrant,
        visual_service=visual_service,
        settings=settings,
        batch_size=settings.embedding_batch_size,
        poll_interval=5,
    )
    worker.priority_mode = getattr(settings, "embedding_priority_mode", False)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, worker.handle_shutdown)
        except NotImplementedError:
            signal.signal(sig, worker.handle_shutdown)

    try:
        await worker.run()
    finally:
        await qdrant.close()
        await db.close()
        logger.info("Resources cleaned up")


if __name__ == "__main__":
    asyncio.run(run_embedding_worker())
