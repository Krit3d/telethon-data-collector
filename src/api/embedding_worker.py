from __future__ import annotations

import asyncio
import logging
import re
import signal
from typing import Any

from sqlalchemy.exc import OperationalError

try:
    import asyncpg.exceptions

    PostgresError = asyncpg.exceptions.PostgresError
except ImportError:
    PostgresError = Exception

from src.config.config import Settings, load_settings
from src.db.database import Database
from src.graph.db.extractor_repo import ExtractorRepository
from src.db.models import Content
from src.embeddings.qdrant_service import QdrantService
from src.embeddings.visual_service import VisualEmbeddingService
from src.parser.creators.core.schemas import AccountMetadata, ContentMetadata

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


class EmbeddingWorker:

    def __init__(
        self,
        db: Database,
        extractor_repo: ExtractorRepository,
        qdrant: QdrantService,
        visual_service: VisualEmbeddingService,
        settings: Settings,
        batch_size: int = 64,
        poll_interval: int = 5,
    ) -> None:
        self.db = db
        self.extractor_repo = extractor_repo
        self.qdrant = qdrant
        self.visual_service = visual_service
        self.settings = settings
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.priority_mode: bool = False
        self._shutdown_event = asyncio.Event()

    def handle_shutdown(self, *args: object) -> None:
        logger.info("Shutdown signal received, finishing current sub-batch before exit")
        self._shutdown_event.set()

    def _clean_text(self, text: str) -> str:
        def _strip_bracket_noise(m: re.Match[str]) -> str:
            inner = (m.group(1) or m.group(2) or "").strip()
            inner_lower = inner.lower()
            if any(w in inner_lower for w in _NOISE_WORDS):
                return ""
            return m.group(0)

        cleaned = _BRACKET_NOISE_RE.sub(_strip_bracket_noise, text)
        cleaned = _HALLUCINATION_RE.sub("", cleaned)
        cleaned = _strip_consecutive_repeats(cleaned)
        cleaned = _WHITESPACE_RE.sub(" ", cleaned)
        cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
        return cleaned.strip()

    def _extract_domain(self, url: str) -> str | None:
        url = url.strip()
        if not url:
            return None
        if "://" in url:
            url = url.split("://", 1)[1]
        if url.startswith("www."):
            url = url[4:]
        url = url.split("/", 1)[0]
        url = url.split("?", 1)[0]
        return url.lower() if url else None

    def _safe_parse_account_metadata(
        self, raw: dict[str, Any] | None
    ) -> AccountMetadata | None:
        if not raw:
            return None
        try:
            return AccountMetadata(**raw)
        except Exception:
            return None

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
        account = post.account

        if account:
            if account.platform:
                parts.append(f"Platform: {account.platform}")

            title = account.title or None
            if not title and account.raw_metadata:
                title = account.raw_metadata.get("full_name")
            if title:
                parts.append(f"Account: {title}")

            if account.username:
                parts.append(f"Username: @{account.username}")

            description = account.description or None
            if not description and account.raw_metadata:
                description = (
                    account.raw_metadata.get("biography")
                    or account.raw_metadata.get("bio")
                )
            if description:
                parts.append(f"Bio: {description}")

            account_meta = self._safe_parse_account_metadata(account.raw_metadata)
            location = None
            if account_meta:
                if account_meta.location:
                    location = account_meta.location
                elif account_meta.geo_data:
                    geo_parts = [
                        p
                        for p in (account_meta.geo_data.city, account_meta.geo_data.country)
                        if p
                    ]
                    if geo_parts:
                        location = ", ".join(geo_parts)
            elif account.raw_metadata:
                raw_loc = account.raw_metadata.get("location")
                if isinstance(raw_loc, str) and raw_loc:
                    location = raw_loc
                elif isinstance(raw_loc, dict):
                    geo_parts = [
                        p
                        for p in (raw_loc.get("city"), raw_loc.get("country"))
                        if p
                    ]
                    if geo_parts:
                        location = ", ".join(geo_parts)
            if location:
                parts.append(f"Location: {location}")

            urls: list[str] = []
            if account_meta:
                if account_meta.external_links:
                    urls.extend(account_meta.external_links[:5])
                if account_meta.website:
                    urls.append(account_meta.website)
                if account_meta.link_in_bio:
                    urls.append(account_meta.link_in_bio)
            elif account.raw_metadata:
                ext_links = account.raw_metadata.get("external_links")
                if isinstance(ext_links, list):
                    urls.extend(str(link) for link in ext_links[:5])
                website = account.raw_metadata.get("website")
                if isinstance(website, str) and website:
                    urls.append(website)
                link_in_bio = account.raw_metadata.get("link_in_bio")
                if isinstance(link_in_bio, str) and link_in_bio:
                    urls.append(link_in_bio)
            if urls:
                domains: list[str] = []
                for u in urls:
                    domain = self._extract_domain(u)
                    if domain:
                        domains.append(domain)
                if domains:
                    unique_domains = list(dict.fromkeys(domains))
                    parts.append(f"Domains: {', '.join(unique_domains)}")

            badges: list[str] = []
            if account_meta:
                if getattr(account_meta, "is_verified", False):
                    badges.append("Verified")
                if getattr(account_meta, "is_business", False):
                    badges.append("Business")
            elif isinstance(account.raw_metadata, dict):
                if account.raw_metadata.get("is_verified"):
                    badges.append("Verified")
                if (
                    account.raw_metadata.get("is_business_account")
                    or account.raw_metadata.get("is_professional_account")
                ):
                    badges.append("Business")
            if badges:
                parts.append(f"Account Type: {' '.join(badges)}")

        content_meta = self._safe_parse_content_metadata(post.raw_metadata)

        post_type = None
        if content_meta:
            post_type = content_meta.post_type
        elif post.raw_metadata:
            pt = post.raw_metadata.get("post_type")
            if pt:
                post_type = str(pt)
        if post_type:
            parts.append(f"Post type: {post_type}")

        author_parts: list[str] = []
        if content_meta and content_meta.author_profile_snapshot:
            snapshot = content_meta.author_profile_snapshot
            if snapshot.title:
                author_parts.append(snapshot.title)
            if snapshot.username:
                author_parts.append(f"@{snapshot.username}")
        elif post.raw_metadata:
            author = post.raw_metadata.get("author_profile_snapshot")
            if isinstance(author, dict):
                if author.get("title"):
                    author_parts.append(str(author["title"]))
                if author.get("username"):
                    author_parts.append(f"@{author['username']}")
        if author_parts:
            parts.append(f"Author: {' '.join(author_parts)}")

        if post.content:
            cleaned = self._clean_text(post.content)
            if cleaned:
                parts.append(f"Post: {cleaned}")

        if post.transcription:
            cleaned_t = self._clean_text(post.transcription)
            if cleaned_t:
                parts.append(f"Transcription: {cleaned_t}")

        accessibility = None
        if content_meta and content_meta.accessibility_caption:
            accessibility = content_meta.accessibility_caption
        elif post.raw_metadata:
            cap = post.raw_metadata.get("accessibility_caption")
            if isinstance(cap, str) and cap:
                accessibility = cap
        if accessibility:
            cleaned_cap = self._clean_text(accessibility)
            if cleaned_cap:
                parts.append(f"Caption: {cleaned_cap}")

        hashtags = None
        if content_meta and content_meta.hashtags:
            hashtags = content_meta.hashtags
        elif post.raw_metadata:
            h = post.raw_metadata.get("hashtags")
            if isinstance(h, list) and h:
                hashtags = [str(x) for x in h]
            elif isinstance(h, str) and h:
                hashtags = [h]
        if hashtags:
            parts.append(f"Hashtags: {', '.join(hashtags)}")

        music_title = None
        music_author = None
        if content_meta:
            music_title = content_meta.music_title
            music_author = content_meta.music_author
        elif post.raw_metadata:
            clips = post.raw_metadata.get("clips_metadata")
            if isinstance(clips, dict):
                music_info = clips.get("clips_music_attribution_info")
                if isinstance(music_info, dict):
                    music_title = music_info.get("song_name")
                    music_author = music_info.get("artist_name")
        if music_title:
            audio = music_title
            if music_author:
                audio = f"{music_title} - {music_author}"
            parts.append(f"Audio: {audio}")

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
        points: list[tuple[int, str, int]] = []
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

            points.append((post.id, stripped_text, post.account_id))
            valid_posts.append(post)

        if skip_ids:
            await self.extractor_repo.mark_content_embedded(skip_ids)
            logger.debug(
                "Marked %d posts as embedded (skipped due to insufficient text)",
                len(skip_ids),
            )

        if not points:
            return

        texts = [p[1] for p in points]
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
                sub_ids = [p[0] for p in sub_batch]
                await self.extractor_repo.mark_content_embedded(sub_ids)
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
                    posts = await self.extractor_repo.get_unembedded_content(
                        limit=self.batch_size, priority_mode=self.priority_mode
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

    extractor_repo = ExtractorRepository(db.async_session, settings)

    visual_service = VisualEmbeddingService(settings)

    worker = EmbeddingWorker(
        db=db,
        extractor_repo=extractor_repo,
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
