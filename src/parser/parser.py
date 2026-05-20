"""Telegram channel parser for scraping and storing channel data.

This module parses Telegram channels, extracts channel metadata and posts,
and stores them in a database with knowledge graph extraction.
"""

import asyncio
import logging
import random
import re
import time
from datetime import timezone
from pathlib import Path
from typing import Any, Callable

from telethon import TelegramClient
from telethon.errors import (
    AuthKeyError,
    ChannelPrivateError,
    FloodWaitError,
    RPCError,
    SessionExpiredError,
    SessionRevokedError,
    UserDeactivatedError,
)
from telethon.tl.types import Channel as TlChannel
from telethon.tl.types import Message

from src.config.config import Settings, load_settings
from src.db.database import Database
from src.db.models import Channel
from src.parser.core.runner import start_workers
from src.parser.core.worker_base import BaseTelegramWorker
from src.parser.core.utils import (
    count_message_comments,
    count_message_reactions,
    get_channel_entity_safe,
    get_full_channel_info,
    normalize_username,
)

logger = logging.getLogger(__name__)


async def _extract_channel_metadata(
    client: TelegramClient,
    channel: Channel,
    db: Database,
    *,
    safe_api_call: Callable[..., Any],
    entity_cache: dict[tuple[int, int | None], Any] | None = None,
) -> tuple[dict[str, Any], TlChannel] | None:
    """Extract channel metadata with comprehensive error handling.

    This function resolves the channel entity and fetches full channel info.
    Relies on Telethon's native session-based caching for access_hash resolution.
    Never stores access_hash in the global PostgreSQL database to avoid
    cross-account token poisoning and FloodWait errors.

    Args:
        client: Telethon client instance.
        channel: Channel object from database with id and username.
        db: Database service for marking channel status.
        safe_api_call: Async function that wraps API calls with retry logic
            and error handling. Format: safe_api_call(name, callable).
        entity_cache: Optional per-worker entity cache for cheap lookups.
    """
    channel_id = channel.id
    entity = None

    # Try direct ID-based resolution first (cheap if session has cached the entity)
    result = await get_channel_entity_safe(
        client,
        channel_id,
        safe_api_call=safe_api_call,
        entity_cache=entity_cache,
    )
    if result.is_success() and result.entity is not None:
        entity = result.entity
        logger.info(
            "Channel id=%s resolved via ID (using Telethon session cache)",
            channel_id,
        )

    # Fallback: If ID resolution failed, try username resolution (expensive)
    # This typically only happens once per session per channel
    if entity is None and channel.username:
        logger.info(
            "Channel id=%s: ID resolution failed, falling back to username resolution",
            channel_id,
        )
        result = await get_channel_entity_safe(
            client,
            channel.username,
            safe_api_call=safe_api_call,
            entity_cache=entity_cache,
        )
        if result.is_success() and result.entity is not None:
            entity = result.entity
        else:
            # Username resolution failed
            if result.shadowbanned:
                await db.mark_channel_pending(channel_id)
            else:
                await db.mark_channel_rejected(channel_id)
                await asyncio.sleep(random.uniform(10, 20))
            return None

    if entity is None:
        # Both ID and username resolution failed
        await db.mark_channel_rejected(channel_id)
        await asyncio.sleep(random.uniform(10, 20))
        return None

    # Check if entity is actually a Channel (not a User, PeerUser, or other type)
    if not isinstance(entity, TlChannel):
        logger.warning("Entity is not a channel, skipping")
        return None

    # Get full channel info using shared utility
    subscribers_count, description = await get_full_channel_info(
        client, entity, safe_api_call=safe_api_call
    )

    channel_data: dict[str, Any] = {
        "id": int(entity.id),
        "username": normalize_username(getattr(entity, "username", None)),
        "title": getattr(entity, "title", "") or "",
        "description": description,
        "subscribers_count": subscribers_count,
    }

    return channel_data, entity


async def _process_message(
    msg: Message,
    entity: TlChannel,
    db: Database,
    *,
    safe_api_call: Callable[..., Any] | None = None,
) -> int | None:
    """Process a single message: save to PostgreSQL and return its ID.

    Enriched metadata extraction for OpenSPG knowledge graph integration.
    Extracts author, forward information, grouped_id, media flags, geo data,
    and builds a structured metadata JSONB field with additional attributes
    for domain-specific processing.

    Args:
        msg: Telethon message object.
        entity: Channel entity.
        db: Database service.
        safe_api_call: Optional async function that wraps API calls with
            retry logic and error handling.

    Returns:
        Post ID if successfully processed, None otherwise.
    """
    if msg.id is None or msg.date is None:
        return None

    # Skip system messages (messages with action) that don't help the OpenSPG graph
    # Use getattr to safely access action attribute (Pylance may not recognize it)
    msg_action = getattr(msg, "action", None)
    if msg_action is not None:
        return None

    # Skip empty messages without content
    msg_content = getattr(msg, "message", None)
    if not msg_content:
        return None

    published_at = msg.date
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    # Extract forward information if available
    fwd_from_channel_id: int | None = None
    if msg.fwd_from and hasattr(msg.fwd_from, "channel_id"):
        fwd_from_channel_id = getattr(msg.fwd_from, "channel_id", None)

    # Extract geo data if available
    geo_lat: float | None = None
    geo_long: float | None = None
    if msg.media and hasattr(msg.media, "geo"):
        geo = getattr(msg.media, "geo", None)
        if geo:
            geo_lat = getattr(geo, "lat", None)
            geo_long = getattr(geo, "long", None)

    # Extract reply information
    reply_to_msg_id: int | None = None
    if msg.reply_to and hasattr(msg.reply_to, "reply_to_msg_id"):
        reply_to_msg_id = getattr(msg.reply_to, "reply_to_msg_id", None)

    # Detect language (placeholder - in production, use a language detection library)
    # Telethon doesn't provide language detection natively
    language: str | None = getattr(msg, "lang", None)

    # Extract numeric metrics and patterns from message text
    text_content = msg_content or ""
    numeric_metrics: dict[str, Any] = {}

    # Extract numbers (prices, amounts, etc.)
    # Match patterns like $100, 100$, 100 USD, etc.
    price_patterns = re.findall(
        r"(?:\$|USD|EUR|GBP|RUB|₽)?\s*[\d,]+(?:\.\d+)?\s*(?:\$|USD|EUR|GBP|RUB|₽)?",
        text_content,
    )
    if price_patterns:
        numeric_metrics["price_patterns"] = price_patterns

    # Extract standalone numbers (integers and floats)
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text_content)
    if numbers:
        numeric_metrics["numbers"] = [
            float(n) if "." in n else int(n) for n in numbers
        ]

    # Count links in the message
    link_count = len(re.findall(r"https?://\S+", text_content))
    if link_count > 0:
        numeric_metrics["link_count"] = link_count

    # Count mentions (Telegram mentions start with @)
    mention_count = len(re.findall(r"@\w+", text_content))
    numeric_metrics["mention_count"] = mention_count

    # Count hashtags
    hashtag_count = len(re.findall(r"#\w+", text_content))
    numeric_metrics["hashtag_count"] = hashtag_count

    # Count words and characters
    numeric_metrics["word_count"] = len(text_content.split())
    numeric_metrics["char_count"] = len(text_content)

    # Build the metadata dictionary for JSONB column (only non-redundant fields for OpenSPG)
    metadata: dict[str, Any] = {
        "reply_to_msg_id": reply_to_msg_id,
        "numeric_metrics": numeric_metrics if numeric_metrics else None,
        "has_links": link_count > 0,
        "has_mentions": mention_count > 0,
        "has_hashtags": hashtag_count > 0,
    }

    # Remove None values from metadata to keep it clean
    metadata = {k: v for k, v in metadata.items() if v is not None}

    post_data: dict[str, Any] = {
        "channel_id": int(entity.id),
        "message_id": int(msg.id),
        "content": getattr(msg, "message", None),
        "published_at": published_at,
        "views": getattr(msg, "views", None),
        "comments_count": count_message_comments(msg),
        "shares_count": getattr(msg, "forwards", None),
        "reactions_count": count_message_reactions(msg),
        # Enriched metadata for OpenSPG knowledge graph (backward compatible)
        "author": getattr(msg, "post_author", None),
        "fwd_from_channel_id": fwd_from_channel_id,
        "grouped_id": getattr(msg, "grouped_id", None),
        "has_media": bool(msg.media),  # Boolean flag only, no downloads
        "geo_lat": geo_lat,
        "geo_long": geo_long,
        "language": language,
        # JSONB raw_metadata column for OpenSPG raw metadata extraction
        "raw_metadata": metadata if metadata else None,
    }

    post = await db.upsert_post(post_data)
    if not (post and post.id and post.content):
        return None

    return int(post.id)


class ParserWorker(BaseTelegramWorker):
    """Async worker that parses channels and extracts knowledge."""

    def __init__(
        self,
        worker_id: int,
        session_path: Path,
        db: Database,
        settings: Settings,
        api_id: int,
        api_hash: str,
        proxy_url: str | None = None,
        device_model: str = "PC 64bit",
        system_version: str = "Windows 10",
        app_version: str = "4.16.8",
        lang_code: str = "en",
        system_lang_code: str = "en-US",
    ):
        """Initialize the parser worker."""
        super().__init__(
            worker_id=worker_id,
            session_path=session_path,
            db=db,
            settings=settings,
            api_id=api_id,
            api_hash=api_hash,
            proxy_url=proxy_url,
            device_model=device_model,
            system_version=system_version,
            app_version=app_version,
            lang_code=lang_code,
            system_lang_code=system_lang_code,
        )
        # Initialize last activity timestamp for Watchdog liveness signaling
        self.last_activity = time.time()

    async def run(self) -> None:
        """Main worker loop: continuously fetch and parse channels from DB queue."""
        await self.connect()

        logger.info("Worker %d: Starting parser loop", self.worker_id)

        try:
            while True:
                # Signal liveness to Watchdog at the start of each loop iteration
                self.last_activity = time.time()
                if not self.is_alive:
                    logger.info(
                        "Worker %d: is_alive=False, terminating worker loop",
                        self.worker_id,
                    )
                    return
                try:
                    channel = await self.db.get_channel_for_parsing()

                    if channel is None:
                        logger.debug(
                            "Worker %d: No channels ready for parsing, sleeping 30s",
                            self.worker_id,
                        )
                        await asyncio.sleep(30)
                        continue

                    logger.info(
                        "Worker %d: Processing channel id=%s username=%s",
                        self.worker_id,
                        channel.id,
                        channel.username,
                    )

                    await self._parse_single_channel(channel)

                except (
                    FloodWaitError,
                    AuthKeyError,
                    UserDeactivatedError,
                    SessionExpiredError,
                    SessionRevokedError,
                ) as e:
                    # Propagate critical exceptions to _worker_runner for session cooldown/ban handling
                    raise
                except Exception as e:
                    logger.error(
                        "Worker %d: Unexpected error in loop: %s",
                        self.worker_id,
                        e,
                        exc_info=True,
                    )
                    await asyncio.sleep(30)
        finally:
            logger.info("Worker %d: Cleanup complete", self.worker_id)

    async def _parse_single_channel(self, channel: Channel) -> None:
        """Fetch, normalize, and persist one channel with its recent posts using safe pagination.

        Implements controlled chunk-based fetching to avoid burst requests that trigger
        FloodWait bans. Messages are fetched in small chunks with natural delays between
        requests.

        When a FloodWaitError (or shadowban/network error caused by it) is bubbled up
        from safe_api_call:
        - Catch it, log it, and safely return the channel to the queue
        - Ensure all local DB resources/sessions are closed
        - Propagate the exception to _worker_runner so the runner knows this session is in cooldown

        Args:
            channel: Channel object from database with id and username.
        """

        channel_id = channel.id
        logger.info(
            "Start channel parse: id=%s username=%s",
            channel_id,
            channel.username,
        )

        try:
            result = await _extract_channel_metadata(
                self.client,  # type: ignore[attr-defined]
                channel,
                self.db,
                safe_api_call=self.safe_api_call,
                entity_cache=self._entity_cache,  # Pass per-worker entity cache
            )
            if result is None:
                await self.db.mark_channel_rejected(channel_id)
                return

            channel_data, entity = result

            await self.db.upsert_channel(channel_data)

            logger.debug(
                "Channel saved: id=%s username=%s title=%s",
                channel_data["id"],
                channel_data["username"],
                channel_data["title"],
            )

            # NATIVE-LIKE EXPLORATION: Occasionally simulate "opening" the channel view
            # by fetching a single message first, mimicking human browsing behavior
            if random.random() < 0.3:  # 30% chance to do an exploration fetch
                try:
                    logger.debug(
                        "Worker %d: Native exploration - fetching latest message from channel %s",
                        self.worker_id,
                        channel_id,
                    )
                    # Wrap API call in safe_api_call to handle FloodWaitError properly
                    exploration_msg = await self.safe_api_call(
                        "exploration_fetch",
                        lambda: self.client.get_messages(  # type: ignore[attr-defined]
                            entity, limit=1
                        ),
                    )
                    # Small delay to simulate reading
                    await asyncio.sleep(random.uniform(1, 3))
                    # Count messages (handle both single Message and list responses)
                    msg_count = 1 if exploration_msg else 0
                    if isinstance(exploration_msg, list):
                        msg_count = len(exploration_msg)
                    self.logger.debug(
                        "Worker %d: Exploration complete - found %d messages",
                        self.worker_id,
                        msg_count,
                    )
                except Exception as e:
                    # Non-critical - log but continue with main parsing
                    self.logger.warning(
                        "Worker %d: Native exploration failed for channel %s: %s",
                        self.worker_id,
                        channel_id,
                        e,
                    )

            # SAFE PAGINATION: Fetch messages in controlled chunks to avoid burst requests
            posts_saved = 0
            chunk_size = 20  # Safe chunk size to avoid triggering FloodWait
            last_msg_id: int = (
                0  # Track last message ID for pagination (initialized to integer 0)
            )
            posts_limit = self.settings.posts_limit

            # SMART SKIP: Fetch the latest known message ID from DB to avoid re-fetching
            # This saves API limits by not requesting messages we already have
            latest_known_id = (
                await self.db.get_latest_message_id(channel_id) or 0
            )
            if latest_known_id > 0:
                self.logger.info(
                    "Channel %s: Latest known message ID is %d, will skip older messages",
                    channel_id,
                    latest_known_id,
                )

            while posts_saved < posts_limit:
                # Calculate how many messages to fetch in this chunk
                remaining = posts_limit - posts_saved
                current_chunk_size = min(chunk_size, remaining)

                # Fetch a chunk of messages with manual pagination
                # Use offset_id to get messages older than the last fetched message
                # Note: offset_id=None means "start from the most recent message"
                # Use min_id to skip messages we already have (optimization)
                # Clean lambda - no conditional default value evaluation traps
                messages = await self.safe_api_call(
                    f"get_messages(channel_id={channel_id}, chunk_size={current_chunk_size})",
                    operation=lambda: self.client.get_messages(  # type: ignore[attr-defined]
                        entity,
                        limit=current_chunk_size,
                        offset_id=last_msg_id,
                        min_id=latest_known_id,
                    ),
                )
                # Signal liveness to Watchdog after fetching a chunk of messages
                self.last_activity = time.time()

                # Handle API call failure
                if messages is None:
                    self.logger.warning(
                        "Worker %d: Failed to fetch messages for channel %s",
                        self.worker_id,
                        channel_id,
                    )
                    break

                # Normalize to list (get_messages may return a single Message or a list)
                if not isinstance(messages, list):
                    messages = [messages]

                # If no messages returned, we've reached the end
                if not messages:
                    break

                # Process messages in this chunk
                for msg in messages:
                    if not isinstance(msg, Message):
                        continue

                    # SMART SKIP: Early exit if we encounter messages we already have
                    # This prevents wasting API limits on already-processed content
                    if msg.id is not None and msg.id <= latest_known_id:
                        self.logger.info(
                            "Channel %s: Reached already known message ID %d (latest known: %d). Stopping pagination.",
                            channel_id,
                            msg.id,
                            latest_known_id,
                        )
                        # Break out of the for loop
                        posts_saved = (
                            posts_limit  # Set to limit to exit while loop
                        )
                        break

                    post_id = await _process_message(
                        msg,
                        entity,
                        self.db,
                        safe_api_call=self.safe_api_call,
                    )
                    # Signal liveness to Watchdog after processing an individual message
                    self.last_activity = time.time()
                    if post_id is None:
                        continue

                    posts_saved += 1

                    # Check if we've reached the limit
                    if posts_saved >= posts_limit:
                        break

                # If we broke out due to smart skip, exit the while loop
                if posts_saved >= posts_limit:
                    break

                # Update last_msg_id for pagination (get messages older than oldest in chunk)
                if messages:
                    oldest_msg = messages[
                        -1
                    ]  # Last message in list is the oldest
                    if (
                        isinstance(oldest_msg, Message)
                        and oldest_msg.id is not None
                    ):
                        last_msg_id = oldest_msg.id

                # If we got fewer messages than requested, we've reached the end
                if len(messages) < current_chunk_size:
                    break

                # Apply lighter delay between chunks to avoid rate limits
                if self.settings.use_natural_delays:
                    await asyncio.sleep(self.natural_delay(base_delay=1.5))
                else:
                    await asyncio.sleep(random.uniform(1.0, 2.5))

            await self.db.mark_channel_parsed(channel_id)

            logger.info(
                "Done channel: id=%s username=%s (posts saved: %s)",
                channel_id,
                channel.username,
                posts_saved,
            )

            # Reduced random delay between channels to avoid rate limits
            await asyncio.sleep(random.uniform(2, 5))

        except FloodWaitError as e:
            # FloodWaitError bubbled up from safe_api_call
            # The worker_base now raises FloodWaitError for delays > 5 seconds
            # instead of sleeping indefinitely inside safe_api_call
            delay = int(getattr(e, "seconds", 0)) or 1
            logger.warning(
                "Worker %d: FloodWaitError in channel %s (delay=%ds). "
                "Marking channel as pending and propagating error.",
                self.worker_id,
                channel_id,
                delay,
            )
            # Safely return channel to queue for later processing
            await self.db.mark_channel_pending(channel_id)
            # Close any local DB resources if needed
            # (Database connections are managed by the pool, no explicit close needed here)
            # Re-raise the exception so _worker_runner can handle session cooldown
            raise

        except (
            ChannelPrivateError,
            UserDeactivatedError,
            AuthKeyError,
            SessionRevokedError,
            SessionExpiredError,
        ) as e:
            if isinstance(
                e,
                (
                    UserDeactivatedError,
                    AuthKeyError,
                    SessionRevokedError,
                    SessionExpiredError,
                ),
            ):
                logger.critical(
                    "Worker %d: Account is DEAD/FROZEN (%s). Terminating worker to protect proxy.",
                    self.worker_id,
                    type(e).__name__,
                )
                self.is_alive = False
                await self.db.mark_channel_rejected(channel_id)
                raise  # Re-raise to propagate fatal session error
            logger.warning(
                "Channel %s: access denied (%s), marking as rejected",
                channel_id,
                type(e).__name__,
            )
            await self.db.mark_channel_rejected(channel_id)

        except (OSError, asyncio.TimeoutError, ConnectionError) as e:
            logger.exception(
                "Channel %s: network error (%s)", channel_id, type(e).__name__
            )
            await self.db.mark_channel_pending(channel_id)  # Return to queue

        except RPCError as e:
            logger.exception(
                "Channel %s: Telethon RPCError (%s)",
                channel_id,
                type(e).__name__,
            )
            await self.db.mark_channel_rejected(channel_id)  # Permanently block

        except Exception:
            logger.exception("Channel %s: unexpected error", channel_id)
            await self.db.mark_channel_rejected(channel_id)  # Permanently block


async def main() -> None:
    """Entry point: discover sessions, read individual configs, spawn worker tasks."""
    settings = load_settings()

    # Install global asyncio exception handler to prevent silent crashes
    loop = asyncio.get_running_loop()

    def global_exception_handler(loop, context):
        msg = context.get("exception", context["message"])
        logging.getLogger("asyncio_global").critical(
            f"Unhandled asyncio exception: {msg}"
        )

    loop.set_exception_handler(global_exception_handler)

    # Print summary of changes/features enabled
    logger.info("=" * 60)
    logger.info("PARSER INITIALIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(
        "- Safe API calls: All API calls wrapped in safe_api_call (FloodWait handled by worker_base)"
    )
    logger.info(
        "- Smart skip: Enabled (min_id=latest_known_id to avoid re-fetching)"
    )
    logger.info(
        "- Metadata enrichment: Enabled (author, fwd_from, grouped_id, has_media, geo, JSONB metadata)"
    )
    logger.info(
        "- FloodWait handling: Session-pool-friendly - sessions return to pool with cooldown"
    )
    logger.info("- Posts limit per channel: %d", settings.posts_limit)
    logger.info("=" * 60)

    logger.info(
        "Starting parser (posts_limit=%d, concurrency from session count)",
        settings.posts_limit,
    )

    # Database is assumed to be initialized by the migration script
    db = Database(settings.db_url)

    # Use the shared runner to start workers
    await start_workers(
        worker_class=ParserWorker,
        settings=settings,
        db=db,
    )


if __name__ == "__main__":
    asyncio.run(main())
