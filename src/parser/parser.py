"""Telegram channel parser for scraping and storing channel data.

This module parses Telegram channels, extracts channel metadata and posts,
and stores them in a database with knowledge graph extraction.
"""

import asyncio
import logging
import random
from datetime import timezone
from pathlib import Path
from typing import Any, Callable

from telethon import TelegramClient
from telethon.errors import (
    AuthKeyError,
    ChannelPrivateError,
    RPCError,
    SessionRevokedError,
    UserDeactivatedError,
)
from telethon.errors.rpcerrorlist import FloodWaitError
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
    fetch_avatar_path,
    get_channel_entity_safe,
    get_full_channel_info,
    normalize_username,
)

logger = logging.getLogger(__name__)


async def _extract_channel_metadata(
    client: TelegramClient,
    channel: Channel,
    db: Database,
    avatars_dir: Path,
    *,
    safe_api_call: Callable[..., Any],
    entity_cache: dict[tuple[int, int | None], Any] | None = None,
) -> tuple[dict[str, Any], TlChannel] | None:
    """Extract channel metadata with comprehensive error handling.

    This function resolves the channel entity, fetches full channel info,
    and downloads the avatar. Implements Zero-Username policy: if access_hash
    is available, use cheap InputPeerChannel resolution. Only use username
    resolution when access_hash is missing (first encounter), then immediately
    save the hash to the DB.

    Args:
        client: Telethon client instance.
        channel: Channel object from database with id and username.
        db: Database service for marking channel status.
        avatars_dir: Directory where avatar files are stored.
        safe_api_call: Async function that wraps API calls with retry logic
            and error handling. Format: safe_api_call(name, callable).
        entity_cache: Optional per-worker entity cache for cheap lookups.
    """
    channel_id = channel.id

    # ZERO-USERNAME POLICY: Prioritize cheap ID+hash resolution to avoid FloodWait
    entity = None
    
    # First, try cheap resolution with access_hash if available (NO USERNAME)
    if channel.access_hash is not None:
        result = await get_channel_entity_safe(
            client,
            channel_id,
            safe_api_call=safe_api_call,
            access_hash=channel.access_hash,
            entity_cache=entity_cache,
        )
        if result.is_success() and result.entity is not None:
            entity = result.entity
            logger.info(
                "Channel id=%s resolved via access_hash (cheap, no FloodWait)",
                channel_id,
            )
        else:
            # Cheap resolution failed - could be invalid hash or account restrictions
            logger.warning(
                "Cheap resolution failed for channel id=%s with access_hash. Error: %s",
                channel_id,
                result.reason or "unknown",
            )
            entity = None

    # Fallback: Only if access_hash is missing OR cheap resolution failed, try username
    # WARNING: This is EXPENSIVE and can trigger FloodWait!
    if entity is None and channel.username:
        logger.info(
            "Channel id=%s: access_hash missing or invalid, falling back to username resolution (EXPENSIVE)",
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
            # IMMEDIATELY save the access_hash to DB for future cheap resolution
            try:
                resolved_hash = getattr(entity, "access_hash", None)
                if resolved_hash is not None:
                    await db.update_channel_access_hash(channel_id, resolved_hash)
                    logger.info(
                        "Updated access_hash for channel id=%s from username resolution",
                        channel_id,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to update access_hash for channel id=%s: %s",
                    channel_id,
                    e,
                )
        else:
            # Username resolution failed
            if result.shadowbanned:
                await db.mark_channel_pending(channel_id)
            else:
                await db.mark_channel_rejected(channel_id)
                await asyncio.sleep(random.uniform(10, 20))
            return None

    if entity is None:
        # Both cheap (hash) and expensive (username) resolution failed
        await db.mark_channel_rejected(channel_id)
        await asyncio.sleep(random.uniform(10, 20))
        return None

    # Get full channel info using shared utility
    subscribers_count, description = await get_full_channel_info(
        client, entity, safe_api_call=safe_api_call
    )

    # Fetch avatar
    avatar_path = await fetch_avatar_path(
        client, entity, avatars_dir, safe_api_call=safe_api_call
    )

    channel_data: dict[str, Any] = {
        "id": int(entity.id),
        "username": normalize_username(getattr(entity, "username", None)),
        "title": getattr(entity, "title", "") or "",
        "description": description,
        "subscribers_count": subscribers_count,
        "avatar_url": avatar_path,
        "access_hash": getattr(entity, "access_hash", None),
    }

    return channel_data, entity


async def _process_message(
    msg: Message,
    entity: TlChannel,
    db: Database,
) -> int | None:
    """Process a single message: save to PostgreSQL and return its ID.

    Args:
        msg: Telethon message object.
        entity: Channel entity.
        db: Database service.

    Returns:
        Post ID if successfully processed, None otherwise.
    """
    if msg.id is None or msg.date is None:
        return None

    published_at = msg.date
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    post_data: dict[str, Any] = {
        "channel_id": int(entity.id),
        "message_id": int(msg.id),
        "content": getattr(msg, "message", None),
        "published_at": published_at,
        "views": getattr(msg, "views", None),
        "comments_count": count_message_comments(msg),
        "shares_count": getattr(msg, "forwards", None),
        "reactions_count": count_message_reactions(msg),
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

    async def run(self) -> None:
        """Main worker loop: continuously fetch and parse channels from DB queue."""
        await self.connect()

        logger.info("Worker %d: Starting parser loop", self.worker_id)

        try:
            while True:
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

                except FloodWaitError as e:
                    delay = int(getattr(e, "seconds", 0)) or 1
                    total_delay = delay + 10
                    logger.warning(
                        "Worker %d: FloodWaitError, sleeping %ds (+10s safety)",
                        self.worker_id,
                        total_delay,
                    )
                    await asyncio.sleep(total_delay)
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
        """Fetch, normalize, and persist one channel with its recent posts.

        Args:
            channel: Channel object from database with id and username.
        """
        from datetime import timezone

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
                self.settings.avatars_dir,
                safe_api_call=self.safe_api_call,
                entity_cache=self._entity_cache,  # Pass per-worker entity cache
            )
            if result is None:
                await self.db.mark_channel_rejected(channel_id)
                return

            channel_data, entity = result

            await self.db.upsert_channel(channel_data)

            logger.info(
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
                    exploration_msg = await self.client.get_messages(  # type: ignore[attr-defined]
                        entity, limit=1
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

            posts_saved = 0

            async for msg in self.client.iter_messages(  # type: ignore[attr-defined]
                entity, limit=self.settings.posts_limit
            ):
                if not isinstance(msg, Message):
                    continue

                post_id = await _process_message(msg, entity, self.db)
                if post_id is None:
                    continue

                posts_saved += 1

            await self.db.mark_channel_parsed(channel_id)

            logger.info(
                "Done channel: id=%s username=%s (posts saved: %s)",
                channel_id,
                channel.username,
                posts_saved,
            )

            # Random delay between channels to avoid rate limits
            await asyncio.sleep(random.uniform(5, 15))

        except FloodWaitError as e:
            delay = int(getattr(e, "seconds", 0)) or 1
            total_delay = delay + 10  # Add 10 seconds safety margin

            logger.warning(
                "Channel %s: FloodWaitError, sleeping %ss (+10s safety)",
                channel_id,
                total_delay,
            )

            await self.db.mark_channel_pending(channel_id)
            await asyncio.sleep(total_delay)

        except (ChannelPrivateError, UserDeactivatedError, AuthKeyError, SessionRevokedError) as e:
            if isinstance(e, (UserDeactivatedError, AuthKeyError, SessionRevokedError)):
                logger.critical(
                    "Worker %d: Account is DEAD/FROZEN (%s). Terminating worker to protect proxy.",
                    self.worker_id,
                    type(e).__name__,
                )
                self.is_alive = False
                await self.db.mark_channel_rejected(channel_id)
                return
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
    logger = logging.getLogger(__name__)

    logger.info(
        "Starting parser (posts_limit=%d, concurrency from session count)",
        settings.posts_limit,
    )

    # Initialize database
    db = Database(settings.db_url)
    await db.init_db()

    # Use the shared runner to start workers
    await start_workers(
        worker_class=ParserWorker,
        settings=settings,
        db=db,
    )


if __name__ == "__main__":
    asyncio.run(main())
