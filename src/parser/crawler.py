"""Distributed Telegram channel crawler using multiple sessions.

This crawler discovers new channels based on recommendations from known channels.
It uses multiple Telegram sessions (from sessions/ directory) with individual proxies
to parallelize the discovery process. Qualifying channels are saved to PostgreSQL
with authorship detection.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, cast

from sqlalchemy import select
from telethon import TelegramClient
from telethon.errors import (
    AuthKeyError,
    ChannelInvalidError,
    FloodWaitError,
    RPCError,
    SessionRevokedError,
    UserDeactivatedError,
)
from telethon.network.connection.tcpintermediate import (
    ConnectionTcpIntermediate,
)
from telethon.network.connection.tcpmtproxy import (
    ConnectionTcpMTProxyRandomizedIntermediate,
)
from telethon.tl.functions.channels import (
    GetChannelRecommendationsRequest,
)
from telethon.tl.types import (
    Channel,
    InputChannel,
    Message,
    messages as MessagesTypes,
)

from src.config.config import Settings, load_settings
from src.db.database import Database
from src.db.models import Channel as ChannelModel
from src.parser.core.utils import (
    get_full_channel_info,
)
from src.parser.core.worker_base import BaseTelegramWorker
from src.parser.core.runner import start_workers
from src.utils.proxy import build_telethon_proxy
from src.parser.core.exceptions import (
    ChannelTaskRejected,
    SessionExpiredError,
    WorkerError,
)

logger = logging.getLogger(__name__)

# Regex for first-person pronouns and characteristic words (Russian)
FIRST_PERSON_REGEX = re.compile(
    r"\b(я|мне|меня|мое|мой|моя|думаю|считаю|пишу|рассказываю|сделал|запустил|работаю|разрабатываю|заметил|поделился)\b",
    re.IGNORECASE | re.UNICODE,
)


class Worker(BaseTelegramWorker):
    """Distributed Telegram channel crawler worker.

    Discovers new channels based on recommendations from known channels.
    Processes channels in parallel using multiple sessions with individual proxies.
    Qualifying channels are saved to database with authorship detection.
    """

    def __init__(
        self,
        worker_id: int,
        session_path: Path,
        db: Database,
        settings: Settings,
        min_subscribers: int,
        delay_min: float,
        delay_max: float,
        api_id: int,
        api_hash: str,
        proxy_url: str | None = None,
        device_model: str = "PC 64bit",
        system_version: str = "Windows 10",
        app_version: str = "4.16.8",
        lang_code: str = "en",
        system_lang_code: str = "en-US",
    ):
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
        self.min_subscribers = min_subscribers
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.processed_count = 0
        self.DAILY_LIMIT = 850

    async def _check_author_content(
        self,
        entity: Channel | InputChannel,
        description: str | None = None,
        posts_to_check: int = 15,
    ) -> bool:
        """Check if channel contains author-generated content (video notes or first-person text)."""
        if not self.client:
            raise RuntimeError("Telegram client is not initialized")

        # Check description for first-person markers
        if description:
            desc_matches = len(FIRST_PERSON_REGEX.findall(description))
            if desc_matches >= 1:
                logger.info(
                    "Worker %d: Channel %s has first-person content in description (%d matches)",
                    self.worker_id,
                    getattr(entity, "username", "unknown"),
                    desc_matches,
                )
                return True

        client = self.client

        try:
            # If already an InputChannel, use it directly; otherwise create one
            if isinstance(entity, InputChannel):
                input_channel = entity
            elif entity.access_hash is None:
                return False
            else:
                input_channel = InputChannel(entity.id, entity.access_hash)
            msgs = await self.safe_api_call(
                f"get_messages(limit={posts_to_check})",
                lambda: client.get_messages(
                    input_channel, limit=posts_to_check
                ),
            )

            if msgs is None:
                return False

            # Ensure messages is always a list
            if isinstance(msgs, Message):
                msgs = [msgs]

            # Check for video notes
            for msg in msgs:
                if getattr(msg, "video_note", None):
                    logger.info(
                        "Worker %d: Channel %s has video note content",
                        self.worker_id,
                        getattr(entity, "username", "unknown"),
                    )

                    return True

            # Check first-person pronouns in posts
            total_matches = 0
            for msg in msgs:
                text = getattr(msg, "message", None) or ""
                total_matches += len(FIRST_PERSON_REGEX.findall(text))

            if total_matches >= 2:
                logger.info(
                    "Worker %d: Channel %s has first-person content (%d matches in posts)",
                    self.worker_id,
                    getattr(entity, "username", "unknown"),
                    total_matches,
                )

                return True

            return False
        except Exception as e:
            # Re-raise critical errors
            if isinstance(
                e, (WorkerError, asyncio.CancelledError, KeyboardInterrupt)
            ):
                raise
            logger.warning(
                "Worker %d: Error checking author content for %s: %s",
                self.worker_id,
                getattr(entity, "username", "unknown"),
                e,
            )
            return False

    async def _save_channel_to_db(
        self,
        channel_id: int,
        username: str | None,
        title: str,
        description: str | None,
        subscribers_count: int | None,
        is_author_blog: bool,
        access_hash: int | None = None,
    ) -> None:
        """Save or update channel in database."""
        channel_data = {
            "id": channel_id,
            "username": username,
            "title": title,
            "description": description,
            "subscribers_count": subscribers_count,
            "avatar_url": None,  # Avatar fetching not needed for recommendations
            "status": "pending",
            "is_author_blog": is_author_blog,
            "access_hash": access_hash,
        }

        await self.db.upsert_channel(channel_data)
        logger.info(
            "Worker %d: Saved channel %s (id=%s, author=%s, subs=%s)",
            self.worker_id,
            username or channel_id,
            channel_id,
            is_author_blog,
            subscribers_count or "hidden",
        )

    async def _process_recommendation(self, rec_channel: Channel) -> bool:
        """
        Process a recommended channel: check filters and save to DB.

        Returns True if channel was saved, False otherwise.
        """
        channel_name = rec_channel.username or str(rec_channel.id)

        # Check if channel already exists in DB
        async with self.db.async_session() as session:
            stmt = select(ChannelModel).where(ChannelModel.id == rec_channel.id)
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing is not None:
                logger.debug(
                    "Worker %d: Channel %s already in DB, skipping",
                    self.worker_id,
                    channel_name,
                )

                return False

        # Get full channel info for subscriber count
        subscribers_count, description = await get_full_channel_info(
            self.client,  # type: ignore[arg-type]
            rec_channel,
            safe_api_call=self.safe_api_call,
        )

        if subscribers_count is None:
            logger.info(
                "Worker %d: Channel %s has no subscriber count, skipping",
                self.worker_id,
                channel_name,
            )

            return False

        if subscribers_count < self.min_subscribers:
            logger.info(
                "Worker %d: Channel %s has %d subscribers (<%d), skipping",
                self.worker_id,
                channel_name,
                subscribers_count,
                self.min_subscribers,
            )

            return False

        # Check for author content (pass description for early detection)
        is_author = await self._check_author_content(
            rec_channel, description=description
        )

        # Save to DB with access_hash
        await self._save_channel_to_db(
            channel_id=rec_channel.id,
            username=rec_channel.username,
            title=getattr(rec_channel, "title", ""),
            description=description,
            subscribers_count=subscribers_count,
            is_author_blog=is_author,
            access_hash=getattr(rec_channel, "access_hash", None),
        )

        return True

    async def _get_recommendations(
        self, entity: Channel | InputChannel, channel_id: int | None = None, operation_name: str = "get_recommendations"
    ) -> list[Channel]:
        """Fetch channel recommendations for a given channel."""
        if not self.client:
            raise RuntimeError("Telegram client is not initialized")

        client = self.client

        try:
            # If already an InputChannel, use it directly; otherwise check for access_hash
            if isinstance(entity, InputChannel):
                input_channel = entity
            elif entity.access_hash is None:
                logger.warning(
                    "Worker %d: Channel %s has no access_hash, cannot get recommendations",
                    self.worker_id,
                    getattr(entity, "username", "unknown"),
                )
                return []
            else:
                input_channel = InputChannel(entity.id, entity.access_hash)

            # Wrap the API call specifically for Telethon errors
            try:
                result = await self.safe_api_call(
                    operation_name,
                    lambda: client(
                        GetChannelRecommendationsRequest(channel=input_channel)
                    ),
                )
            except ChannelInvalidError:
                logger.warning(
                    "Worker %d: Channel id=%s is invalid (ChannelInvalidError), rejecting task",
                    self.worker_id,
                    channel_id,
                )
                raise ChannelTaskRejected(f"Channel {channel_id} is invalid")
            except RPCError as e:
                if "Invalid channel" in str(e):
                    logger.warning(
                        "Worker %d: Channel id=%s is invalid (RPCError: Invalid channel), rejecting task",
                        self.worker_id,
                        channel_id,
                    )
                    raise ChannelTaskRejected(
                        f"Channel {channel_id} is invalid"
                    )
                # Re-raise other RPC errors
                raise

            result = cast(MessagesTypes.Chats, result)

            recommended_channels: list[Channel] = []

            if result.chats:
                for chat in result.chats:
                    if (
                        isinstance(chat, Channel)
                        and getattr(chat, "broadcast", False)
                        and getattr(chat, "username", None)
                    ):
                        recommended_channels.append(chat)

            logger.info(
                "Worker %d: Got %d recommendations for %s",
                self.worker_id,
                len(recommended_channels),
                getattr(entity, "username", "unknown"),
            )

            return recommended_channels

        except Exception as e:
            error_text = str(e)
            if "method that is not available for frozen accounts" in error_text:
                logger.critical(
                    f"Worker {self.worker_id}: ACCOUNT FROZEN by Telegram. Raising SessionExpiredError."
                )
                raise SessionExpiredError("Account frozen by Telegram")
            logger.error(
                "Worker %d: Failed to get recommendations for %s: %s",
                self.worker_id,
                getattr(entity, "username", "unknown") or channel_id,
                e,
            )
            raise

    async def _claim_channel(
        self, require_hash: bool = False
    ) -> ChannelModel | None:
        """
        Claim a random pending channel from the database for processing.

        Marks the channel as 'processing' to prevent other workers from
        claiming it simultaneously.

        Args:
            require_hash: If True, only claim channels with non-null access_hash.
        """
        channel = await self.db.get_random_pending_channel(
            require_hash=require_hash
        )
        if channel is None:
            return None

        logger.info(
            "Worker %d: Claimed channel id=%s username=%s for processing",
            self.worker_id,
            channel.id,
            channel.username,
        )

        return channel

    async def _mark_processed(self, channel_id: int) -> None:
        """Mark a channel as successfully processed."""
        await self.db.mark_channel_processed(channel_id)

    async def _mark_rejected(self, channel_id: int) -> None:
        """Mark a channel as rejected (could not be processed)."""
        await self.db.mark_channel_rejected(channel_id)

    async def run(self) -> None:
        """Main worker loop: claim channels, get recommendations, process them."""
        # Build proxy configuration
        proxy_config = None

        if self.proxy_url:
            try:
                proxy_config = build_telethon_proxy(self.proxy_url)
            except ValueError as e:
                logger.error(
                    "Worker %d: Invalid proxy URL %s: %s",
                    self.worker_id,
                    self.proxy_url,
                    e,
                )
                return

        # Create Telethon client with device parameters from session JSON
        client_kwargs: dict[str, Any] = {
            "device_model": self.device_model,
            "system_version": self.system_version,
            "app_version": self.app_version,
            "lang_code": self.lang_code,
            "system_lang_code": self.system_lang_code,
            "use_ipv6": False,
            "timeout": 60,
            "connection": ConnectionTcpIntermediate,
            "connection_retries": 10,
            "retry_delay": 5,
        }

        if proxy_config:
            if proxy_config.pop("is_mtproxy", False):
                client_kwargs["connection"] = (
                    ConnectionTcpMTProxyRandomizedIntermediate
                )
                client_kwargs["proxy"] = (
                    proxy_config["addr"],
                    proxy_config["port"],
                    proxy_config["secret"],
                )
            else:
                client_kwargs["proxy"] = proxy_config

        # Log absolute session path for debugging
        session_abs_path = self.session_path.with_suffix("").absolute()
        logger.info(
            "Worker %d: Attempting to load session from: %s",
            self.worker_id,
            session_abs_path,
        )

        self.client = TelegramClient(
            str(session_abs_path),
            self.api_id,
            self.api_hash,
            **client_kwargs,
        )

        try:
            # Log MTProxy connection if applicable
            if proxy_config and proxy_config.get("is_mtproxy"):
                logger.info(
                    "Worker %d: Connecting to MTProxy %s:%d",
                    self.worker_id,
                    proxy_config["addr"],
                    proxy_config["port"],
                )

            await self.client.connect()

            if not await self.client.is_user_authorized():
                logger.critical(
                    "Worker %d: SESSION UNAUTHORIZED. Path: %s. "
                    "Interactive login is impossible in Docker. Check your session files!",
                    self.worker_id,
                    self.session_path,
                )
                return

            logger.info("Worker %d: Starting processing loop", self.worker_id)

            while True:
                if not self.is_alive:
                    logger.info(
                        "Worker %d: is_alive=False, terminating worker loop",
                        self.worker_id,
                    )
                    return
                if getattr(self, "is_frozen", False):
                    return
                if self.processed_count >= self.DAILY_LIMIT:
                    logger.info(
                        "Worker %d reached daily limit of %d channels. Stopping worker to protect session.",
                        self.worker_id,
                        self.DAILY_LIMIT,
                    )
                    return
                channel = None

                try:
                    # Claim a random pending channel
                    channel = await self._claim_channel(
                        require_hash=self.safe_mode
                    )

                    if channel is None:
                        if self.safe_mode:
                            logger.info(
                                "Worker %d: Safe mode cooldown - sleeping for 300 seconds",
                                self.worker_id,
                            )
                            await asyncio.sleep(300)
                            self.safe_mode = False
                        else:
                            logger.info(
                                "Worker %d: No pending channels available, waiting...",
                                self.worker_id,
                            )
                            await asyncio.sleep(30)
                        continue

                    # Resolve entity with access_hash-first priority (more reliable)
                    entity = None
                    resolved_entity_type = None  # 'hash' or 'username'

                    # Step 1: Try to resolve by access_hash first (faster, no rate limits)
                    if channel.access_hash is not None:
                        try:
                            input_channel = InputChannel(channel.id, channel.access_hash)
                            # Test the access_hash by calling _get_recommendations directly
                            recommendations = await self._get_recommendations(
                                input_channel, channel.id, operation_name="resolve_by_hash"
                            )
                            # Success - use the InputChannel as entity
                            entity = input_channel
                            resolved_entity_type = 'hash'
                        except (RPCError, ValueError, ChannelTaskRejected) as e:
                            # The stored access_hash is invalid/stale or other RPC error
                            logger.warning(
                                "Worker %d: Access hash failed for channel %s, switching to PLAN B (resolve by username)",
                                self.worker_id,
                                channel.username or channel.id,
                            )
                            entity = None

                    # Step 2: Fallback to username resolution if hash failed or not available
                    if entity is None and channel.username:
                        try:
                            # Resolve by username using safe_api_call directly
                            entity = await self.safe_api_call(
                                "resolve_by_username",
                                lambda: self.client.get_entity(channel.username)  # type: ignore[arg-type]
                            )
                            # Check if entity is a broadcast channel
                            if isinstance(entity, Channel) and getattr(entity, "broadcast", False):
                                resolved_entity_type = 'username'
                                # Fetch recommendations separately after successful username resolution
                                recommendations = await self._get_recommendations(
                                    entity, channel.id, operation_name="get_recommendations"
                                )
                            else:
                                # Not a broadcast channel or invalid
                                logger.error(
                                    "Worker %d: PLAN B failed. Channel %s is unreachable.",
                                    self.worker_id,
                                    channel.username or channel.id,
                                )
                                await self._mark_processed(channel.id)
                                continue
                        except (RPCError, ValueError, ChannelTaskRejected) as e:
                            # Username resolution also failed
                            logger.error(
                                "Worker %d: PLAN B failed. Channel %s is unreachable.",
                                self.worker_id,
                                channel.username or channel.id,
                            )
                            await self._mark_processed(channel.id)
                            continue

                    if entity is None:
                        # Could not resolve channel by any method (no access_hash and no username)
                        await self._mark_processed(channel.id)
                        continue

                    # Process each recommendation
                    saved_count = 0

                    for rec_channel in recommendations:
                        try:
                            saved = await self._process_recommendation(
                                rec_channel
                            )
                            if saved:
                                saved_count += 1
                        except Exception as e:
                            logger.error(
                                "Worker %d: Error processing recommendation %s: %s",
                                self.worker_id,
                                getattr(rec_channel, "username", "unknown"),
                                e,
                            )

                    logger.info(
                        "Worker %d: Processed channel id=%s, saved %d new channels",
                        self.worker_id,
                        channel.id,
                        saved_count,
                    )

                    # Mark original channel as processed
                    await self._mark_processed(channel.id)
                    self.processed_count += 1

                    # IMPORTANT: Update the channel's access_hash in the database
                    # with the session-local correct hash from the resolved entity.
                    # This ensures future workers use the correct hash for this session.
                    if hasattr(entity, 'access_hash') and entity.access_hash is not None:
                        try:
                            await self.db.update_channel_access_hash(
                                channel.id,
                                entity.access_hash
                            )
                            logger.debug(
                                "Worker %d: Updated access_hash for channel %s in DB",
                                self.worker_id,
                                channel.id,
                            )
                        except Exception as e:
                            logger.warning(
                                "Worker %d: Failed to update access_hash for %s: %s",
                                self.worker_id,
                                channel.id,
                                e,
                            )

                except FloodWaitError as e:
                    if channel:
                        await self.db.mark_channel_pending(channel.id)

                    delay = int(getattr(e, "seconds", 0)) or 1
                    logger.warning(
                        "Worker %d: FloodWaitError, sleeping %ds",
                        self.worker_id,
                        delay + 10,
                    )
                    await asyncio.sleep(delay + 10)

                except Exception as e:
                    if channel:
                        await self.db.mark_channel_pending(channel.id)

                    logger.error(
                        "Worker %d: Error in loop (reverted channel to pending): %s",
                        self.worker_id,
                        type(e).__name__,
                    )
                    await asyncio.sleep(15)

        except (AuthKeyError, UserDeactivatedError, SessionRevokedError) as e:
            logger.critical(
                f"Worker {self.worker_id}: SESSION DEAD. Error: {type(e).__name__}. File: {self.session_path}"
            )
            self.is_alive = False
            return

        except KeyboardInterrupt:
            logger.info("Worker %d: Stopped by user", self.worker_id)
        except Exception as e:
            logger.exception("Worker %d: Fatal error: %s", self.worker_id, e)
        finally:
            if self.client:
                await self.client.disconnect()  # type: ignore
                logger.info("Worker %d: Disconnected", self.worker_id)


async def main() -> None:
    """Entry point: discover sessions, read individual configs, spawn worker tasks."""
    settings = load_settings()
    logger = logging.getLogger(__name__)

    logger.info("Starting distributed crawler via core runner...")

    db = Database(settings.db_url)
    await db.init_db()
    await db.reset_orphaned_processing_channels()

    worker_args = {
        "min_subscribers": 3000,
        "delay_min": settings.crawler_delay_min,
        "delay_max": settings.crawler_delay_max,
    }

    try:
        await start_workers(Worker, settings, db, worker_args=worker_args)
    finally:
        await db.close()
        logger.info("Database connections closed.")


if __name__ == "__main__":
    asyncio.run(main())
