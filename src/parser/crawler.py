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
    get_channel_entity_safe,
    get_full_channel_info,
)
from src.parser.core.worker_base import BaseTelegramWorker
from src.parser.core.runner import start_workers
from src.parser.core.exceptions import (
    ChannelTaskRejected,
    SessionExpiredError,
    WorkerError,
)

logger = logging.getLogger(__name__)

# Fatal session exceptions that should bubble up to the runner without being caught locally
FATAL_SESSION_EXCEPTIONS = (
    AuthKeyError,
    UserDeactivatedError,
    SessionRevokedError,
    SessionExpiredError,
)

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
        # Per-worker entity cache to avoid cross-account access_hash poisoning
        self._entity_cache: dict[tuple[int, int | None], Any] = {}

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
            "status": "pending",
            "is_author_blog": is_author_blog,
            "access_hash": access_hash,
        }

        await self.db.upsert_channel(channel_data)
        logger.info(
            "Saved channel @%s (id=%s, author=%s, subs=%s)",
            username or channel_id,
            channel_id,
            is_author_blog,
            subscribers_count or "hidden",
        )

    async def _process_recommendation(
        self, rec_channel: Channel, client: TelegramClient | None = None
    ) -> bool:
        """
        Process a recommended channel: check filters and save to DB.

        Args:
            rec_channel: The recommended channel to process.
            client: Optional TelegramClient to use. If None, uses self.client.

        Returns True if channel was saved, False otherwise.
        """
        channel_name = rec_channel.username or str(rec_channel.id)
        active_client = client or self.client

        if not active_client:
            raise RuntimeError("Telegram client is not initialized")

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
            active_client,
            rec_channel,
            safe_api_call=self.safe_api_call,
        )

        if subscribers_count is None:
            logger.debug(
                "Worker %d: Channel %s has no subscriber count, skipping",
                self.worker_id,
                channel_name,
            )

            return False

        if subscribers_count < self.min_subscribers:
            logger.debug(
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
        self,
        entity: Channel | InputChannel,
        channel_id: int | None = None,
        operation_name: str = "get_recommendations",
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
                logger.debug(
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
                logger.info(
                    "Worker %d: Channel id=%s is invalid, rejecting task",
                    self.worker_id,
                    channel_id,
                )
                raise ChannelTaskRejected(f"Channel {channel_id} is invalid")
            except RPCError as e:
                if "Invalid channel" in str(e):
                    logger.info(
                        "Worker %d: Channel id=%s is invalid, rejecting task",
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
                    "Worker %d: ACCOUNT FROZEN by Telegram",
                    self.worker_id,
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
        # Connect using base class logic (handles proxy, client creation, auth check)
        await self.connect()

        # Ensure client is initialized after connect()
        if self.client is None:
            raise RuntimeError(
                "Telegram client is not initialized after connect()"
            )

        client = self.client  # Local reference for type safety

        try:
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

                    # Resolve entity using session-aware safe lookup
                    entity = None
                    recommendations: list[Channel] = []

                    # Step 1: Resolve entity using session-aware safe lookup
                    result = await get_channel_entity_safe(
                        client,
                        channel.id,
                        safe_api_call=self.safe_api_call,
                        entity_cache=self._entity_cache,
                    )

                    # Fallback to username only if ID resolution failed and username exists
                    if not result.is_success() and channel.username:
                        result = await get_channel_entity_safe(
                            client,
                            channel.username,
                            safe_api_call=self.safe_api_call,
                            entity_cache=self._entity_cache,
                        )

                    entity = result.entity

                    # If entity is resolved successfully, fetch the recommendations
                    if entity is not None:
                        recommendations = await self._get_recommendations(
                            entity,
                            channel.id,
                            operation_name="get_recommendations",
                        )
                    else:
                        # Could not resolve channel by any method
                        logger.info(
                            "Worker %d: Channel %s could not be resolved, marking as processed",
                            self.worker_id,
                            channel.username or channel.id,
                        )
                        await self._mark_processed(channel.id)
                        continue

                    # Process each recommendation
                    saved_count = 0

                    for rec_channel in recommendations:
                        try:
                            saved = await self._process_recommendation(
                                rec_channel, client=client
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

                except FloodWaitError as e:
                    # Revert channel to pending and re-raise for session pool runner to handle
                    if channel:
                        await self.db.mark_channel_pending(channel.id)

                    logger.warning(
                        "Worker %d: FloodWaitError, re-raising for pool runner",
                        self.worker_id,
                    )
                    raise  # Re-raise to let session pool runner handle cooldown

                except FATAL_SESSION_EXCEPTIONS:
                    # Fatal session exceptions: do NOT catch - let them bubble up naturally
                    # The runner will mark the session as permanently banned
                    raise

                except Exception as e:
                    # Revert channel to pending for transient/non-fatal errors
                    if channel:
                        await self.db.mark_channel_pending(channel.id)

                    logger.error(
                        "Worker %d: Error in loop (reverted channel to pending): %s",
                        self.worker_id,
                        type(e).__name__,
                    )
                    raise  # Re-raise to let runner handle them

        finally:
            # Guaranteed disconnect on exit
            await self.disconnect()


async def main() -> None:
    """Entry point: discover sessions, read individual configs, spawn worker tasks."""
    settings = load_settings()

    # Install global asyncio exception handler to prevent silent crashes
    loop = asyncio.get_running_loop()

    def global_exception_handler(loop, context):
        msg = context.get("exception", context["message"])
        logging.getLogger("asyncio_global").critical(
            "Unhandled asyncio exception: %s", msg
        )

    loop.set_exception_handler(global_exception_handler)

    logger.info("Starting distributed crawler via core runner...")

    # Database is assumed to be initialized by the migration script
    db = Database(settings.db_url)
    await db.reset_orphaned_processing_channels()

    worker_args = {
        "min_subscribers": 3000,
        "delay_min": settings.crawler_delay_min,
        "delay_max": settings.crawler_delay_max,
    }

    try:
        await start_workers(
            Worker,
            settings,
            db,
            worker_args=worker_args,
            ignore_concurrency_limit=True,
        )
    finally:
        await db.close()
        logger.info("Database connections closed.")


if __name__ == "__main__":
    asyncio.run(main())
