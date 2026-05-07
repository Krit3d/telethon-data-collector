"""Distributed Telegram channel crawler using multiple sessions.

This crawler discovers new channels based on recommendations from known channels.
It uses multiple Telegram sessions (from sessions/ directory) with individual proxies
to parallelize the discovery process. Qualifying channels are saved to PostgreSQL
with authorship detection.
"""

import asyncio
import logging
import random
import re
from pathlib import Path
from typing import Any, Literal, cast

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
    GetFullChannelRequest,
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
from src.utils.proxy import build_telethon_proxy

logger = logging.getLogger(__name__)

# Regex for first-person pronouns and characteristic words (Russian)
FIRST_PERSON_REGEX = re.compile(
    r"\b(я|мне|меня|мое|мой|моя|думаю|считаю|пишу|рассказываю|сделал|запустил|работаю|разрабатываю|заметил|поделился)\b",
    re.IGNORECASE | re.UNICODE,
)


class ChannelTaskRejected(Exception):
    """Exception raised when a channel task should be rejected due to invalid channel."""
    pass


class Worker:
    """Async worker that processes channels using a single Telegram session."""

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
        self.worker_id = worker_id
        self.session_path = session_path
        self.db = db
        self.settings = settings
        self.min_subscribers = min_subscribers
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.api_id = api_id
        self.api_hash = api_hash
        self.proxy_url = proxy_url
        self.device_model = device_model
        self.system_version = system_version
        self.app_version = app_version
        self.lang_code = lang_code
        self.system_lang_code = system_lang_code
        self.client: TelegramClient | None = None
        self.consecutive_shadowbans = 0
        self.safe_mode = False
        self.is_frozen = False

    async def _random_delay(self) -> None:
        """Sleep for a random duration within configured range."""
        delay = random.uniform(self.delay_min, self.delay_max)
        logger.info(
            "Worker %d: sleeping for %.1f seconds", self.worker_id, delay
        )
        await asyncio.sleep(delay)

    async def _call_api(self, operation: Any) -> Any:
        """Execute a Telethon API call with random delay and error handling."""
        if self.client and not self.client.is_connected():
            logger.warning(
                "Worker %d: Client disconnected. Reconnecting...",
                self.worker_id,
            )
            await self.client.connect()

        await self._random_delay()

        try:
            return await operation()
        except FloodWaitError as e:
            delay = int(getattr(e, "seconds", 0)) or 1

            logger.warning(
                "Worker %d: FloodWaitError, sleeping %ds + 10s",
                self.worker_id,
                delay,
            )

            await asyncio.sleep(delay + 10)
            raise
        except (OSError, asyncio.TimeoutError, ConnectionError) as e:
            logger.warning(
                "Worker %d: Network error (%s), retrying after backoff",
                self.worker_id,
                type(e).__name__,
            )

            await asyncio.sleep(10)
            raise
        except RPCError as e:
            logger.error("Worker %d: RPC error: %s", self.worker_id, e)
            raise

    async def _get_channel_entity_safe(
        self, channel_id: int | str
    ) -> Channel | Literal["SHADOWBANNED"] | None:
        """Safely get channel entity by ID or username."""
        if not self.client:
            raise RuntimeError("Telegram client is not initialized")

        client = self.client

        try:
            entity = await self._call_api(lambda: client.get_entity(channel_id))

            if isinstance(entity, Channel) and getattr(
                entity, "broadcast", False
            ):
                return entity

        except ValueError as e:
            if "No user has" in str(e):
                logger.warning(
                    "Worker %d: Shadowban suspected for global search. Target: %s",
                    self.worker_id,
                    channel_id,
                )
                return "SHADOWBANNED"

            logger.warning(
                "Worker %d: ValueError resolving %s: %s",
                self.worker_id,
                channel_id,
                e,
            )

        except Exception as e:
            if "disconnected" in str(e).lower() or isinstance(
                e, (ConnectionError, OSError)
            ):
                raise

            logger.warning(
                "Worker %d: Failed to resolve channel %s: %s",
                self.worker_id,
                channel_id,
                e,
            )

        return None

    async def _get_full_channel_info(
        self, entity: Channel | InputChannel
    ) -> tuple[int | None, str | None]:
        """Get subscriber count and description for a channel."""
        if not self.client:
            raise RuntimeError("Telegram client is not initialized")

        client = self.client

        try:
            # If already an InputChannel, use it directly; otherwise create one
            if isinstance(entity, InputChannel):
                input_channel = entity
            elif entity.access_hash is None:
                return None, None
            else:
                input_channel = InputChannel(entity.id, entity.access_hash)
            full = await self._call_api(
                lambda: client(GetFullChannelRequest(input_channel))
            )
            participants_count = getattr(
                getattr(full, "full_chat", None), "participants_count", None
            )
            description = getattr(
                getattr(full, "full_chat", None), "about", None
            )

            if not isinstance(participants_count, int):
                participants_count = None

            return participants_count, description
        except Exception as e:
            logger.warning(
                "Worker %d: Error getting full channel for %s: %s",
                self.worker_id,
                getattr(entity, "username", "unknown"),
                e,
            )

            return None, None

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
            msgs = await self._call_api(
                lambda: client.get_messages(input_channel, limit=posts_to_check)
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
        subscribers_count, description = await self._get_full_channel_info(
            rec_channel
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
        self, entity: Channel | InputChannel, channel_id: int | None = None
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
                result = await self._call_api(
                    lambda: client(
                        GetChannelRecommendationsRequest(channel=input_channel)
                    )
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
                    raise ChannelTaskRejected(f"Channel {channel_id} is invalid")
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
                    f"Worker {self.worker_id}: ACCOUNT FROZEN by Telegram. Disconnecting."
                )
                self.is_frozen = True
                if self.client and self.client.is_connected():
                    await self.client.disconnect()  # type: ignore
                return []
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
                if getattr(self, "is_frozen", False):
                    return
                channel = None

                try:
                    # Claim a random pending channel
                    channel = await self._claim_channel(
                        require_hash=self.safe_mode
                    )

                    if channel is None:
                        logger.info(
                            "Worker %d: No pending channels available, waiting...",
                            self.worker_id,
                        )
                        await asyncio.sleep(30)
                        continue

                    # Get channel entity - use InputChannel if we have access_hash
                    entity = None
                    if channel.access_hash is not None:
                        # Ideal scenario: we have hash, create InputChannel without network calls!
                        from telethon.tl.types import InputChannel

                        entity = InputChannel(channel.id, channel.access_hash)
                    else:
                        # Fallback: if no hash (e.g., seed channels), resolve by username or ID
                        identifier = (
                            channel.username if channel.username else channel.id
                        )
                        entity = await self._get_channel_entity_safe(identifier)

                    if entity == "SHADOWBANNED":
                        # Return channel back
                        await self.db.mark_channel_pending(channel.id)
                        self.consecutive_shadowbans += 1

                        # Enable safe mode to only process channels with access_hash
                        if not self.safe_mode:
                            self.safe_mode = True
                            logger.warning(
                                "Worker %d switched to SAFE MODE. Will only process channels with access_hash.",
                                self.worker_id,
                            )

                        if self.consecutive_shadowbans >= 3:
                            logger.error(
                                "Worker %d is SHADOWBANNED. Suspending for 3 hours to protect DB.",
                                self.worker_id,
                            )
                            await asyncio.sleep(10800)  # Sleeping for 3 hours
                            self.consecutive_shadowbans = (
                                0  # Try again after sleep
                            )
                        else:
                            await asyncio.sleep(
                                60
                            )  # Small pause before next try
                        continue

                    # Reset counter for any successful resolving (even if entity is None, but not banned)
                    self.consecutive_shadowbans = 0

                    # Safety check: if safe_mode is on but channel has no access_hash, skip it
                    if self.safe_mode and channel.access_hash is None:
                        logger.warning(
                            "Worker %d: Channel id=%s has no access_hash in safe mode, returning to pending",
                            self.worker_id,
                            channel.id,
                        )
                        await self.db.mark_channel_pending(channel.id)
                        continue

                    if entity is None:
                        logger.warning(
                            "Worker %d: Could not resolve channel id=%s, marking as rejected",
                            self.worker_id,
                            channel.id,
                        )
                        await self._mark_rejected(channel.id)
                        continue

                    # Get recommendations
                    try:
                        recommendations = await self._get_recommendations(entity, channel.id)
                    except ChannelTaskRejected as e:
                        # Mark channel as rejected and skip to next
                        await self._mark_rejected(channel.id)
                        logger.info(
                            "Worker %d: Channel id=%s rejected, skipping",
                            self.worker_id,
                            channel.id,
                        )
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

    global settings
    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info(
        "Starting distributed crawler (delay=%d-%ds, min_subscribers=%d)",
        settings.crawler_delay_min,
        settings.crawler_delay_max,
        3000,
    )

    # Initialize database
    db = Database(settings.db_url)
    await db.init_db()
    await db.reset_orphaned_processing_channels()

    # Scan sessions directory
    sessions_dir = settings.session_dir
    if not sessions_dir.exists():
        logger.error("Sessions directory %s does not exist", sessions_dir)
        return

    session_files = sorted(sessions_dir.glob("*.session"))
    if not session_files:
        logger.error("No .session files found in %s", sessions_dir)
        return

    logger.info(
        "Found %d session files: %s",
        len(session_files),
        [f.name for f in session_files],
    )

    # Create and spawn workers
    workers: list[Worker] = []

    for i, session_path in enumerate(session_files):
        # Look for accompanying .json config file
        json_path = session_path.with_suffix(".json")

        api_id = settings.api_id
        api_hash = settings.api_hash
        proxy_url = settings.proxy_url
        device_model = "PC 64bit"
        system_version = "Windows 10"
        app_version = "4.16.8"
        lang_code = "en"
        system_lang_code = "en-US"

        if json_path.exists():
            try:
                import json

                with json_path.open(encoding="utf-8") as f:
                    config = json.load(f)

                api_id = (
                    config.get("api_id")
                    or config.get("app_id")
                    or settings.api_id
                )
                api_hash = (
                    config.get("api_hash")
                    or config.get("app_hash")
                    or settings.api_hash
                )
                proxy_url = (
                    config.get("proxy_url")
                    if "proxy_url" in config
                    else settings.proxy_url
                )
                device_model = (
                    config.get("device_model")
                    or config.get("device")
                    or device_model
                )
                system_version = (
                    config.get("system_version")
                    or config.get("sdk")
                    or system_version
                )
                app_version = config.get("app_version", app_version)
                lang_code = config.get("lang_code", lang_code)
                system_lang_code = config.get(
                    "system_lang_code", system_lang_code
                )

                try:
                    api_id = int(api_id)
                except (ValueError, TypeError):
                    api_id = settings.api_id
                if not isinstance(api_hash, str):
                    api_hash = settings.api_hash

                logger.info(
                    "Loaded config for %s from %s (api_id=%s, proxy=%s, device=%s)",
                    session_path.name,
                    json_path.name,
                    api_id if api_id else "default",
                    "yes" if proxy_url else "no",
                    device_model,
                )
            except Exception as e:
                logger.warning(
                    "Failed to read %s: %s, using global settings",
                    json_path.name,
                    e,
                )
        else:
            logger.debug(
                "No config file for %s, using global settings",
                session_path.name,
            )

        worker = Worker(
            worker_id=i,
            session_path=session_path,
            db=db,
            settings=settings,
            min_subscribers=3000,
            delay_min=settings.crawler_delay_min,
            delay_max=settings.crawler_delay_max,
            api_id=api_id,
            api_hash=api_hash,
            proxy_url=proxy_url,
            device_model=device_model,
            system_version=system_version,
            app_version=app_version,
            lang_code=lang_code,
            system_lang_code=system_lang_code,
        )
        workers.append(worker)

    logger.info("Spawning %d workers", len(workers))

    # Run all workers concurrently
    tasks = [asyncio.create_task(worker.run()) for worker in workers]

    if not tasks:
        logger.error("No workers to run. Check your session directory.")
        await db.close()
        logger.info("Database connections closed")
        return

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping workers...")

        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error("Global error in gather: %s", e, exc_info=True)
    finally:
        await db.close()
        logger.info("Database connections closed")


if __name__ == "__main__":
    asyncio.run(main())
