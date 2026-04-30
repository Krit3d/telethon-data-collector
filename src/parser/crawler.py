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
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

from python_socks import ProxyType
from sqlalchemy import select
from telethon import TelegramClient
from telethon.errors import FloodWaitError, RPCError
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

logger = logging.getLogger(__name__)

# Regex for first-person pronouns (Russian)
FIRST_PERSON_REGEX = re.compile(
    r"\b(я|мне|меня|мое|мой|моя|думаю|считаю)\b", re.IGNORECASE | re.UNICODE
)


def _build_telethon_proxy(proxy_url: str | None) -> dict[str, Any] | None:
    """Build Telethon proxy configuration from URL string."""
    if not proxy_url:
        return None

    parsed = urlparse(proxy_url)
    if not parsed.scheme or not parsed.hostname or not parsed.port:
        raise ValueError(
            "Invalid PROXY_URL. Use e.g. socks5://user:pass@ip:port, "
            "http://ip:port, or mtproxy://secret@ip:port"
        )

    scheme = parsed.scheme.lower()

    if scheme in {"socks5", "socks5h"}:
        proxy_type = ProxyType.SOCKS5
    elif scheme == "socks4":
        proxy_type = ProxyType.SOCKS4
    elif scheme in {"http", "https"}:
        proxy_type = ProxyType.HTTP
    elif scheme in {"mtproxy", "mtproto"}:
        query_secret = parse_qs(parsed.query).get("secret", [None])[0]
        mtproxy_secret = (
            parsed.username
            or parsed.password
            or parsed.path.strip("/")
            or query_secret
        )

        if not mtproxy_secret:
            raise ValueError(
                "MTProxy requires a secret. Use mtproxy://secret@ip:port "
                "or mtproxy://ip:port?secret=..."
            )

        return {
            "addr": parsed.hostname,
            "port": parsed.port,
            "secret": mtproxy_secret,
            "is_mtproxy": True,
        }
    elif scheme == "vless":
        raise ValueError(
            "Telethon does not support VLESS directly. Run a local "
            "sing-box/xray client and set PROXY_URL to its local SOCKS "
            "endpoint (e.g. socks5://127.0.0.1:2080)."
        )
    else:
        raise ValueError(
            "Unsupported proxy scheme in PROXY_URL. Supported: socks5, "
            "socks5h, socks4, http, https, mtproxy, mtproto. For VLESS use "
            "a local bridge and point PROXY_URL to socks5://127.0.0.1:2080."
        )

    return {
        "proxy_type": proxy_type,
        "addr": parsed.hostname,
        "port": parsed.port,
        "username": parsed.username,
        "password": parsed.password,
        "rdns": scheme == "socks5h",
    }


class Worker:
    """Async worker that processes channels using a single Telegram session."""

    def __init__(
        self,
        worker_id: int,
        session_path: Path,
        proxy_url: str | None,
        db: Database,
        settings: Settings,
        min_subscribers: int,
        delay_min: float,
        delay_max: float,
    ):
        self.worker_id = worker_id
        self.session_path = session_path
        self.proxy_url = proxy_url
        self.db = db
        self.settings = settings
        self.min_subscribers = min_subscribers
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.client: TelegramClient | None = None

    async def _random_delay(self) -> None:
        """Sleep for a random duration within configured range."""
        delay = random.uniform(self.delay_min, self.delay_max)
        logger.info(
            "Worker %d: sleeping for %.1f seconds", self.worker_id, delay
        )
        await asyncio.sleep(delay)

    async def _call_api(self, operation: Any) -> Any:
        """Execute a Telethon API call with random delay and error handling."""
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

            await asyncio.sleep(5)
            raise
        except RPCError as e:
            logger.error("Worker %d: RPC error: %s", self.worker_id, e)
            raise

    async def _get_channel_entity_safe(
        self, channel_id: int | str
    ) -> Channel | None:
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
        except Exception as e:
            logger.warning(
                "Worker %d: Failed to resolve channel %s: %s",
                self.worker_id,
                channel_id,
                e,
            )

        return None

    async def _get_full_channel_info(
        self, entity: Channel
    ) -> tuple[int | None, str | None]:
        """Get subscriber count and description for a channel."""
        if not self.client:
            raise RuntimeError("Telegram client is not initialized")

        client = self.client

        try:
            if entity.access_hash is None:
                return None, None

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
        self, entity: Channel, posts_to_check: int = 15
    ) -> bool:
        """Check if channel contains author-generated content (video notes or first-person text)."""
        if not self.client:
            raise RuntimeError("Telegram client is not initialized")

        client = self.client

        try:
            if entity.access_hash is None:
                return False

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

            # Check first-person pronouns
            total_matches = 0
            for msg in msgs:
                text = getattr(msg, "message", None) or ""
                total_matches += len(FIRST_PERSON_REGEX.findall(text))

            if total_matches >= 3:
                logger.info(
                    "Worker %d: Channel %s has first-person content (%d matches)",
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

        # Check for author content
        is_author = await self._check_author_content(rec_channel)

        # Save to DB
        await self._save_channel_to_db(
            channel_id=rec_channel.id,
            username=rec_channel.username,
            title=getattr(rec_channel, "title", ""),
            description=description,
            subscribers_count=subscribers_count,
            is_author_blog=is_author,
        )

        return True

    async def _get_recommendations(self, entity: Channel) -> list[Channel]:
        """Fetch channel recommendations for a given channel."""
        if not self.client:
            raise RuntimeError("Telegram client is not initialized")

        client = self.client

        try:
            if entity.access_hash is None:
                logger.warning(
                    "Worker %d: Channel %s has no access_hash, cannot get recommendations",
                    self.worker_id,
                    getattr(entity, "username", "unknown"),
                )

                return []

            input_channel = InputChannel(entity.id, entity.access_hash)
            result = await self._call_api(
                lambda: client(
                    GetChannelRecommendationsRequest(channel=input_channel)
                )
            )

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
            logger.error(
                "Worker %d: Failed to get recommendations for %s: %s",
                self.worker_id,
                getattr(entity, "username", "unknown"),
                e,
            )

            return []

    async def _claim_channel(self) -> ChannelModel | None:
        """
        Claim a random pending channel from the database for processing.

        Marks the channel as 'processing' to prevent other workers from
        claiming it simultaneously.
        """
        channel = await self.db.get_random_pending_channel()
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
                proxy_config = _build_telethon_proxy(self.proxy_url)
            except ValueError as e:
                logger.error(
                    "Worker %d: Invalid proxy URL %s: %s",
                    self.worker_id,
                    self.proxy_url,
                    e,
                )
                return

        # Create Telethon client
        client_kwargs: dict[str, Any] = {}

        if proxy_config:
            if proxy_config.pop("is_mtproxy", False):
                from telethon.network.connection.tcpmtproxy import (
                    ConnectionTcpMTProxyRandomizedIntermediate,
                )

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

        self.client = TelegramClient(
            str(self.session_path),
            self.settings.api_id,
            self.settings.api_hash,
            **client_kwargs,
        )

        try:
            await self.client.connect()

            if not await self.client.is_user_authorized():
                logger.info(
                    "Worker %d: Session not authorized, starting interactive login...",
                    self.worker_id,
                )
                await self.client.start()  # type: ignore
                logger.info("Worker %d: Login successful", self.worker_id)
            else:
                logger.info(
                    "Worker %d: Session already authorized", self.worker_id
                )

            logger.info("Worker %d: Starting processing loop", self.worker_id)

            while True:
                try:
                    # Claim a random pending channel
                    channel = await self._claim_channel()

                    if channel is None:
                        logger.info(
                            "Worker %d: No pending channels available, waiting...",
                            self.worker_id,
                        )
                        await asyncio.sleep(30)
                        continue

                    # Get channel entity
                    identifier = (
                        channel.username if channel.username else channel.id
                    )
                    entity = await self._get_channel_entity_safe(identifier)

                    if entity is None:
                        logger.warning(
                            "Worker %d: Could not resolve channel id=%s, marking as rejected",
                            self.worker_id,
                            channel.id,
                        )
                        await self._mark_rejected(channel.id)
                        continue

                    # Get recommendations
                    recommendations = await self._get_recommendations(entity)

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
                    delay = int(getattr(e, "seconds", 0)) or 1
                    logger.warning(
                        "Worker %d: FloodWaitError, sleeping %ds + 10s",
                        self.worker_id,
                        delay,
                    )
                    await asyncio.sleep(delay + 10)
                except Exception as e:
                    logger.error(
                        "Worker %d: Unexpected error in loop: %s",
                        self.worker_id,
                        e,
                        exc_info=True,
                    )
                    await asyncio.sleep(30)

        except KeyboardInterrupt:
            logger.info("Worker %d: Stopped by user", self.worker_id)
        except Exception as e:
            logger.exception("Worker %d: Fatal error: %s", self.worker_id, e)
        finally:
            if self.client:
                await self.client.disconnect()  # type: ignore
                logger.info("Worker %d: Disconnected", self.worker_id)


async def main() -> None:
    """Entry point: discover sessions, read proxies, spawn worker tasks."""

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

    # Scan sessions directory
    sessions_dir = settings.session_dir
    if not sessions_dir.exists():
        logger.error("Sessions directory %s does not exist", sessions_dir)
        return

    session_files = list(sessions_dir.glob("*.session"))
    if not session_files:
        logger.error("No .session files found in %s", sessions_dir)
        return

    logger.info(
        "Found %d session files: %s",
        len(session_files),
        [f.name for f in session_files],
    )

    # Read proxies from file
    proxies_file = Path("proxies.txt")
    proxies: list[str | None] = []

    if proxies_file.exists():
        lines = proxies_file.read_text(encoding="utf-8").strip().split("\n")
        proxies = [line.strip() for line in lines if line.strip()]
        logger.info("Loaded %d proxies from %s", len(proxies), proxies_file)
    else:
        logger.warning(
            "proxies.txt not found, all workers will run without proxy"
        )
        proxies = []

    # Map sessions to proxies (1:1, cycle proxies if fewer than sessions)
    worker_configs: list[tuple[Path, str | None]] = []

    for i, session_file in enumerate(session_files):
        proxy = proxies[i % len(proxies)] if proxies else None

        worker_configs.append((session_file, proxy))

    # Create and spawn workers
    workers: list[Worker] = []

    for i, (session_path, proxy_url) in enumerate(worker_configs):
        worker = Worker(
            worker_id=i,
            session_path=session_path,
            proxy_url=proxy_url,
            db=db,
            settings=settings,
            min_subscribers=3000,
            delay_min=settings.crawler_delay_min,
            delay_max=settings.crawler_delay_max,
        )
        workers.append(worker)

    logger.info("Spawning %d workers", len(workers))

    # Run all workers concurrently
    tasks = [asyncio.create_task(worker.run()) for worker in workers]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping workers...")

        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await db.close()
        logger.info("Database connections closed")


if __name__ == "__main__":
    asyncio.run(main())
