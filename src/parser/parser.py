"""Telegram channel parser for scraping and storing channel data.

This module provides functionality to parse Telegram channels, extract
channel metadata and posts, and store them in a database.
"""

import asyncio
import logging
from datetime import timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from telethon import TelegramClient
from telethon.errors import RPCError
from telethon.errors import AuthKeyError, ChannelPrivateError, SessionRevokedError, UserDeactivatedError
from telethon.errors.rpcerrorlist import FloodWaitError
from telethon.network.connection.tcpintermediate import (
    ConnectionTcpIntermediate,
)
from telethon.network.connection.tcpmtproxy import (
    ConnectionTcpMTProxyRandomizedIntermediate,
)
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.types import Channel as TlChannel
from telethon.tl.types import InputChannel
from telethon.tl.types import InputPeerChannel
from telethon.tl.types import Message

from src.db.database import Database
from src.db.models import Channel
from src.config.config import Settings, load_settings
from src.utils.proxy import build_telethon_proxy

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def _with_telethon_retries(
    op_name: str,
    fn: Callable[[], "asyncio.Future[T]"] | Callable[[], Any],
    *,
    network_retries: int,
    base_delay_s: float,
) -> T:
    """Execute a Telethon operation with retry and FloodWait handling.

    Args:
        op_name: Human-readable operation name for log messages.
        fn: Sync or async callable that performs the target operation.
        network_retries: Maximum retry attempts for transient network errors.
        base_delay_s: Base delay in seconds for exponential backoff.

    Returns:
        The result produced by the callable `fn`.

    Raises:
        RPCError: Re-raised for non-network Telethon RPC failures.
        OSError: Re-raised when transient network retries are exhausted.
        asyncio.TimeoutError: Re-raised when retries are exhausted.
        ConnectionError: Re-raised when retries are exhausted.
    """

    attempt = 0

    while True:
        try:
            result = fn()

            if asyncio.iscoroutine(result):
                return await result  # type: ignore[return-value]

            return result  # type: ignore[return-value]

        except FloodWaitError as e:
            delay = int(getattr(e, "seconds", 0)) or 1
            logger.warning("%s: FloodWaitError, sleeping %ss", op_name, delay)
            await asyncio.sleep(delay)

        except (OSError, asyncio.TimeoutError, ConnectionError) as e:
            if attempt >= network_retries:
                logger.exception(
                    "%s: network error, retries exhausted", op_name
                )
                raise

            delay = base_delay_s * (2**attempt)
            attempt += 1

            logger.warning(
                "%s: network error (%s), retry %s/%s in %.1fs",
                op_name,
                type(e).__name__,
                attempt,
                network_retries,
                delay,
            )

            await asyncio.sleep(delay)
            await asyncio.sleep(10)

        except RPCError:
            # Non-network Telethon errors: log and bubble up (channel-specific handler will catch)
            logger.exception("%s: Telethon RPCError", op_name)
            raise


def format_tg_id(channel_id: int) -> int:
    """Ensures that the channel ID is prefixed with -100 if this is not the case."""
    s_id = str(channel_id)

    if not s_id.startswith("-100"):
        clean_id = s_id.lstrip("-")

        return int(f"-100{clean_id}")

    return channel_id


def _normalize_username(username: str | None) -> str | None:
    """Normalize a Telegram username by removing a leading at-sign.

    Args:
        username: Raw username string, potentially with a leading `@`.

    Returns:
        Username without a leading `@`, or `None` when no username is provided.
    """

    if not username:
        return None

    return username[1:] if username.startswith("@") else username


def _message_reactions_count(message: Message) -> int | None:
    """Calculate total reactions count for a message.

    Args:
        message: Telethon message object to inspect.

    Returns:
        Sum of all reaction counters if available, otherwise `None`.
    """

    reactions = getattr(message, "reactions", None)

    if not reactions or not getattr(reactions, "results", None):
        return None

    total = 0

    for r in reactions.results:
        c = getattr(r, "count", None)

        if isinstance(c, int):
            total += c

    return total


def _message_comments_count(message: Message) -> int | None:
    """Extract comments count from a message replies metadata.

    Args:
        message: Telethon message object to inspect.

    Returns:
        Number of replies/comments when present, otherwise `None`.
    """

    replies = getattr(message, "replies", None)
    count = getattr(replies, "replies", None) if replies else None

    return count if isinstance(count, int) else None


async def _fetch_avatar_path(
    client: TelegramClient,
    entity: TlChannel | InputPeerChannel,
    avatars_dir: Path,
    *,
    network_retries: int,
    base_delay_s: float,
) -> str | None:
    """Download and return a channel avatar path.

    Args:
        client: Initialized Telethon client instance.
        entity: Telegram channel entity used as avatar source.
        avatars_dir: Directory where avatar files are stored.
        network_retries: Maximum retry attempts for transient network failures.
        base_delay_s: Base delay in seconds for exponential retry backoff.

    Returns:
        Absolute or relative file path returned by Telethon if avatar download
        succeeds, otherwise `None`.
    """

    entity_id = getattr(entity, "id", None) or getattr(
        entity, "channel_id", "unknown"
    )

    if not getattr(entity, "photo", None):
        logger.info("Channel %s has no profile photo, skipping", entity_id)
        return None

    avatars_dir.mkdir(parents=True, exist_ok=True)
    target_file = avatars_dir / f"{entity_id}.jpg"

    async def _dl() -> str | None:
        result = await client.download_profile_photo(
            entity, file=str(target_file)
        )

        if result is None:
            return None

        return str(result)

    try:
        return await _with_telethon_retries(
            "download_profile_photo",
            _dl,
            network_retries=network_retries,
            base_delay_s=base_delay_s,
        )
    except Exception:
        logger.exception("Failed to download avatar")
        return None


async def _extract_channel_metadata(
    client: TelegramClient,
    channel: Channel,
    db: Database,
    settings: Settings,
) -> tuple[dict[str, Any], TlChannel] | None:
    """Fetch and extract channel metadata and entity.

    Args:
        client: Connected Telethon client.
        channel: Channel object from database with id and username.
        db: Database gateway for marking rejected channels.
        settings: Runtime settings with retry configuration.

    Returns:
        Tuple of (channel_data dict, channel entity) or None if skipped.
    """

    channel_id = channel.id
    formatted_id = format_tg_id(channel_id)

    try:
        # Use username if available, otherwise fall back to formatted numeric ID
        identifier = channel.username if channel.username else formatted_id

        entity = await _with_telethon_retries(
            f"get_entity({formatted_id})",
            lambda: client.get_entity(identifier),
            network_retries=settings.network_retries,
            base_delay_s=settings.network_retry_base_delay_s,
        )
    except ValueError as e:
        logger.error(f"Could not find entity for {channel.id}: {e}")
        await db.mark_channel_rejected(channel.id)
        return None

    if not isinstance(entity, TlChannel):
        logger.warning(
            "Skipping non-channel entity: %s (%s)",
            channel_id,
            type(entity).__name__,
        )

        return None

    if entity.access_hash is None:
        full = None
    else:
        channel_id_val = entity.id
        access_hash = entity.access_hash
        full = await _with_telethon_retries(
            f"GetFullChannelRequest({channel_id})",
            lambda: client(
                GetFullChannelRequest(InputChannel(channel_id_val, access_hash))
            ),
            network_retries=settings.network_retries,
            base_delay_s=settings.network_retry_base_delay_s,
        )

    subscribers_count = getattr(
        getattr(full, "full_chat", None), "participants_count", None
    )
    if not isinstance(subscribers_count, int):
        subscribers_count = None

    avatar_path = await _fetch_avatar_path(
        client,
        entity,
        settings.avatars_dir,
        network_retries=settings.network_retries,
        base_delay_s=settings.network_retry_base_delay_s,
    )

    channel_data: dict[str, Any] = {
        "id": int(entity.id),
        "username": _normalize_username(getattr(entity, "username", None)),
        "title": getattr(entity, "title", "") or "",
        "description": getattr(getattr(full, "full_chat", None), "about", None),
        "subscribers_count": subscribers_count,
        "avatar_url": avatar_path,
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
        "comments_count": _message_comments_count(msg),
        "shares_count": getattr(msg, "forwards", None),
        "reactions_count": _message_reactions_count(msg),
    }

    post = await db.upsert_post(post_data)
    if not (post and post.id and post.content):
        return None

    return int(post.id)


async def _parse_single_channel(
    client: TelegramClient,
    db: Database,
    channel: Channel,
    settings: Settings,
) -> None:
    """Fetch, normalize, and persist one channel with its recent posts.

    Args:
        client: Connected Telethon client used for Telegram API calls.
        db: Database gateway used to upsert channels and posts.
        channel: Channel object from database with id and username.
        settings: Runtime settings containing limits and retry configuration.
    """

    channel_id = channel.id
    logger.info(
        "Start channel parse: id=%s username=%s", channel_id, channel.username
    )

    try:
        result = await _extract_channel_metadata(client, channel, db, settings)
        if result is None:
            await db.mark_channel_rejected(channel_id)
            return

        channel_data, entity = result

        await db.upsert_channel(channel_data)

        logger.info(
            "Channel saved: id=%s username=%s title=%s",
            channel_data["id"],
            channel_data["username"],
            channel_data["title"],
        )

        posts_saved = 0

        async for msg in client.iter_messages(
            entity, limit=settings.posts_limit
        ):
            if not isinstance(msg, Message):
                continue

            post_id = await _process_message(msg, entity, db)
            if post_id is None:
                continue

            posts_saved += 1

        await db.mark_channel_parsed(channel_id)

        logger.info(
            "Done channel: id=%s username=%s (posts saved: %s)",
            channel_id,
            channel.username,
            posts_saved,
        )

    except FloodWaitError as e:
        delay = int(getattr(e, "seconds", 0)) or 1
        total_delay = delay + 10  # Add 10 seconds safety margin

        logger.warning(
            "Channel %s: FloodWaitError, sleeping %ss (+10s safety)",
            channel_id,
            total_delay,
        )

        await asyncio.sleep(total_delay)
        # After FloodWait, mark as parsed to avoid getting stuck
        await db.mark_channel_parsed(channel_id)

    except (ChannelPrivateError, UserDeactivatedError) as e:
        logger.warning(
            "Channel %s: access denied (%s), marking as rejected",
            channel_id,
            type(e).__name__,
        )
        await db.mark_channel_rejected(channel_id)

    except (OSError, asyncio.TimeoutError, ConnectionError) as e:
        logger.exception(
            "Channel %s: network error (%s)", channel_id, type(e).__name__
        )
        await db.mark_channel_processed(channel_id)

    except RPCError as e:
        logger.exception(
            "Channel %s: Telethon RPCError (%s)",
            channel_id,
            type(e).__name__,
        )
        await db.mark_channel_processed(channel_id)

    except Exception:
        logger.exception("Channel %s: unexpected error", channel_id)
        await db.mark_channel_processed(channel_id)


class ParserWorker:
    """Async worker that processes channels using a single Telegram session."""

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
        self.worker_id = worker_id
        self.session_path = session_path
        self.db = db
        self.settings = settings
        self.api_id = api_id
        self.api_hash = api_hash
        self.proxy_url = proxy_url
        self.device_model = device_model
        self.system_version = system_version
        self.app_version = app_version
        self.lang_code = lang_code
        self.system_lang_code = system_lang_code
        self.client: TelegramClient | None = None

    async def run(self) -> None:
        """Main worker loop: continuously fetch and parse channels from DB queue."""

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
            "request_retries": self.settings.network_retries,
            "connection_retries": self.settings.network_retries,
            "retry_delay": self.settings.network_retry_base_delay_s,
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

            if self.proxy_url:
                logger.info(
                    "Worker %d: Connecting to Telegram via proxy",
                    self.worker_id,
                )
            else:
                logger.info(
                    "Worker %d: Connecting to Telegram directly (no proxy)",
                    self.worker_id,
                )

            if not await self.client.is_user_authorized():
                logger.critical(
                    "Worker %d: SESSION UNAUTHORIZED. Path: %s. "
                    "Interactive login is impossible in Docker. Check your session files!",
                    self.worker_id, self.session_path
                )
                return

            logger.info("Worker %d: Starting parser loop", self.worker_id)

            while True:
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

                    await _parse_single_channel(
                        self.client, self.db, channel, self.settings
                    )

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

    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info(
        "Starting parser (posts_limit=%d, concurrency from session count)",
        settings.posts_limit,
    )

    # Initialize database
    db = Database(settings.db_url)
    await db.init_db()

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
    workers: list[ParserWorker] = []

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

                api_id = config.get("api_id") or config.get("app_id") or settings.api_id
                api_hash = config.get("api_hash") or config.get("app_hash") or settings.api_hash
                proxy_url = config.get("proxy_url") if "proxy_url" in config else settings.proxy_url
                device_model = config.get("device_model") or config.get("device") or device_model
                system_version = config.get("system_version") or config.get("sdk") or system_version
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

        worker = ParserWorker(
            worker_id=i,
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
        workers.append(worker)

    logger.info("Spawning %d parser workers", len(workers))

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
