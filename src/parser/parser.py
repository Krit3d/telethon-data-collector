import asyncio
import logging
from datetime import timezone
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import parse_qs, urlparse

from python_socks import ProxyType
from telethon import TelegramClient
from telethon.errors import RPCError
from telethon.errors.rpcerrorlist import FloodWaitError
from telethon.network.connection.tcpmtproxy import (
    ConnectionTcpMTProxyRandomizedIntermediate,
)
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.types import Channel as TlChannel
from telethon.tl.types import InputPeerChannel
from telethon.tl.types import Message

from src.db.database import Database
from src.config.config import Settings, load_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _build_telethon_proxy(proxy_url: str | None) -> dict[str, Any] | None:
    """Build a Telethon-compatible proxy configuration from a URL string.

    Args:
        proxy_url: Proxy URL from settings, or `None` to disable proxying.

    Returns:
        A dictionary with Telethon proxy fields when a valid proxy URL is
        provided, otherwise `None`.

    Raises:
        ValueError: If the URL is malformed or uses an unsupported scheme.
    """

    if not proxy_url:
        return None

    parsed = urlparse(proxy_url)
    if not parsed.scheme or not parsed.hostname or not parsed.port:
        raise ValueError(
            "Invalid PROXY_URL. Use e.g. socks5://user:pass@ip:port, http://ip:port, or mtproxy://secret@ip:port"
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
                "MTProxy requires a secret. Use mtproxy://secret@ip:port or mtproxy://ip:port?secret=..."
            )

        return {
            "addr": parsed.hostname,
            "port": parsed.port,
            "secret": mtproxy_secret,
            "is_mtproxy": True,
        }
    elif scheme == "vless":
        raise ValueError(
            "Telethon does not support VLESS directly. Run a local sing-box/xray client and set PROXY_URL to its local SOCKS endpoint (e.g. socks5://127.0.0.1:1080)."
        )
    else:
        raise ValueError(
            "Unsupported proxy scheme in PROXY_URL. Supported: socks5, socks5h, socks4, http, https, mtproxy, mtproto. For VLESS use a local bridge and point PROXY_URL to socks5://127.0.0.1:1080."
        )

    return {
        "proxy_type": proxy_type,
        "addr": parsed.hostname,
        "port": parsed.port,
        "username": parsed.username,
        "password": parsed.password,
        "rdns": scheme == "socks5h",
    }


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

        except RPCError:
            # Non-network Telethon errors: log and bubble up (channel-specific handler will catch)
            logger.exception("%s: Telethon RPCError", op_name)
            raise


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

    avatars_dir.mkdir(parents=True, exist_ok=True)

    target_file = avatars_dir / f"{entity.id}.jpg"

    async def _dl() -> str | None:
        # Telethon may return None if no photo / cannot download
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


async def _parse_single_channel(
    client: TelegramClient,
    db: Database,
    channel_ref: str,
    settings: Settings,
    sem: asyncio.Semaphore,
) -> None:
    """Fetch, normalize, and persist one channel with its recent posts.

    Args:
        client: Connected Telethon client used for Telegram API calls.
        db: Database gateway used to upsert channels and posts.
        channel_ref: Channel username/link/id reference from settings.
        settings: Runtime settings containing limits and retry configuration.
        sem: Concurrency semaphore to limit parallel channel parsing tasks.

    Returns:
        `None`. All parsed data is persisted through side effects in `db`.
    """

    async with sem:
        logger.info("Start channel parse: %s", channel_ref)

        try:
            entity = await _with_telethon_retries(
                f"get_entity({channel_ref})",
                lambda: client.get_entity(channel_ref),
                network_retries=settings.network_retries,
                base_delay_s=settings.network_retry_base_delay_s,
            )

            if not isinstance(entity, TlChannel):
                logger.warning(
                    "Skipping non-channel entity: %s (%s)",
                    channel_ref,
                    type(entity).__name__,
                )

                return

            full = await _with_telethon_retries(
                f"GetFullChannelRequest({channel_ref})",
                lambda: client(GetFullChannelRequest(entity)),
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
                "username": _normalize_username(
                    getattr(entity, "username", None)
                ),
                "title": getattr(entity, "title", "") or "",
                "description": getattr(
                    getattr(full, "full_chat", None), "about", None
                ),
                "subscribers_count": subscribers_count,
                "avatar_url": avatar_path,
            }

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

                if msg.id is None or msg.date is None:
                    continue

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

                await db.upsert_post(post_data)
                posts_saved += 1

            logger.info(
                "Done channel: %s (posts saved: %s)", channel_ref, posts_saved
            )

        except FloodWaitError as e:
            delay = int(getattr(e, "seconds", 0)) or 1
            logger.warning(
                "Channel %s: FloodWaitError, sleeping %ss", channel_ref, delay
            )
            await asyncio.sleep(delay)

        except (OSError, asyncio.TimeoutError, ConnectionError) as e:
            logger.exception(
                "Channel %s: network error (%s)", channel_ref, type(e).__name__
            )

        except RPCError as e:
            logger.exception(
                "Channel %s: Telethon RPCError (%s)",
                channel_ref,
                type(e).__name__,
            )

        except Exception:
            logger.exception("Channel %s: unexpected error", channel_ref)


async def main() -> None:
    """Run parser entrypoint: connect, parse configured channels, and shutdown."""

    settings = load_settings()
    logger.info(
        "Parser starting (channels=%s posts=%s concurrency=%s)",
        len(settings.channels),
        settings.posts_limit,
        settings.concurrency,
    )

    db = Database(settings.db_url)
    await db.init_db()

    settings.session_dir.mkdir(parents=True, exist_ok=True)
    session_path = str(settings.session_dir / "telethon")

    sem = asyncio.Semaphore(settings.concurrency)
    proxy = _build_telethon_proxy(settings.proxy_url)
    client_kwargs: dict[str, Any] = {
        "request_retries": settings.network_retries,
        "connection_retries": settings.network_retries,
        "retry_delay": settings.network_retry_base_delay_s,
    }

    if proxy:
        if proxy.pop("is_mtproxy", False):
            client_kwargs["connection"] = (
                ConnectionTcpMTProxyRandomizedIntermediate
            )
            client_kwargs["proxy"] = (
                proxy["addr"],
                proxy["port"],
                proxy["secret"],
            )
        else:
            client_kwargs["proxy"] = proxy

    client = TelegramClient(
        session_path,
        settings.api_id,
        settings.api_hash,
        **client_kwargs,
    )
    try:
        if settings.proxy_url:
            logger.info("Connecting to Telegram via proxy")
        else:
            logger.info("Connecting to Telegram directly (no proxy)")

        await client.connect()
        logger.info("Connected to Telegram")

        if not await client.is_user_authorized():
            logger.info("Session is not authorized, starting interactive login")
            await client.start()
            logger.info("Interactive login completed")
        else:
            logger.info("Session already authorized")

        tasks = [
            asyncio.create_task(
                _parse_single_channel(client, db, ch, settings, sem)
            )
            for ch in settings.channels
        ]
        await asyncio.gather(*tasks)
    finally:
        await client.disconnect()

    await db.close()
    logger.info("Parser finished")


if __name__ == "__main__":
    asyncio.run(main())
