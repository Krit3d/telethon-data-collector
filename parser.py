import argparse
import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import urlparse

from dotenv import load_dotenv
from python_socks import ProxyType
from telethon import TelegramClient
from telethon.errors import RPCError
from telethon.errors.rpcerrorlist import FloodWaitError
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.types import Channel as TlChannel
from telethon.tl.types import InputPeerChannel
from telethon.tl.types import Message

from database import Database

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class Settings:
    api_id: int
    api_hash: str
    db_url: str
    session_name: str
    channels: list[str]
    posts_limit: int
    concurrency: int
    network_retries: int
    network_retry_base_delay_s: float
    proxy_url: str | None
    avatars_dir: Path


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _parse_channels_env(value: str | None) -> list[str]:
    if not value:
        return []

    raw = value.replace("\r\n", "\n").replace("\r", "\n")
    parts: list[str] = []

    for token in raw.replace(",", "\n").split("\n"):
        t = token.strip()

        if not t:
            continue
        elif t.startswith("https://t.me/"):
            t = t.removeprefix("https://t.me/").strip("/")
        elif t.startswith("@"):
            t = t[1:]

        parts.append(t)

    # preserve order, remove duplicates
    seen: set[str] = set()
    out: list[str] = []

    for ch in parts:
        if ch in seen:
            continue

        seen.add(ch)
        out.append(ch)

    return out


def _load_channels_from_file(path: Path) -> list[str]:
    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8")

    return _parse_channels_env(content)


def _load_settings() -> Settings:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Telegram channels parser (Telethon)"
    )
    parser.add_argument(
        "--posts", type=int, default=None, help="Posts per channel"
    )
    parser.add_argument(
        "--concurrency", type=int, default=None, help="Max parallel channels"
    )
    parser.add_argument(
        "--channels-limit",
        type=int,
        default=None,
        help="Limit number of channels to parse",
    )
    parser.add_argument(
        "--channels-file",
        type=str,
        default=os.getenv("CHANNELS_FILE", "channels.txt"),
        help="Path to file with channels list",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level",
    )

    args = parser.parse_args()
    _setup_logging(args.log_level)

    api_id_raw = os.getenv("API_ID")
    api_hash = os.getenv("API_HASH")
    db_url = os.getenv("DB_URL")
    session_name = os.getenv("SESSION_NAME", "telethon")

    if not api_id_raw or not api_hash or not db_url:
        raise SystemExit(
            "Missing env vars. Required: API_ID, API_HASH, DB_URL (see .env.example)"
        )

    channels = _load_channels_from_file(Path(args.channels_file))
    channels_env = _parse_channels_env(os.getenv("CHANNELS"))

    if channels_env:
        # env has priority
        channels = channels_env

    if not channels:
        raise SystemExit(
            "No channels provided. Set CHANNELS env var or create channels.txt"
        )

    if args.channels_limit is not None:
        channels = channels[: max(0, args.channels_limit)]

    posts_limit = int(
        args.posts if args.posts is not None else os.getenv("POSTS_LIMIT", 10)
    )
    concurrency = int(
        args.concurrency
        if args.concurrency is not None
        else os.getenv("CONCURRENCY", 10)
    )
    network_retries = int(os.getenv("NETWORK_RETRIES", 5))
    network_retry_base_delay_s = float(
        os.getenv("NETWORK_RETRY_BASE_DELAY_S", 1.0)
    )
    proxy_url = os.getenv("PROXY_URL", "").strip() or None
    avatars_dir = Path(os.getenv("AVATARS_DIR", "avatars"))

    return Settings(
        api_id=int(api_id_raw),
        api_hash=api_hash,
        db_url=db_url,
        session_name=session_name,
        channels=channels,
        posts_limit=posts_limit,
        concurrency=max(1, concurrency),
        network_retries=max(0, network_retries),
        network_retry_base_delay_s=max(0.1, network_retry_base_delay_s),
        proxy_url=proxy_url,
        avatars_dir=avatars_dir,
    )


def _build_telethon_proxy(proxy_url: str | None) -> dict[str, Any] | None:
    if not proxy_url:
        return None

    parsed = urlparse(proxy_url)
    if not parsed.scheme or not parsed.hostname or not parsed.port:
        raise ValueError(
            "Invalid PROXY_URL. Use e.g. socks5://user:pass@ip:port or http://ip:port"
        )

    scheme = parsed.scheme.lower()
    if scheme in {"socks5", "socks5h"}:
        proxy_type = ProxyType.SOCKS5
    elif scheme == "socks4":
        proxy_type = ProxyType.SOCKS4
    elif scheme in {"http", "https"}:
        proxy_type = ProxyType.HTTP
    else:
        raise ValueError(
            "Unsupported proxy scheme in PROXY_URL. Supported: socks5, socks5h, socks4, http, https"
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
    if not username:
        return None

    return username[1:] if username.startswith("@") else username


def _message_reactions_count(message: Message) -> int | None:
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
    settings = _load_settings()
    logger.info(
        "Parser starting (channels=%s posts=%s concurrency=%s)",
        len(settings.channels),
        settings.posts_limit,
        settings.concurrency,
    )

    db = Database(settings.db_url)
    await db.init_db()

    sem = asyncio.Semaphore(settings.concurrency)
    proxy = _build_telethon_proxy(settings.proxy_url)

    client = TelegramClient(
        settings.session_name,
        settings.api_id,
        settings.api_hash,
        request_retries=settings.network_retries,
        connection_retries=settings.network_retries,
        retry_delay=settings.network_retry_base_delay_s,
        proxy=proxy,
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
