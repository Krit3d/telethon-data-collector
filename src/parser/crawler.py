"""Autonomous Telegram channel crawler using Telethon.

This crawler discovers new channels based on recommendations from known channels.
It applies filters for subscriber count and author-generated content, and
appends qualifying channels to channels.txt.

Safety features:
- Strict random delays (15-30s) between all Telegram API calls.
- FloodWaitError handling with mandatory extra 10s wait.
- Single account usage with optional proxy support.
- BFS traversal with automatic restart on dead ends.
"""

import asyncio
import logging
import random
import re
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar, cast

from dotenv import load_dotenv

load_dotenv()
from pydantic import ValidationError
from telethon import TelegramClient
from telethon.errors import FloodWaitError, RPCError
from telethon.tl.functions.channels import (
    GetChannelRecommendationsRequest,
    GetFullChannelRequest,
)
from telethon.tl.types import Channel, InputChannel, Message, messages


# Proxy support (copied from parser.py to maintain consistency)
from python_socks import ProxyType
from urllib.parse import parse_qs, urlparse
from telethon.network.connection.tcpmtproxy import (
    ConnectionTcpMTProxyRandomizedIntermediate,
)


from src.config.config import Settings, load_settings

logger = logging.getLogger(__name__)
T = TypeVar("T")

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


class SafeCrawler:
    """Autonomous BFS crawler for Telegram channel discovery."""

    def __init__(
        self,
        client: TelegramClient,
        channels_file: Path,
        min_subscribers: int = 3000,
        posts_to_check: int = 15,
    ):
        self.client = client
        self.channels_file = channels_file
        self.min_subscribers = min_subscribers
        self.posts_to_check = posts_to_check

        # Persistent set of all known good usernames, without duplicates
        self.known_usernames: set[str] = set()
        self._load_channels()

        if not self.known_usernames:
            logger.warning("No seed channels found in %s.", self.channels_file)

        # BFS queue and sets for current session
        self.queue: deque[Channel] = deque()
        self.visited_ids: set[int] = set()
        self.queued_ids: set[int] = set()
        self.rejected_ids: set[int] = set()  # memory for filter garbage

        # List for safe restart (strings from file)
        self.seed_usernames: list[str] = list(self.known_usernames)

        logger.info(
            "Crawler initialized with %d seed channels",
            len(self.seed_usernames),
        )

    def _normalize(self, username: str) -> str:
        """Normalize a channel reference to a lowercase username without @ or URL."""
        username = username.strip()

        if username.startswith("@"):
            username = username[1:]
        if username.startswith("https://t.me/"):
            username = username.split("/")[-1]

        return username.lower()

    def _load_channels(self) -> None:
        """Load known channels from file."""
        if not self.channels_file.exists():
            return

        with open(self.channels_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                norm = self._normalize(line)
                if norm:
                    self.known_usernames.add(norm)

    def _append_channel(self, channel: str) -> None:
        """Append a new channel to the channels file (normalized, without @)."""
        with open(self.channels_file, "a", encoding="utf-8") as f:
            f.write(channel + "\n")

    async def _call_api(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Execute a Telethon API call with random delay and robust error handling.

        Adds a random 15-30 second delay before each attempt.
        Handles FloodWaitError by sleeping specified time + 10 seconds (retries indefinitely).
        Retries network errors with exponential backoff (max 5 attempts).
        """
        max_retries = 5
        attempt = 0

        while True:
            try:
                # Mandatory random delay before ANY API request
                await asyncio.sleep(random.uniform(15, 30))

                return await operation()
            except FloodWaitError as e:
                delay = int(getattr(e, "seconds", 0)) or 1
                logger.warning("⏳ FloodWaitError: sleeping %ds + 10s", delay)
                await asyncio.sleep(delay + 10)
                # retry indefinitely after FloodWait
            except (OSError, asyncio.TimeoutError, ConnectionError) as e:
                if attempt >= max_retries:
                    logger.error(
                        "❌ Max retries exceeded for network error: %s", e
                    )
                    raise

                attempt += 1
                backoff = 2**attempt * 5

                logger.warning(
                    "🌐 Network error (%s): retry %d/%d in %.1fs",
                    type(e).__name__,
                    attempt,
                    max_retries,
                    backoff,
                )
                await asyncio.sleep(backoff)
            except RPCError as e:
                logger.error("❌ RPC error: %s", e)
                raise

    async def _get_recommendations(
        self, channel_entity: Channel
    ) -> list[Channel]:
        """Fetch channel recommendations for a given channel."""
        try:
            # Build input channel - handle None access_hash
            access_hash = channel_entity.access_hash
            if access_hash is None:
                logger.warning(
                    "⚠️ Channel %s has no access_hash, cannot get recommendations",
                    getattr(channel_entity, "username", "unknown"),
                )

                return []

            input_channel = InputChannel(channel_entity.id, access_hash)
            result = await self._call_api(
                lambda: self.client(
                    GetChannelRecommendationsRequest(channel=input_channel)
                )
            )
            result = cast(messages.Chats, result)

            recommended_channels: list[Channel] = []

            # The result structure: channels are in result.chats
            if result.chats:
                for chat in result.chats:
                    if (
                        isinstance(chat, Channel)
                        and getattr(chat, "broadcast", False)
                        and getattr(chat, "username", None)
                    ):
                        recommended_channels.append(chat)

            logger.info(
                "📈 Got %d recommendations for %s",
                len(recommended_channels),
                getattr(channel_entity, "username", "unknown"),
            )

            return recommended_channels
        except Exception as e:
            logger.error(
                "❌ Failed to get recommendations for %s: %s",
                getattr(channel_entity, "username", "unknown"),
                e,
            )

            return []

    async def _passes_filters(self, entity: Channel) -> bool:
        """Apply filters directly to a Channel entity. Returns True if channel qualifies."""
        channel_name = entity.username or str(entity.id)

        if not entity.access_hash:
            return False

        # Filter 1: Subscriber count >= 3000
        try:
            access_hash = entity.access_hash
            if access_hash is None:
                logger.info("Channel %s has no access_hash", channel_name)

                return False

            input_channel = InputChannel(entity.id, access_hash)
            full = await self._call_api(
                lambda: self.client(GetFullChannelRequest(input_channel))
            )
            participants_count = getattr(
                getattr(full, "full_chat", None), "participants_count", None
            )

            if not isinstance(participants_count, int):
                logger.info("Channel %s has no participant count", channel_name)

                return False
            if participants_count < self.min_subscribers:
                logger.info(
                    "⏭️ Channel %s has %d subscribers (<%d)",
                    channel_name,
                    participants_count,
                    self.min_subscribers,
                )

                return False

            logger.info(
                "✅ Size filter passed: %s (%d subs)",
                channel_name,
                participants_count,
            )
        except Exception as e:
            logger.error(
                "❌ Error getting full channel for %s: %s", channel_name, e
            )

            return False

        # Filter 2: Author content check (video notes or first-person text)
        try:
            msgs = await self._call_api(
                lambda: self.client.get_messages(
                    input_channel, limit=self.posts_to_check
                )
            )

            if msgs is None:
                logger.info("Channel %s returned no messages", channel_name)

                return False

            # Ensure messages is always a list (get_messages can return a single Message)
            if isinstance(msgs, Message):
                msgs = [msgs]

            # Check for video notes
            for msg in msgs:
                if getattr(msg, "video_note", None):
                    logger.info(
                        "🎥 Channel %s has video note content", channel_name
                    )

                    return True

            # Check first-person pronouns
            total_matches = 0

            for msg in msgs:
                text = getattr(msg, "message", None) or ""
                total_matches += len(FIRST_PERSON_REGEX.findall(text))

                if total_matches >= 3:
                    logger.info(
                        "✍️ Channel %s has first-person content (%d matches)",
                        channel_name,
                        total_matches,
                    )

                    return True

            logger.info(
                "⏭️ Channel %s lacks author content (video notes: no, pronouns: %d)",
                channel_name,
                total_matches,
            )

            return False
        except Exception as e:
            logger.error(
                "❌ Error getting messages for %s: %s", channel_name, e
            )

            return False

    async def _seed_queue_with_random(self) -> bool:
        """Resolves a random known channel to start/restart the BFS."""
        if not self.seed_usernames:
            return False

        random.shuffle(self.seed_usernames)

        for username in self.seed_usernames:
            try:
                entity = await self._call_api(
                    lambda: self.client.get_entity(username)
                )

                if isinstance(entity, Channel) and getattr(
                    entity, "broadcast", False
                ):
                    if entity.id in self.visited_ids:
                        continue

                    self.queue.append(entity)
                    self.queued_ids.add(entity.id)
                    logger.info("🔄 Seeded queue with channel: %s", username)

                    return True
            except Exception as e:
                logger.warning("⚠️ Failed to resolve seed %s: %s", username, e)

        return False

    async def crawl(self) -> None:
        """Main infinite BFS loop."""
        logger.info(
            "🚀 Starting crawler. Known channels: %d", len(self.known_usernames)
        )

        if not await self._seed_queue_with_random():
            logger.error("❌ Could not resolve any seed channels. Exiting.")
            return

        while True:
            try:
                if not self.queue:
                    logger.info(
                        "🚧 Queue is empty (dead end). Restarting from a random seed..."
                    )

                    if not await self._seed_queue_with_random():
                        logger.error("❌ Cannot restart, seeds exhausted.")
                        break

                    continue

                current_channel = self.queue.popleft()

                if current_channel.id in self.visited_ids:
                    continue

                self.visited_ids.add(current_channel.id)
                logger.info(
                    "📍 Processing channel: %s (queue size: %d)",
                    current_channel.username,
                    len(self.queue),
                )

                recs = await self._get_recommendations(current_channel)

                for rec_channel in recs:
                    if not rec_channel.username:
                        continue

                    if (
                        rec_channel.id in self.visited_ids
                        or rec_channel.id in self.rejected_ids
                    ):
                        continue

                    norm_username = self._normalize(rec_channel.username)

                    if norm_username not in self.known_usernames:
                        if await self._passes_filters(rec_channel):
                            self.known_usernames.add(norm_username)
                            self._append_channel(norm_username)

                            logger.info(
                                "🌟 SUCCESS! Added new author channel: %s",
                                norm_username,
                            )

                            if rec_channel.id not in self.queued_ids:
                                self.queue.append(rec_channel)
                                self.queued_ids.add(rec_channel.id)
                        else:
                            self.rejected_ids.add(rec_channel.id)
                    else:
                        if rec_channel.id not in self.queued_ids:
                            self.queue.append(rec_channel)
                            self.queued_ids.add(rec_channel.id)

            except FloodWaitError as e:
                delay = int(getattr(e, "seconds", 0)) or 1
                logger.warning(
                    "🌊 Global FloodWaitError: sleeping %ds + 10s", delay
                )
                await asyncio.sleep(delay + 10)
            except Exception as e:
                logger.error(
                    "❌ Unexpected error in crawl loop: %s", e, exc_info=True
                )
                await asyncio.sleep(30)


async def main() -> None:
    """Entry point for the crawler."""

    print("Скрипт запущен, проверяю настройки...")  # Прямой принт в консоль
    try:
        settings = load_settings()
        print("Настройки загружены успешно!")
    except Exception as e:
        print(f"Ошибка при загрузке настроек: {e}")
        return

    # Load configuration from environment, .env file, and CLI arguments
    try:
        settings = load_settings()
    except SystemExit:
        # load_settings() calls sys.exit on validation errors or missing channels
        return

    # Configure logging (already configured by load_settings, but ensure it's set)
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Use settings
    API_ID = settings.api_id
    API_HASH = settings.api_hash
    CHANNELS_FILE = settings.channels_file
    SESSION_DIR = settings.session_dir
    PROXY_URL = settings.proxy_url

    # Ensure session directory exists
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    session_path = SESSION_DIR / "crawler.session"

    # Build proxy configuration if needed
    proxy_config: dict[str, Any] | None = None

    if PROXY_URL:
        try:
            proxy_config = _build_telethon_proxy(PROXY_URL)
        except ValueError as e:
            logger.error("❌ Invalid PROXY_URL: %s", e)
            return

    # Create Telethon client
    client_kwargs: dict[str, Any] = {}

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

    client = TelegramClient(
        str(session_path), API_ID, API_HASH, **client_kwargs
    )

    try:
        await client.connect()
        if not await client.is_user_authorized():
            logger.info(
                "🔐 Session not authorized. Starting interactive login..."
            )
            await client.start()  # type: ignore
            logger.info("✅ Login successful")
        else:
            logger.info("✅ Session already authorized")

        logger.info("🚀 Starting autonomous crawler")
        crawler = SafeCrawler(client, CHANNELS_FILE)

        if not crawler.known_usernames:
            logger.error(
                "❌ No seed channels available. Please add channels to channels.txt"
            )
            return

        await crawler.crawl()
    except KeyboardInterrupt:
        logger.info("🛑 Crawler stopped by user")
    except Exception as e:
        logger.exception("❌ Fatal error: %s", e)
    finally:
        await client.disconnect()  # type: ignore
        logger.info("👋 Client disconnected")


if __name__ == "__main__":
    asyncio.run(main())
