"""Telegram account parser.

Scrapes account metadata and content via Telethon, stores to PostgreSQL,
and triggers knowledge graph extraction.
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
from telethon.tl.types import PeerChannel

from src.config.config import Settings, load_settings
from src.db.database import Database
from src.db.models import Account
from src.parser.telegram.core.runner import start_workers
from src.parser.telegram.core.worker_base import BaseTelegramWorker
from src.parser.telegram.core.utils import (
    count_message_comments,
    count_message_reactions,
    get_channel_entity_safe,
    get_full_channel_info,
    normalize_username,
)

logger = logging.getLogger(__name__)


def _preprocess_text_for_regex(text: str, max_length: int = 4096) -> str:
    """Sanitize text to prevent ReDoS: truncate, remove invalid chars, normalize whitespace."""
    if not text:
        return ""

    text = text[:max_length]

    # Remove null bytes and control characters, keep newlines and tabs
    text = "".join(
        char for char in text if char.isprintable() or char in ("\n", "\t")
    )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def _detect_language(text: str) -> str:
    """Detect text language: returns 'ru' if Cyrillic chars outnumber Latin, else 'en'."""
    if not text:
        return "en"

    cyrillic_count = 0
    latin_count = 0

    for char in text:
        cp = ord(char)
        if 0x0400 <= cp <= 0x04FF:  # Cyrillic Unicode block
            cyrillic_count += 1
        elif ("a" <= char.lower() <= "z") or ("A" <= char <= "Z"):
            latin_count += 1

    return "ru" if cyrillic_count > latin_count else "en"


async def _extract_account_metadata(
    client: TelegramClient,
    account: Account,
    db: Database,
    *,
    safe_api_call: Callable[..., Any],
    entity_cache: dict[tuple[int, int | None], Any] | None = None,
) -> tuple[dict[str, Any], TlChannel] | None:
    """Resolve account entity, fetch metadata (subscribers, description), return data and TL entity."""
    account_id = account.id
    entity = None

    result = await get_channel_entity_safe(
        client,
        account_id,
        safe_api_call=safe_api_call,
        entity_cache=entity_cache,
    )
    if result.is_success() and result.entity is not None:
        entity = result.entity
        logger.info("Account id=%s resolved via ID", account_id)

    if entity is None and account.username:
        logger.info("Account id=%s: falling back to username", account_id)
        result = await get_channel_entity_safe(
            client,
            account.username,
            safe_api_call=safe_api_call,
            entity_cache=entity_cache,
        )
        if result.is_success() and result.entity is not None:
            entity = result.entity
        else:
            if result.shadowbanned:
                try:
                    await asyncio.wait_for(
                        db.mark_account_pending(account_id), timeout=15.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "DB timeout: mark_account_pending %s", account_id
                    )
            else:
                try:
                    await asyncio.wait_for(
                        db.mark_account_rejected(account_id), timeout=15.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "DB timeout: mark_account_rejected %s", account_id
                    )
                await asyncio.sleep(random.uniform(10, 20))
            return None

    if entity is None:
        try:
            await asyncio.wait_for(
                db.mark_account_rejected(account_id), timeout=15.0
            )
        except asyncio.TimeoutError:
            logger.warning("DB timeout: mark_account_rejected %s", account_id)
        await asyncio.sleep(random.uniform(10, 20))
        return None

    if not isinstance(entity, TlChannel):
        logger.warning("Entity is not a channel, skipping")
        return None

    subscribers_count, description = await get_full_channel_info(
        client, entity, safe_api_call=safe_api_call
    )

    account_data: dict[str, Any] = {
        "id": int(entity.id),
        "username": normalize_username(getattr(entity, "username", None)),
        "title": getattr(entity, "title", "") or "",
        "description": description,
        "subscribers_count": subscribers_count,
    }

    return account_data, entity


async def _process_message(
    msg: Message,
    entity: TlChannel,
    db: Database,
    *,
    safe_api_call: Callable[..., Any] | None = None,
) -> int | None:
    """Process a Telegram message: extract content/metadata, upsert to DB, return content ID or None."""
    if msg.id is None or msg.date is None:
        return None

    if getattr(msg, "action", None) is not None:
        return None

    msg_content = getattr(msg, "message", None)
    if not msg_content:
        return None

    published_at = msg.date
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    # Dedicated DB columns (not in raw_metadata)
    fwd_from_channel_id: int | None = None
    if msg.fwd_from and msg.fwd_from.from_id:
        if isinstance(msg.fwd_from.from_id, PeerChannel):
            fwd_from_channel_id = msg.fwd_from.from_id.channel_id

    # Geo-coordinates as a dictionary
    geo: dict[str, float] | None = None
    if msg.media and hasattr(msg.media, "geo"):
        geo_obj = getattr(msg.media, "geo", None)
        if geo_obj:
            lat = getattr(geo_obj, "lat", None)
            long_val = getattr(geo_obj, "long", None)
            if lat is not None and long_val is not None:
                geo = {"lat": lat, "long": long_val}

    reply_to_msg_id: int | None = None
    if msg.reply_to and hasattr(msg.reply_to, "reply_to_msg_id"):
        reply_to_msg_id = getattr(msg.reply_to, "reply_to_msg_id", None)

    author: str | None = getattr(msg, "post_author", None)

    # Pre-process text to prevent ReDoS
    try:
        text_content = _preprocess_text_for_regex(msg_content, max_length=4096)
    except Exception as e:
        logger.warning(
            "Text pre-processing failed for message_id=%s: %s", msg.id, e
        )
        text_content = ""

    try:
        language: str = _detect_language(text_content)
    except Exception as e:
        logger.warning(
            "Language detection failed for message_id=%s: %s", msg.id, e
        )
        language = "en"

    # Text metrics
    text_metrics: dict[str, int] = {}
    try:
        text_metrics["word_count"] = len(text_content.split())
        text_metrics["char_count"] = len(text_content)
    except Exception as e:
        logger.warning(
            "Word/char count failed for message_id=%s: %s", msg.id, e
        )

    try:
        text_metrics["link_count"] = len(
            re.findall(r"https?://[^\s<>]{1,500}", text_content)
        )
    except Exception as e:
        logger.warning("Link counting failed for message_id=%s: %s", msg.id, e)
        text_metrics["link_count"] = 0

    try:
        text_metrics["mention_count"] = len(
            re.findall(r"@\w{1,50}", text_content)
        )
    except Exception as e:
        logger.warning(
            "Mention counting failed for message_id=%s: %s", msg.id, e
        )
        text_metrics["mention_count"] = 0

    try:
        text_metrics["hashtag_count"] = len(
            re.findall(r"#\w{1,50}", text_content)
        )
    except Exception as e:
        logger.warning(
            "Hashtag counting failed for message_id=%s: %s", msg.id, e
        )
        text_metrics["hashtag_count"] = 0

    # Numeric metrics
    numeric_metrics: dict[str, Any] = {}

    # Extract price patterns (as strings)
    try:
        price_patterns: list[str] = []
        # Match prices with currency symbol/prefix first
        price_patterns.extend(
            re.findall(
                r"(?:[$₽]|USD|EUR|GBP|RUB)\s*\d{1,10}(?:[.,]\d{1,2})?",
                text_content,
            )
        )
        # Match prices with currency symbol/suffix
        price_patterns.extend(
            re.findall(
                r"\d{1,10}(?:[.,]\d{1,2})?\s*(?:[$₽]|USD|EUR|GBP|RUB)",
                text_content,
            )
        )
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_prices = [
            p for p in price_patterns if not (p in seen or seen.add(p))
        ]
        if unique_prices:
            numeric_metrics["prices"] = unique_prices
    except Exception as e:
        logger.warning(
            "Price pattern extraction failed for message_id=%s: %s", msg.id, e
        )

    # Extract numbers with validation
    try:
        numbers = re.findall(r"\b\d{1,10}(?:\.\d{1,5})?\b", text_content)
        if numbers:
            parsed_numbers: list[float | int] = []
            for n in numbers[:100]:  # Limit to 100 numbers for performance
                try:
                    if "." in n:
                        val = float(n)
                    else:
                        val = int(n)
                    if -1e15 < val < 1e15:
                        parsed_numbers.append(val)
                except (ValueError, OverflowError):
                    continue
            if parsed_numbers:
                numeric_metrics["numbers"] = parsed_numbers
    except Exception as e:
        logger.warning(
            "Number extraction failed for message_id=%s: %s", msg.id, e
        )

    # Build raw_metadata for JSONB column
    metadata: dict[str, Any] = {
        "language": language,
        "geo": geo,
        "author": author,
        "text_metrics": text_metrics if text_metrics else None,
        "numeric_metrics": numeric_metrics if numeric_metrics else None,
        "reply_to_msg_id": reply_to_msg_id,
    }

    # Filter out None values, but keep "language" and "author" even if None
    volatile_metadata_keys = {"author", "language"}
    metadata = {
        k: v
        for k, v in metadata.items()
        if v is not None or k in volatile_metadata_keys
    }

    # Build content data with dedicated columns
    content_data: dict[str, Any] = {
        "account_id": int(entity.id),
        "message_id": int(msg.id),
        "content": getattr(msg, "message", None),
        "published_at": published_at,
        "views": getattr(msg, "views", None),
        "comments_count": count_message_comments(msg),
        "shares_count": getattr(msg, "forwards", None),
        "reactions_count": count_message_reactions(msg),
        "fwd_from_channel_id": fwd_from_channel_id,
        "grouped_id": getattr(msg, "grouped_id", None),
        "has_media": bool(msg.media),
        "raw_metadata": metadata if metadata else None,
    }

    # Database insert with 15.0s timeout
    try:
        content = await asyncio.wait_for(db.upsert_content(content_data), timeout=15.0)
    except asyncio.TimeoutError:
        logger.warning(
            "Database timeout for message_id=%s in account %s",
            content_data.get("message_id"),
            content_data.get("account_id"),
        )
        return None
    except Exception as e:
        logger.error(
            "Database error for message_id=%s: %s",
            content_data.get("message_id"),
            e,
            exc_info=True,
        )
        return None

    if not (content and content.id and content.content):
        return None

    return int(content.id)


class ParserWorker(BaseTelegramWorker):
    """Telethon worker that parses accounts from DB, fetches content, and stores to PostgreSQL."""

    def __init__(
        self,
        worker_id: int,
        session_path: Path,
        db: Database,
        settings: Settings,
        api_id: int,
        api_hash: str,
        session_index: int = 0,
        total_sessions: int = 1,
        proxy_url: str | None = None,
        device_model: str = "PC 64bit",
        system_version: str = "Windows 10",
        app_version: str = "4.16.8",
        lang_code: str = "en",
        system_lang_code: str = "en-US",
    ):
        """Initialize parser worker with session, DB, and Telegram client configuration."""
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
        self.session_index = session_index
        self.total_sessions = total_sessions
        self.last_activity = time.time()
        self.channels_parsed_count = 0

    async def run(self) -> None:
        """Main loop: process accounts from DB, handle rate limits and auth errors."""
        try:
            await asyncio.wait_for(self.connect(), timeout=45.0)
        except asyncio.TimeoutError as e:
            logger.critical("Worker %d: Connection timeout during worker startup", self.worker_id)
            self.is_alive = False
            raise ConnectionError("Connection timed out during worker startup") from e

        logger.info(
            "Worker %d: Starting parser loop (session %d/%d)",
            self.worker_id,
            self.session_index,
            self.total_sessions,
        )

        try:
            while True:
                self.last_activity = time.time()
                if not self.is_alive:
                    logger.info("Worker %d: Terminating loop", self.worker_id)
                    return

                session_limit = getattr(
                    self.settings, "channels_per_session_limit", 5
                )
                if self.channels_parsed_count >= session_limit:
                    logger.info(
                        "Worker %d: Session quota reached", self.worker_id
                    )
                    return

                try:
                    logger.info(
                        "Worker %d (session %d/%d): Requesting account for parsing",
                        self.worker_id,
                        self.session_index,
                        self.total_sessions,
                    )
                    try:
                        account = await asyncio.wait_for(
                            self.db.get_account_for_parsing(
                                session_index=self.session_index,
                                total_sessions=self.total_sessions,
                            ),
                            timeout=15.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Worker %d: DB timeout get_account", self.worker_id
                        )
                        await asyncio.sleep(10)
                        continue

                    if account is None:
                        logger.debug(
                            "Worker %d (session %d/%d): No accounts in shard, sleeping",
                            self.worker_id,
                            self.session_index,
                            self.total_sessions,
                        )
                        await asyncio.sleep(30)
                        continue

                    logger.info(
                        "Worker %d: Processing account %s",
                        self.worker_id,
                        account.id,
                    )

                    await self._parse_single_account(account)

                    self.channels_parsed_count += 1

                except (
                    FloodWaitError,
                    AuthKeyError,
                    UserDeactivatedError,
                    SessionExpiredError,
                    SessionRevokedError,
                ) as e:
                    raise
                except (OSError, ConnectionError, asyncio.TimeoutError) as e:
                    logger.critical(
                        "Worker %d: Connection-level error, propagating to runner: %s",
                        self.worker_id,
                        e,
                        exc_info=True,
                    )
                    raise
                except Exception as e:
                    logger.error(
                        "Worker %d: Error: %s", self.worker_id, e, exc_info=True
                    )
                    await asyncio.sleep(30)
        finally:
            logger.info("Worker %d: Cleanup", self.worker_id)

    async def _parse_single_account(self, account: Account) -> None:
        """Parse single account: upsert metadata, fetch recent content, update account status in DB."""
        account_id = account.id

        try:
            result = await _extract_account_metadata(
                self.client,  # type: ignore[attr-defined]
                account,
                self.db,
                safe_api_call=self.safe_api_call,
                entity_cache=self._entity_cache,
            )
            if result is None:
                return

            account_data, entity = result

            try:
                await asyncio.wait_for(
                    self.db.upsert_account(account_data), timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Worker %d: DB timeout upsert_account %s",
                    self.worker_id,
                    account_id,
                )
                try:
                    await asyncio.wait_for(
                        self.db.mark_account_pending(account_id), timeout=15.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Worker %d: DB timeout mark_pending %s",
                        self.worker_id,
                        account_id,
                    )
                return

            # Occasional native exploration (30% chance)
            if random.random() < 0.3:
                try:
                    exploration_msg = await self.safe_api_call(
                        "exploration_fetch",
                        lambda: self.client.get_messages(  # type: ignore[attr-defined]
                            entity, limit=1
                        ),
                    )
                    await asyncio.sleep(random.uniform(1, 3))
                except Exception as e:
                    self.logger.warning(
                        "Worker %d: Exploration failed: %s", self.worker_id, e
                    )

            content_saved = 0
            chunk_size = 20
            last_msg_id: int = 0
            content_limit = self.settings.posts_limit

            try:
                latest_known_id = (
                    await asyncio.wait_for(
                        self.db.get_latest_message_id(account_id), timeout=15.0
                    )
                    or 0
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Worker %d: DB timeout get_latest_id %s",
                    self.worker_id,
                    account_id,
                )
                try:
                    await asyncio.wait_for(
                        self.db.mark_account_pending(account_id), timeout=15.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Worker %d: DB timeout mark_pending %s",
                        self.worker_id,
                        account_id,
                    )
                return

            while content_saved < content_limit:
                remaining = content_limit - content_saved
                current_chunk_size = min(chunk_size, remaining)

                messages = await self.safe_api_call(
                    f"get_messages(account_id={account_id}, chunk_size={current_chunk_size})",
                    operation=lambda: self.client.get_messages(  # type: ignore[attr-defined]
                        entity,
                        limit=current_chunk_size,
                        offset_id=last_msg_id,
                        min_id=latest_known_id,
                    ),
                )
                self.last_activity = time.time()

                if messages is None:
                    break

                if not isinstance(messages, list):
                    messages = [messages]

                if not messages:
                    break

                for msg in messages:
                    if not isinstance(msg, Message):
                        continue

                    if msg.id is not None and msg.id <= latest_known_id:
                        content_saved = content_limit
                        break

                    content_id = await _process_message(
                        msg,
                        entity,
                        self.db,
                        safe_api_call=self.safe_api_call,
                    )
                    self.last_activity = time.time()
                    if content_id is None:
                        continue

                    content_saved += 1

                    if content_saved >= content_limit:
                        break

                if content_saved >= content_limit:
                    break

                if messages:
                    oldest_msg = messages[-1]
                    if (
                        isinstance(oldest_msg, Message)
                        and oldest_msg.id is not None
                    ):
                        last_msg_id = oldest_msg.id

                if len(messages) < current_chunk_size:
                    break

                if self.settings.use_natural_delays:
                    await asyncio.sleep(self.natural_delay(base_delay=1.5))
                else:
                    await asyncio.sleep(random.uniform(1.0, 2.5))

            try:
                await asyncio.wait_for(
                    self.db.mark_account_parsed(account_id), timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Worker %d: DB timeout mark_parsed %s",
                    self.worker_id,
                    account_id,
                )
                try:
                    await asyncio.wait_for(
                        self.db.mark_account_pending(account_id), timeout=15.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Worker %d: DB timeout mark_pending %s",
                        self.worker_id,
                        account_id,
                    )

            logger.info(
                "Worker %d: Done account %s (%d content items)",
                self.worker_id,
                account_id,
                content_saved,
            )
            await asyncio.sleep(random.uniform(15.0, 45.0))

        except FloodWaitError as e:
            delay = int(getattr(e, "seconds", 0)) or 1
            logger.warning(
                "Worker %d: FloodWait %ds, account %s",
                self.worker_id,
                delay,
                account_id,
            )
            try:
                await asyncio.wait_for(
                    self.db.mark_account_pending(account_id), timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Worker %d: DB timeout mark_pending %s",
                    self.worker_id,
                    account_id,
                )
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
                    "Worker %d: Account DEAD/FROZEN (%s)",
                    self.worker_id,
                    type(e).__name__,
                )
                self.is_alive = False
                try:
                    await asyncio.wait_for(
                        self.db.mark_account_rejected(account_id), timeout=15.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Worker %d: DB timeout mark_rejected %s",
                        self.worker_id,
                        account_id,
                    )
                raise
            logger.warning(
                "Worker %d: Access denied %s: %s",
                self.worker_id,
                account_id,
                type(e).__name__,
            )
            try:
                await asyncio.wait_for(
                    self.db.mark_account_rejected(account_id), timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Worker %d: DB timeout mark_rejected %s",
                    self.worker_id,
                    account_id,
                )

        except (OSError, asyncio.TimeoutError, ConnectionError) as e:
            logger.exception(
                "Worker %d: Network error %s: %s",
                self.worker_id,
                account_id,
                type(e).__name__,
            )
            try:
                await asyncio.wait_for(
                    self.db.mark_account_pending(account_id), timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Worker %d: DB timeout mark_pending %s",
                    self.worker_id,
                    account_id,
                )

        except RPCError as e:
            logger.exception(
                "Worker %d: RPCError %s: %s",
                self.worker_id,
                account_id,
                type(e).__name__,
            )
            try:
                await asyncio.wait_for(
                    self.db.mark_account_rejected(account_id), timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Worker %d: DB timeout mark_rejected %s",
                    self.worker_id,
                    account_id,
                )

        except Exception:
            logger.exception(
                "Worker %d: Unexpected error %s", self.worker_id, account_id
            )
            try:
                await asyncio.wait_for(
                    self.db.mark_account_rejected(account_id), timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Worker %d: DB timeout mark_rejected %s",
                    self.worker_id,
                    account_id,
                )


async def main() -> None:
    """Entry point: load config, initialize DB, and start parser workers."""
    settings = load_settings()

    loop = asyncio.get_running_loop()

    def global_exception_handler(loop, context):
        msg = context.get("exception", context["message"])
        logging.getLogger("asyncio_global").critical(
            f"Unhandled asyncio exception: {msg}"
        )

    loop.set_exception_handler(global_exception_handler)

    logger.info("Starting parser (posts_limit=%d)", settings.posts_limit)

    db = Database(settings.db_url)

    await start_workers(
        worker_class=ParserWorker,
        settings=settings,
        db=db,
    )


if __name__ == "__main__":
    asyncio.run(main())
