"""Shared utility functions for Telegram channel and message processing."""

from __future__ import annotations

import logging
import asyncio
from typing import Any, Callable

from telethon import TelegramClient
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.types import Channel as TlChannel
from telethon.tl.types import InputChannel, Message

logger = logging.getLogger(__name__)

# Global in-memory cache for entity resolution (legacy, kept for backward compatibility)
# Key: (channel_id, None) tuple, Value: resolved TlChannel entity
# Note: access_hash is NOT used in the cache key to prevent cross-account token poisoning.
# Each Telethon session stores its own access_hash in its .session SQLite file.
# Limited to 10000 entries to prevent unbounded memory growth; oldest entries are evicted
_entity_cache: dict[tuple[int, int | None], TlChannel] = {}
_MAX_CACHE_SIZE = 10000


def normalize_username(username: str | None) -> str | None:
    """Normalize a Telegram username by removing a leading at-sign.

    Args:
        username: Raw username string, potentially with a leading `@`.

    Returns:
        Username without a leading `@`, or `None` when no username is provided.
    """

    if not username:
        return None
    return username[1:] if username.startswith("@") else username


def count_message_reactions(message: Message) -> int | None:
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
        count = getattr(r, "count", None)
        if isinstance(count, int):
            total += count
    return total


def count_message_comments(message: Message) -> int | None:
    """Extract comments count from a message replies metadata.

    Args:
        message: Telethon message object to inspect.

    Returns:
        Number of replies/comments when present, otherwise `None`.
    """

    replies = getattr(message, "replies", None)
    count = getattr(replies, "replies", None) if replies else None
    return count if isinstance(count, int) else None


async def get_channel_entity_safe(
    client: TelegramClient,
    channel_id: int | str,
    *,
    safe_api_call: Callable[..., Any] | None = None,
    access_hash: int | None = None,
    entity_cache: dict[tuple[int, int | None], Any] | None = None,
) -> ChannelResolutionResult:
    """Safely get channel entity by ID or username using Telethon session cache.

    This function resolves channel entities relying on Telethon's native SQLite
    session cache for access_hash lookup. The access_hash parameter is ignored
    to prevent cross-account token poisoning.

    Resolution flow:
    1. Check worker's in-memory entity_cache using (channel_id, None)
    2. Try direct client.get_entity(channel_id) - cheap if session has cached the entity
    3. Fall back to username resolution if ID-only resolution fails (expensive)

    Args:
        client: Telethon client instance.
        channel_id: Channel ID (int) or username (str).
        safe_api_call: Optional async function that wraps API calls with
            retry logic and error handling. Format: safe_api_call(name, callable).
        access_hash: Ignored. Kept for signature compatibility only.
        entity_cache: Optional per-worker entity cache dict to check before global cache.

    Returns:
        ChannelResolutionResult containing the entity or error status.
    """
    # access_hash is ignored to prevent cross-account token poisoning.
    # Telethon's native session cache handles access_hash lookup internally.

    # Check worker's entity cache first if provided (per-worker cache)
    # Use (channel_id, None) as cache key to avoid access_hash poisoning
    if entity_cache is not None and isinstance(channel_id, int):
        cache_key = (channel_id, None)
        if cache_key in entity_cache:
            cached_entity = entity_cache[cache_key]
            logger.debug(
                "Cache HIT for channel id=%s (from worker cache)",
                channel_id,
            )
            return ChannelResolutionResult(entity=cached_entity)

    # Check global cache next (shared across workers)
    if isinstance(channel_id, int):
        cache_key = (channel_id, None)
        if cache_key in _entity_cache:
            cached_entity = _entity_cache[cache_key]
            logger.debug(
                "Cache HIT for channel id=%s (from global cache)",
                channel_id,
            )
            return ChannelResolutionResult(entity=cached_entity)

    # Resolve entity: Telethon will use its session cache for cheap lookup
    async def _fetch_entity() -> Any:
        """Internal helper to fetch entity using the appropriate method."""
        if isinstance(channel_id, int):
            # SQLite session cache lookup by ID: cache miss triggers a network request, wrap in wait_for to prevent hangs
            return await asyncio.wait_for(client.get_entity(channel_id), timeout=30.0)
        # Username-based lookup requires network call
        if safe_api_call:
            # safe_api_call internally handles pre-request sleep and enforces its own strict socket-level timeout
            return await safe_api_call(
                f"get_entity({channel_id})",
                lambda: client.get_entity(channel_id),
            )
        # No safe_api_call provided: wrap raw call to protect against socket hangs
        return await asyncio.wait_for(client.get_entity(channel_id), timeout=30.0)

    try:
        entity = await _fetch_entity()

        if isinstance(entity, TlChannel) and getattr(
            entity, "broadcast", False
        ):
            # Log resolution method
            if isinstance(channel_id, str):
                logger.debug(
                    "Resolved entity by username: %s",
                    channel_id,
                )
            else:
                logger.debug(
                    "Resolved entity by ID (using session cache): id=%s",
                    channel_id,
                )
            # Cache the result using (channel_id, None) to avoid access_hash poisoning
            if isinstance(channel_id, int):
                cache_key = (channel_id, None)
                _entity_cache[cache_key] = entity
                if entity_cache is not None:
                    entity_cache[cache_key] = entity
                # Simple eviction: if cache exceeds max size, clear it
                if len(_entity_cache) > _MAX_CACHE_SIZE:
                    _entity_cache.clear()
            return ChannelResolutionResult(entity=entity)
        return ChannelResolutionResult(
            entity=None, reason="not_a_broadcast_channel"
        )

    except asyncio.TimeoutError:
        logger.warning("Entity resolution timed out for channel %s after 30 seconds", channel_id)
        return ChannelResolutionResult(entity=None, reason="timeout")

    except ValueError as e:
        if "No user has" in str(e):
            logger.debug(
                "Shadowban suspected for global search. Target: %s",
                channel_id,
            )
            return ChannelResolutionResult(shadowbanned=True)
        logger.debug(
            "ValueError resolving %s: %s",
            channel_id,
            e,
        )
        return ChannelResolutionResult(entity=None, reason="value_error")

    except Exception as e:
        # Re-raise connection errors only when using safe_api_call
        if safe_api_call and (
            "disconnected" in str(e).lower()
            or isinstance(e, (ConnectionError, OSError))
        ):
            raise
        logger.debug(
            "Failed to resolve channel %s: %s",
            channel_id,
            e,
        )
        return ChannelResolutionResult(entity=None, reason="unknown_error")


async def get_full_channel_info(
    client: TelegramClient,
    entity: TlChannel | InputChannel,
    *,
    safe_api_call: Callable[..., Any] | None = None,
) -> tuple[int | None, str | None]:
    """Get subscriber count and description for a channel.

    Args:
        client: Telethon client instance.
        entity: Channel entity (TlChannel or InputChannel).
        safe_api_call: Optional async function that wraps API calls with
            retry logic and error handling.

    Returns:
        Tuple of (subscribers_count, description). Count may be None if unavailable.
    """

    try:
        if isinstance(entity, InputChannel):
            input_channel = entity
            entity_id = input_channel.channel_id
        elif entity.access_hash is None:
            return None, None
        else:
            input_channel = InputChannel(entity.id, entity.access_hash)
            entity_id = entity.id

        if safe_api_call:
            # safe_api_call internally handles pre-request sleep and enforces its own strict socket-level timeout
            full = await safe_api_call(
                f"GetFullChannelRequest({entity_id})",
                lambda: client(GetFullChannelRequest(input_channel)),
                rpc_error_fatal=True,
            )
        else:
            # No safe_api_call provided: wrap raw call to protect against socket hangs
            full = await asyncio.wait_for(
                client(GetFullChannelRequest(input_channel)),
                timeout=30.0,
            )

        if full is None:
            return None, None

        participants_count = getattr(
            getattr(full, "full_chat", None), "participants_count", None
        )
        description = getattr(getattr(full, "full_chat", None), "about", None)

        if not isinstance(participants_count, int):
            participants_count = None

        return participants_count, description
    except asyncio.TimeoutError:
        logger.warning(
            "Fetching full channel info timed out for entity %s after 30 seconds",
            getattr(entity, "username", entity_id),
        )
        return None, None
    except Exception as e:
        logger.debug(
            "Error getting full channel for %s: %s",
            getattr(entity, "username", "unknown"),
            e,
        )
        return None, None


class ChannelResolutionResult:
    """Result of a channel entity resolution attempt."""

    def __init__(
        self,
        entity: TlChannel | None = None,
        shadowbanned: bool = False,
        reason: str | None = None,
    ) -> None:
        """Initialize resolution result.

        Args:
            entity: Resolved channel entity, or None if resolution failed.
            shadowbanned: True if shadowban was detected.
            reason: Optional reason for failure (for debugging).
        """

        self.entity = entity
        self.shadowbanned = shadowbanned
        self.reason = reason

    def is_success(self) -> bool:
        """Check if resolution was successful."""
        return self.entity is not None
