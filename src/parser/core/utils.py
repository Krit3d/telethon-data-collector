"""Shared utility functions for Telegram channel and message processing."""

from __future__ import annotations

import logging
from typing import Any, Callable

from telethon import TelegramClient
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.types import Channel as TlChannel
from telethon.tl.types import InputChannel, InputPeerChannel, Message

logger = logging.getLogger(__name__)

# Global in-memory cache for entity resolution (legacy, kept for backward compatibility)
# Key: (channel_id, access_hash) tuple, Value: resolved TlChannel entity
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
    """Safely get channel entity by ID or username with caching and cheap resolution.

    This function attempts to resolve a channel entity. If safe_api_call is
    provided, it uses that for error handling and retries. Otherwise, it
    uses the client's get_entity directly.

    Optimizations:
    - Checks worker's entity cache first (if provided), then global cache
    - If access_hash is provided, constructs InputPeerChannel directly (cheap, no FloodWait)
    - Falls back to username-based resolution only when necessary

    Args:
        client: Telethon client instance.
        channel_id: Channel ID (int) or username (str).
        safe_api_call: Optional async function that wraps API calls with
            retry logic and error handling. Format: safe_api_call(name, callable).
        access_hash: Optional access_hash for cheap resolution via InputPeerChannel.
        entity_cache: Optional per-worker entity cache dict to check before global cache.

    Returns:
        ChannelResolutionResult containing the entity or error status.
    """

    # Check worker's entity cache first if provided (per-worker cache)
    if entity_cache is not None and isinstance(channel_id, int):
        cache_key = (channel_id, access_hash)
        if cache_key in entity_cache:
            cached_entity = entity_cache[cache_key]
            logger.info(
                "Cache HIT for channel id=%s (Cheap - from worker cache)",
                channel_id,
            )
            return ChannelResolutionResult(entity=cached_entity)

    # Check global cache next (shared across workers)
    if isinstance(channel_id, int):
        cache_key = (channel_id, access_hash)
        if cache_key in _entity_cache:
            cached_entity = _entity_cache[cache_key]
            logger.info(
                "Cache HIT for channel id=%s (Cheap - from global cache)",
                channel_id,
            )
            return ChannelResolutionResult(entity=cached_entity)

    # Try cheap resolution first if we have access_hash
    if access_hash is not None and isinstance(channel_id, int):
        try:
            input_peer = InputPeerChannel(channel_id, access_hash)
            # We need to fetch the full entity, but using InputPeerChannel is cheap
            # Let's use client.get_entity with the InputPeer directly
            if safe_api_call:
                entity = await safe_api_call(
                    f"get_entity(InputPeerChannel({channel_id}, hash))",
                    lambda: client.get_entity(input_peer),
                )
            else:
                entity = await client.get_entity(input_peer)

            if isinstance(entity, TlChannel) and getattr(
                entity, "broadcast", False
            ):
                logger.info(
                    "Resolved entity by ID+HASH (Cheap): id=%s",
                    channel_id,
                )
                # Cache with the access_hash we used in both global and worker cache
                if isinstance(channel_id, int) and access_hash is not None:
                    cache_key_with_hash = (channel_id, access_hash)
                    _entity_cache[cache_key_with_hash] = entity
                    if entity_cache is not None:
                        entity_cache[cache_key_with_hash] = entity
                    # Simple eviction: if cache exceeds max size, clear it (oldest entries dropped)
                    if len(_entity_cache) > _MAX_CACHE_SIZE:
                        _entity_cache.clear()
                return ChannelResolutionResult(entity=entity)
            return ChannelResolutionResult(
                entity=None, reason="not_a_broadcast_channel"
            )
        except Exception as e:
            # Cheap resolution failed, fall through to username or ID-only resolution
            logger.debug(
                "Cheap resolution failed for id=%s: %s. Will try other methods.",
                channel_id,
                e,
            )

    async def _fetch_entity() -> Any:
        """Internal helper to fetch entity using the appropriate method."""
        if safe_api_call:
            return await safe_api_call(
                f"get_entity({channel_id})",
                lambda: client.get_entity(channel_id),
            )
        return await client.get_entity(channel_id)

    try:
        entity = await _fetch_entity()

        if isinstance(entity, TlChannel) and getattr(
            entity, "broadcast", False
        ):
            # Determine if this was a username-based resolution (expensive)
            if isinstance(channel_id, str):
                logger.info(
                    "Resolving entity by USERNAME (Expensive!): %s",
                    channel_id,
                )
            else:
                logger.info(
                    "Resolving entity by ID-only (Expensive!): id=%s",
                    channel_id,
                )
            # Cache the result for future use if we got an access_hash
            if isinstance(channel_id, int) and getattr(entity, "access_hash", None):
                cache_key_with_hash = (channel_id, entity.access_hash)
                _entity_cache[cache_key_with_hash] = entity
                if entity_cache is not None:
                    entity_cache[cache_key_with_hash] = entity
                # Simple eviction: if cache exceeds max size, clear it
                if len(_entity_cache) > _MAX_CACHE_SIZE:
                    _entity_cache.clear()
            return ChannelResolutionResult(entity=entity)
        return ChannelResolutionResult(
            entity=None, reason="not_a_broadcast_channel"
        )

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
            full = await safe_api_call(
                f"GetFullChannelRequest({entity_id})",
                lambda: client(GetFullChannelRequest(input_channel)),
                rpc_error_fatal=True,
            )
        else:
            full = await client(GetFullChannelRequest(input_channel))

        if full is None:
            return None, None

        participants_count = getattr(
            getattr(full, "full_chat", None), "participants_count", None
        )
        description = getattr(getattr(full, "full_chat", None), "about", None)

        if not isinstance(participants_count, int):
            participants_count = None

        return participants_count, description
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
