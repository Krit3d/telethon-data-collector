"""Shared utility functions for Telegram channel and message processing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from telethon import TelegramClient
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.types import Channel as TlChannel
from telethon.tl.types import InputChannel, InputPeerChannel, Message

logger = logging.getLogger(__name__)


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


async def fetch_avatar_path(
    client: TelegramClient,
    entity: TlChannel | InputPeerChannel,
    avatars_dir: Path,
    *,
    safe_api_call: Callable[..., Any] | None = None,
) -> str | None:
    """Download and return a channel avatar path.

    Args:
        client: Initialized Telethon client instance.
        entity: Telegram channel entity used as avatar source.
        avatars_dir: Directory where avatar files are stored.
        safe_api_call: Optional async function that wraps API calls with
            retry logic and error handling. Format: safe_api_call(name, callable).

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

    async def _download() -> str | None:
        result = await client.download_profile_photo(
            entity, file=str(target_file)
        )
        return str(result) if result else None

    try:
        if safe_api_call:
            return await safe_api_call("download_avatar", _download)
        else:
            return await _download()
    except Exception:
        logger.exception("Failed to download avatar for channel %s", entity_id)
        return None


async def get_channel_entity_safe(
    client: TelegramClient,
    channel_id: int | str,
    *,
    safe_api_call: Callable[..., Any] | None = None,
) -> ChannelResolutionResult:
    """Safely get channel entity by ID or username.

    This function attempts to resolve a channel entity. If safe_api_call is
    provided, it uses that for error handling and retries. Otherwise, it
    uses the client's get_entity directly.

    Args:
        client: Telethon client instance.
        channel_id: Channel ID (int) or username (str).
        safe_api_call: Optional async function that wraps API calls with
            retry logic and error handling. Format: safe_api_call(name, callable).

    Returns:
        ChannelResolutionResult containing the entity or error status.
    """

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
            return ChannelResolutionResult(entity=entity)
        return ChannelResolutionResult(
            entity=None, reason="not_a_broadcast_channel"
        )

    except ValueError as e:
        if "No user has" in str(e):
            logger.warning(
                "Shadowban suspected for global search. Target: %s",
                channel_id,
            )
            return ChannelResolutionResult(shadowbanned=True)
        logger.warning(
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
        logger.warning(
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
        logger.warning(
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
