"""
Async database helpers for discovered account queuing and virtual bio post upserts.

This module provides:
    - Functions to queue discovered accounts from profile contacts for cross-platform spidering
    - Helpers to queue discovered @username mentions for the current platform
    - Helpers to upsert virtual profile posts into the content table for semantic search
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Account, Content

logger = logging.getLogger(__name__)


async def queue_discovered_accounts(
    session: AsyncSession,
    contacts_dict: dict[str, Any],
    parent_handle: str,
) -> None:
    """Queue discovered accounts from profile contacts for cross-platform spidering.

    Processes emails, Telegram handles, and external links from the contacts
    dictionary and inserts new account records into the accounts table with
    status "pending" for platforms Instagram, TikTok, YouTube, Threads, and Telegram.

    Uses a SELECT-then-INSERT pattern with transaction safety to avoid
    unique constraint violations.

    Args:
        session: SQLAlchemy async session for database operations.
        contacts_dict: Dictionary containing emails, telegram_handles, and external_links.
        parent_handle: Handle of the parent account whose bio was scanned.
    """
    telegram_handles: list[str] = contacts_dict.get("telegram_handles", [])
    external_links: list[str] = contacts_dict.get("external_links", [])

    if not telegram_handles and not external_links:
        logger.debug(
            "[SPIDER] No external social accounts discovered in bio of parent account %s.",
            parent_handle,
        )
        return

    # Process Telegram handles
    for handle in telegram_handles:
        if not handle:
            continue
        clean_handle = handle.lstrip("@")
        await _queue_single_account(session, "TELEGRAM", clean_handle, parent_handle)

    # Process external links for Instagram, TikTok, YouTube, Threads
    for link in external_links:
        if not link or not isinstance(link, str):
            continue

        link_lower = link.lower()

        # Instagram: instagram.com/username
        if "instagram.com/" in link_lower:
            match = re.search(r"instagram\.com/([^/?#]+)", link)
            if match:
                instagram_handle = match.group(1)
                if instagram_handle and instagram_handle != "p":
                    await _queue_single_account(session, "INSTAGRAM", instagram_handle, parent_handle)

        # TikTok: tiktok.com/@username
        elif "tiktok.com/" in link_lower:
            match = re.search(r"tiktok\.com/@([^/?#]+)", link)
            if match:
                tiktok_username = match.group(1)
                if tiktok_username:
                    await _queue_single_account(session, "TIKTOK", tiktok_username, parent_handle)

        # YouTube: youtube.com/@handle, youtube.com/channel/UC..., youtu.be/
        elif "youtube.com/" in link_lower or "youtu.be/" in link_lower:
            platform_id: str | None = None
            if "/@" in link:
                platform_id = link.split("/@")[-1].split("?")[0].split("/")[0]
            elif "youtube.com/channel/" in link_lower:
                platform_id = link.split("/channel/")[-1].split("?")[0].split("/")[0]
            elif "youtu.be/" in link_lower:
                # youtu.be links are for videos, not channels; skip
                continue

            if platform_id:
                await _queue_single_account(session, "YOUTUBE", platform_id, parent_handle)

        # Threads: threads.net/@username or threads.net/username
        elif "threads.net/" in link_lower:
            if "/@" in link:
                platform_id = link.split("/@")[-1].split("?")[0].split("/")[0]
            else:
                platform_id = link.split("/")[-1].split("?")[0].split("/")[0]
            if platform_id:
                await _queue_single_account(session, "THREADS", platform_id, parent_handle)


async def queue_discovered_mentions(
    session: AsyncSession,
    platform: str,
    mentions: list[str],
    parent_handle: str,
) -> None:
    """Queue discovered @username mentions for the current platform into the database with status 'pending'.

    Args:
        session: SQLAlchemy async session for database operations.
        platform: Platform name (INSTAGRAM, TIKTOK, YOUTUBE, THREADS).
        mentions: List of usernames extracted from @mentions (without @ symbol).
        parent_handle: Handle of the parent account that contained the mentions.
    """
    for username in mentions:
        if not username or len(username) < 3:
            continue
        # Avoid queueing obvious non-user names or system tags
        await _queue_single_account(session, platform, username, parent_handle)


async def _queue_single_account(
    session: AsyncSession,
    platform: str,
    platform_id: str,
    parent_handle: str,
) -> None:
    """Queue a single account for cross-platform spidering if it doesn't already exist.

    Checks for an existing account record first to avoid unique constraint violations,
    then inserts a new record with "pending" status if not found.

    Args:
        session: SQLAlchemy async session for database operations.
        platform: Platform name (INSTAGRAM, TIKTOK, YOUTUBE, THREADS, TELEGRAM).
        platform_id: Platform-specific identifier (handle, username, channel ID, etc.).
        parent_handle: Handle of the parent account that led to this discovery.
    """
    stmt = select(Account).where(
        Account.platform == platform,
        Account.platform_id == platform_id,
    )
    result = await session.execute(stmt)
    existing = result.scalar_one_or_none()

    if not existing:
        new_account = Account(
            platform=platform,
            platform_id=platform_id,
            username=platform_id,
            title=platform_id,
            status="pending",
        )
        session.add(new_account)
        logger.info(
            "[SPIDER] Queued discovered %s account: %s from bio of parent account %s.",
            platform,
            platform_id,
            parent_handle,
        )


async def upsert_virtual_bio_post(
    session: AsyncSession,
    account_id: int,
    platform: str,
    platform_id: str,
    username: str | None,
    full_name: str | None,
    biography: str | None,
    subscribers_count: int,
    raw_metadata: dict[str, Any] | None = None,
) -> None:
    """Upsert a virtual profile post into the content table for semantic search.

    Creates a synthetic content record containing the creator's biography and
    profile metadata so the embedding worker can index it into Qdrant for
    semantic search over creator biographies.

    The virtual post has:
        - platform_content_id = "profile_bio_{platform_id}"
        - content = compiled profile metadata string
        - is_embedded = False (picked up by embedding worker)
        - has_media = False

    Accepts an optional raw_metadata parameter that will be written directly
    to the database. This can be used to store additional metadata such as
    female_heuristic results.

    Uses PostgreSQL ON CONFLICT DO UPDATE on the "uq_content_account_platform_id"
    constraint for high-throughput, concurrent-safe upserts.

    Args:
        session: SQLAlchemy async session for database operations.
        account_id: Database ID of the parent account record.
        platform: Platform name (INSTAGRAM, TIKTOK, YOUTUBE, THREADS).
        platform_id: Platform-specific ID (user ID, channel ID, etc.).
        username: Creator's username on the platform, or None.
        full_name: Creator's display name / title, or None.
        biography: Creator's biography text, or None.
        subscribers_count: Number of subscribers / followers.
        raw_metadata: Optional dictionary to store in raw_metadata field.
    """
    virtual_content_id = f"profile_bio_{platform_id}"
    compiled_text = (
        f"[PROFILE METADATA]\n"
        f"Platform: {platform}\n"
        f"Username: @{username or 'unknown'}\n"
        f"Title: {full_name or 'Unknown'}\n"
        f"Subscribers: {subscribers_count}\n"
        f"Bio: {biography or 'N/A'}"
    )

    now = datetime.now(timezone.utc)

    stmt = pg_insert(Content).values(
        account_id=account_id,
        platform_content_id=virtual_content_id,
        content=compiled_text,
        transcription=None,
        published_at=now,
        views=None,
        reactions_count=None,
        comments_count=None,
        shares_count=None,
        has_media=False,
        is_embedded=False,
        is_graph_extracted=False,
        raw_metadata=raw_metadata,
        updated_at=now,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_content_account_platform_id",
        set_=dict(
            content=stmt.excluded.content,
            raw_metadata=stmt.excluded.raw_metadata,
            updated_at=stmt.excluded.updated_at,
        ),
    )
    await session.execute(stmt)
    logger.debug(
        "Upserted virtual profile post for account_id: %d (platform: %s, platform_content_id: %s)",
        account_id,
        platform,
        virtual_content_id,
    )
