"""
Async database helpers for discovered account queuing and virtual bio post upserts.

This module provides:
    - Functions to queue discovered accounts from profile contacts for cross-platform spidering
    - Helpers to queue discovered @username mentions for the current platform
    - Helpers to upsert virtual profile posts into the content table for semantic search
    - Centralized metadata collection pipeline for parsed accounts
"""

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Account, Content
from src.parser.creators.core.utils import parse_profile_contacts

logger = logging.getLogger(__name__)

# Well-known multi-link / link-in-bio domains for classification
LINK_IN_BIO_DOMAINS = {
    "linktr.ee",
    "taplink.cc",
    "beacons.ai",
    "msha.ke",
    "solo.to",
    "lu.ma",
    "lnk.bio",
    "campsite.bio",
    "carrd.co",
    "instabio.cc",
}


def generate_deterministic_id(platform: str, platform_id: str) -> int:
    """
    Generate a deterministic, positive 63-bit integer hash from platform and platform_id.
    Forces MSB to 0 to fit in signed 64-bit BigInt and to prevent collision with negative Telegram IDs.
    """
    key = f"{platform.upper()}:{platform_id.lower()}".encode("utf-8")
    hash_bytes = hashlib.sha256(key).digest()
    return int.from_bytes(hash_bytes[:8], byteorder="big") & 0x7FFFFFFFFFFFFFFF


def _parse_url_domain(url: str) -> str | None:
    """
    Parse URL and return the normalized domain (lowercase, without port).
    
    Args:
        url: The URL to parse.
        
    Returns:
        Normalized domain string or None if parsing fails.
    """
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return None
        return parsed.netloc.lower()
    except Exception:
        return None


def _extract_platform_info(url: str) -> tuple[str | None, str | None]:
    """
    Extract platform type and platform_id from a URL based on domain matching.
    
    Returns:
        Tuple of (platform, platform_id) or (None, None) if no match.
    """
    domain = _parse_url_domain(url)
    if not domain:
        return None, None
    
    url_lower = url.lower()
    
    # TELEGRAM: Filter out non-user actions
    if any(d in domain for d in ["t.me", "telegram.me", "telegram.dog", "tglink.ru"]):
        # Skip non-user actions
        non_user_paths = ["/addstickers", "/addemoji", "/joinchat", "/join"]
        if any(path in url_lower for path in non_user_paths):
            return None, None
        
        # Extract username from path
        match = re.search(r"t\.me/([^/?#]+)", url_lower)
        if match:
            username = match.group(1)
            if username:
                return "TELEGRAM", username
        return None, None
    
    # VK: vk.com, vk.ru, vkontakte.ru
    if any(d in domain for d in ["vk.com", "vk.ru", "vkontakte.ru"]):
        match = re.search(r"vk\.com/([^/?#]+)", url_lower)
        if match:
            return "VK", match.group(1)
        match = re.search(r"vk\.ru/([^/?#]+)", url_lower)
        if match:
            return "VK", match.group(1)
        match = re.search(r"vkontakte\.ru/([^/?#]+)", url_lower)
        if match:
            return "VK", match.group(1)
        return None, None
    
    # RUTUBE: rutube.ru
    if "rutube.ru" in domain:
        match = re.search(r"rutube\.ru/([^/?#]+)", url_lower)
        if match:
            return "RUTUBE", match.group(1)
        return None, None
    
    # YANDEX_DZEN: dzen.ru, zen.yandex.ru, zen.yandex.com
    if any(d in domain for d in ["dzen.ru", "zen.yandex.ru", "zen.yandex.com"]):
        match = re.search(r"zen\.yandex\.[a-z]+/([^/?#]+)", url_lower)
        if match:
            return "YANDEX_DZEN", match.group(1)
        match = re.search(r"dzen\.ru/([^/?#]+)", url_lower)
        if match:
            return "YANDEX_DZEN", match.group(1)
        return None, None
    
    # INSTAGRAM: instagram.com, instagr.am - skip post paths
    if any(d in domain for d in ["instagram.com", "instagr.am"]):
        match = re.search(r"instagram\.com/([^/?#]+)", url_lower)
        if match:
            username = match.group(1)
            if username and username != "p":
                return "INSTAGRAM", username
        match = re.search(r"instagr\.am/([^/?#]+)", url_lower)
        if match:
            return "INSTAGRAM", match.group(1)
        return None, None
    
    # TIKTOK: tiktok.com
    if "tiktok.com" in domain:
        match = re.search(r"tiktok\.com/@([^/?#]+)", url)
        if match:
            return "TIKTOK", match.group(1)
        return None, None
    
    # YOUTUBE: youtube.com, youtu.be
    if any(d in domain for d in ["youtube.com", "youtu.be"]):
        if "/@" in url:
            handle = url.split("/@")[-1].split("?")[0].split("/")[0]
            if handle:
                return "YOUTUBE", handle
        elif "youtube.com/channel/" in url_lower:
            channel_id = url.split("/channel/")[-1].split("?")[0].split("/")[0]
            if channel_id:
                return "YOUTUBE", channel_id
        # youtu.be links are for videos, not channels; skip
        return None, None
    
    # THREADS: threads.net
    if "threads.net" in domain:
        if "/@" in url:
            username = url.split("/@")[-1].split("?")[0].split("/")[0]
            if username:
                return "THREADS", username
        else:
            username = url.split("/")[-1].split("?")[0].split("/")[0]
            if username:
                return "THREADS", username
        return None, None
    
    # LINK_IN_BIO: Check if domain or subdomains match LINK_IN_BIO_DOMAINS
    for bio_domain in LINK_IN_BIO_DOMAINS:
        if domain == bio_domain or domain.endswith("." + bio_domain):
            return "LINK_IN_BIO", url
    
    # WEBSITE: Any other valid http/https link
    if url_lower.startswith("http://") or url_lower.startswith("https://"):
        return "WEBSITE", url
    
    return None, None


async def queue_discovered_accounts(
    session: AsyncSession,
    contacts_dict: dict[str, Any],
    parent_handle: str,
    status: str = "pending",
) -> None:
    """Queue discovered accounts from profile contacts for cross-platform spidering.

    Processes emails, Telegram handles, and external links from the contacts
    dictionary and inserts new account records into the accounts table with
    the specified status for multiple platforms including Telegram, VK, Rutube,
    Yandex Dzen, Instagram, TikTok, YouTube, Threads, and link-in-bio services.

    Uses domain-based parsing to classify external links into platform types
    and extracts platform-specific identifiers using regex patterns.

    Args:
        session: SQLAlchemy async session for database operations.
        contacts_dict: Dictionary containing emails, telegram_handles, and external_links.
        parent_handle: Handle of the parent account whose bio was scanned.
        status: Status to assign to newly discovered accounts (default: "pending").
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
        await _queue_single_account(session, "TELEGRAM", clean_handle, parent_handle, status)

    # Process external links with comprehensive platform classification
    for link in external_links:
        if not link or not isinstance(link, str):
            continue

        platform, platform_id = _extract_platform_info(link)
        
        if platform and platform_id and platform not in ("WEBSITE", "LINK_IN_BIO"):
            await _queue_single_account(session, platform, platform_id, parent_handle, status)


async def queue_discovered_mentions(
    session: AsyncSession,
    platform: str,
    mentions: list[str],
    parent_handle: str,
    status: str = "pending",
) -> None:
    """Queue discovered @username mentions for the current platform into the database.

    Args:
        session: SQLAlchemy async session for database operations.
        platform: Platform name (INSTAGRAM, TIKTOK, YOUTUBE, THREADS).
        mentions: List of usernames extracted from @mentions (without @ symbol).
        parent_handle: Handle of the parent account that contained the mentions.
        status: Status to assign to newly discovered accounts (default: "pending").
    """
    for username in mentions:
        if not username or len(username) < 3:
            continue
        # Avoid queueing obvious non-user names or system tags
        await _queue_single_account(session, platform, username, parent_handle, status)


async def _queue_single_account(
    session: AsyncSession,
    platform: str,
    platform_id: str,
    parent_handle: str,
    status: str = "pending",
) -> None:
    """Queue a single account for cross-platform spidering if it doesn't already exist.

    Checks for an existing account record first to avoid unique constraint violations,
    then inserts a new record with the specified status if not found.

    Args:
        session: SQLAlchemy async session for database operations.
        platform: Platform name (INSTAGRAM, TIKTOK, YOUTUBE, THREADS, TELEGRAM, VK, etc.).
        platform_id: Platform-specific identifier (handle, username, channel ID, etc.).
        parent_handle: Handle of the parent account that led to this discovery.
        status: Status to assign to the account (default: "pending").
    """
    stmt = select(Account).where(
        Account.platform == platform,
        Account.platform_id == platform_id,
    )
    result = await session.execute(stmt)
    existing = result.scalar_one_or_none()

    if not existing:
        generated_id = generate_deterministic_id(platform, platform_id)
        try:
            new_account = Account(
                id=generated_id,
                platform=platform,
                platform_id=platform_id,
                username=platform_id,
                title=platform_id,
                status=status,
            )
            session.add(new_account)
            await session.flush()
            logger.info(
                "[SPIDER] Queued discovered %s account: %s from bio of parent account %s.",
                platform,
                platform_id,
                parent_handle,
            )
        except IntegrityError as e:
            # Handle race condition where another process inserted the same account
            logger.debug(
                "Integrity error while queuing %s account %s (likely duplicate): %s",
                platform,
                platform_id,
                e,
            )


async def update_account_profile_metadata(
    session: AsyncSession,
    account_id: int,
    biography: str | None,
    subscribers_count: int | None = None,
    is_author_blog: bool | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Centralized async helper to update account profile metadata and trigger cross-platform discovery.
    
    Extracts contact information from biography, structures all parsed data into raw_metadata,
    updates the Account record, and queues discovered accounts for cross-platform spidering.
    
    Args:
        session: SQLAlchemy async session for database operations.
        account_id: Database ID of the account to update.
        biography: Creator's biography text to extract contacts from.
        subscribers_count: Number of subscribers / followers (optional).
        is_author_blog: Whether the account is identified as an author blog (optional).
        extra_meta: Additional metadata dictionary to merge into raw_metadata (optional).
        
    Returns:
        The built raw_metadata dictionary for use by platform-specific parsers.
    """
    # Extract contacts from biography using the imported function
    contacts: dict[str, Any] = {}
    if biography:
        contacts = parse_profile_contacts(biography, None)
    
    # Build raw_metadata dictionary
    raw_metadata: dict[str, Any] = {
        "contacts": {
            "emails": contacts.get("emails", []),
            "telegram_handles": contacts.get("telegram_handles", []),
            "external_links": contacts.get("external_links", []),
        },
    }
    
    # Add location if available in extra_meta
    if extra_meta:
        if "location" in extra_meta:
            raw_metadata["location"] = extra_meta["location"]
        if "language" in extra_meta:
            raw_metadata["language"] = extra_meta["language"]
        # Merge any additional extra_meta fields
        for key, value in extra_meta.items():
            if key not in raw_metadata:
                raw_metadata[key] = value
    
    # Select Account record by account_id
    stmt = select(Account).where(Account.id == account_id)
    result = await session.execute(stmt)
    account = result.scalar_one_or_none()
    
    if account:
        # Update Account fields
        if biography is not None:
            account.description = biography
        if subscribers_count is not None:
            account.subscribers_count = subscribers_count
        if is_author_blog is not None:
            account.is_author_blog = is_author_blog
        
        account.raw_metadata = raw_metadata
        await session.flush()
        
        logger.info(
            "Updated profile metadata for account_id: %d",
            account_id,
        )
        
        # Trigger cross-platform discovery using extracted contacts
        if contacts:
            # Get parent handle from account
            parent_handle = account.username or account.platform_id or str(account_id)
            await queue_discovered_accounts(session, contacts, parent_handle)
    else:
        logger.warning(
            "Account with id %d not found for metadata update",
            account_id,
        )
    
    return raw_metadata


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
