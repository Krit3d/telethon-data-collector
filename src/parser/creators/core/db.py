"""
Async database helpers for discovered account queuing and virtual bio post upserts.

This module provides:
    - Functions to queue discovered accounts from profile contacts for cross-platform spidering
    - Helpers to queue discovered @username mentions for the current platform
    - Helpers to upsert virtual profile posts into the content table for semantic search
    - Centralized metadata collection pipeline for parsed accounts
    - Generic async functions for account upsertion, duplicate merging, and bulk content operations
"""

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any

from urllib.parse import urlparse

from sqlalchemy import select, update, delete, or_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Account, Content, Comment
from src.parser.creators.core.contacts import parse_profile_contacts, compile_author_metadata_dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported platforms taxonomy
# ---------------------------------------------------------------------------

SUPPORTED_PLATFORMS: frozenset[str] = frozenset({
    "TELEGRAM",
    "VK",
    "RUTUBE",
    "YANDEX_DZEN",
    "INSTAGRAM",
    "TIKTOK",
    "YOUTUBE",
    "THREADS",
    "LINK_IN_BIO",
    "WEBSITE",
})

# Well-known multi-link / link-in-bio domains for classification
LINK_IN_BIO_DOMAINS: frozenset[str] = frozenset({
    "linktr.ee",
    "taplink.cc",
    "beacons.ai",
    "msha.ke",
    "solo.to",
    "lu.ma",
    "lnk.bio",
    "campsite.bio",
    "carrd.co",
})

# Telegram domains for platform detection
TELEGRAM_DOMAINS: frozenset[str] = frozenset({
    "t.me",
    "telegram.me",
    "telegram.dog",
    "tglink.ru",
})

# VK domains for platform detection
VK_DOMAINS: frozenset[str] = frozenset({
    "vk.com",
    "vk.ru",
    "vkontakte.ru",
})

# Yandex Dzen domains for platform detection
YANDEX_DZEN_DOMAINS: frozenset[str] = frozenset({
    "dzen.ru",
    "zen.yandex.ru",
    "zen.yandex.com",
})

# Instagram domains for platform detection
INSTAGRAM_DOMAINS: frozenset[str] = frozenset({
    "instagram.com",
    "instagr.am",
})

# Non-user action paths to filter out for Telegram
TELEGRAM_NON_USER_PATHS: frozenset[str] = frozenset({
    "/addstickers",
    "/addemoji",
    "/joinchat",
    "/join",
})


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

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

    Supports the full SUPPORTED_PLATFORMS taxonomy:
    - TELEGRAM: t.me, telegram.me, telegram.dog, tglink.ru
    - VK: vk.com, vk.ru, vkontakte.ru
    - RUTUBE: rutube.ru
    - YANDEX_DZEN: dzen.ru, zen.yandex.ru, zen.yandex.com
    - INSTAGRAM: instagram.com, instagr.am
    - TIKTOK: tiktok.com
    - YOUTUBE: youtube.com, youtu.be
    - THREADS: threads.net
    - LINK_IN_BIO: Known multi-link domains
    - WEBSITE: Any other valid http/https URL

    Returns:
        Tuple of (platform, platform_id) or (None, None) if no match.
    """
    domain = _parse_url_domain(url)
    if not domain:
        return None, None

    url_lower = url.lower()

    # TELEGRAM: Filter out non-user actions
    if any(d in domain for d in TELEGRAM_DOMAINS):
        # Skip non-user actions
        if any(path in url_lower for path in TELEGRAM_NON_USER_PATHS):
            return None, None

        # Extract username from path
        match = re.search(r"t\.me/([^/?#]+)", url_lower)
        if match:
            username = match.group(1)
            if username:
                return "TELEGRAM", username
        return None, None

    # VK: vk.com, vk.ru, vkontakte.ru
    if any(d in domain for d in VK_DOMAINS):
        # Try each domain pattern
        for pattern in [r"vk\.com/([^/?#]+)", r"vk\.ru/([^/?#]+)", r"vkontakte\.ru/([^/?#]+)"]:
            match = re.search(pattern, url_lower)
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
    if any(d in domain for d in YANDEX_DZEN_DOMAINS):
        # Try zen.yandex.* pattern first
        match = re.search(r"zen\.yandex\.[a-z]+/([^/?#]+)", url_lower)
        if match:
            return "YANDEX_DZEN", match.group(1)
        # Then dzen.ru pattern
        match = re.search(r"dzen\.ru/([^/?#]+)", url_lower)
        if match:
            return "YANDEX_DZEN", match.group(1)
        return None, None

    # INSTAGRAM: instagram.com, instagr.am - skip post paths
    if any(d in domain for d in INSTAGRAM_DOMAINS):
        # Skip paths like /p/ (posts)
        if "/p/" in url_lower:
            return None, None
        match = re.search(r"instagram\.com/([^/?#]+)", url_lower)
        if match:
            return "INSTAGRAM", match.group(1)
        match = re.search(r"instagr\.am/([^/?#]+)", url_lower)
        if match:
            return "INSTAGRAM", match.group(1)
        return None, None

    # TIKTOK: tiktok.com - extract handle starting with @
    if "tiktok.com" in domain:
        match = re.search(r"tiktok\.com/@([^/?#]+)", url)
        if match:
            return "TIKTOK", match.group(1)
        return None, None

    # YOUTUBE: youtube.com, youtu.be
    if any(d in domain for d in ["youtube.com", "youtu.be"]):
        # Extract @username
        if "/@" in url:
            handle = url.split("/@")[-1].split("?")[0].split("/")[0]
            if handle:
                return "YOUTUBE", handle
        # Extract channel ID
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

    # WEBSITE: Any other valid http/https URL
    parsed_url = urlparse(url)
    if parsed_url.scheme in ("http", "https"):
        return "WEBSITE", url

    return None, None


# ---------------------------------------------------------------------------
# Generic async duplicate-aware account upsertion
# ---------------------------------------------------------------------------

async def upsert_and_deduplicate_account(
    session: AsyncSession,
    platform: str,
    platform_id: str,
    username: str | None,
    title: str,
    description: str | None,
    subscribers_count: int | None,
    status: str,
) -> int:
    """
    Upsert an account with duplicate detection and merging.

    This function:
    1. Queries matching accounts by BOTH platform_id and username for the given platform
    2. If no account exists, creates a new one and returns its ID
    3. If exactly one account exists, updates its fields and returns its ID
    4. If multiple accounts exist (duplicates), finds the primary account
       (preferring numeric platform_id over username-based platform_id),
       merges duplicates into it, and returns the primary ID

    Args:
        session: SQLAlchemy async session for database operations
        platform: Platform name from SUPPORTED_PLATFORMS
        platform_id: Platform-specific identifier (handle, channel ID, etc.)
        username: Account username on the platform, or None
        title: Account title/name
        description: Account description/bio, or None
        subscribers_count: Number of subscribers/followers, or None
        status: Account status (pending, processing, parsed, rejected)

    Returns:
        The account ID (primary ID after potential deduplication)
    """
    # Validate platform
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"Unsupported platform: {platform}. Must be one of {SUPPORTED_PLATFORMS}")

    # Query for existing accounts matching platform_id or username
    conditions = []
    if platform_id:
        conditions.append(
            (Account.platform == platform) & (Account.platform_id == platform_id)
        )
    if username:
        conditions.append(
            (Account.platform == platform) & (Account.username == username)
        )

    if not conditions:
        # No platform_id or username provided, cannot match
        generated_id = generate_deterministic_id(platform, platform_id or username or title)
        new_account = Account(
            id=generated_id,
            platform=platform,
            platform_id=platform_id or "",
            username=username,
            title=title,
            description=description,
            subscribers_count=subscribers_count,
            status=status,
        )
        session.add(new_account)
        await session.flush()
        return generated_id

    # Build OR query
    stmt = select(Account).where(
        Account.platform == platform,
        or_(*conditions),
    )
    result = await session.execute(stmt)
    existing_accounts = list(result.scalars().all())

    if not existing_accounts:
        # No existing account, create new one
        generated_id = generate_deterministic_id(platform, platform_id or username or title)
        new_account = Account(
            id=generated_id,
            platform=platform,
            platform_id=platform_id or "",
            username=username,
            title=title,
            description=description,
            subscribers_count=subscribers_count,
            status=status,
        )
        session.add(new_account)
        await session.flush()
        logger.info(
            "Created new account: platform=%s, platform_id=%s, username=%s, id=%d",
            platform,
            platform_id,
            username,
            generated_id,
        )
        return generated_id

    if len(existing_accounts) == 1:
        # Exactly one account exists, update it
        account = existing_accounts[0]
        account.platform_id = platform_id or account.platform_id
        account.username = username or account.username
        account.title = title
        account.description = description if description is not None else account.description
        account.subscribers_count = (
            subscribers_count if subscribers_count is not None else account.subscribers_count
        )
        account.status = status
        await session.flush()
        logger.info(
            "Updated existing account: platform=%s, platform_id=%s, id=%d",
            platform,
            platform_id,
            account.id,
        )
        return account.id

    # Multiple accounts exist - need to deduplicate
    # Find primary account: prefer numeric platform_id over username-based platform_id
    primary_account = None
    for account in existing_accounts:
        if account.platform_id and account.platform_id.isdigit():
            primary_account = account
            break

    # If no account with numeric platform_id found, use the first one
    if primary_account is None:
        primary_account = existing_accounts[0]

    primary_id = primary_account.id

    # Update primary account with the most recent data
    primary_account.platform_id = platform_id or primary_account.platform_id
    primary_account.username = username or primary_account.username
    primary_account.title = title
    primary_account.description = description if description is not None else primary_account.description
    primary_account.subscribers_count = (
        subscribers_count if subscribers_count is not None else primary_account.subscribers_count
    )
    primary_account.status = status

    # Reassign content and comments from duplicate accounts to primary account
    duplicate_ids = [acc.id for acc in existing_accounts if acc.id != primary_id]
    if duplicate_ids:
        # Update content.account_id for all content from duplicate accounts
        await session.execute(
            update(Content)
            .where(Content.account_id.in_(duplicate_ids))
            .values(account_id=primary_id)
        )

        # Update comments.account_id for all comments from duplicate accounts
        await session.execute(
            update(Comment)
            .where(Comment.account_id.in_(duplicate_ids))
            .values(account_id=primary_id)
        )

        # Delete duplicate accounts
        await session.execute(
            delete(Account).where(Account.id.in_(duplicate_ids))
        )

        logger.info(
            "Merged %d duplicate accounts into primary account %d for platform %s",
            len(duplicate_ids),
            primary_id,
            platform,
        )

    await session.flush()
    return primary_id


# ---------------------------------------------------------------------------
# Account profile metadata update with cross-platform discovery
# ---------------------------------------------------------------------------

async def update_account_profile_metadata(
    session: AsyncSession,
    account_id: int,
    platform: str,
    biography: str | None,
    external_url: str | None = None,
    location: str | None = None,
    language: str | None = None,
    geo_data: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
    category: str | None = None,
    raw_profile_payload: dict[str, Any] | None = None,
    subscribers_count: int | None = None,
) -> dict[str, Any]:
    """
    Update account profile metadata and trigger cross-platform discovery.

    This function:
    1. Uses parse_profile_contacts(biography, external_url) to extract contacts
    2. Compiles an OpenSPG-compliant metadata dictionary using compile_author_metadata_dict
    3. Updates the Account's raw_metadata and description with compiled values
    4. Automatically runs queue_discovered_accounts to queue newly discovered accounts as "pending"
    5. Returns the compiled metadata dictionary

    Args:
        session: SQLAlchemy async session for database operations
        account_id: Database ID of the account to update
        platform: Platform name from SUPPORTED_PLATFORMS
        biography: Creator's biography text to extract contacts from
        external_url: External URL from profile, or None
        location: Human-readable location string, or None
        language: Language code or label, or None
        geo_data: Structured geographic data dictionary, or None
        extra_meta: Additional metadata dictionary to merge, or None
        category: Optional category string for the account
        raw_profile_payload: Optional raw JSON payload from the platform API
        subscribers_count: Optional subscriber count for metrics_history

    Returns:
        The compiled metadata dictionary
    """
    # Extract contacts from biography and external_url
    contacts: dict[str, Any] = {}
    if biography or external_url:
        contacts = parse_profile_contacts(biography, external_url)

    # Get account to access username/platform_id for profile link compilation
    stmt = select(Account).where(Account.id == account_id)
    result = await session.execute(stmt)
    account = result.scalar_one_or_none()

    if not account:
        logger.warning("Account with id %d not found for metadata update", account_id)
        return {}

    # Compile OpenSPG-compliant metadata with new parameters
    username = account.username or account.platform_id
    compiled_metadata = compile_author_metadata_dict(
        platform=platform,
        username=username,
        biography=biography,
        contacts_dict=contacts,
        extra_links=contacts.get("external_links", []),
        location=location,
        language=language,
        geo_data=geo_data,
        category=category,
        raw_profile_payload=raw_profile_payload,
    )

    # Build metrics_history block if subscribers_count is provided
    if subscribers_count is not None:
        metrics_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "subscribers_count": subscribers_count,
        }
        # Initialize or append to metrics_history
        if "metrics_history" not in compiled_metadata:
            compiled_metadata["metrics_history"] = []
        compiled_metadata["metrics_history"].append(metrics_entry)

    # Merge extra_meta if provided
    if extra_meta:
        for key, value in extra_meta.items():
            if key not in compiled_metadata or not compiled_metadata[key]:
                compiled_metadata[key] = value

    # Update account fields
    account.description = biography if biography is not None else account.description
    account.raw_metadata = compiled_metadata
    if subscribers_count is not None:
        account.subscribers_count = subscribers_count
    await session.flush()

    logger.info("Updated profile metadata for account_id: %d", account_id)

    # Trigger cross-platform discovery using extracted contacts
    if contacts:
        parent_handle = account.username or account.platform_id or str(account_id)
        # Get parent category for inheritance
        parent_category = category
        if not parent_category:
            raw_metadata = account.raw_metadata
            if raw_metadata is not None:
                parent_category = raw_metadata.get("category")
        await queue_discovered_accounts(
            session, contacts, parent_handle, category=parent_category
        )

    return compiled_metadata


# ---------------------------------------------------------------------------
# Bulk content upsert
# ---------------------------------------------------------------------------

async def bulk_upsert_content(
    session: AsyncSession,
    content_values: list[dict[str, Any]],
) -> None:
    """
    Execute a bulk PostgreSQL insert of Content items with ON CONFLICT DO UPDATE.

    This function performs a bulk upsert on the content table using the
    ON CONFLICT (account_id, platform_content_id) DO UPDATE clause.

    Updated fields on conflict:
    - content
    - transcription
    - views
    - reactions_count
    - comments_count
    - raw_metadata
    - updated_at

    Args:
        session: SQLAlchemy async session for database operations
        content_values: List of dictionaries containing content data.
            Each dict should have keys matching Content model fields.
            Required keys: account_id, platform_content_id, published_at
            Optional keys: content, transcription, views, reactions_count,
                          comments_count, raw_metadata, etc.
    """
    if not content_values:
        return

    now = datetime.now(timezone.utc)

    # Prepare values for insert
    insert_values = []
    for values in content_values:
        # Ensure required fields are present
        if "account_id" not in values or "platform_content_id" not in values:
            logger.warning("Skipping content value missing required fields: %s", values)
            continue

        # Set defaults
        prepared = {
            "account_id": values["account_id"],
            "platform_content_id": values["platform_content_id"],
            "content": values.get("content"),
            "transcription": values.get("transcription"),
            "published_at": values.get("published_at", now),
            "views": values.get("views"),
            "reactions_count": values.get("reactions_count"),
            "comments_count": values.get("comments_count"),
            "shares_count": values.get("shares_count"),
            "raw_metadata": values.get("raw_metadata"),
            "updated_at": now,
            "created_at": now,
            "is_embedded": values.get("is_embedded", False),
            "is_graph_extracted": values.get("is_graph_extracted", False),
            "has_media": values.get("has_media", False),
        }
        insert_values.append(prepared)

    if not insert_values:
        return

    # Bulk insert with ON CONFLICT DO UPDATE
    stmt = pg_insert(Content).values(insert_values)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_content_account_platform_id",
        set_=dict(
            content=stmt.excluded.content,
            transcription=stmt.excluded.transcription,
            views=stmt.excluded.views,
            reactions_count=stmt.excluded.reactions_count,
            comments_count=stmt.excluded.comments_count,
            raw_metadata=stmt.excluded.raw_metadata,
            updated_at=stmt.excluded.updated_at,
        ),
    )
    await session.execute(stmt)
    await session.flush()

    logger.debug("Bulk upserted %d content items", len(insert_values))


# ---------------------------------------------------------------------------
# Account queuing for cross-platform spidering
# ---------------------------------------------------------------------------

async def queue_discovered_accounts(
    session: AsyncSession,
    contacts_dict: dict[str, Any],
    parent_handle: str,
    status: str = "pending",
    category: str | None = None,
) -> None:
    """
    Queue discovered accounts from profile contacts for cross-platform spidering.

    Processes emails, Telegram handles (channels only, not personal), and
    external links from the contacts dictionary and inserts new account records
    into the accounts table with the specified status for multiple platforms
    including Telegram, VK, Rutube, Yandex Dzen, Instagram, TikTok, YouTube,
    Threads, and link-in-bio services.

    Uses domain-based parsing to classify external links into platform types
    and extracts platform-specific identifiers using regex patterns.

    Implements category inheritance: newly queued accounts inherit the parent's
    category in their raw_metadata.

    Args:
        session: SQLAlchemy async session for database operations
        contacts_dict: Dictionary containing emails, telegram_handles, external_links,
            and external_platforms
        parent_handle: Handle of the parent account whose bio was scanned
        status: Status to assign to newly discovered accounts (default: "pending")
        category: Optional category to inherit to newly queued accounts
    """
    # Use telegram_channels from contacts dict (not telegram_handles)
    # This prevents personal advertising profiles from cluttering the queue
    telegram_channels: list[str] = []
    contacts = contacts_dict.get("contacts", {})
    if isinstance(contacts, dict):
        telegram_channels = contacts.get("telegram_channels", [])
    external_links: list[str] = contacts_dict.get("external_links", [])
    external_platforms: dict[str, str] = contacts_dict.get("external_platforms", {})

    if not telegram_channels and not external_links and not external_platforms:
        logger.debug(
            "[SPIDER] No external social accounts discovered in bio of parent account %s.",
            parent_handle,
        )
        return

    # Process Telegram channels (NOT personal handles)
    for handle in telegram_channels:
        if not handle:
            continue
        clean_handle = handle.lstrip("@")
        await _queue_single_account(
            session, "TELEGRAM", clean_handle, parent_handle, status, category
        )

    # Process external platforms from structured external_platforms map
    platform_mapping = {
        "vk": "VK",
        "youtube": "YOUTUBE",
        "threads": "THREADS",
        "tiktok": "TIKTOK",
    }
    for platform_slug, handle in external_platforms.items():
        if not handle:
            continue
        platform_name = platform_mapping.get(platform_slug.lower())
        if platform_name:
            await _queue_single_account(
                session, platform_name, handle, parent_handle, status, category
            )

    # Process external links with comprehensive platform classification
    for link in external_links:
        if not link or not isinstance(link, str):
            continue

        platform, platform_id = _extract_platform_info(link)

        if platform and platform_id and platform not in ("WEBSITE", "LINK_IN_BIO"):
            await _queue_single_account(
                session, platform, platform_id, parent_handle, status, category
            )


async def queue_discovered_mentions(
    session: AsyncSession,
    platform: str,
    mentions: list[str],
    parent_handle: str,
    status: str = "pending",
) -> None:
    """
    Queue discovered @username mentions for the current platform into the database.

    Args:
        session: SQLAlchemy async session for database operations
        platform: Platform name (INSTAGRAM, TIKTOK, YOUTUBE, THREADS)
        mentions: List of usernames extracted from @mentions (without @ symbol)
        parent_handle: Handle of the parent account that contained the mentions
        status: Status to assign to newly discovered accounts (default: "pending")
    """
    for username in mentions:
        if not username or len(username) < 3:
            continue
        # Avoid queuing obvious non-user names or system tags
        await _queue_single_account(session, platform, username, parent_handle, status)


async def _queue_single_account(
    session: AsyncSession,
    platform: str,
    platform_id: str,
    parent_handle: str,
    status: str = "pending",
    category: str | None = None,
) -> None:
    """
    Queue a single account for cross-platform spidering if it doesn't already exist.

    Checks for an existing account record first to avoid unique constraint violations,
    then inserts a new record with the specified status if not found.
    Implements category inheritance from parent account.

    Args:
        session: SQLAlchemy async session for database operations
        platform: Platform name from SUPPORTED_PLATFORMS
        platform_id: Platform-specific identifier (handle, username, channel ID, etc.)
        parent_handle: Handle of the parent account that led to this discovery
        status: Status to assign to the account (default: "pending")
        category: Optional category to inherit to the new account
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
            # Build raw_metadata with category inheritance
            raw_metadata = {}
            if category is not None:
                raw_metadata["category"] = category

            new_account = Account(
                id=generated_id,
                platform=platform,
                platform_id=platform_id,
                username=platform_id,
                title=platform_id,
                status=status,
                raw_metadata=raw_metadata if raw_metadata else None,
            )
            session.add(new_account)
            await session.flush()
            logger.info(
                "[SPIDER] Queued discovered %s account: %s from bio of parent account %s (category: %s).",
                platform,
                platform_id,
                parent_handle,
                category or "none",
            )
        except IntegrityError as e:
            # Handle race condition where another process inserted the same account
            logger.debug(
                "Integrity error while queuing %s account %s (likely duplicate): %s",
                platform,
                platform_id,
                e,
            )


# ---------------------------------------------------------------------------
# Virtual bio post upsert for semantic search
# ---------------------------------------------------------------------------

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
    """
    Upsert a virtual profile post into the content table for semantic search.

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
        session: SQLAlchemy async session for database operations
        account_id: Database ID of the parent account record
        platform: Platform name (INSTAGRAM, TIKTOK, YOUTUBE, THREADS)
        platform_id: Platform-specific ID (user ID, channel ID, etc.)
        username: Creator's username on the platform, or None
        full_name: Creator's display name / title, or None
        biography: Creator's biography text, or None
        subscribers_count: Number of subscribers / followers
        raw_metadata: Optional dictionary to store in raw_metadata field
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
