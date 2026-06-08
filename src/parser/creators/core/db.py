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
from src.parser.creators.core.contacts import parse_profile_contacts, compile_author_metadata, URL_PATTERN
from src.parser.creators.core.schemas import (
    AccountMetadata,
    MetricsEntry,
    ContentMetadata,
    Contacts,
    ExternalPlatforms,
)

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

    # TELEGRAM: Handle both standard usernames and invite links
    if any(d in domain for d in TELEGRAM_DOMAINS):
        # Filter out non-user actions (except invite links)
        if any(path in url_lower for path in TELEGRAM_NON_USER_PATHS):
            return None, None

        # Check for invite links: t.me/+hash or t.me/joinchat/hash
        # Preserve original case for invite hashes (they are case-sensitive)
        plus_match = re.search(r"t\.me/\+([A-Za-z0-9_\-]{6,})", url)
        if plus_match:
            return "TELEGRAM", "+" + plus_match.group(1)

        joinchat_match = re.search(r"t\.me/joinchat/([A-Za-z0-9_\-]{6,})", url)
        if joinchat_match:
            # Standardize on +hash format for consistency
            return "TELEGRAM", "+" + joinchat_match.group(1)

        # Standard username extraction (supports optional /s/ prefix for web-view links)
        match = re.search(r"t\.me/(?:s/)?([^/?#]+)", url_lower)
        if match:
            username = match.group(1)
            if username and username not in ("joinchat", "join"):
                # Filter out Telegram bot usernames (case-insensitive)
                # Bot usernames end with "bot" (e.g., "mybot", "newsbot")
                username_lower = username.lower()
                if username_lower.endswith("bot"):
                    return None, None
                return "TELEGRAM", username
        return None, None

    # VK: vk.com, vk.ru, vkontakte.ru
    if any(d in domain for d in VK_DOMAINS):
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
        match = re.search(r"zen\.yandex\.[a-z]+/([^/?#]+)", url_lower)
        if match:
            return "YANDEX_DZEN", match.group(1)
        match = re.search(r"dzen\.ru/([^/?#]+)", url_lower)
        if match:
            return "YANDEX_DZEN", match.group(1)
        return None, None

    # INSTAGRAM: instagram.com, instagr.am - skip post paths
    if any(d in domain for d in INSTAGRAM_DOMAINS):
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
        if "/@" in url:
            handle = url.split("/@")[-1].split("?")[0].split("/")[0]
            if handle:
                return "YOUTUBE", handle
        elif "youtube.com/channel/" in url_lower:
            channel_id = url.split("/channel/")[-1].split("?")[0].split("/")[0]
            if channel_id:
                return "YOUTUBE", channel_id
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


def _convert_dict_to_account_metadata(contacts_dict: dict[str, Any]) -> AccountMetadata:
    """
    Convert a legacy contacts dictionary to AccountMetadata model.

    This helper function handles the conversion from the old dictionary format
    returned by parse_profile_contacts() to the new AccountMetadata Pydantic model.

    Args:
        contacts_dict: Dictionary with keys like 'emails', 'telegram_handles',
                      'external_links', 'external_platforms'.

    Returns:
        AccountMetadata model instance.
    """
    emails = contacts_dict.get("emails", [])
    telegram_handles = contacts_dict.get("telegram_handles", [])
    external_links = contacts_dict.get("external_links", [])
    external_platforms_dict = contacts_dict.get("external_platforms", {})

    # Classify Telegram handles (using simple heuristic: treat all as channels for now)
    # Filter out bot usernames (case-insensitive): usernames ending with "bot"
    telegram_channels = []
    for handle in telegram_handles:
        if not handle:
            continue
        username = handle.lstrip("@")
        # Bot usernames end with "bot" or "_bot" (case-insensitive)
        # Note: endswith("bot") catches both "bot" and "_bot" endings
        if username.lower().endswith("bot"):
            continue
        telegram_channels.append(username)
    telegram_personal: list[str] = []

    # Build ExternalPlatforms
    external_platforms = ExternalPlatforms(
        vk=external_platforms_dict.get("vk"),
        youtube=external_platforms_dict.get("youtube"),
        threads=external_platforms_dict.get("threads"),
        tiktok=external_platforms_dict.get("tiktok"),
    )

    # Build Contacts
    contacts = Contacts(
        emails=[e for e in emails if e],
        telegram_channels=telegram_channels,
        telegram_personal=telegram_personal,
    )

    # Build AccountMetadata
    return AccountMetadata(
        contacts=contacts,
        external_platforms=external_platforms,
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )


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
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"Unsupported platform: {platform}. Must be one of {SUPPORTED_PLATFORMS}")

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

    stmt = select(Account).where(
        Account.platform == platform,
        or_(*conditions),
    )
    result = await session.execute(stmt)
    existing_accounts = list(result.scalars().all())

    if not existing_accounts:
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

    primary_account = None
    for account in existing_accounts:
        if account.platform_id and account.platform_id.isdigit():
            primary_account = account
            break

    if primary_account is None:
        primary_account = existing_accounts[0]

    primary_id = primary_account.id

    primary_account.platform_id = platform_id or primary_account.platform_id
    primary_account.username = username or primary_account.username
    primary_account.title = title
    primary_account.description = description if description is not None else primary_account.description
    primary_account.subscribers_count = (
        subscribers_count if subscribers_count is not None else primary_account.subscribers_count
    )
    primary_account.status = status

    duplicate_ids = [acc.id for acc in existing_accounts if acc.id != primary_id]
    if duplicate_ids:
        await session.execute(
            update(Content)
            .where(Content.account_id.in_(duplicate_ids))
            .values(account_id=primary_id)
        )

        await session.execute(
            update(Comment)
            .where(Comment.account_id.in_(duplicate_ids))
            .values(account_id=primary_id)
        )

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
    2. Compiles an OpenSPG-compliant metadata using compile_author_metadata
       (returns an AccountMetadata Pydantic model)
    3. Updates the Account's raw_metadata with compiled metadata (exclude_none=True)
    4. Automatically runs queue_discovered_accounts to queue newly discovered accounts as "pending"
    5. Returns the compiled metadata as a dictionary

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
        The compiled metadata dictionary (with None values excluded)
    """
    contacts: dict[str, Any] = {}
    if biography or external_url:
        contacts = parse_profile_contacts(biography, external_url)

    stmt = select(Account).where(Account.id == account_id)
    result = await session.execute(stmt)
    account = result.scalar_one_or_none()

    if not account:
        logger.warning("Account with id %d not found for metadata update", account_id)
        return {}

    username = account.username or account.platform_id

    # Resolve category with inheritance logic
    resolved_category = category
    if resolved_category is None:
        # Check if account has raw_metadata with category
        if account.raw_metadata and isinstance(account.raw_metadata, dict):
            resolved_category = account.raw_metadata.get("category")
        # If still None, fall back to "unknown"
        if resolved_category is None:
            resolved_category = "unknown"

    compiled_metadata = compile_author_metadata(
        platform=platform,
        username=username,
        biography=biography,
        contacts_dict=contacts,
        extra_links=contacts.get("external_links", []),
        location=location,
        language=language,
        geo_data=geo_data,
        category=resolved_category,
        raw_profile_payload=raw_profile_payload,
    )

    if subscribers_count is not None:
        metrics_entry = MetricsEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            subscribers_count=subscribers_count,
            posts_count=None,
        )
        compiled_metadata.metrics_history.append(metrics_entry)

    if extra_meta:
        for key, value in extra_meta.items():
            if hasattr(compiled_metadata, key):
                current_value = getattr(compiled_metadata, key)
                if not current_value:
                    setattr(compiled_metadata, key, value)
            else:
                logger.debug(
                    "Extra meta key '%s' not found in AccountMetadata model, skipping",
                    key,
                )

    account.description = biography if biography is not None else account.description
    account.raw_metadata = compiled_metadata.model_dump(exclude_none=True)

    if subscribers_count is not None:
        account.subscribers_count = subscribers_count
    await session.flush()

    logger.info("Updated profile metadata for account_id: %d", account_id)

    if contacts:
        parent_handle = account.username or account.platform_id or str(account_id)
        await queue_discovered_accounts(
            session, compiled_metadata, parent_handle, status="pending", category=resolved_category
        )

    return compiled_metadata.model_dump(exclude_none=True)


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

    insert_values = []
    for values in content_values:
        if "account_id" not in values or "platform_content_id" not in values:
            logger.warning("Skipping content value missing required fields: %s", values)
            continue

        raw_metadata = values.get("raw_metadata")
        if raw_metadata is not None:
            raw_metadata = _clean_content_raw_metadata(raw_metadata)

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
            "raw_metadata": raw_metadata,
            "updated_at": now,
            "created_at": now,
            "is_embedded": values.get("is_embedded", False),
            "is_graph_extracted": values.get("is_graph_extracted", False),
            "has_media": values.get("has_media", False),
        }
        insert_values.append(prepared)

    if not insert_values:
        return

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


def _clean_content_raw_metadata(raw_metadata: dict[str, Any] | ContentMetadata | Any) -> dict[str, Any] | None:
    """
    Clean and serialize raw_metadata to prevent database bloat.

    This helper function:
    1. If raw_metadata is a ContentMetadata model, dump it with exclude_none=True
    2. If raw_metadata is a dict, attempt to validate it against ContentMetadata
       - If validation succeeds, dump with exclude_none=True
       - If validation fails, filter out None values manually
    3. If raw_metadata is neither, return None

    Args:
        raw_metadata: The raw_metadata to clean (can be dict, ContentMetadata, or other).

    Returns:
        Cleaned dictionary with None values excluded, or None.
    """
    if raw_metadata is None:
        return None

    if isinstance(raw_metadata, ContentMetadata):
        return raw_metadata.model_dump(exclude_none=True)

    if isinstance(raw_metadata, dict):
        try:
            validated = ContentMetadata.model_validate(raw_metadata)
            return validated.model_dump(exclude_none=True)
        except Exception:
            return {k: v for k, v in raw_metadata.items() if v is not None}

    logger.warning(
        "Unexpected raw_metadata type: %s, expected dict or ContentMetadata",
        type(raw_metadata).__name__,
    )
    return None


# ---------------------------------------------------------------------------
# Account queuing for cross-platform spidering
# ---------------------------------------------------------------------------

async def queue_discovered_accounts(
    session: AsyncSession,
    metadata: AccountMetadata | dict[str, Any],
    parent_handle: str,
    status: str = "pending",
    category: str | None = None,
) -> None:
    """
    Queue discovered accounts from profile contacts for cross-platform spidering.

    Accepts either an AccountMetadata model or a raw dictionary (for backward compatibility).
    If a dict is passed, it will be converted to AccountMetadata using best-effort conversion.

    Processes contacts from AccountMetadata model:
    - Telegram channels from metadata.contacts.telegram_channels
    - External platforms from metadata.external_platforms (VK, YouTube, Threads, TikTok)
    - Scans URLs from metadata.biography, metadata.website, and metadata.link_in_bio
      using _extract_platform_info to detect secondary social platforms

    Implements category inheritance: newly queued accounts inherit the parent's
    category in their raw_metadata.

    Args:
        session: SQLAlchemy async session for database operations
        metadata: AccountMetadata Pydantic model or dict containing contacts and platforms
        parent_handle: Handle of the parent account whose bio was scanned
        status: Status to assign to newly discovered accounts (default: "pending")
        category: Optional category to inherit to newly queued accounts
    """
    # Convert dict to AccountMetadata if needed (backward compatibility)
    if isinstance(metadata, dict):
        metadata = _convert_dict_to_account_metadata(metadata)

    telegram_channels: list[str] = []
    external_platforms: ExternalPlatforms | None = None
    biography: str | None = None
    website: str | None = None
    link_in_bio: str | None = None

    if metadata.contacts:
        telegram_channels = metadata.contacts.telegram_channels

    external_platforms = metadata.external_platforms
    biography = metadata.biography
    website = metadata.website
    link_in_bio = metadata.link_in_bio

    # Track queued platform+id pairs to avoid duplicates
    queued_accounts: set[tuple[str, str]] = set()

    # Helper to queue account if not already queued
    async def _queue_if_new(platform: str, platform_id: str) -> None:
        key = (platform, platform_id)
        if key not in queued_accounts:
            queued_accounts.add(key)
            await _queue_single_account(
                session, platform, platform_id, parent_handle, status, category
            )

    # Queue Telegram channels
    for handle in telegram_channels:
        if not handle:
            continue
        clean_handle = handle.lstrip("@")
        await _queue_if_new("TELEGRAM", clean_handle)

    # Queue standard external platforms
    if external_platforms:
        platform_mapping = {
            "vk": "VK",
            "youtube": "YOUTUBE",
            "threads": "THREADS",
            "tiktok": "TIKTOK",
        }

        for platform_slug, platform_name in platform_mapping.items():
            handle = getattr(external_platforms, platform_slug, None)
            if handle:
                await _queue_if_new(platform_name, handle)

    # Scan URLs from biography, website, and link_in_bio for additional platforms
    urls_to_scan: list[str] = []

    # Extract URLs from biography
    if biography:
        bio_urls = URL_PATTERN.findall(biography)
        urls_to_scan.extend(bio_urls)

    # Add website URL if present
    if website:
        urls_to_scan.append(website)

    # Add link_in_bio URL if present
    if link_in_bio:
        urls_to_scan.append(link_in_bio)

    # Deduplicate URLs while preserving order
    seen_urls: set[str] = set()
    unique_urls: list[str] = []
    for url in urls_to_scan:
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_urls.append(url)

    # Extract platform info from each URL and queue discovered accounts
    for url in unique_urls:
        platform, platform_id = _extract_platform_info(url)
        if platform and platform_id:
            # Skip WEBSITE and LINK_IN_BIO - they are not social platform accounts
            if platform in ("WEBSITE", "LINK_IN_BIO"):
                continue
            await _queue_if_new(platform, platform_id)

    if not queued_accounts:
        logger.debug(
            "[SPIDER] No external social accounts discovered in bio of parent account %s.",
            parent_handle,
        )


async def queue_discovered_mentions(
    session: AsyncSession,
    platform: str,
    mentions: list[str],
    parent_handle: str,
    status: str = "pending",
    category: str | None = None,
) -> None:
    """
    Queue discovered @username mentions for the current platform into the database.

    Args:
        session: SQLAlchemy async session for database operations
        platform: Platform name (INSTAGRAM, TIKTOK, YOUTUBE, THREADS)
        mentions: List of usernames extracted from @mentions (without @ symbol)
        parent_handle: Handle of the parent account that contained the mentions
        status: Status to assign to newly discovered accounts (default: "pending")
        category: Optional category to inherit to newly queued accounts
    """
    for username in mentions:
        if not username or len(username) < 3:
            continue
        await _queue_single_account(session, platform, username, parent_handle, status, category)


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
            # Safe dict assignment: only set category if provided, otherwise None
            raw_metadata = {"category": category} if category else None

            new_account = Account(
                id=generated_id,
                platform=platform,
                platform_id=platform_id,
                username=platform_id,
                title=platform_id,
                status=status,
                raw_metadata=raw_metadata,
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
    raw_metadata: AccountMetadata | dict[str, Any] | None = None,
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

    Accepts an optional raw_metadata parameter that can be:
        - AccountMetadata model (will be dumped with exclude_none=True)
        - dict (will be attempted to parse as AccountMetadata, or cleaned with exclude_none)
        - None (no raw_metadata will be stored)

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
        raw_metadata: Optional AccountMetadata model or dict to store in raw_metadata field
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

    processed_metadata = _clean_account_raw_metadata(raw_metadata)

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
        raw_metadata=processed_metadata,
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


def _clean_account_raw_metadata(
    raw_metadata: AccountMetadata | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Clean and serialize raw_metadata for account/virtual bio post.

    This helper function:
    1. If raw_metadata is an AccountMetadata model, dump it with exclude_none=True
    2. If raw_metadata is a dict, attempt to validate it against AccountMetadata
       - If validation succeeds, dump with exclude_none=True
       - If validation fails, filter out None values manually
    3. If raw_metadata is None, return None

    Args:
        raw_metadata: The raw_metadata to clean (can be AccountMetadata, dict, or None).

    Returns:
        Cleaned dictionary with None values excluded, or None.
    """
    if raw_metadata is None:
        return None

    if isinstance(raw_metadata, AccountMetadata):
        return raw_metadata.model_dump(exclude_none=True)

    if isinstance(raw_metadata, dict):
        try:
            validated = AccountMetadata.model_validate(raw_metadata)
            return validated.model_dump(exclude_none=True)
        except Exception:
            return {k: v for k, v in raw_metadata.items() if v is not None}

    logger.warning(
        "Unexpected raw_metadata type: %s, expected AccountMetadata, dict, or None",
        type(raw_metadata).__name__,
    )
    return None
