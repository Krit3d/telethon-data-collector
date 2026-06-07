"""
Contact information extraction helpers for social media profile parsing.

This module provides:
    - Regex patterns for extracting emails, Telegram handles, URLs, and @mentions
    - Functions to extract and validate contact information from text
    - Helpers to parse profile contacts from biography and external URL
    - Uniform author metadata compiler for OpenSPG-compliant output
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any

from src.parser.creators.core.schemas import (
    AccountMetadata,
    Contacts,
    ExternalPlatforms,
    GeoData,
    MetricsEntry,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for contact extraction
# ---------------------------------------------------------------------------

# Robust email regex pattern
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b",
    re.IGNORECASE,
)

# Telegram handle patterns: @username, t.me/username, https://telegram.me/username
TELEGRAM_HANDLE_PATTERN = re.compile(
    r"(?:@|t\.me/|telegram\.me/|https?://t\.me/|https?://telegram\.me/)"
    r"([A-Za-z0-9_]{5,32})",
    re.IGNORECASE,
)

# Telegram invite link patterns: t.me/+hash, t.me/joinchat/hash
TELEGRAM_INVITE_PATTERN = re.compile(
    r"t\.me/(?:\+|joinchat/)([A-Za-z0-9_\-]{6,})",
    re.IGNORECASE,
)

# URL pattern for extracting external links
URL_PATTERN = re.compile(
    r"https?://[^\s<>\"{}|\\^`\[\]]+",
    re.IGNORECASE,
)

# Regex pattern for extracting @username mentions (general purpose)
MENTION_PATTERN = re.compile(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([a-zA-Z0-9_\.]{1,30})")

# Common social media domains to exclude when extracting external links
SOCIAL_MEDIA_DOMAINS: frozenset[str] = frozenset({
    "instagram.com",
    "tiktok.com",
    "youtube.com",
    "youtu.be",
    "threads.net",
    "threads.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "t.me",
    "telegram.me",
    "wa.me",
    "whatsapp.com",
})

# External platform domains for structured extraction
# Maps platform slug to list of domains to match
EXTERNAL_PLATFORM_DOMAINS: dict[str, list[str]] = {
    "vk": ["vk.com", "vk.ru", "vkontakte.ru"],
    "youtube": ["youtube.com", "youtu.be"],
    "threads": ["threads.net", "threads.com"],
    "tiktok": ["tiktok.com"],
}

# Keywords to identify personal Telegram contacts (PR, management, advertising)
TELEGRAM_PERSONAL_KEYWORDS: frozenset[str] = frozenset({
    "manager", "pr", "admin", "sales", "cooperation",
    "reklama", "advertising", "write", "contact",
})

# Keywords to identify Telegram channels (TGK)
TELEGRAM_CHANNEL_KEYWORDS: frozenset[str] = frozenset({
    "channel", "канал", "тгк", "телега", "блог", "blog",
    "t.me/joinchat", "t.me/+", "telegram.me/joinchat",
})

# Russian phrases indicating personal contact for advertising
TELEGRAM_PERSONAL_PHRASES: list[str] = [
    "по рекламе",
    "пишите",
    "сотрудничество",
    "реклама",
    "advertising",
    "for ads",
]


def extract_emails(text: str | None) -> list[str]:
    """Extract all email addresses from the provided text.

    Uses a robust regex pattern to find email addresses in various formats.
    Returns a deduplicated list of lowercase emails.

    Args:
        text: The text to search for email addresses. Can be None.

    Returns:
        A list of unique lowercase email addresses. Returns empty list if
        text is None or no emails are found.
    """
    if not text:
        return []

    emails = EMAIL_PATTERN.findall(text)
    return list(set(email.lower() for email in emails))


def extract_external_platforms(
    text: str | None,
) -> dict[str, str]:
    """Extract and classify social media links into structured external_platforms map.

    Parses the text for URLs matching known platform domains (VK, YouTube,
    Threads, TikTok) and extracts clean handles/usernames from those URLs.

    Args:
        text: The text to search for platform URLs. Can be None.

    Returns:
        A dictionary mapping platform slug to extracted handle/username.
        Example: {"vk": "username", "youtube": "@channel_name"}
    """
    if not text:
        return {}

    external_platforms: dict[str, str] = {}
    urls = URL_PATTERN.findall(text)

    for url in urls:
        url_lower = url.lower()

        # VK: vk.com/username or vk.com/id12345
        for domain in EXTERNAL_PLATFORM_DOMAINS["vk"]:
            if domain in url_lower:
                match = re.search(rf"{re.escape(domain)}/([^/?#]+)", url_lower)
                if match:
                    handle = match.group(1)
                    if handle and handle not in ("id", "club", "public"):
                        external_platforms["vk"] = handle
                        break
                break

        # YouTube: youtube.com/@handle or youtu.be/channel_id
        for domain in EXTERNAL_PLATFORM_DOMAINS["youtube"]:
            if domain in url_lower:
                # Try @handle format first (new YouTube format)
                match = re.search(rf"{re.escape(domain)}/@([^/?#]+)", url_lower)
                if match:
                    external_platforms["youtube"] = f"@{match.group(1)}"
                    break
                # Try /channel/ format
                match = re.search(rf"{re.escape(domain)}/channel/([^/?#]+)", url_lower)
                if match:
                    external_platforms["youtube"] = match.group(1)
                    break
                # Try /c/ format
                match = re.search(rf"{re.escape(domain)}/c/([^/?#]+)", url_lower)
                if match:
                    external_platforms["youtube"] = match.group(1)
                    break
                # Try /user/ format
                match = re.search(rf"{re.escape(domain)}/user/([^/?#]+)", url_lower)
                if match:
                    external_platforms["youtube"] = match.group(1)
                    break
                break

        # Threads: threads.net/@username
        for domain in EXTERNAL_PLATFORM_DOMAINS["threads"]:
            if domain in url_lower:
                match = re.search(rf"{re.escape(domain)}/@?([^/?#]+)", url_lower)
                if match:
                    handle = match.group(1)
                    if handle:
                        external_platforms["threads"] = handle
                    break
                break

        # TikTok: tiktok.com/@username
        for domain in EXTERNAL_PLATFORM_DOMAINS["tiktok"]:
            if domain in url_lower:
                match = re.search(rf"{re.escape(domain)}/@([^/?#]+)", url_lower)
                if match:
                    external_platforms["tiktok"] = match.group(1)
                break

    return external_platforms


def extract_telegram_handles(text: str | None) -> list[str]:
    """Extract Telegram handles and invite links from text.

    Finds Telegram usernames from various formats:
    - @username
    - t.me/username
    - telegram.me/username
    - https://t.me/username
    - https://telegram.me/username

    Also extracts Telegram invite links:
    - t.me/+hash (invite hash with + prefix)
    - t.me/joinchat/hash (legacy joinchat format)

    Filters out:
    - Invalid matches like "joinchat" or "join" as usernames
    - Bot accounts (usernames ending with "bot", case-insensitive)

    Args:
        text: The text to search for Telegram handles. Can be None.

    Returns:
        A deduplicated list of clean Telegram identifiers (usernames and invite hashes).
        Returns empty list if text is None or no handles are found.
    """
    if not text:
        return []

    handles: set[str] = set()

    # Extract standard usernames
    username_matches = TELEGRAM_HANDLE_PATTERN.findall(text)
    for match in username_matches:
        normalized = match.lower()
        # Filter out invalid usernames like "joinchat" or "join"
        if normalized in ("joinchat", "join"):
            continue
        # Filter out bot accounts (usernames ending with "bot")
        if normalized.endswith("bot"):
            continue
        handles.add(normalized)

    # Extract invite links (+hash or joinchat/hash)
    for plus_match in re.finditer(r"t\.me/\+([A-Za-z0-9_\-]{6,})", text, re.IGNORECASE):
        handles.add("+" + plus_match.group(1))

    for joinchat_match in re.finditer(r"t\.me/joinchat/([A-Za-z0-9_\-]{6,})", text, re.IGNORECASE):
        handles.add("joinchat/" + joinchat_match.group(1))

    return list(handles)


def extract_mentions(text: str | None) -> list[str]:
    """Extract all @username mentions from text, normalizing them to lowercase and removing the @ symbol.

    Args:
        text: The text to search for @username mentions. Can be None.

    Returns:
        A deduplicated list of usernames without the @ symbol (lowercase).
        Returns empty list if text is None or no mentions are found.
    """
    if not text:
        return []

    matches = MENTION_PATTERN.findall(text)
    return list(set(match.lower() for match in matches if match))


def extract_external_links(
    text: str | None,
    exclude_domains: set[str] | None = None,
) -> list[str]:
    """Extract all HTTP/HTTPS links from text.

    Excludes social network domains and optionally additional domains
    specified in exclude_domains parameter. Also excludes domains that
    are classified into external_platforms to avoid duplication.

    Args:
        text: The text to search for URLs. Can be None.
        exclude_domains: Optional set of domains to exclude.
            Defaults to common social media domains.

    Returns:
        A deduplicated list of external links. Returns empty list if
        text is None or no links are found.
    """
    if not text:
        return []

    # Build extended exclude list including external platform domains
    extended_excludes = SOCIAL_MEDIA_DOMAINS.copy()
    for domains in EXTERNAL_PLATFORM_DOMAINS.values():
        extended_excludes = extended_excludes.union(set(domains))
    if exclude_domains:
        extended_excludes = extended_excludes.union(exclude_domains)

    urls = URL_PATTERN.findall(text)

    external_links: list[str] = []
    for url in urls:
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
        if domain_match:
            domain = domain_match.group(1).lower()
            if not any(excluded in domain for excluded in extended_excludes):
                external_links.append(url)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_links: list[str] = []
    for link in external_links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)

    return unique_links


def parse_profile_contacts(
    biography: str | None,
    external_url: str | None = None,
) -> dict[str, Any]:
    """Parse profile contacts from biography text and external URL.

    Combines multiple extraction functions to parse emails, Telegram handles,
    external links, and structured external platforms from a social media
    profile's biography and optional external URL field.

    Args:
        biography: The biography or description text of the profile. Can be None.
        external_url: Optional external URL directly from the profile API.
            This is added to the text to parse for external links.

    Returns:
        A structured dictionary containing:
        - emails: List of extracted email addresses
        - telegram_handles: List of extracted Telegram usernames
        - external_links: List of external links (excluding social media and external platforms)
        - external_platforms: Dict mapping platform slug to handle (vk, youtube, threads, tiktok)
        - raw_bio: The original biography text
    """
    combined_text = biography or ""

    if external_url:
        if combined_text:
            combined_text = f"{combined_text}\n{external_url}"
        else:
            combined_text = external_url

    emails = extract_emails(combined_text)
    telegram_handles = extract_telegram_handles(combined_text)
    external_platforms = extract_external_platforms(combined_text)
    external_links = extract_external_links(combined_text)

    # Also extract links from raw_external_url directly if provided
    if external_url and external_url not in external_links:
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", external_url)
        if domain_match:
            domain = domain_match.group(1).lower()
            # Check against extended excludes (including external platform domains)
            extended_excludes = SOCIAL_MEDIA_DOMAINS.copy()
            for domains in EXTERNAL_PLATFORM_DOMAINS.values():
                extended_excludes = extended_excludes.union(set(domains))
            if not any(excluded in domain for excluded in extended_excludes):
                external_links.append(external_url)

    return {
        "emails": emails,
        "telegram_handles": telegram_handles,
        "external_links": external_links,
        "external_platforms": external_platforms,
        "raw_bio": biography or "",
    }


def normalize_telegram_handle(handle: str) -> str:
    """Normalize a Telegram handle to clean format.

    For standard usernames:
    - Removes @, t.me/, telegram.me/ prefixes
    - Converts to lowercase

    For invite links (starting with + or containing joinchat/):
    - Removes domain prefixes (t.me/, https://t.me/, etc.)
    - Preserves the original case of the invite hash
    - Telegram invite hashes are case-sensitive

    Args:
        handle: The Telegram handle or invite link to normalize.

    Returns:
        Normalized Telegram identifier (username in lowercase, invite hash in original case).
    """
    handle = handle.strip()

    # Check if it's an invite link (preserve case for invite hashes)
    is_invite = handle.startswith("+") or "/joinchat/" in handle or handle.startswith("joinchat/")

    if handle.startswith("@"):
        handle = handle[1:]

    for prefix in [
        "t.me/",
        "telegram.me/",
        "https://t.me/",
        "https://telegram.me/",
        "http://t.me/",
        "http://telegram.me/",
    ]:
        if handle.lower().startswith(prefix):
            handle = handle[len(prefix):]
            break

    # For invite links, preserve original case; for usernames, convert to lowercase
    if is_invite:
        return handle
    return handle.lower()


def is_valid_email(email: str) -> bool:
    """Check if a string is a valid email address.

    Args:
        email: The string to validate.

    Returns:
        True if the string matches a valid email pattern, False otherwise.
    """
    return bool(EMAIL_PATTERN.fullmatch(email))


def is_valid_telegram_handle(handle: str) -> bool:
    """Check if a string is a valid Telegram username.

    Telegram usernames must:
    - Be 5-32 characters long
    - Contain only letters, numbers, and underscores
    - Start with a letter or underscore

    Args:
        handle: The username to validate.

    Returns:
        True if the string is a valid Telegram username, False otherwise.
    """
    normalized = normalize_telegram_handle(handle)

    if not (5 <= len(normalized) <= 32):
        return False

    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{4,31}", normalized))


def classify_telegram_handles(
    biography: str | None,
    handles: list[str],
) -> tuple[list[str], list[str]]:
    """Classify Telegram handles into channels (TGK) vs personal contacts.

    Analyzes the biography text and handle names to determine if a Telegram
    handle is likely a channel (TGK) or a personal contact (PR, management, etc.).

    Personal profiles are identified by keywords like:
        - manager, pr, admin, sales, cooperation, reklama, write
        - Russian phrases: "по рекламе", "пишите", "сотрудничество"

    Channels (TGK) are identified by keywords like:
        - channel, канал, тгк, телега, блог, blog
        - Invite formats: t.me/joinchat/..., t.me/+...

    Args:
        biography: The biography text to analyze for context. Can be None.
        handles: List of Telegram handles to classify.

    Returns:
        A tuple of two lists: (telegram_channels, telegram_personal)
    """
    if not handles:
        return ([], [])

    bio_lower = biography.lower() if biography else ""
    telegram_channels: list[str] = []
    telegram_personal: list[str] = []

    for handle in handles:
        if not handle:
            continue

        handle_lower = handle.lower()
        normalized_handle = normalize_telegram_handle(handle)

        # Check if handle or bio context indicates personal contact
        is_personal = False
        is_channel = False

        # Check handle name for personal keywords
        for keyword in TELEGRAM_PERSONAL_KEYWORDS:
            if keyword in handle_lower:
                is_personal = True
                break

        # Check biography for personal contact phrases
        if not is_personal and biography:
            for phrase in TELEGRAM_PERSONAL_PHRASES:
                if phrase in bio_lower:
                    # Check if this phrase is near the current handle
                    # Simple check: if phrase exists in bio, mark as personal contact
                    is_personal = True
                    break

        # Check handle name for channel keywords
        if not is_personal:
            for keyword in TELEGRAM_CHANNEL_KEYWORDS:
                if keyword in handle_lower:
                    is_channel = True
                    break

        # Check for invite link formats (channels) - always classify as channel
        if handle and (handle.startswith("+") or handle.startswith("joinchat/") or
                      "joinchat" in handle_lower or "/+" in handle_lower or "t.me/+" in handle_lower):
            is_channel = True

        # Default classification: if no clear personal indicators, treat as channel
        if not is_personal and not is_channel:
            is_channel = True

        if is_personal:
            telegram_personal.append(normalized_handle)
        else:
            telegram_channels.append(normalized_handle)

    # Deduplicate while preserving order
    telegram_channels = list(dict.fromkeys(telegram_channels))
    telegram_personal = list(dict.fromkeys(telegram_personal))

    return (telegram_channels, telegram_personal)


# ---------------------------------------------------------------------------
# Platform profile link templates
# ---------------------------------------------------------------------------

PLATFORM_PROFILE_LINKS: dict[str, str] = {
    "INSTAGRAM": "https://instagram.com/{username}",
    "TIKTOK": "https://www.tiktok.com/@{username}",
    "YOUTUBE": "https://youtube.com/@{username}",
    "THREADS": "https://threads.net/@{username}",
}

# Link-in-bio domains to identify services like taplink, linktree, etc.
LINK_IN_BIO_DOMAINS: frozenset[str] = frozenset({
    "linktr.ee",
    "taplink.cc",
    "beacons.ai",
    "linkin.bio",
    "bio.link",
    "lnk.bio",
    "solo.to",
    "campsite.bio",
    "linkbio.co",
    "my.link",
})


# ---------------------------------------------------------------------------
# Uniform author metadata compiler
# ---------------------------------------------------------------------------


def compile_author_metadata(
    platform: str,
    username: str | None,
    biography: str | None,
    contacts_dict: dict[str, Any],
    extra_links: list[str] | None = None,
    location: str | None = None,
    language: str | None = None,
    geo_data: dict[str, Any] | None = None,
    category: str | None = None,
    raw_profile_payload: dict[str, Any] | None = None,
) -> AccountMetadata:
    """Compile a standardized, OpenSPG-compliant author profile metadata.

    This helper normalizes contact information and constructs a uniform
    metadata structure suitable for cross-platform storage and downstream
    processing. Returns an AccountMetadata Pydantic model instance.

    The profile link is formatted according to the platform convention:

    - **INSTAGRAM**: ``https://instagram.com/{username}``
    - **TIKTOK**:   ``https://www.tiktok.com/@{username}``
    - **YOUTUBE**:  ``https://youtube.com/@{username}``
    - **THREADS**:  ``https://threads.net/@{username}``

    Args:
        platform: Platform identifier string (e.g. ``"INSTAGRAM"``, ``"YOUTUBE"``).
        username: Creator's username on the platform, or ``None``.
        biography: Profile biography / description text, or ``None``.
        contacts_dict: Dictionary containing extracted contacts with keys
            ``emails`` (list of strings), ``telegram_handles`` (list of strings),
            ``external_links`` (list of strings), ``external_platforms`` (dict).
        extra_links: Optional list of additional external links to include.
        location: Optional human-readable location string.
        language: Optional language code or label.
        geo_data: Optional dictionary with structured geographic data.
        category: Optional category string for the account.
        raw_profile_payload: Optional raw JSON payload from the platform API.

    Returns:
        An AccountMetadata Pydantic model instance with the following fields:

        - ``profile_url`` (str): Full URL to the creator's profile.
        - ``biography`` (str): The biography text.
        - ``contacts`` (Contacts): Structured contacts with emails, phones,
          telegram_channels, and telegram_personal lists.
        - ``external_platforms`` (ExternalPlatforms): Structured map of platform slugs.
        - ``link_in_bio`` (str): First matched link-in-bio URL if any.
        - ``website`` (str): Primary website URL if available.
        - ``language`` (str): Language information, if provided.
        - ``location`` (str): Location string, if provided.
        - ``geo_data`` (GeoData): Structured geo data, if provided.
        - ``category`` (str): Category string, if provided.
        - ``raw_profile_payload`` (dict): Raw JSON payload, if provided.
        - ``extracted_at`` (str): Current UTC timestamp in ISO 8601 format.
    """
    # Build profile URL
    profile_url: str = ""
    if username:
        template = PLATFORM_PROFILE_LINKS.get(platform.upper() if platform else "", "")
        if template:
            profile_url = template.format(username=username)
        else:
            # Fallback: lowercase platform name as subdomain
            safe_platform = (platform or "unknown").lower().replace(" ", "")
            profile_url = f"https://{safe_platform}.com/{username}"

    # Classify Telegram handles into channels vs personal contacts
    telegram_handles = contacts_dict.get("telegram_handles", []) if contacts_dict else []
    telegram_channels: list[str] = []
    telegram_personal: list[str] = []

    if telegram_handles:
        telegram_channels, telegram_personal = classify_telegram_handles(
            biography, telegram_handles
        )

    # Extract emails
    emails = contacts_dict.get("emails", []) if contacts_dict else []
    if not isinstance(emails, list):
        emails = []

    # External links: combine from contacts_dict and extra_links parameter
    external_links: list[str] = []
    links_from_dict = contacts_dict.get("external_links", []) if contacts_dict else []
    if isinstance(links_from_dict, list):
        external_links.extend(links_from_dict)
    if extra_links:
        external_links.extend(extra_links)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_external: list[str] = []
    for link in external_links:
        if link and link not in seen:
            seen.add(link)
            unique_external.append(link)

    # Identify link-in-bio URL (first match from known link-in-bio domains)
    link_in_bio: str | None = None
    website: str | None = None
    remaining_external_links: list[str] = []

    for link in unique_external:
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", link)
        if domain_match:
            domain = domain_match.group(1).lower()
            if any(bio_domain in domain for bio_domain in LINK_IN_BIO_DOMAINS):
                if link_in_bio is None:
                    link_in_bio = link
                continue
            # First non-link-in-bio, non-social link becomes website
            if website is None and domain not in SOCIAL_MEDIA_DOMAINS:
                website = link
                continue
        remaining_external_links.append(link)

    # Get external_platforms from contacts_dict and convert to ExternalPlatforms model
    external_platforms_dict = (
        contacts_dict.get("external_platforms", {}) if contacts_dict else {}
    )
    if not isinstance(external_platforms_dict, dict):
        external_platforms_dict = {}

    # Build ExternalPlatforms model
    external_platforms = ExternalPlatforms(
        vk=external_platforms_dict.get("vk"),
        youtube=external_platforms_dict.get("youtube"),
        threads=external_platforms_dict.get("threads"),
        tiktok=external_platforms_dict.get("tiktok"),
    )

    # Check for unsupported platforms (platforms in external_links not in known platforms)
    unsupported: dict[str, str] | None = None
    for link in remaining_external_links:
        # Try to extract platform name from URL
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", link)
        if domain_match:
            domain = domain_match.group(1).lower()
            # Skip if already in external_platforms or social media domains
            if domain not in SOCIAL_MEDIA_DOMAINS:
                if unsupported is None:
                    unsupported = {}
                # Use domain as platform name
                platform_name = domain.split(".")[0]
                unsupported[platform_name] = link

    if unsupported:
        external_platforms.unsupported = unsupported

    # Build Contacts model
    contacts = Contacts(
        emails=[email.lower().strip() for email in emails if email and isinstance(email, str)],
        telegram_channels=telegram_channels,
        telegram_personal=telegram_personal,
    )

    # Build GeoData model if provided
    geo_data_model: GeoData | None = None
    if geo_data and isinstance(geo_data, dict):
        geo_data_model = GeoData(
            city=geo_data.get("city"),
            country=geo_data.get("country"),
            coordinates=geo_data.get("coordinates"),
        )

    # Build AccountMetadata model
    return AccountMetadata(
        profile_url=profile_url or None,
        biography=biography or None,
        category=category,
        language=language,
        location=location,
        contacts=contacts,
        external_platforms=external_platforms,
        link_in_bio=link_in_bio,
        website=website,
        geo_data=geo_data_model,
        metrics_history=[],  # Empty by default, can be populated separately
        raw_profile_payload=raw_profile_payload,
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )


def compile_author_metadata_dict(
    platform: str,
    username: str | None,
    biography: str | None,
    contacts_dict: dict[str, Any],
    extra_links: list[str] | None = None,
    location: str | None = None,
    language: str | None = None,
    geo_data: dict[str, Any] | None = None,
    category: str | None = None,
    raw_profile_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile author metadata and return as a dictionary.

    Convenience wrapper around compile_author_metadata that returns a dict
    instead of an AccountMetadata model. Useful for backward compatibility
    or when a dict output is required.

    Args:
        Same as compile_author_metadata.

    Returns:
        A dictionary representation of AccountMetadata with None values excluded.
    """
    account_metadata = compile_author_metadata(
        platform=platform,
        username=username,
        biography=biography,
        contacts_dict=contacts_dict,
        extra_links=extra_links,
        location=location,
        language=language,
        geo_data=geo_data,
        category=category,
        raw_profile_payload=raw_profile_payload,
    )
    return account_metadata.model_dump(exclude_none=True)
