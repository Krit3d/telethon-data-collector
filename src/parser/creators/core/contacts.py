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
from typing import Any

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


def extract_telegram_handles(text: str | None) -> list[str]:
    """Extract Telegram handles from text.

    Finds Telegram usernames from various formats:
    - @username
    - t.me/username
    - telegram.me/username
    - https://t.me/username
    - https://telegram.me/username

    Normalizes them to clean usernames without @ or URL prefixes.

    Args:
        text: The text to search for Telegram handles. Can be None.

    Returns:
        A deduplicated list of clean Telegram usernames (without @ or URL).
        Returns empty list if text is None or no handles are found.
    """
    if not text:
        return []

    matches = TELEGRAM_HANDLE_PATTERN.findall(text)
    return list(set(match.lower() for match in matches))


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
    specified in exclude_domains parameter.

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

    effective_excludes = SOCIAL_MEDIA_DOMAINS.copy()
    if exclude_domains:
        effective_excludes = effective_excludes.union(exclude_domains)

    urls = URL_PATTERN.findall(text)

    external_links: list[str] = []
    for url in urls:
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
        if domain_match:
            domain = domain_match.group(1).lower()
            if not any(excluded in domain for excluded in effective_excludes):
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
) -> dict[str, list[str] | str]:
    """Parse profile contacts from biography text and external URL.

    Combines multiple extraction functions to parse emails, Telegram handles,
    and external links from a social media profile's biography and optional
    external URL field. Also extracts platform-specific handles from known
    social URLs (Instagram, TikTok, YouTube, Threads).

    Args:
        biography: The biography or description text of the profile. Can be None.
        external_url: Optional external URL directly from the profile API.
            This is added to the text to parse for external links.

    Returns:
        A structured dictionary containing:
        - emails: List of extracted email addresses
        - telegram_handles: List of extracted Telegram usernames
        - external_links: List of external links (excluding social media)
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
    external_links = extract_external_links(combined_text)

    # Also extract links from raw_external_url directly if provided
    if external_url and external_url not in external_links:
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", external_url)
        if domain_match:
            domain = domain_match.group(1).lower()
            if not any(excluded in domain for excluded in SOCIAL_MEDIA_DOMAINS):
                external_links.append(external_url)

    return {
        "emails": emails,
        "telegram_handles": telegram_handles,
        "external_links": external_links,
        "raw_bio": biography or "",
    }


def normalize_telegram_handle(handle: str) -> str:
    """Normalize a Telegram handle to clean username format.

    Removes @, t.me/, telegram.me/ prefixes and converts to lowercase.

    Args:
        handle: The Telegram handle to normalize.

    Returns:
        Normalized Telegram username without prefixes.
    """
    handle = handle.strip()

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
            handle = handle[len(prefix) :]
            break

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


# ---------------------------------------------------------------------------
# Platform profile link templates
# ---------------------------------------------------------------------------

PLATFORM_PROFILE_LINKS: dict[str, str] = {
    "INSTAGRAM": "https://instagram.com/{username}",
    "TIKTOK": "https://www.tiktok.com/@{username}",
    "YOUTUBE": "https://youtube.com/@{username}",
    "THREADS": "https://threads.net/@{username}",
}


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
) -> dict[str, Any]:
    """Compile a standardized, OpenSPG-compliant author profile metadata dict.

    This helper normalizes contact information and constructs a uniform
    metadata structure suitable for cross-platform storage and downstream
    processing.

    Emails are prefixed with ``email:`` and Telegram handles with ``telegram:``
    to create a machine-readable contact format.

    The profile link is formatted according to the platform convention:

    - **INSTAGRAM**: ``https://instagram.com/{username}``
    - **TIKTOK**:   ``https://www.tiktok.com/@{username}``
    - **YOUTUBE**:  ``https://youtube.com/@{username}``
    - **THREADS**:  ``https://threads.net/@{username}``

    All ``None`` values are stripped from the output to produce clean JSON.

    Args:
        platform: Platform identifier string (e.g. ``"INSTAGRAM"``, ``"YOUTUBE"``).
        username: Creator's username on the platform, or ``None``.
        biography: Profile biography / description text, or ``None``.
        contacts_dict: Dictionary containing extracted contacts with keys
            ``emails`` (list of strings) and ``telegram_handles`` (list of strings).
        extra_links: Optional list of additional external links to include.
        location: Optional human-readable location string.
        language: Optional language code or label.
        geo_data: Optional dictionary with structured geographic data.

    Returns:
        A dictionary with the following keys (all values guaranteed non-``None``):

        - ``profile_link`` (str): Full URL to the creator's profile.
        - ``bio_description`` (str): The biography text.
        - ``external_links`` (list[str]): All external / non-social links.
        - ``contacts`` (list[str]): Normalized contact strings
          (``email:...`` and ``telegram:@...``).
        - ``advertising_contacts`` (list[str]): Same as ``contacts``; kept for
          OpenSPG schema compatibility.
        - ``language`` (str): Language information, if provided.
        - ``location`` (str): Location string, if provided.
        - ``geo_data`` (dict): Structured geo data, if provided.
    """
    # Build profile link
    profile_link: str = ""
    if username:
        template = PLATFORM_PROFILE_LINKS.get(platform.upper() if platform else "", "")
        if template:
            profile_link = template.format(username=username)
        else:
            # Fallback: lowercase platform name as subdomain
            safe_platform = (platform or "unknown").lower().replace(" ", "")
            profile_link = f"https://{safe_platform}.com/{username}"

    # Normalize contacts to standardized format
    contacts: list[str] = []

    emails = contacts_dict.get("emails", []) if contacts_dict else []
    if isinstance(emails, list):
        for email in emails:
            if email and isinstance(email, str):
                contacts.append(f"email:{email.lower().strip()}")

    telegram_handles = contacts_dict.get("telegram_handles", []) if contacts_dict else []
    if isinstance(telegram_handles, list):
        for handle in telegram_handles:
            if handle and isinstance(handle, str):
                normalized = normalize_telegram_handle(handle)
                if normalized:
                    contacts.append(f"telegram:@{normalized}")

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

    # Build result dict, only including non-None values
    result: dict[str, Any] = {
        "profile_link": profile_link,
        "bio_description": biography or "",
        "external_links": unique_external,
        "contacts": contacts,
        "advertising_contacts": contacts.copy(),
    }

    if language is not None:
        result["language"] = language
    if location is not None:
        result["location"] = location
    if geo_data is not None:
        result["geo_data"] = geo_data

    return result
