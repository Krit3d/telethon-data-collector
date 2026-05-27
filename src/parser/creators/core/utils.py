"""
Utility functions for parsing metadata and contact channels from social media profiles.

This module provides robust extraction functions for emails, Telegram handles,
and external links from profile descriptions and biographies.
"""

import re
from typing import Any


# Robust email regex pattern
# Matches most common email formats including special characters
EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b',
    re.IGNORECASE
)

# Telegram handle patterns
# Matches @username, t.me/username, and https://telegram.me/username
TELEGRAM_HANDLE_PATTERN = re.compile(
    r'(?:@|t\.me/|telegram\.me/|https?://t\.me/|https?://telegram\.me/)'
    r'([A-Za-z0-9_]{5,32})',
    re.IGNORECASE
)

# URL pattern for extracting external links
URL_PATTERN = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+',
    re.IGNORECASE
)

# Common social media domains to exclude when extracting external links
SOCIAL_MEDIA_DOMAINS = {
    'instagram.com',
    'tiktok.com',
    'youtube.com',
    'youtu.be',
    'threads.net',
    'threads.com',
    'facebook.com',
    'twitter.com',
    'x.com',
    'linkedin.com',
    't.me',
    'telegram.me',
    'wa.me',
    'whatsapp.com',
}


def extract_emails(text: str | None) -> list[str]:
    """
    Extract all email addresses from the provided text.

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

    # Find all email matches
    emails = EMAIL_PATTERN.findall(text)

    # Deduplicate and convert to lowercase
    unique_emails = list(set(email.lower() for email in emails))

    return unique_emails


def extract_telegram_handles(text: str | None) -> list[str]:
    """
    Extract Telegram handles from text.

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

    # Find all Telegram handle matches
    matches = TELEGRAM_HANDLE_PATTERN.findall(text)

    # Deduplicate and convert to lowercase for consistency
    # Telegram usernames are case-insensitive
    unique_handles = list(set(match.lower() for match in matches))

    return unique_handles


def extract_external_links(
    text: str | None,
    exclude_domains: set[str] | None = None
) -> list[str]:
    """
    Extract all HTTP/HTTPS links from text.

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

    # Use default exclude domains if not provided
    if exclude_domains is None:
        exclude_domains = SOCIAL_MEDIA_DOMAINS.copy()
    else:
        exclude_domains = SOCIAL_MEDIA_DOMAINS.union(exclude_domains)

    # Find all URL matches
    urls = URL_PATTERN.findall(text)

    # Filter out social media domains
    external_links = []
    for url in urls:
        # Extract domain from URL
        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if domain_match:
            domain = domain_match.group(1).lower()
            # Check if domain should be excluded
            if not any(excluded in domain for excluded in exclude_domains):
                external_links.append(url)

    # Deduplicate while preserving order
    seen = set()
    unique_links = []
    for link in external_links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)

    return unique_links


def parse_profile_contacts(
    bio: str | None,
    raw_external_url: str | None = None
) -> dict[str, Any]:
    """
    Parse profile contacts from biography text and external URL.

    Combines multiple extraction functions to parse emails, Telegram handles,
    and external links from a social media profile's biography and optional
    external URL field.

    Args:
        bio: The biography or description text of the profile. Can be None.
        raw_external_url: Optional external URL directly from the profile API.
                         This is added to the text to parse for external links.

    Returns:
        A structured dictionary containing:
        - emails: List of extracted email addresses
        - telegram_handles: List of extracted Telegram usernames
        - external_links: List of external links (excluding social media)
        - raw_bio: The original bio text
    """
    # Combine bio and raw_external_url for comprehensive parsing
    combined_text = bio or ""

    if raw_external_url:
        # Add the external URL to the text for link extraction
        if combined_text:
            combined_text = f"{combined_text}\n{raw_external_url}"
        else:
            combined_text = raw_external_url

    # Extract all contact information
    emails = extract_emails(combined_text)
    telegram_handles = extract_telegram_handles(combined_text)
    external_links = extract_external_links(combined_text)

    # Also extract links from raw_external_url directly if provided
    if raw_external_url and raw_external_url not in external_links:
        # Check if it's a valid external link (not a social media domain)
        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', raw_external_url)
        if domain_match:
            domain = domain_match.group(1).lower()
            if not any(excluded in domain for excluded in SOCIAL_MEDIA_DOMAINS):
                external_links.append(raw_external_url)

    # Build the result dictionary
    result = {
        "emails": emails,
        "telegram_handles": telegram_handles,
        "external_links": external_links,
        "raw_bio": bio,
    }

    return result


def normalize_telegram_handle(handle: str) -> str:
    """
    Normalize a Telegram handle to clean username format.

    Removes @, t.me/, telegram.me/ prefixes and converts to lowercase.

    Args:
        handle: The Telegram handle to normalize.

    Returns:
        Normalized Telegram username without prefixes.
    """
    # Remove common prefixes
    handle = handle.strip()

    # Remove @ prefix
    if handle.startswith('@'):
        handle = handle[1:]

    # Remove t.me/ or telegram.me/ prefixes
    for prefix in ['t.me/', 'telegram.me/', 'https://t.me/', 'https://telegram.me/',
                   'http://t.me/', 'http://telegram.me/']:
        if handle.lower().startswith(prefix):
            handle = handle[len(prefix):]
            break

    return handle.lower()


def is_valid_email(email: str) -> bool:
    """
    Check if a string is a valid email address.

    Args:
        email: The string to validate.

    Returns:
        True if the string matches a valid email pattern, False otherwise.
    """
    return bool(EMAIL_PATTERN.fullmatch(email))


def is_valid_telegram_handle(handle: str) -> bool:
    """
    Check if a string is a valid Telegram username.

    Telegram usernames must:
    - Be 5-32 characters long
    - Contain only letters, numbers, and underscores
    - Start with a letter or underscore

    Args:
        handle: The username to validate.

    Returns:
        True if the string is a valid Telegram username, False otherwise.
    """
    # Normalize first
    normalized = normalize_telegram_handle(handle)

    # Check length (5-32 characters)
    if not (5 <= len(normalized) <= 32):
        return False

    # Check pattern: letters, numbers, underscores; must start with letter or underscore
    return bool(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{4,31}', normalized))
