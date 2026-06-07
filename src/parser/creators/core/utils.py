"""
Utility functions for cross-platform social media profile parsing.

This module is a facade that re-exports all public functions and constants
from the modularized submodules:
    - text: Text processing and validation logic
    - contacts: Contact parsing regexes and functions
    - db: SQLAlchemy database transaction helpers
    - instagram_helpers: Instagram-specific JSON payload parsers

All original functionality is preserved. Existing imports from
src.parser.creators.core.utils will continue to work unchanged.
"""

# ---------------------------------------------------------------------------
# Re-exports from text module
# ---------------------------------------------------------------------------

from src.parser.creators.core.text import (
    SLOP_STOP_WORDS,
    is_russian_text,
    is_slop_or_theme_page,
    parse_published_at,
    clean_vtt_content,
)

# ---------------------------------------------------------------------------
# Re-exports from contacts module
# ---------------------------------------------------------------------------

from src.parser.creators.core.contacts import (
    EMAIL_PATTERN,
    TELEGRAM_HANDLE_PATTERN,
    MENTION_PATTERN,
    URL_PATTERN,
    SOCIAL_MEDIA_DOMAINS,
    extract_emails,
    extract_telegram_handles,
    extract_mentions,
    extract_external_links,
    parse_profile_contacts,
    normalize_telegram_handle,
    is_valid_email,
    is_valid_telegram_handle,
    compile_author_metadata_dict,
)

# ---------------------------------------------------------------------------
# Re-exports from db module
# ---------------------------------------------------------------------------

from src.parser.creators.core.db import (
    upsert_and_deduplicate_account,
    queue_discovered_accounts,
    queue_discovered_mentions,
    _queue_single_account,
    upsert_virtual_bio_post,
)

# ---------------------------------------------------------------------------
# Re-exports from instagram_helpers module
# ---------------------------------------------------------------------------

from src.parser.creators.core.instagram_helpers import (
    extract_instagram_subscribers,
    extract_instagram_content_text,
    extract_instagram_published_at,
    extract_instagram_video_url,
    extract_instagram_metrics,
    build_instagram_author_metadata,
)

# ---------------------------------------------------------------------------
# Module-level dunder to expose all public symbols
# ---------------------------------------------------------------------------

__all__ = [
    # Text module
    "SLOP_STOP_WORDS",
    "is_russian_text",
    "is_slop_or_theme_page",
    "parse_published_at",
    "clean_vtt_content",
    # Contacts module
    "EMAIL_PATTERN",
    "TELEGRAM_HANDLE_PATTERN",
    "MENTION_PATTERN",
    "URL_PATTERN",
    "SOCIAL_MEDIA_DOMAINS",
    "extract_emails",
    "extract_telegram_handles",
    "extract_mentions",
    "extract_external_links",
    "parse_profile_contacts",
    "normalize_telegram_handle",
    "is_valid_email",
    "is_valid_telegram_handle",
    "compile_author_metadata_dict",
    # DB module
    "upsert_and_deduplicate_account",
    "queue_discovered_accounts",
    "queue_discovered_mentions",
    "_queue_single_account",
    "upsert_virtual_bio_post",
    # Instagram helpers module
    "extract_instagram_subscribers",
    "extract_instagram_content_text",
    "extract_instagram_published_at",
    "extract_instagram_video_url",
    "extract_instagram_metrics",
    "build_instagram_author_metadata",
]
