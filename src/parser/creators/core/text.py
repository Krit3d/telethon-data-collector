"""
Text processing and validation helpers for social media profile parsing.

This module provides:
    - Cyrillic (Russian) text detection
    - AI-slop / theme-page / meme-page detection via stop-words
    - Female creator detection based on Russian biography indicators
    - Timezone-aware datetime parser for published_at timestamps
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop-words for AI-slop / theme-page / meme-page detection (global, English-first)
# ---------------------------------------------------------------------------

SLOP_STOP_WORDS: frozenset[str] = frozenset({
    # AI generation / synthetic content
    "ai art",
    "midjourney",
    "stable diffusion",
    "dall-e",
    "dalle",
    "chatgpt",
    "openai",
    "generative ai",
    "neural network",
    "нейросеть",
    "нейросети",
    "генерация",
    "ai generated",
    "synthetic",
    "made with ai",
    # Compilations / meme aggregators / theme pages
    "meme",
    "memes",
    "humor",
    "funny",
    "lol",
    "lmao",
    "нарезка",
    "нарезки",
    "мемы",
    "мем",
    "юмор",
    "приколы",
    "прикол",
    "смешно",
    "смешное",
    # Spam / aggregation / low-value
    "compilation",
    "best of",
    "top 10",
    "highlights",
    "gaming clips",
    "funny moments",
    "pubg",
    "fortnite",
    "dota",
    "csgo",
    "videos daily",
    "subscribe",
    "подпишись",
    "взаимно",
    "паблик",
    "админ",
    "по вопросам рекламы",
    "сливы",
    "слив",
    "shina",
    # Low-effort content labels
    "гороскоп",
    "гороскопы",
    "анекдоты",
    "анекдот",
    "фильмы на вечер",
    "лучшие фильмы",
    "кино на вечер",
    "треки",
    "музыка",
    "сохрани",
})

# ---------------------------------------------------------------------------
# Female indicator patterns for biography scanning
# ---------------------------------------------------------------------------

# Russian female indicator patterns for biography scanning
FEMALE_INDICATOR_PATTERNS: list[re.Pattern[str]] = [
    # Direct female role indicators
    re.compile(r"\bблогерша\b", re.IGNORECASE),
    re.compile(r"\bмама\b", re.IGNORECASE),
    re.compile(r"\bосновательница\b", re.IGNORECASE),
    re.compile(r"\bсоздала\b", re.IGNORECASE),
    re.compile(r"\bавторка\b", re.IGNORECASE),
    re.compile(r"\bэкспертка\b", re.IGNORECASE),
    re.compile(r"\bдевушка\b", re.IGNORECASE),
    # Verbs ending in "ась" or "ла" (female past tense indicators)
    re.compile(r"\b\w+[аэоуыяёю]ла\b", re.IGNORECASE),
    re.compile(r"\b\w+ась\b", re.IGNORECASE),
    re.compile(r"\b\w+лась\b", re.IGNORECASE),
]


def is_russian_text(text: str | None) -> bool:
    """Check if text contains any Cyrillic (Russian) characters.

    Args:
        text: Text to check, or None.

    Returns:
        True if text contains at least one Cyrillic character, False otherwise.
    """
    if not text:
        return False
    return bool(re.search(r"[а-яА-ЯёЁ]", text))


def is_slop_or_theme_page(username: str | None, biography: str | None) -> bool:
    """Check if account is an AI-slop, meme-page, or compilation channel.

    Scans the username and biography for stop-words that indicate
    non-author accounts such as AI generation pages, meme aggregators,
    video compilation channels, and spam accounts.

    Args:
        username: Platform username (without @ prefix), or None.
        biography: User biography / description text, or None.

    Returns:
        True if any stop-word from SLOP_STOP_WORDS is found in either
        the username or biography (case-insensitive), False otherwise.
    """
    search_text = ""

    if username:
        search_text += username.lower() + " "

    if biography:
        search_text += biography.lower()

    return any(stop_word in search_text for stop_word in SLOP_STOP_WORDS)


def detect_female_creator(biography: str | None) -> bool:
    """Detect if a creator is likely female based on Russian biography text.

    Scans the biography for Russian female indicators such as:
    - Female role words: "блогерша", "мама", "основательница", "создала",
      "авторка", "экспертка", "девушка"
    - Verbs ending in "ась" or "ла" (female past tense forms)

    Args:
        biography: The biography text to scan, or None.

    Returns:
        True if female indicators are detected, False otherwise.
    """
    if not biography:
        return False

    for pattern in FEMALE_INDICATOR_PATTERNS:
        if pattern.search(biography):
            return True

    return False


# ---------------------------------------------------------------------------
# Timezone-aware datetime parser
# ---------------------------------------------------------------------------


def parse_published_at(timestamp: Any) -> datetime:
    """Parse a timestamp into a timezone-aware UTC datetime.

    Handles multiple input formats:

    - **Unix epochs** (int or float): Seconds since 1970-01-01 UTC.
    - **ISO-8601 strings**: Including strings ending with ``"Z"`` (treated as
      UTC) or containing explicit timezone offsets (``+HH:MM``, ``-HH:MM``).
    - **Fallback**: If the input is ``None``, empty, or cannot be parsed, the
      function returns ``datetime.now(timezone.utc)``.

    The returned datetime is always timezone-aware and normalized to UTC.

    Args:
        timestamp: The timestamp value to parse. Can be an integer, float,
            ISO-8601 formatted string, or ``None``.

    Returns:
        A timezone-aware ``datetime`` object in UTC. If parsing fails, the
        current UTC time is returned as a fallback.
    """
    # Fallback for None or empty
    if timestamp is None:
        return datetime.now(timezone.utc)

    if isinstance(timestamp, str) and not timestamp.strip():
        return datetime.now(timezone.utc)

    # --- Unix epoch (int or float) ----------------------------------------
    if isinstance(timestamp, (int, float)):
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            logger.debug("Failed to parse numeric timestamp: %r", timestamp)
            return datetime.now(timezone.utc)

    # --- String (ISO-8601 or similar) ------------------------------------
    if isinstance(timestamp, str):
        ts = timestamp.strip()

        # Replace 'Z' suffix with explicit UTC offset '+00:00'
        if ts.endswith("Z") or ts.endswith("z"):
            ts = ts[:-1] + "+00:00"

        # Common fractional-second pattern: ensure it parses correctly
        # Python 3.7+ fromisoformat supports most ISO formats but not all
        # (e.g. trailing Z is handled above).
        try:
            # Try fromisoformat first (Python 3.11+ handles more formats)
            parsed = datetime.fromisoformat(ts)
            # If naive, assume UTC
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            # Convert to UTC
            return parsed.astimezone(timezone.utc)
        except (ValueError, OverflowError):
            pass

        # Fallback: try common patterns manually
        # Pattern: 2023-01-15T10:30:00+00:00 or similar
        try:
            # Attempt parsing with strptime for common formats
            for fmt in (
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
            ):
                try:
                    parsed = datetime.strptime(ts, fmt)
                    if parsed.tzinfo is None:
                        return parsed.replace(tzinfo=timezone.utc)
                    return parsed.astimezone(timezone.utc)
                except ValueError:
                    continue
        except Exception:
            pass

        logger.debug("Failed to parse timestamp string: %r", timestamp)

    # --- Final fallback ---------------------------------------------------
    return datetime.now(timezone.utc)
