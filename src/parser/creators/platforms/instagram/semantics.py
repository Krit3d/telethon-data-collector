import logging
import re

from src.parser.creators.core.utils import parse_profile_contacts

logger = logging.getLogger(__name__)

TELEGRAM_PATTERN = (
    r"@\w+|"
    r"t\.me/[a-zA-Z0-9_\+\-]+|"
    r"telegram\.(?:me|dog)/[a-zA-Z0-9_\+\-]+|"
    r"\b(?:тг|тгк|телеграм|tg|telegram|канал)\b"
)


def has_sufficient_semantics(
    bio: str | None,
    external_url: str | None,
    caption: str | None,
    hashtags: list[str],
    has_target_semantics: bool,
) -> bool:
    if not has_target_semantics:
        return False

    caption_len = len(caption) if caption else 0
    if caption_len <= 120 and len(hashtags) < 3:
        return False

    bio = bio or ""
    external_url = external_url or ""
    combined = (bio + " " + external_url).lower()

    contacts = parse_profile_contacts(combined)
    has_email = len(contacts.get("emails", [])) > 0
    has_external_platform = len(contacts.get("external_platforms", {})) > 0

    has_telegram = bool(re.search(TELEGRAM_PATTERN, combined, re.IGNORECASE))

    if not (has_email or has_telegram or has_external_platform):
        return False

    return True
