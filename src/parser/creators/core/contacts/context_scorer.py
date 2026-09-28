import re
from enum import Enum

from .constants import (
    CHANNEL_EXACT, CHANNEL_PHRASES, CHANNEL_PREFIXES,
    COMMERCIAL_EXACT, COMMERCIAL_PHRASES, COMMERCIAL_PREFIXES,
    DEFAULT_CONTEXT_WINDOW, PERSONAL_EXACT, PERSONAL_PHRASES, PERSONAL_PREFIXES,
)

_WORD_RE = re.compile(r"\w+")


class ContactRole(str, Enum):
    COMMERCIAL = "commercial"
    PERSONAL = "personal"
    CHANNEL = "channel"
    UNKNOWN = "unknown"


ROLE_PRIORITY: dict[ContactRole, int] = {
    ContactRole.COMMERCIAL: 4,
    ContactRole.PERSONAL: 3,
    ContactRole.CHANNEL: 2,
    ContactRole.UNKNOWN: 1,
}


def get_context_window(text: str, start_pos: int, end_pos: int, window_size: int = DEFAULT_CONTEXT_WINDOW) -> str:
    text_lower = text.lower()
    before = text_lower[max(0, start_pos - window_size):start_pos]
    after = text_lower[end_pos:min(len(text_lower), end_pos + window_size)]
    return f"{before} {after}"


def _matches(
    exact: frozenset[str],
    prefixes: frozenset[str],
    phrases: frozenset[str],
    context: str,
    tokens: list[str],
) -> bool:
    for phrase in phrases:
        if phrase in context:
            return True
    for token in tokens:
        if token in exact:
            return True
    for token in tokens:
        if len(token) >= 3:
            for prefix in prefixes:
                if token.startswith(prefix):
                    return True
    return False


def score_context(context_window: str) -> ContactRole:
    context = context_window.lower()
    clean_context = context.replace("_", " ").replace("-", " ").replace(".", " ")
    tokens = _WORD_RE.findall(clean_context)
    if _matches(COMMERCIAL_EXACT, COMMERCIAL_PREFIXES, COMMERCIAL_PHRASES, context, tokens):
        return ContactRole.COMMERCIAL
    if _matches(CHANNEL_EXACT, CHANNEL_PREFIXES, CHANNEL_PHRASES, context, tokens):
        return ContactRole.CHANNEL
    if _matches(PERSONAL_EXACT, PERSONAL_PREFIXES, PERSONAL_PHRASES, context, tokens):
        return ContactRole.PERSONAL
    return ContactRole.UNKNOWN