import re

from dataclasses import dataclass
from urllib.parse import urlparse

from ..constants import (
    AT_MARKER_WINDOW, BARE_TELEGRAM_TRIGGERS, DEFAULT_CONTEXT_WINDOW,
    TELEGRAM_BLACKLISTED_HANDLES, TELEGRAM_PLATFORM_MARKERS, TELEGRAM_SERVICE_TOKENS,
)
from ..context_scorer import ROLE_PRIORITY, ContactRole, get_context_window, score_context


@dataclass(slots=True, frozen=True)
class ExtractedTelegram:
    handle: str
    role: ContactRole
    is_bot: bool
    is_invite: bool
    raw_match: str
    start: int
    end: int
    is_explicit: bool = False


_URL_DOMAINS = frozenset(("t.me", "telegram.me", "telegram.dog", "tglink.ru"))
_URL_START_PREFIXES = (
    "https://", "http://",
    "t.me/", "telegram.me/", "telegram.dog/", "tglink.ru/",
)

_SEPARATOR_CHARS = frozenset(" \t:—–-→|/\"'«»“”")

_URL_PREFIXES = (
    "https://t.me/", "http://t.me/", "t.me/",
    "telegram.me/", "telegram.dog/",
)


def normalize_telegram_handle(handle: str) -> str:
    value = handle.strip()
    if value.startswith("@"):
        value = value[1:]
    for prefix in _URL_PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    value = value.split("?", 1)[0].split("#", 1)[0]
    value = value.rstrip("/")
    value = value.rstrip(".,;:!?()[]{}<>\"' ")
    is_invite = value.startswith("+") or "joinchat" in value
    if is_invite:
        return value
    return value.lower()


def _is_valid_username(candidate: str) -> bool:
    if not (5 <= len(candidate) <= 32):
        return False
    if candidate.startswith("_") or candidate.endswith("_"):
        return False
    if "__" in candidate:
        return False
    has_letter = False
    for c in candidate:
        if not (c.isascii() and (c.isalnum() or c == "_")):
            return False
        if c.isalpha() and c.isascii():
            has_letter = True
    return has_letter


def _is_valid_invite_hash(candidate: str) -> bool:
    if not (5 <= len(candidate) <= 32):
        return False
    for c in candidate:
        if not (c.isascii() and (c.isalnum() or c in "-_")):
            return False
    return True


def is_valid_telegram_handle(handle: str) -> bool:
    normalized = normalize_telegram_handle(handle)
    if normalized in TELEGRAM_BLACKLISTED_HANDLES:
        return False
    is_invite = normalized.startswith("+") or "joinchat" in normalized
    if is_invite:
        hash_part = normalized
        if normalized.startswith("+"):
            hash_part = normalized[1:]
        elif "joinchat" in normalized:
            hash_part = normalized.split("/", 1)[-1]
        return _is_valid_invite_hash(hash_part)
    return _is_valid_username(normalized)


def _is_commercial_bot(handle: str, text: str, start: int, end: int, window_size: int) -> bool:
    context = get_context_window(text, start, end, window_size)
    return (
        score_context(context) is ContactRole.COMMERCIAL
        or score_context(handle) is ContactRole.COMMERCIAL
    )


def _merge_candidate(seen: dict[str, ExtractedTelegram], candidate: ExtractedTelegram) -> None:
    existing = seen.get(candidate.handle)
    if existing is None or ROLE_PRIORITY[candidate.role] > ROLE_PRIORITY[existing.role]:
        seen[candidate.handle] = candidate


def _tokenize(text: str) -> list[tuple[int, int, str]]:
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        start = i
        while i < n and not text[i].isspace():
            i += 1
        tokens.append((start, i, text[start:i]))
    return tokens


def _sentence_boundary_pos(text: str) -> int:
    for i in range(len(text) - 1, -1, -1):
        if text[i] in "!?…":
            return i + 1
    return 0


def _word_tokens(text: str) -> list[str]:
    words = []
    for _, _, token in _tokenize(text):
        for part in re.split(r"[\s\-/.,;:!?()\[\]{}<>\"'«»…]+", token):
            if part and part.isalnum():
                words.append(part)
    return words


def _has_platform_marker_nearby(text: str, start: int, end: int, window: int = AT_MARKER_WINDOW) -> bool:
    before = text[max(0, start - window):start]
    after = text[end:min(len(text), end + window)]
    boundary = _sentence_boundary_pos(before)
    if boundary:
        before = before[boundary:]
    window_text = (before + " " + after).lower()
    words = _word_tokens(window_text)
    return any(marker in words for marker in TELEGRAM_PLATFORM_MARKERS)


def _find_url_start(text: str, from_pos: int) -> int | None:
    best = None
    for prefix in _URL_START_PREFIXES:
        idx = text.find(prefix, from_pos)
        while idx != -1:
            if idx == 0 or not (text[idx - 1].isalnum() or text[idx - 1] == "_"):
                if best is None or idx < best:
                    best = idx
                break
            idx = text.find(prefix, idx + 1)
    return best


def _extract_urls(text: str) -> list[tuple[int, int, str]]:
    results = []
    n = len(text)
    pos = 0
    while pos < n:
        start = _find_url_start(text, pos)
        if start is None:
            break
        end = start
        while end < n and not text[end].isspace():
            end += 1
        results.append((start, end, text[start:end]))
        pos = end
    return results


def _extract_handle_from_url(raw: str) -> tuple[str, bool] | None:
    candidate = raw.strip()
    candidate = candidate.rstrip(".,;:!?)]}>\"' ")
    if not (candidate.startswith("http://") or candidate.startswith("https://")):
        candidate = "https://" + candidate
    parsed = urlparse(candidate)
    if parsed.netloc.lower() not in _URL_DOMAINS:
        return None
    path = parsed.path
    if path.startswith("/"):
        path = path[1:]
    if path.startswith("c/"):
        return None
    if path.startswith("s/"):
        path = path[2:]
    is_invite = path.startswith("+") or "joinchat/" in path
    if is_invite:
        if path.startswith("+"):
            hash_part = path[1:].split("/", 1)[0]
            return "+" + hash_part, True
        if "joinchat/" in path:
            hash_part = path.split("joinchat/", 1)[1].split("/", 1)[0]
            return "joinchat/" + hash_part, True
        return None
    first_seg = path.split("/", 1)[0]
    first_seg = first_seg.lstrip("@")
    first_seg = first_seg.rstrip(".,;:!?)]}>\"' ")
    if not first_seg:
        return None
    return first_seg, False


def _skip_to_handle(tokens: list[tuple[int, int, str]], j: int) -> tuple[int, bool]:
    saw_service = False
    while j < len(tokens):
        t = tokens[j][2]
        t_lower = t.lower()
        if t_lower in TELEGRAM_SERVICE_TOKENS:
            saw_service = True
            j += 1
            continue
        if all(c in _SEPARATOR_CHARS for c in t):
            j += 1
            continue
        break
    return j, saw_service


def _process_at_mentions(
    text: str,
    tokens: list[tuple[int, int, str]],
    seen: dict[str, ExtractedTelegram],
    context_window_size: int,
) -> None:
    for start, end, token in tokens:
        at_pos = token.find("@")
        if at_pos == -1:
            continue
        if at_pos > 0:
            prev = token[at_pos - 1]
        elif start > 0:
            prev = text[start - 1]
        else:
            prev = ""
        if prev and (prev.isalnum() or prev in ".-"):
            continue
        candidate = token[at_pos + 1:]
        candidate = candidate.rstrip(".,;:!?)]}>\"' ")
        normalized = normalize_telegram_handle(candidate)
        if not is_valid_telegram_handle(normalized):
            continue
        if not _has_platform_marker_nearby(text, start, end):
            continue
        context = get_context_window(text, start, end, context_window_size)
        role = score_context(context)
        is_bot = normalized.lower().endswith("bot")
        if is_bot and not _is_commercial_bot(normalized, text, start, end, context_window_size):
            continue
        _merge_candidate(seen, ExtractedTelegram(normalized, role, is_bot, False, token, start, end))


def _process_bare_triggers(
    text: str,
    tokens: list[tuple[int, int, str]],
    seen: dict[str, ExtractedTelegram],
    context_window_size: int,
) -> None:
    for i, (start, end, token) in enumerate(tokens):
        cleaned = token.lstrip("([«\"' ")
        trigger_word = cleaned.rstrip(":—–-→|/\"'«»“” ").lower()
        if trigger_word in BARE_TELEGRAM_TRIGGERS:
            j, saw_service = _skip_to_handle(tokens, i + 1)
            if j >= len(tokens):
                continue
            hstart, hend, htoken = tokens[j]
            if any(ord(c) > 127 for c in htoken):
                continue
            candidate = htoken.lstrip("@")
            candidate = candidate.rstrip(".,;:!?)]}>\"' ")
            normalized = normalize_telegram_handle(candidate)
            if not is_valid_telegram_handle(normalized):
                continue
            context = get_context_window(text, hstart, hend, context_window_size)
            role = score_context(context)
            is_channel_prefix = trigger_word == "тгк" or saw_service
            if is_channel_prefix and role is not ContactRole.COMMERCIAL:
                role = ContactRole.CHANNEL
            is_bot = normalized.lower().endswith("bot")
            if is_bot and not _is_commercial_bot(normalized, text, hstart, hend, context_window_size):
                continue
            _merge_candidate(
                seen, ExtractedTelegram(normalized, role, is_bot, False, token, hstart, hend, is_channel_prefix)
            )
            continue
        sep_positions = [(cleaned.find(s), s) for s in (":", "/", "—", "-")]
        sep_positions = [(p, s) for p, s in sep_positions if p != -1]
        if not sep_positions:
            continue
        sep_pos, _ = min(sep_positions)
        trigger_part = cleaned[:sep_pos]
        potential_handle = cleaned[sep_pos + 1:]
        if trigger_part.lower() not in BARE_TELEGRAM_TRIGGERS:
            continue
        if potential_handle:
            candidate = potential_handle.lstrip("@")
            candidate = candidate.rstrip(".,;:!?)]}>\"' ")
            normalized = normalize_telegram_handle(candidate)
            if not is_valid_telegram_handle(normalized):
                continue
            context = get_context_window(text, start, end, context_window_size)
            role = score_context(context)
            is_channel_prefix = trigger_part.lower() == "тгк"
            if is_channel_prefix and role is not ContactRole.COMMERCIAL:
                role = ContactRole.CHANNEL
            is_bot = normalized.lower().endswith("bot")
            if is_bot and not _is_commercial_bot(normalized, text, start, end, context_window_size):
                continue
            _merge_candidate(
                seen, ExtractedTelegram(normalized, role, is_bot, False, token, start, end, is_channel_prefix)
            )
            continue


def extract_telegram_contacts(text: str | None, context_window_size: int = DEFAULT_CONTEXT_WINDOW) -> list[ExtractedTelegram]:
    if not text:
        return []
    seen: dict[str, ExtractedTelegram] = {}

    for start, end, raw in _extract_urls(text):
        parsed = _extract_handle_from_url(raw)
        if parsed is None:
            continue
        handle, is_invite = parsed
        normalized = normalize_telegram_handle(handle)
        if not is_valid_telegram_handle(normalized):
            continue
        role = ContactRole.CHANNEL if is_invite else score_context(
            get_context_window(text, start, end, context_window_size)
        )
        is_bot = normalized.lower().endswith("bot")
        if is_bot and not _is_commercial_bot(normalized, text, start, end, context_window_size):
            continue
        _merge_candidate(seen, ExtractedTelegram(normalized, role, is_bot, is_invite, raw, start, end, True))

    tokens = _tokenize(text)
    _process_at_mentions(text, tokens, seen, context_window_size)
    _process_bare_triggers(text, tokens, seen, context_window_size)

    return list(seen.values())