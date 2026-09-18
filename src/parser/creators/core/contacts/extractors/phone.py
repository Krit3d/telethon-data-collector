import bisect
import re
from dataclasses import dataclass

import phonenumbers
from phonenumbers import Leniency, PhoneNumberFormat, PhoneNumberMatcher, PhoneNumberType

from ..constants import PHONE_TRIGGER_WORDS, WHATSAPP_TRIGGER_WORDS
from ..context_scorer import ROLE_PRIORITY, ContactRole, get_context_window, score_context

CIS_REGIONS: tuple[str, ...] = ("RU", "UA", "KZ", "BY", "KG")


@dataclass(slots=True, frozen=True)
class ExtractedPhone:
    phone: str
    is_whatsapp: bool
    role: ContactRole
    raw_match: str
    start: int
    end: int


_WHATSAPP_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?(?:wa\.me/(?:\+)?(\d{7,15})|api\.whatsapp\.com/send/?\?phone=(?:\+|%2B)?(\d{7,15}))",
    re.IGNORECASE,
)

_PHONE_TRIGGER_RE = re.compile(
    r"\b(?:whats\s*app|" + "|".join(re.escape(w) for w in PHONE_TRIGGER_WORDS) + r")\b",
    re.IGNORECASE,
)

_WHATSAPP_TRIGGER_RE = re.compile(
    r"\b(?:whats\s*app|w/a|" + "|".join(re.escape(w) for w in WHATSAPP_TRIGGER_WORDS) + r")\b",
    re.IGNORECASE,
)

def normalize_phone(phone_raw: str, default_region: str | None = "RU") -> str | None:
    if phone_raw.startswith("+"):
        regions: tuple[str | None, ...] = (None,)
    elif default_region:
        regions = (default_region,) + tuple(r for r in CIS_REGIONS if r != default_region)
    else:
        regions = CIS_REGIONS
    for region in regions:
        try:
            parsed = phonenumbers.parse(phone_raw, region)
        except phonenumbers.NumberParseException:
            continue
        if phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(parsed, PhoneNumberFormat.E164)
    return None


def _spans_overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return max(first[0], second[0]) < min(first[1], second[1])


def _register_span(occupied_spans: list[tuple[int, int]], span: tuple[int, int]) -> bool:
    index = bisect.bisect_left(occupied_spans, span)
    if index > 0 and _spans_overlap(occupied_spans[index - 1], span):
        return False
    if index < len(occupied_spans) and _spans_overlap(occupied_spans[index], span):
        return False
    occupied_spans.insert(index, span)
    return True


def _ordered_regions(default_region: str | None) -> tuple[str, ...]:
    if default_region and default_region in phonenumbers.SUPPORTED_REGIONS:
        return (default_region,) + tuple(r for r in CIS_REGIONS if r != default_region)
    return CIS_REGIONS


def _merge_candidate(seen: dict[str, ExtractedPhone], candidate: ExtractedPhone) -> None:
    existing = seen.get(candidate.phone)
    if existing is None or ROLE_PRIORITY[candidate.role] > ROLE_PRIORITY[existing.role]:
        seen[candidate.phone] = candidate


def extract_phones(
    text: str | None,
    context_window_size: int = 60,
    default_region: str | None = None,
) -> list[ExtractedPhone]:
    if not text:
        return []
    digit_count = 0
    for c in text:
        if c.isdigit():
            digit_count += 1
            if digit_count >= 7:
                break
    else:
        return []
    seen: dict[str, ExtractedPhone] = {}
    occupied_spans: list[tuple[int, int]] = []

    for match in _WHATSAPP_URL_PATTERN.finditer(text):
        span = (match.start(), match.end())
        if not _register_span(occupied_spans, span):
            continue
        number = match.group(1) or match.group(2)
        normalized = normalize_phone(f"+{number}")
        if normalized is None:
            continue
        context = get_context_window(text, match.start(), match.end(), context_window_size)
        role = score_context(context)
        _merge_candidate(seen, ExtractedPhone(normalized, True, role, match.group(0), match.start(), match.end()))

    for region in _ordered_regions(default_region):
        for match in PhoneNumberMatcher(text, region, leniency=Leniency.VALID):
            if not phonenumbers.is_valid_number(match.number):
                continue
            span = (match.start, match.end)
            if not _register_span(occupied_spans, span):
                continue
            normalized = phonenumbers.format_number(match.number, PhoneNumberFormat.E164)
            if normalized in seen:
                continue
            context = get_context_window(text, match.start, match.end, context_window_size)
            has_trigger = _PHONE_TRIGGER_RE.search(context) is not None
            if phonenumbers.number_type(match.number) not in (PhoneNumberType.MOBILE, PhoneNumberType.FIXED_LINE_OR_MOBILE) and not has_trigger:
                continue
            is_whatsapp = _WHATSAPP_TRIGGER_RE.search(context) is not None
            role = score_context(context)
            _merge_candidate(seen, ExtractedPhone(normalized, is_whatsapp, role, match.raw_string, match.start, match.end))

    return list(seen.values())