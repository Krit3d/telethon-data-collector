import mimetypes
import re
from dataclasses import dataclass

from ..context_scorer import ROLE_PRIORITY, ContactRole, get_context_window, score_context
from ..normalizer import deobfuscate_text, is_media_asset


@dataclass(slots=True, frozen=True)
class ExtractedEmail:
    email: str
    role: ContactRole
    raw_match: str
    start: int
    end: int


_SUBDOMAIN_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_DOMAIN_LABEL = r"[A-Za-z0-9][A-Za-z0-9-]{0,61}[A-Za-z0-9]"

EMAIL_PATTERN = re.compile(
    r"(?<![\w.-])"
    r"([A-Za-z0-9._%+-]+@"
    r"(?:" + _SUBDOMAIN_LABEL + r"\.)*"
    + _DOMAIN_LABEL + r"\."
    r"[A-Za-z]{2,24})"
    r"(?![A-Za-z0-9_@])",
    re.IGNORECASE,
)

_ARCHIVE_MIME_TYPES: frozenset[str] = frozenset((
    "application/zip",
    "application/x-tar",
    "application/pdf",
    "application/octet-stream",
))


def _is_file_asset_tld(tld: str) -> bool:
    mime_type = mimetypes.guess_type(f"file.{tld}")[0]
    if mime_type is None:
        return False
    if mime_type.startswith(("image/", "video/", "audio/", "font/")):
        return True
    return mime_type in _ARCHIVE_MIME_TYPES


def is_valid_email(email: str) -> bool:
    if EMAIL_PATTERN.fullmatch(email) is None:
        return False
    local_part, _, domain = email.partition("@")
    if not (local_part and local_part[0].isalnum() and local_part[-1].isalnum()):
        return False
    for segment in domain.split("."):
        if not segment or segment[0] == "-" or segment[-1] == "-":
            return False
    tld = domain.rsplit(".", 1)[-1].lower()
    if not tld.isalpha():
        return False
    if not 2 <= len(tld) <= 24:
        return False
    if _is_file_asset_tld(tld):
        return False
    return True


def extract_emails(text: str | None, context_window_size: int = 60) -> list[ExtractedEmail]:
    if not text:
        return []
    cleaned = deobfuscate_text(text)
    candidates: dict[str, ExtractedEmail] = {}
    for match in EMAIL_PATTERN.finditer(cleaned):
        raw = match.group(0)
        email = raw.lower()
        if not is_valid_email(email):
            continue
        _, _, domain = email.partition("@")
        if is_media_asset(domain) or is_media_asset(email):
            continue
        role = score_context(get_context_window(cleaned, match.start(), match.end(), context_window_size))
        candidate = ExtractedEmail(email, role, raw, match.start(), match.end())
        existing = candidates.get(email)
        if existing is None or ROLE_PRIORITY[candidate.role] > ROLE_PRIORITY[existing.role]:
            candidates[email] = candidate
    return list(candidates.values())