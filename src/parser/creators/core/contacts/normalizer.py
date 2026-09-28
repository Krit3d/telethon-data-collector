import mimetypes
import os
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PARAMS = frozenset(("fbclid", "igsh", "gclid", "yclid", "_openstat"))

_ARCHIVE_EXTENSIONS = frozenset((
    ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz", ".tgz", ".tbz2", ".zst",
))

_ZERO_WIDTH_CHARS = frozenset(("\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\ufeff"))

_OBFUSCATION_PATTERNS = (
    (re.compile(r"\[at\]"), "@"),
    (re.compile(r"\(at\)"), "@"),
    (re.compile(r"\[ @ \]"), "@"),
)

_TLD_TYPO_PATTERN = re.compile(r"(?<=[A-Za-z0-9]),\s*(com|ru|net|org)(?![A-Za-z0-9])")


def clean_tracking_params(url: str) -> str:
    parsed = urlsplit(url)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.startswith("utm_") and key not in TRACKING_PARAMS
    ]
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))


def is_valid_web_url(url: str) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        return False
    hostname = parsed.hostname
    if not hostname or "." not in hostname:
        return False
    if not all(c.isalnum() or c in ".-_" for c in hostname):
        return False
    tld = hostname.split(".")[-1]
    if len(tld) < 2 or not tld.isalpha():
        return False
    return True


def normalize_url(url: str) -> str:
    url = url.strip().strip(".,;:!?)]}>\"'«»“” ")
    if url and "://" not in url:
        url = "https://" + url
    url = clean_tracking_params(url)
    if not is_valid_web_url(url):
        return ""
    return url


_MEDIA_RAW_EXTENSIONS = frozenset((
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic", ".heif", ".svg", ".bmp", ".ico",
    ".avif", ".tiff", ".tif", ".jfif",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".3gp", ".flv",
    ".wmv", ".ogv",
    ".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a", ".wma", ".opus", ".mid", ".midi",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
))

_MEDIA_EXTENSIONS = _MEDIA_RAW_EXTENSIONS | _ARCHIVE_EXTENSIONS


def is_media_asset(target: str) -> bool:
    ext = os.path.splitext(target)[1].lower()
    if ext in _MEDIA_EXTENSIONS:
        return True
    mime_type, _ = mimetypes.guess_type(target)
    if mime_type is not None and mime_type.startswith(("image/", "video/", "audio/")):
        return True
    return False


def deobfuscate_text(text: str | None) -> str:
    if text is None:
        return ""
    for char in _ZERO_WIDTH_CHARS:
        text = text.replace(char, "")
    for pattern, replacement in _OBFUSCATION_PATTERNS:
        text = pattern.sub(replacement, text)
    text = _TLD_TYPO_PATTERN.sub(lambda m: "." + m.group(1), text)
    return "\n".join(" ".join(line.split()) for line in text.split("\n"))


def deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


__all__ = [
    "clean_tracking_params",
    "normalize_url",
    "is_valid_web_url",
    "is_media_asset",
    "deobfuscate_text",
    "deduplicate_preserve_order",
]