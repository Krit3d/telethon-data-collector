import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from ..constants import LINK_IN_BIO_DOMAINS, PLATFORM_HANDLE_RULES, SOCIAL_MEDIA_DOMAINS
from ..context_scorer import ContactRole, score_context
from ..normalizer import is_media_asset, is_valid_web_url, normalize_url


@dataclass(slots=True, frozen=True)
class ExtractedLink:
    url: str
    title: str | None
    platform: str | None
    is_bio_link: bool
    role: ContactRole


EXTERNAL_PLATFORM_DOMAINS: dict[str, tuple[str, ...]] = {
    "vk": ("vk.com", "vk.ru", "vkontakte.ru", "vk.me"),
    "youtube": ("youtube.com", "youtu.be"),
    "threads": ("threads.net", "threads.com"),
    "tiktok": ("tiktok.com", "vm.tiktok.com"),
    "rutube": ("rutube.ru",),
    "dzen": ("dzen.ru", "zen.yandex.ru", "zen.yandex.com"),
    "ok": ("ok.ru", "odnoklassniki.ru"),
}

_URL_FINDER = re.compile(
    r"(?<![\w@/.-])(?:https?://[^\s<>\"'«»“”]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s<>\"'«»“”]*)?)",
    re.IGNORECASE,
)

def _extract_host(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _host_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def _extract_platform_handle(platform: str, host: str, path: str) -> str | None:
    parts = [p for p in path.split("/") if p]
    if not parts:
        return None
    rules = PLATFORM_HANDLE_RULES.get(platform)
    if not rules:
        return parts[0].lstrip("@").rstrip(".,;:!?)]}>\"'«»“”") or None
    if host in rules.get("ignore_hosts", ()):
        return None
    first = parts[0].lstrip("@").rstrip(".,;:!?)]}>\"'«»“”")
    pattern = rules.get("content_prefix_pattern")
    if pattern and re.match(pattern, first):
        return None
    if first in rules.get("system_sections", frozenset()):
        return None
    prefix_handlers = rules.get("prefix_handlers", {})
    if first in prefix_handlers:
        index = prefix_handlers[first]
        return parts[index] if len(parts) > index else None
    if rules.get("allow_at_prefix") and parts[0].startswith("@"):
        return first or None
    return first or None


def extract_bio_links(raw_bio_links: list[dict[str, Any]] | list[str] | None) -> list[ExtractedLink]:
    if not raw_bio_links:
        return []
    result: list[ExtractedLink] = []
    for item in raw_bio_links:
        if isinstance(item, dict):
            url = item.get("url")
            title = item.get("title")
            raw_url = str(url or "").strip(".,;:!?)]}>\"'«»“” ")
            if not raw_url or not is_valid_web_url(normalize_url(raw_url)) or "l.instagram.com" in raw_url:
                lynx_url = item.get("lynx_url")
                if lynx_url:
                    target = parse_qs(urlparse(str(lynx_url)).query).get("u", [None])[0]
                    if target:
                        url = unquote(target)
                    else:
                        url = raw_url
                else:
                    url = raw_url
        elif isinstance(item, str):
            url = item
            title = None
        else:
            continue
        if not url:
            continue
        normalized = normalize_url(str(url))
        if not normalized or not is_valid_web_url(normalized):
            continue
        title_text = str(title) if title is not None else ""
        role = score_context(title_text.lower()) if title_text else ContactRole.UNKNOWN
        host = _extract_host(normalized)
        is_bio_link = any(_host_matches(host, d) for d in LINK_IN_BIO_DOMAINS)
        platform = next(
            (p for p, domains in EXTERNAL_PLATFORM_DOMAINS.items() if any(_host_matches(host, d) for d in domains)),
            None,
        )
        result.append(ExtractedLink(normalized, title_text or None, platform, is_bio_link, role))
    return result


def extract_external_platforms(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    result: dict[str, str] = {}
    for match in _URL_FINDER.finditer(text):
        normalized = normalize_url(match.group(0))
        host = _extract_host(normalized)
        for platform, domains in EXTERNAL_PLATFORM_DOMAINS.items():
            if any(_host_matches(host, d) for d in domains):
                handle = _extract_platform_handle(platform, host, urlparse(normalized).path)
                if handle:
                    result[platform] = handle
                break
    return result


def extract_external_links(text: str | None, exclude_domains: frozenset[str] | None = None) -> list[str]:
    if not text:
        return []
    excluded = SOCIAL_MEDIA_DOMAINS | LINK_IN_BIO_DOMAINS | (exclude_domains or frozenset())
    result: list[str] = []
    seen: set[str] = set()
    for match in _URL_FINDER.finditer(text):
        cleaned = normalize_url(match.group(0))
        if not is_valid_web_url(cleaned):
            continue
        if any(_host_matches(_extract_host(cleaned), d) for d in excluded):
            continue
        if is_media_asset(urlparse(cleaned).path) or is_media_asset(cleaned):
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result