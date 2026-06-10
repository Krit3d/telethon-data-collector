import logging
import re
from datetime import datetime, timezone
from typing import Any

from src.parser.creators.core.schemas import (
    AccountMetadata,
    Contacts,
    ExternalPlatforms,
    GeoData
)

logger = logging.getLogger(__name__)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b",
    re.IGNORECASE,
)

TELEGRAM_URL_PATTERN = re.compile(
    r"(?:t\.me/|telegram\.me/|telegram\.dog/|tglink\.ru/"
    r"|https?://t\.me/|https?://telegram\.me/|https?://telegram\.dog/|https?://tglink\.ru/)"
    r"([A-Za-z0-9_]{5,32})",
    re.IGNORECASE,
)

TELEGRAM_CONTEXTUAL_PATTERN = re.compile(
    r"(?:tg|тг|telegram|телега|канал|channel|pr|ads|manager"
    r"|реклама|рекламе|сотрудничество|связь|контакты|пишите"
    r"|contact|cooperation|collabs|dm)"
    r"[\s:\-]{0,15}?@([A-Za-z0-9_]{5,32})",
    re.IGNORECASE,
)

TELEGRAM_INVITE_PATTERN = re.compile(
    r"t\.me/(?:\+|joinchat/)([A-Za-z0-9_\-]{6,})",
    re.IGNORECASE,
)

URL_PATTERN = re.compile(
    r"https?://[^\s<>\"{}|\\^`\[\]]+",
    re.IGNORECASE,
)

MENTION_PATTERN = re.compile(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([a-zA-Z0-9_\.]{1,30})")

PHONE_PATTERN = re.compile(
    r"\b(?:\+?7|8)[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}\b"
)

SOCIAL_MEDIA_DOMAINS: frozenset[str] = frozenset({
    "instagram.com",
    "tiktok.com",
    "youtube.com",
    "youtu.be",
    "threads.net",
    "threads.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "t.me",
    "telegram.me",
    "wa.me",
    "whatsapp.com",
    "vk.com",
    "vk.ru",
    "vkontakte.ru",
    "dzen.ru",
    "zen.yandex.ru",
    "zen.yandex.com",
    "rutube.ru",
    "ok.ru",
    "odnoklassniki.ru",
})

EXTERNAL_PLATFORM_DOMAINS: dict[str, list[str]] = {
    "vk": ["vk.com", "vk.ru", "vkontakte.ru"],
    "youtube": ["youtube.com", "youtu.be"],
    "threads": ["threads.net", "threads.com"],
    "tiktok": ["tiktok.com"],
}

TELEGRAM_PERSONAL_KEYWORDS: frozenset[str] = frozenset({
    "manager", "pr", "admin", "sales", "cooperation",
    "reklama", "advertising", "write", "contact",
})

TELEGRAM_CHANNEL_KEYWORDS: frozenset[str] = frozenset({
    "channel", "канал", "тгк", "телега", "блог", "blog",
    "t.me/joinchat", "t.me/+", "telegram.me/joinchat",
})

TELEGRAM_PERSONAL_PHRASES: list[str] = [
    "по рекламе",
    "пишите",
    "сотрудничество",
    "реклама",
    "advertising",
    "for ads",
]

_EMAIL_OBFUSCATION_RE = re.compile(
    r"\s*\[at\]\s*|\s*\(at\)\s*|\s*\[\s*@\s*\]\s*|\s+@\s+",
    re.IGNORECASE,
)


def extract_emails(text: str | None) -> list[str]:
    if not text:
        return []

    normalized = _EMAIL_OBFUSCATION_RE.sub("@", text)
    emails = EMAIL_PATTERN.findall(normalized)
    return list(set(email.lower() for email in emails))


def extract_external_platforms(
    text: str | None,
) -> dict[str, str]:
    if not text:
        return {}

    external_platforms: dict[str, str] = {}
    urls = URL_PATTERN.findall(text)

    for url in urls:
        url_lower = url.lower()

        for domain in EXTERNAL_PLATFORM_DOMAINS["vk"]:
            if domain in url_lower:
                match = re.search(rf"{re.escape(domain)}/([^/?#]+)", url_lower)
                if match:
                    handle = match.group(1)
                    if handle and handle not in ("id", "club", "public"):
                        external_platforms["vk"] = handle
                        break
                break

        for domain in EXTERNAL_PLATFORM_DOMAINS["youtube"]:
            if domain in url_lower:
                match = re.search(rf"{re.escape(domain)}/@([^/?#]+)", url_lower)
                if match:
                    external_platforms["youtube"] = f"@{match.group(1)}"
                    break
                match = re.search(rf"{re.escape(domain)}/channel/([^/?#]+)", url_lower)
                if match:
                    external_platforms["youtube"] = match.group(1)
                    break
                match = re.search(rf"{re.escape(domain)}/c/([^/?#]+)", url_lower)
                if match:
                    external_platforms["youtube"] = match.group(1)
                    break
                match = re.search(rf"{re.escape(domain)}/user/([^/?#]+)", url_lower)
                if match:
                    external_platforms["youtube"] = match.group(1)
                    break
                break

        for domain in EXTERNAL_PLATFORM_DOMAINS["threads"]:
            if domain in url_lower:
                match = re.search(rf"{re.escape(domain)}/@?([^/?#]+)", url_lower)
                if match:
                    handle = match.group(1)
                    if handle:
                        external_platforms["threads"] = handle
                    break
                break

        for domain in EXTERNAL_PLATFORM_DOMAINS["tiktok"]:
            if domain in url_lower:
                match = re.search(rf"{re.escape(domain)}/@([^/?#]+)", url_lower)
                if match:
                    external_platforms["tiktok"] = match.group(1)
                break

    return external_platforms


def extract_telegram_handles(text: str | None) -> list[str]:
    if not text:
        return []

    handles: set[str] = set()

    url_matches = TELEGRAM_URL_PATTERN.findall(text)
    for match in url_matches:
        normalized = match.lower()
        if normalized in ("joinchat", "join"):
            continue
        if normalized.endswith("bot"):
            continue
        handles.add(normalized)

    contextual_matches = TELEGRAM_CONTEXTUAL_PATTERN.findall(text)
    for match in contextual_matches:
        normalized = match.lower()
        if normalized in ("joinchat", "join"):
            continue
        if normalized.endswith("bot"):
            continue
        handles.add(normalized)

    for plus_match in re.finditer(r"t\.me/\+([A-Za-z0-9_\-]{6,})", text, re.IGNORECASE):
        handles.add("+" + plus_match.group(1))

    for joinchat_match in re.finditer(r"t\.me/joinchat/([A-Za-z0-9_\-]{6,})", text, re.IGNORECASE):
        handles.add("joinchat/" + joinchat_match.group(1))

    return list(handles)


def extract_mentions(text: str | None) -> list[str]:
    if not text:
        return []

    matches = MENTION_PATTERN.findall(text)
    return list(set(match.lower() for match in matches if match))


def extract_phones(text: str | None) -> list[str]:
    if not text:
        return []

    matches = PHONE_PATTERN.findall(text)
    seen: set[str] = set()
    unique_phones: list[str] = []
    for phone in matches:
        cleaned = "".join(c for c in phone if c.isdigit() or c == "+")
        if cleaned not in seen:
            seen.add(cleaned)
            unique_phones.append(cleaned)
    return unique_phones


def extract_external_links(
    text: str | None,
    exclude_domains: set[str] | None = None,
) -> list[str]:
    if not text:
        return []

    extended_excludes = SOCIAL_MEDIA_DOMAINS.copy()
    for domains in EXTERNAL_PLATFORM_DOMAINS.values():
        extended_excludes = extended_excludes.union(set(domains))
    if exclude_domains:
        extended_excludes = extended_excludes.union(exclude_domains)

    urls = URL_PATTERN.findall(text)

    external_links: list[str] = []
    for url in urls:
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
        if domain_match:
            domain = domain_match.group(1).lower()
            if not any(excluded in domain for excluded in extended_excludes):
                external_links.append(url)

    seen: set[str] = set()
    unique_links: list[str] = []
    for link in external_links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)

    return unique_links


def parse_profile_contacts(
    biography: str | None,
    external_url: str | None = None,
) -> dict[str, Any]:
    combined_text = biography or ""

    if external_url:
        if combined_text:
            combined_text = f"{combined_text}\n{external_url}"
        else:
            combined_text = external_url

    emails = extract_emails(combined_text)
    telegram_handles = extract_telegram_handles(combined_text)
    phones = extract_phones(combined_text)
    external_platforms = extract_external_platforms(combined_text)
    external_links = extract_external_links(combined_text)

    if external_url and external_url not in external_links:
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", external_url)
        if domain_match:
            domain = domain_match.group(1).lower()
            extended_excludes = SOCIAL_MEDIA_DOMAINS.copy()
            for domains in EXTERNAL_PLATFORM_DOMAINS.values():
                extended_excludes = extended_excludes.union(set(domains))
            if not any(excluded in domain for excluded in extended_excludes):
                external_links.append(external_url)

    return {
        "emails": emails,
        "phones": phones,
        "telegram_handles": telegram_handles,
        "external_links": external_links,
        "external_platforms": external_platforms,
        "raw_bio": biography or "",
    }


def normalize_telegram_handle(handle: str) -> str:
    handle = handle.strip()

    is_invite = handle.startswith("+") or "/joinchat/" in handle or handle.startswith("joinchat/")

    if handle.startswith("@"):
        handle = handle[1:]

    for prefix in [
        "t.me/",
        "telegram.me/",
        "https://t.me/",
        "https://telegram.me/",
        "http://t.me/",
        "http://telegram.me/",
    ]:
        if handle.lower().startswith(prefix):
            handle = handle[len(prefix):]
            break

    if is_invite:
        return handle
    return handle.lower()


def is_valid_email(email: str) -> bool:
    return bool(EMAIL_PATTERN.fullmatch(email))


def is_valid_telegram_handle(handle: str) -> bool:
    normalized = normalize_telegram_handle(handle)

    if not (5 <= len(normalized) <= 32):
        return False

    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{4,31}", normalized))


def classify_telegram_handles(
    biography: str | None,
    handles: list[str],
) -> tuple[list[str], list[str]]:
    if not handles:
        return ([], [])

    bio_lower = biography.lower() if biography else ""
    telegram_channels: list[str] = []
    telegram_personal: list[str] = []

    for handle in handles:
        if not handle:
            continue

        handle_lower = handle.lower()
        normalized_handle = normalize_telegram_handle(handle)

        is_personal = False
        is_channel = False

        for keyword in TELEGRAM_PERSONAL_KEYWORDS:
            if keyword in handle_lower:
                is_personal = True
                break

        if not is_personal and biography:
            for phrase in TELEGRAM_PERSONAL_PHRASES:
                if phrase in bio_lower:
                    is_personal = True
                    break

        if not is_personal:
            for keyword in TELEGRAM_CHANNEL_KEYWORDS:
                if keyword in handle_lower:
                    is_channel = True
                    break

        if handle and (handle.startswith("+") or handle.startswith("joinchat/") or
                      "joinchat" in handle_lower or "/+" in handle_lower or "t.me/+" in handle_lower):
            is_channel = True

        if not is_personal and not is_channel:
            is_channel = True

        if is_personal:
            telegram_personal.append(normalized_handle)
        else:
            telegram_channels.append(normalized_handle)

    telegram_channels = list(dict.fromkeys(telegram_channels))
    telegram_personal = list(dict.fromkeys(telegram_personal))

    return (telegram_channels, telegram_personal)


PLATFORM_PROFILE_LINKS: dict[str, str] = {
    "INSTAGRAM": "https://instagram.com/{username}",
    "TIKTOK": "https://www.tiktok.com/@{username}",
    "YOUTUBE": "https://youtube.com/@{username}",
    "THREADS": "https://threads.net/@{username}",
}

LINK_IN_BIO_DOMAINS: frozenset[str] = frozenset({
    "linktr.ee",
    "taplink.cc",
    "beacons.ai",
    "msha.ke",
    "solo.to",
    "lu.ma",
    "lnk.bio",
    "campsite.bio",
    "carrd.co",
    "linkin.bio",
    "bio.link",
    "linkbio.co",
    "my.link",
})


def compile_author_metadata(
    platform: str,
    username: str | None,
    biography: str | None,
    contacts_dict: dict[str, Any],
    extra_links: list[str] | None = None,
    location: str | None = None,
    language: str | None = None,
    geo_data: dict[str, Any] | None = None,
    category: str | None = None,
    raw_profile_payload: dict[str, Any] | None = None,
) -> AccountMetadata:
    profile_url: str = ""
    if username:
        template = PLATFORM_PROFILE_LINKS.get(platform.upper() if platform else "", "")
        if template:
            profile_url = template.format(username=username)
        else:
            safe_platform = (platform or "unknown").lower().replace(" ", "")
            profile_url = f"https://{safe_platform}.com/{username}"

    telegram_handles = contacts_dict.get("telegram_handles", []) if contacts_dict else []
    telegram_channels: list[str] = []
    telegram_personal: list[str] = []

    if telegram_handles:
        telegram_channels, telegram_personal = classify_telegram_handles(
            biography, telegram_handles
        )

    emails = contacts_dict.get("emails", []) if contacts_dict else []
    if not isinstance(emails, list):
        emails = []

    phones = contacts_dict.get("phones", []) if contacts_dict else []
    if not isinstance(phones, list):
        phones = []

    external_links: list[str] = []
    links_from_dict = contacts_dict.get("external_links", []) if contacts_dict else []
    if isinstance(links_from_dict, list):
        external_links.extend(links_from_dict)
    if extra_links:
        external_links.extend(extra_links)

    seen: set[str] = set()
    unique_external: list[str] = []
    for link in external_links:
        if link and link not in seen:
            seen.add(link)
            unique_external.append(link)

    link_in_bio: str | None = None
    website: str | None = None
    remaining_external_links: list[str] = []

    for link in unique_external:
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", link)
        if domain_match:
            domain = domain_match.group(1).lower()
            if any(bio_domain in domain for bio_domain in LINK_IN_BIO_DOMAINS):
                if link_in_bio is None:
                    link_in_bio = link
                continue
            if website is None and domain not in SOCIAL_MEDIA_DOMAINS:
                website = link
                continue
        remaining_external_links.append(link)

    external_platforms_dict = (
        contacts_dict.get("external_platforms", {}) if contacts_dict else {}
    )
    if not isinstance(external_platforms_dict, dict):
        external_platforms_dict = {}

    external_platforms = ExternalPlatforms(
        vk=external_platforms_dict.get("vk"),
        youtube=external_platforms_dict.get("youtube"),
        threads=external_platforms_dict.get("threads"),
        tiktok=external_platforms_dict.get("tiktok"),
    )

    contacts = Contacts(
        emails=[email.lower().strip() for email in emails if email and isinstance(email, str)],
        phones=[p for p in phones if p and isinstance(p, str)],
        telegram_channels=telegram_channels,
        telegram_personal=telegram_personal,
    )

    geo_data_model: GeoData | None = None
    if geo_data and isinstance(geo_data, dict):
        geo_data_model = GeoData(
            city=geo_data.get("city"),
            country=geo_data.get("country"),
            coordinates=geo_data.get("coordinates"),
        )

    return AccountMetadata(
        profile_url=profile_url or None,
        biography=biography or None,
        category=category,
        language=language,
        location=location,
        contacts=contacts,
        external_platforms=external_platforms,
        link_in_bio=link_in_bio,
        website=website,
        geo_data=geo_data_model,
        metrics_history=[],
        raw_profile_payload=raw_profile_payload,
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )


def compile_author_metadata_dict(
    platform: str,
    username: str | None,
    biography: str | None,
    contacts_dict: dict[str, Any],
    extra_links: list[str] | None = None,
    location: str | None = None,
    language: str | None = None,
    geo_data: dict[str, Any] | None = None,
    category: str | None = None,
    raw_profile_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    account_metadata = compile_author_metadata(
        platform=platform,
        username=username,
        biography=biography,
        contacts_dict=contacts_dict,
        extra_links=extra_links,
        location=location,
        language=language,
        geo_data=geo_data,
        category=category,
        raw_profile_payload=raw_profile_payload,
    )
    return account_metadata.model_dump(exclude_none=True)
