import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any

from urllib.parse import urlparse

from src.parser.creators.core.schemas import (
    AccountMetadata,
    ContentMetadata,
    Contacts,
)

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS: frozenset[str] = frozenset({
    "TELEGRAM",
    "VK",
    "RUTUBE",
    "YANDEX_DZEN",
    "INSTAGRAM",
    "TIKTOK",
    "YOUTUBE",
    "THREADS",
    "OK",
    "LINK_IN_BIO",
    "WEBSITE",
})

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

TELEGRAM_DOMAINS: frozenset[str] = frozenset({
    "t.me",
    "telegram.me",
    "telegram.dog",
    "tglink.ru",
})

VK_DOMAINS: frozenset[str] = frozenset({
    "vk.com",
    "vk.ru",
    "vkontakte.ru",
})

YANDEX_DZEN_DOMAINS: frozenset[str] = frozenset({
    "dzen.ru",
    "zen.yandex.ru",
    "zen.yandex.com",
})

INSTAGRAM_DOMAINS: frozenset[str] = frozenset({
    "instagram.com",
    "instagr.am",
})

TELEGRAM_NON_USER_PATHS: frozenset[str] = frozenset({
    "/addstickers",
    "/addemoji",
    "/share",
    "/socks",
    "/proxy",
    "/setlanguage",
    "/bg",
    "/addtheme",
    "/invoice",
})

OK_DOMAINS: frozenset[str] = frozenset({
    "ok.ru",
    "odnoklassniki.ru",
})


def generate_deterministic_id(platform: str, platform_id: str) -> int:
    key = f"{platform.upper()}:{platform_id.lower()}".encode("utf-8")
    hash_bytes = hashlib.sha256(key).digest()
    return int.from_bytes(hash_bytes[:8], byteorder="big") & 0x7FFFFFFFFFFFFFFF


def parse_url_domain(url: str) -> str | None:
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return None
        return parsed.netloc.lower()
    except Exception:
        return None


def extract_platform_info(url: str) -> tuple[str | None, str | None]:
    domain = parse_url_domain(url)
    if not domain:
        return None, None

    url_lower = url.lower()

    if any(d in domain for d in TELEGRAM_DOMAINS):
        if any(path in url_lower for path in TELEGRAM_NON_USER_PATHS):
            return None, None

        telegram_domain_regex = "|".join(re.escape(d) for d in TELEGRAM_DOMAINS)

        plus_match = re.search(rf"(?:{telegram_domain_regex})/\+([A-Za-z0-9_\-]{{6,}})", url, re.IGNORECASE)
        if plus_match:
            return "TELEGRAM", "+" + plus_match.group(1)

        joinchat_match = re.search(rf"(?:{telegram_domain_regex})/joinchat/([A-Za-z0-9_\-]{{6,}})", url, re.IGNORECASE)
        if joinchat_match:
            return "TELEGRAM", "+" + joinchat_match.group(1)

        match = re.search(rf"(?:{telegram_domain_regex})/(?:s/)?([^/?#]+)", url_lower)
        if match:
            username = match.group(1)
            if username and username not in ("joinchat", "join"):
                username_lower = username.lower()
                if username_lower.endswith("bot"):
                    return None, None
                return "TELEGRAM", username
        return None, None

    if any(d in domain for d in VK_DOMAINS):
        vk_skip_paths = {"/wall", "/photo", "/video", "/album", "/topic", "/doc", "/clip"}
        if any(path in url_lower for path in vk_skip_paths):
            return None, None

        for pattern in [r"vk\.com/([^/?#]+)", r"vk\.ru/([^/?#]+)", r"vkontakte\.ru/([^/?#]+)"]:
            match = re.search(pattern, url_lower)
            if match:
                return "VK", match.group(1)
        return None, None

    if "rutube.ru" in domain:
        match = re.search(r"rutube\.ru/([^/?#]+)", url_lower)
        if match:
            return "RUTUBE", match.group(1)
        return None, None

    if any(d in domain for d in YANDEX_DZEN_DOMAINS):
        id_match = re.search(r"(?:dzen\.ru|zen\.yandex\.[a-z]+)/id/([^/?#]+)", url_lower)
        if id_match:
            return "YANDEX_DZEN", f"id/{id_match.group(1)}"

        match = re.search(r"zen\.yandex\.[a-z]+/([^/?#]+)", url_lower)
        if match:
            return "YANDEX_DZEN", match.group(1)
        match = re.search(r"dzen\.ru/([^/?#]+)", url_lower)
        if match:
            return "YANDEX_DZEN", match.group(1)
        return None, None

    if any(d in domain for d in INSTAGRAM_DOMAINS):
        instagram_non_user_paths = {"/p/", "/reel/", "/reels/", "/stories/"}
        if any(path in url_lower for path in instagram_non_user_paths):
            return None, None
        match = re.search(r"instagram\.com/([^/?#]+)", url_lower)
        if match:
            return "INSTAGRAM", match.group(1)
        match = re.search(r"instagr\.am/([^/?#]+)", url_lower)
        if match:
            return "INSTAGRAM", match.group(1)
        return None, None

    if "tiktok.com" in domain:
        match = re.search(r"tiktok\.com/@([^/?#]+)", url)
        if match:
            return "TIKTOK", match.group(1)
        return None, None

    if any(d in domain for d in ["youtube.com", "youtu.be"]):
        if "/@" in url:
            handle = url.split("/@")[-1].split("?")[0].split("/")[0]
            if handle:
                return "YOUTUBE", handle
        elif "youtube.com/channel/" in url_lower:
            channel_id = url.split("/channel/")[-1].split("?")[0].split("/")[0]
            if channel_id:
                return "YOUTUBE", channel_id
        elif "youtube.com/c/" in url_lower:
            custom_handle = url.split("/c/")[-1].split("?")[0].split("/")[0]
            if custom_handle:
                return "YOUTUBE", custom_handle
        elif "youtube.com/user/" in url_lower:
            username = url.split("/user/")[-1].split("?")[0].split("/")[0]
            if username:
                return "YOUTUBE", username
        return None, None

    if "threads.net" in domain:
        if "/@" in url:
            username = url.split("/@")[-1].split("?")[0].split("/")[0]
            if username:
                return "THREADS", username
        else:
            username = url.split("/")[-1].split("?")[0].split("/")[0]
            if username:
                return "THREADS", username
        return None, None

    if any(d in domain for d in OK_DOMAINS):
        generic_paths = {"profile", "group", "live", "messages"}
        path_match = re.search(r"ok\.ru/([^/?#]+)", url_lower)
        if not path_match:
            path_match = re.search(r"odnoklassniki\.ru/([^/?#]+)", url_lower)
        if path_match:
            username = path_match.group(1)
            if username and username not in generic_paths:
                return "OK", username
        return None, None

    for bio_domain in LINK_IN_BIO_DOMAINS:
        if domain == bio_domain or domain.endswith("." + bio_domain):
            return "LINK_IN_BIO", url

    parsed_url = urlparse(url)
    if parsed_url.scheme in ("http", "https"):
        return "WEBSITE", url

    return None, None


def convert_dict_to_account_metadata(contacts_dict: dict[str, Any]) -> AccountMetadata:
    emails = contacts_dict.get("emails", [])
    telegram_handles = contacts_dict.get("telegram_handles", [])
    external_platforms_dict = contacts_dict.get("external_platforms", {})

    telegram_channels = []
    for handle in telegram_handles:
        if not handle:
            continue
        username = handle.lstrip("@")
        if username.lower().endswith("bot"):
            continue
        telegram_channels.append(username)
    telegram_personal: list[str] = []

    external_platforms: dict[str, str | None] = {
        k: v for k, v in external_platforms_dict.items() if isinstance(v, str | type(None))
    } if external_platforms_dict else {}

    contacts = Contacts(
        emails=[e for e in emails if e],
        telegram_channels=telegram_channels,
        telegram_personal=telegram_personal,
    )

    return AccountMetadata(
        contacts=contacts,
        external_platforms=external_platforms,
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )


def clean_content_raw_metadata(raw_metadata: dict[str, Any] | ContentMetadata | Any) -> dict[str, Any] | None:
    if raw_metadata is None:
        return None

    result: dict[str, Any] = {}

    if isinstance(raw_metadata, ContentMetadata):
        result = raw_metadata.model_dump(exclude_none=True)
    elif isinstance(raw_metadata, dict):
        try:
            validated = ContentMetadata.model_validate(raw_metadata)
            result = validated.model_dump(exclude_none=True)
        except Exception:
            result = {k: v for k, v in raw_metadata.items() if v is not None}
    else:
        logger.warning(
            "Unexpected raw_metadata type: %s, expected dict or ContentMetadata",
            type(raw_metadata).__name__,
        )
        return None

    result["schema_version"] = 1
    return result


def clean_account_raw_metadata(
    raw_metadata: AccountMetadata | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if raw_metadata is None:
        return None

    result: dict[str, Any] = {}

    if isinstance(raw_metadata, AccountMetadata):
        result = raw_metadata.model_dump(exclude_none=True)
    elif isinstance(raw_metadata, dict):
        try:
            validated = AccountMetadata.model_validate(raw_metadata)
            result = validated.model_dump(exclude_none=True)
        except Exception:
            result = {k: v for k, v in raw_metadata.items() if v is not None}
    else:
        logger.warning(
            "Unexpected raw_metadata type: %s, expected AccountMetadata, dict, or None",
            type(raw_metadata).__name__,
        )
        return None

    result["schema_version"] = 1
    return result
