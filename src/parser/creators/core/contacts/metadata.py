import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from ..schemas import AccountMetadata, Contacts
from .constants import (
    LINK_IN_BIO_DOMAINS,
    PLATFORM_PROFILE_LINKS,
    SOCIAL_MEDIA_DOMAINS,
)
from .engine import ContactEngine, ExtractionPayload
from .extractors import (
    extract_bio_links,
    extract_external_links,
    extract_external_platforms,
    extract_telegram_contacts,
    normalize_telegram_handle,
)
from .normalizer import deduplicate_preserve_order, normalize_url

_MENTION_PATTERN = re.compile(r"(?<![\w.-])@([A-Za-z0-9_.]{1,30})")


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item and isinstance(item, str)]


def extract_mentions(text: str | None) -> list[str]:
    if not text:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for match in _MENTION_PATTERN.finditer(text):
        username = match.group(1).strip(".,!?-_")
        if not username:
            continue
        username = username.lower()
        if username not in seen:
            seen.add(username)
            result.append(username)
    return result


def extract_telegram_handles(text: str | None) -> list[str]:
    if not text:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for extracted in extract_telegram_contacts(text):
        handle = normalize_telegram_handle(extracted.handle)
        if handle and handle not in seen:
            seen.add(handle)
            result.append(handle)
    return result


def parse_profile_contacts(biography: str | None, external_url: str | None = None) -> dict[str, Any]:
    engine = ContactEngine()
    contacts = engine.extract_contacts(ExtractionPayload(biography=biography, external_url=external_url))
    combined = biography or ""
    if external_url:
        normalized_external = normalize_url(external_url)
        combined = f"{combined}\n{normalized_external}" if combined else normalized_external
    mentions = extract_mentions(combined)
    external_platforms = extract_external_platforms(combined)
    external_links = extract_external_links(combined)
    return {
        "emails": contacts.emails,
        "advertising_emails": contacts.advertising_emails,
        "phones": contacts.phones,
        "telegram_handles": contacts.telegram_handles,
        "mentions": mentions,
        "external_links": external_links,
        "external_platforms": external_platforms,
        "raw_bio": biography or "",
    }


def _merge_contacts_dict(contacts: Contacts, contacts_dict: dict[str, Any] | None) -> Contacts:
    if not contacts_dict:
        return contacts
    return Contacts(
        emails=deduplicate_preserve_order(contacts.emails + _as_str_list(contacts_dict.get("emails"))),
        phones=deduplicate_preserve_order(contacts.phones + _as_str_list(contacts_dict.get("phones"))),
        telegram_handles=deduplicate_preserve_order(contacts.telegram_handles + _as_str_list(contacts_dict.get("telegram_handles"))),
        telegram_channels=deduplicate_preserve_order(contacts.telegram_channels + _as_str_list(contacts_dict.get("telegram_channels"))),
        telegram_personal=deduplicate_preserve_order(contacts.telegram_personal + _as_str_list(contacts_dict.get("telegram_personal"))),
        advertising_emails=deduplicate_preserve_order(contacts.advertising_emails + _as_str_list(contacts_dict.get("advertising_emails"))),
        advertising_telegrams=deduplicate_preserve_order(contacts.advertising_telegrams + _as_str_list(contacts_dict.get("advertising_telegrams"))),
    )


def compile_author_metadata(
    platform: str,
    username: str | None,
    biography: str | None,
    contacts_dict: dict[str, Any],
    extra_links: list[str] | None = None,
    raw_profile_payload: dict[str, Any] | None = None,
    context_text: str | None = None,
    posts_content: list[str] | None = None,
    transcriptions: list[str] | None = None,
) -> AccountMetadata:
    profile_url: str = ""
    if username:
        template = PLATFORM_PROFILE_LINKS.get(platform.upper() if platform else "", "")
        if template:
            profile_url = template.format(username=username)
        else:
            safe_platform = (platform or "unknown").lower().replace(" ", "")
            profile_url = f"https://{safe_platform}.com/{username}"

    raw_bio_links: list[dict[str, Any]] | None = None
    if isinstance(raw_profile_payload, dict):
        raw_links = raw_profile_payload.get("bio_links")
        if isinstance(raw_links, list):
            raw_bio_links = [item for item in raw_links if isinstance(item, dict)]

    engine = ContactEngine()
    if biography or context_text or posts_content or contacts_dict or raw_bio_links:
        payload = ExtractionPayload(
            biography=biography,
            context_text=context_text,
            raw_bio_links=raw_bio_links,
            author_username=username,
            platform=platform,
            posts_content=posts_content,
            transcriptions=transcriptions,
        )
        contacts = _merge_contacts_dict(engine.extract_contacts(payload), contacts_dict)
    else:
        contacts = Contacts()

    external_links: list[str] = []
    links_from_dict = contacts_dict.get("external_links", []) if contacts_dict else []
    if isinstance(links_from_dict, list):
        external_links.extend(links_from_dict)
    if extra_links:
        external_links.extend(extra_links)

    bio_link_urls: list[str] = []
    if raw_bio_links:
        bio_link_urls = [link.url for link in extract_bio_links(raw_bio_links)]

    if bio_link_urls:
        external_links.extend(bio_link_urls)

    seen: set[str] = set()
    unique_external: list[str] = []
    for link in external_links:
        if link:
            normalized_link = normalize_url(link)
            if normalized_link not in seen:
                seen.add(normalized_link)
                unique_external.append(normalized_link)

    link_in_bio: str | None = None
    website: str | None = None
    remaining_external_links: list[str] = []

    for link in unique_external:
        host = (urlsplit(link).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host:
            if any(bio_domain in host for bio_domain in LINK_IN_BIO_DOMAINS):
                if link_in_bio is None:
                    link_in_bio = link
                    continue
            elif website is None and host not in SOCIAL_MEDIA_DOMAINS:
                website = link
                continue
        remaining_external_links.append(link)

    external_platforms_dict = contacts_dict.get("external_platforms", {}) if contacts_dict else {}
    if not isinstance(external_platforms_dict, dict):
        external_platforms_dict = {}
    external_platforms: dict[str, str | None] = {
        key: value
        for key, value in external_platforms_dict.items()
        if isinstance(value, str | type(None))
    } if external_platforms_dict else {}

    for url in unique_external:
        for pl_key, pl_val in extract_external_platforms(url).items():
            if pl_key not in external_platforms or external_platforms[pl_key] is None:
                external_platforms[pl_key] = pl_val

    return AccountMetadata(
        profile_url=profile_url or None,
        biography=biography or None,
        contacts=contacts,
        external_platforms=external_platforms,
        link_in_bio=link_in_bio,
        website=website,
        external_links=remaining_external_links,
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
    raw_profile_payload: dict[str, Any] | None = None,
    context_text: str | None = None,
    posts_content: list[str] | None = None,
    transcriptions: list[str] | None = None,
) -> dict[str, Any]:
    account_metadata = compile_author_metadata(
        platform=platform,
        username=username,
        biography=biography,
        contacts_dict=contacts_dict,
        extra_links=extra_links,
        raw_profile_payload=raw_profile_payload,
        context_text=context_text,
        posts_content=posts_content,
        transcriptions=transcriptions,
    )
    return account_metadata.model_dump(exclude_none=True)