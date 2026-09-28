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
    extract_phones,
    extract_telegram_contacts,
    normalize_telegram_handle,
)
from .normalizer import deduplicate_preserve_order, is_valid_web_url, normalize_url

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


def parse_profile_contacts(
    biography: str | None,
    external_url: str | None = None,
    author_username: str | None = None,
) -> dict[str, Any]:
    engine = ContactEngine()
    contacts = engine.extract_contacts(
        ExtractionPayload(
            biography=biography,
            external_url=external_url,
            author_username=author_username,
        )
    )
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
        "telegram_channels": contacts.telegram_channels,
        "telegram_personal": contacts.telegram_personal,
        "advertising_telegrams": contacts.advertising_telegrams,
        "mentions": mentions,
        "external_links": external_links,
        "external_platforms": external_platforms,
        "raw_bio": biography or "",
    }


def extract_structural_links(
    biography: str | None,
    raw_profile_payload: dict[str, Any] | None = None,
    extra_links: list[str] | None = None,
    contacts_dict: dict[str, Any] | None = None,
) -> tuple[str | None, str | None, dict[str, str | None], list[str]]:
    raw_bio_links: list[dict[str, Any]] | None = None
    if isinstance(raw_profile_payload, dict):
        raw_links = raw_profile_payload.get("bio_links")
        if isinstance(raw_links, list):
            raw_bio_links = [item for item in raw_links if isinstance(item, dict)]

    collected: list[str] = []
    if isinstance(raw_profile_payload, dict):
        external_url = raw_profile_payload.get("external_url")
        if isinstance(external_url, str) and external_url:
            collected.append(external_url)
    if raw_bio_links:
        collected.extend(link.url for link in extract_bio_links(raw_bio_links))
    if biography:
        collected.extend(extract_external_links(biography))
    if extra_links:
        collected.extend(extra_links)
    if contacts_dict:
        links_from_dict = contacts_dict.get("external_links", [])
        if isinstance(links_from_dict, list):
            collected.extend(links_from_dict)

    seen: set[str] = set()
    unique_external: list[str] = []
    for link in collected:
        if link:
            normalized_link = normalize_url(link)
            if normalized_link and is_valid_web_url(normalized_link) and normalized_link not in seen:
                seen.add(normalized_link)
                unique_external.append(normalized_link)

    link_in_bio: str | None = None
    website: str | None = None
    remaining_external_links: list[str] = []

    website_candidates: list[str] = []
    for link in unique_external:
        extracted_phones = extract_phones(link)
        if extracted_phones and contacts_dict is not None:
            existing_phones = contacts_dict.get("phones")
            if not isinstance(existing_phones, list):
                existing_phones = []
                contacts_dict["phones"] = existing_phones
            for extracted in extracted_phones:
                if extracted.phone not in existing_phones:
                    existing_phones.append(extracted.phone)
        host = (urlsplit(link).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        is_bio = any(host == d or host.endswith("." + d) for d in LINK_IN_BIO_DOMAINS)
        is_social = any(host == d or host.endswith("." + d) for d in SOCIAL_MEDIA_DOMAINS)
        if is_bio:
            if link_in_bio is None:
                link_in_bio = link
            continue
        if not is_social and not extracted_phones:
            website_candidates.append(link)
            continue

    if website_candidates:
        website = min(
            website_candidates,
            key=lambda candidate: (
                len(urlsplit(candidate).path),
                bool(urlsplit(candidate).query),
            ),
        )
        for candidate in website_candidates:
            if candidate != website:
                remaining_external_links.append(candidate)

    external_platforms: dict[str, str | None] = {}
    if contacts_dict:
        external_platforms_dict = contacts_dict.get("external_platforms", {})
        if isinstance(external_platforms_dict, dict):
            external_platforms = {
                key: value
                for key, value in external_platforms_dict.items()
                if isinstance(value, str | type(None))
            }

    if biography:
        for pl_key, pl_val in extract_external_platforms(biography).items():
            if pl_key not in external_platforms or external_platforms[pl_key] is None:
                external_platforms[pl_key] = pl_val

    for url in unique_external:
        for pl_key, pl_val in extract_external_platforms(url).items():
            if pl_key not in external_platforms or external_platforms[pl_key] is None:
                external_platforms[pl_key] = pl_val

    return link_in_bio, website, external_platforms, remaining_external_links


def _merge_contacts_dict(contacts: Contacts, contacts_dict: dict[str, Any] | None) -> Contacts:
    if not contacts_dict:
        return contacts
    merged_advertising = deduplicate_preserve_order(
        contacts.advertising_telegrams + _as_str_list(contacts_dict.get("advertising_telegrams"))
    )
    raw_channels = deduplicate_preserve_order(
        contacts.telegram_channels + _as_str_list(contacts_dict.get("telegram_channels"))
    )
    merged_channels = [h for h in raw_channels if h not in merged_advertising]
    raw_personal = deduplicate_preserve_order(
        contacts.telegram_personal + _as_str_list(contacts_dict.get("telegram_personal"))
    )
    merged_personal = [h for h in raw_personal if h not in merged_advertising and h not in merged_channels]
    merged_handles = deduplicate_preserve_order(
        contacts.telegram_handles
        + _as_str_list(contacts_dict.get("telegram_handles"))
        + _as_str_list(contacts_dict.get("other_telegrams"))
        + merged_advertising
        + merged_channels
        + merged_personal
    )
    return Contacts(
        emails=deduplicate_preserve_order(contacts.emails + _as_str_list(contacts_dict.get("emails"))),
        phones=deduplicate_preserve_order(contacts.phones + _as_str_list(contacts_dict.get("phones"))),
        telegram_handles=merged_handles,
        telegram_channels=merged_channels,
        telegram_personal=merged_personal,
        advertising_emails=deduplicate_preserve_order(contacts.advertising_emails + _as_str_list(contacts_dict.get("advertising_emails"))),
        advertising_telegrams=merged_advertising,
    )


def _enforce_telegram_exclusivity(contacts: Contacts) -> Contacts:
    advertising = deduplicate_preserve_order(contacts.advertising_telegrams)
    channels = [h for h in deduplicate_preserve_order(contacts.telegram_channels) if h not in advertising]
    personal = [h for h in deduplicate_preserve_order(contacts.telegram_personal) if h not in advertising and h not in channels]
    handles = deduplicate_preserve_order(contacts.telegram_handles + advertising + channels + personal)
    return Contacts(
        emails=contacts.emails,
        phones=contacts.phones,
        telegram_handles=handles,
        telegram_channels=channels,
        telegram_personal=personal,
        advertising_emails=contacts.advertising_emails,
        advertising_telegrams=advertising,
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

    business_email = raw_profile_payload.get("business_email") if isinstance(raw_profile_payload, dict) else None
    business_phone = raw_profile_payload.get("business_phone_number") if isinstance(raw_profile_payload, dict) else None
    trusted_emails = [business_email] if isinstance(business_email, str) and business_email.strip() else []
    trusted_phones = [business_phone] if isinstance(business_phone, str) and business_phone.strip() else []

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
            trusted_emails=trusted_emails,
            trusted_phones=trusted_phones,
        )
        contacts = _merge_contacts_dict(engine.extract_contacts(payload), contacts_dict)
    else:
        contacts = Contacts()

    contacts = _enforce_telegram_exclusivity(contacts)

    link_in_bio, website, external_platforms, remaining_external_links = extract_structural_links(
        biography=biography,
        raw_profile_payload=raw_profile_payload,
        extra_links=extra_links,
        contacts_dict=contacts_dict,
    )

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