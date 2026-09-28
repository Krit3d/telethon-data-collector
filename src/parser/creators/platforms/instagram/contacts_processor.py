import logging
from typing import Any

from src.parser.creators.core.contacts import (
    compile_author_metadata,
    extract_mentions,
    parse_profile_contacts,
)
from src.parser.creators.core.contacts.normalizer import deduplicate_preserve_order
from src.parser.creators.core.db.discovery_repo import (
    queue_discovered_accounts,
    queue_discovered_mentions,
)

logger = logging.getLogger(__name__)


def _empty_contacts(profile_biography: str | None) -> dict[str, Any]:
    return {
        "emails": [],
        "advertising_emails": [],
        "phones": [],
        "telegram_handles": [],
        "telegram_channels": [],
        "telegram_personal": [],
        "advertising_telegrams": [],
        "external_links": [],
        "external_platforms": {},
        "raw_bio": profile_biography or "",
    }


def _extract_username_str(element: Any) -> str | None:
    if isinstance(element, str):
        cleaned = element.strip().lstrip("@").strip()
        return cleaned or None
    if isinstance(element, dict):
        for key in ("username", "handle"):
            value = element.get(key)
            if isinstance(value, str):
                cleaned = value.strip().lstrip("@").strip()
                if cleaned:
                    return cleaned
        user = element.get("user")
        if isinstance(user, dict):
            value = user.get("username")
            if isinstance(value, str):
                cleaned = value.strip().lstrip("@").strip()
                if cleaned:
                    return cleaned
        ig_artist = element.get("ig_artist")
        if isinstance(ig_artist, dict):
            value = ig_artist.get("username")
            if isinstance(value, str):
                cleaned = value.strip().lstrip("@").strip()
                if cleaned:
                    return cleaned
    return None


def _sanitize_username(value: Any, parent_lower: str) -> str | None:
    username = _extract_username_str(value)
    if not username:
        return None
    username = username.lower()
    if not (3 <= len(username) <= 30):
        return None
    if not all(c.isalnum() or c in "._" for c in username):
        return None
    if username == parent_lower:
        return None
    return username


def _collect_from_list(value: Any, parent_lower: str, target: set[str]) -> None:
    if not isinstance(value, list):
        return
    for element in value:
        username = _sanitize_username(element, parent_lower)
        if username:
            target.add(username)


def _collect_usertags(value: Any, parent_lower: str, target: set[str]) -> None:
    if isinstance(value, dict):
        _collect_from_list(value.get("in"), parent_lower, target)
    elif isinstance(value, list):
        _collect_from_list(value, parent_lower, target)


async def process_and_queue_discovered_contacts(
    session_maker: Any,
    parent_username: str,
    profile_biography: str | None,
    profile_external_url: str | None,
    items_data: list[dict[str, Any]],
    enable_contact_extraction: bool = False,
) -> tuple[dict[str, Any], int]:
    parent_lower = parent_username.lower()
    aggregated_mentions: set[str] = set()
    spider_count: int = 0

    if profile_biography:
        for mention in extract_mentions(profile_biography):
            username = _sanitize_username(mention, parent_lower)
            if username:
                aggregated_mentions.add(username)

    for item_data in items_data:
        raw_item = item_data.get("item")
        item = raw_item if isinstance(raw_item, dict) else item_data

        for mention in extract_mentions(item_data.get("content_text") or ""):
            username = _sanitize_username(mention, parent_lower)
            if username:
                aggregated_mentions.add(username)

        for key in ("coauthor_producers", "invited_coauthor_producers", "coauthors"):
            _collect_from_list(item.get(key), parent_lower, aggregated_mentions)

        _collect_usertags(item.get("usertags"), parent_lower, aggregated_mentions)
        _collect_usertags(item.get("fb_user_tags"), parent_lower, aggregated_mentions)
        _collect_from_list(item.get("tagged_users"), parent_lower, aggregated_mentions)

        carousel_media = item.get("carousel_media")
        if isinstance(carousel_media, list):
            for slide in carousel_media:
                if not isinstance(slide, dict):
                    continue
                _collect_usertags(slide.get("usertags"), parent_lower, aggregated_mentions)
                _collect_usertags(slide.get("fb_user_tags"), parent_lower, aggregated_mentions)

        mashup_info = item.get("mashup_info")
        if isinstance(mashup_info, dict):
            original_media = mashup_info.get("original_media")
            if isinstance(original_media, dict):
                _collect_from_list(
                    [original_media.get("user")], parent_lower, aggregated_mentions
                )

        clips_metadata = item.get("clips_metadata")
        if isinstance(clips_metadata, dict):
            original_sound_info = clips_metadata.get("original_sound_info")
            if isinstance(original_sound_info, dict):
                _collect_from_list(
                    [original_sound_info.get("ig_artist")],
                    parent_lower,
                    aggregated_mentions,
                )

    if aggregated_mentions:
        try:
            async with session_maker() as session:
                spider_count += await queue_discovered_mentions(
                    session=session,
                    platform="INSTAGRAM",
                    mentions=list(aggregated_mentions),
                    parent_handle=parent_username,
                    status="pending",
                )
                await session.commit()
        except Exception as e:
            logger.warning(
                "Failed to queue discovered mentions from Instagram content for %s: %s",
                parent_username,
                e,
            )

    if not enable_contact_extraction:
        return _empty_contacts(profile_biography), spider_count

    aggregated_emails: list[str] = []
    aggregated_advertising_emails: list[str] = []
    aggregated_phones: list[str] = []
    aggregated_telegram_handles: list[str] = []
    aggregated_telegram_channels: list[str] = []
    aggregated_telegram_personal: list[str] = []
    aggregated_advertising_telegrams: list[str] = []
    aggregated_external_links: list[str] = []
    aggregated_external_platforms: dict[str, str] = {}

    bio_contacts = parse_profile_contacts(
        profile_biography, profile_external_url, author_username=parent_username
    )
    for email in bio_contacts.get("emails", []):
        if email and email not in aggregated_emails:
            aggregated_emails.append(email)
    for email in bio_contacts.get("advertising_emails", []):
        if email and email not in aggregated_advertising_emails:
            aggregated_advertising_emails.append(email)
    for handle in bio_contacts.get("telegram_handles", []):
        if handle and handle not in aggregated_telegram_handles:
            aggregated_telegram_handles.append(handle)
    for phone in bio_contacts.get("phones", []):
        if phone and phone not in aggregated_phones:
            aggregated_phones.append(phone)
    for handle in bio_contacts.get("telegram_channels", []):
        if handle and handle not in aggregated_telegram_channels:
            aggregated_telegram_channels.append(handle)
    for handle in bio_contacts.get("telegram_personal", []):
        if handle and handle not in aggregated_telegram_personal:
            aggregated_telegram_personal.append(handle)
    for handle in bio_contacts.get("advertising_telegrams", []):
        if handle and handle not in aggregated_advertising_telegrams:
            aggregated_advertising_telegrams.append(handle)
    for link in bio_contacts.get("external_links", []):
        if link and link not in aggregated_external_links:
            aggregated_external_links.append(link)
    for platform_slug, handle in bio_contacts.get("external_platforms", {}).items():
        if handle and platform_slug not in aggregated_external_platforms:
            aggregated_external_platforms[platform_slug] = handle

    for item_data in items_data:
        content_text = item_data.get("content_text")
        if not content_text:
            continue

        contacts_dict = parse_profile_contacts(content_text, author_username=parent_username)

        for email in contacts_dict.get("emails", []):
            if email and email not in aggregated_emails:
                aggregated_emails.append(email)

        for email in contacts_dict.get("advertising_emails", []):
            if email and email not in aggregated_advertising_emails:
                aggregated_advertising_emails.append(email)

        for handle in contacts_dict.get("telegram_handles", []):
            if handle and handle not in aggregated_telegram_handles:
                aggregated_telegram_handles.append(handle)

        for phone in contacts_dict.get("phones", []):
            if phone and phone not in aggregated_phones:
                aggregated_phones.append(phone)

        for handle in contacts_dict.get("telegram_channels", []):
            if handle and handle not in aggregated_telegram_channels:
                aggregated_telegram_channels.append(handle)

        for handle in contacts_dict.get("telegram_personal", []):
            if handle and handle not in aggregated_telegram_personal:
                aggregated_telegram_personal.append(handle)

        for handle in contacts_dict.get("advertising_telegrams", []):
            if handle and handle not in aggregated_advertising_telegrams:
                aggregated_advertising_telegrams.append(handle)

        for link in contacts_dict.get("external_links", []):
            if link and link not in aggregated_external_links:
                aggregated_external_links.append(link)

        for platform_slug, handle in contacts_dict.get("external_platforms", {}).items():
            if handle and platform_slug not in aggregated_external_platforms:
                aggregated_external_platforms[platform_slug] = handle

    context_parts: list[str] = []
    if profile_biography:
        context_parts.append(profile_biography)
    for item_data in items_data:
        item_content = item_data.get("content_text")
        if item_content:
            context_parts.append(item_content)
    context_text: str | None = "\n".join(context_parts) if context_parts else None

    aggregated_telegram_personal = [
        h for h in aggregated_telegram_personal if h not in aggregated_advertising_telegrams
    ]
    aggregated_telegram_channels = [
        h
        for h in aggregated_telegram_channels
        if h not in aggregated_advertising_telegrams and h not in aggregated_telegram_personal
    ]
    aggregated_telegram_handles = deduplicate_preserve_order(
        aggregated_telegram_handles
        + aggregated_advertising_telegrams
        + aggregated_telegram_personal
        + aggregated_telegram_channels
    )

    aggregated_contacts: dict[str, Any] = {
        "emails": aggregated_emails,
        "advertising_emails": aggregated_advertising_emails,
        "phones": aggregated_phones,
        "telegram_handles": aggregated_telegram_handles,
        "telegram_channels": aggregated_telegram_channels,
        "telegram_personal": aggregated_telegram_personal,
        "advertising_telegrams": aggregated_advertising_telegrams,
        "external_links": aggregated_external_links,
        "external_platforms": aggregated_external_platforms,
        "raw_bio": profile_biography or "",
    }

    has_contacts = any([
        aggregated_emails,
        aggregated_telegram_handles,
        aggregated_external_links,
        aggregated_external_platforms,
    ])

    if has_contacts:
        try:
            async with session_maker() as session:
                compiled_meta = compile_author_metadata(
                    platform="INSTAGRAM",
                    username=parent_username,
                    biography=profile_biography,
                    contacts_dict=aggregated_contacts,
                    context_text=context_text,
                )
                spider_count += await queue_discovered_accounts(
                    session=session,
                    metadata=compiled_meta,
                    parent_handle=parent_username,
                    status="pending",
                )
                await session.commit()
        except Exception as e:
            logger.warning(
                "Failed to queue discovered contacts from Instagram content for %s: %s",
                parent_username,
                e,
            )

    return aggregated_contacts, spider_count
