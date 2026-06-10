import logging
from typing import Any

from src.parser.creators.core.contacts import parse_profile_contacts, extract_mentions
from src.parser.creators.core.db.discovery_repo import (
    queue_discovered_accounts,
    queue_discovered_mentions,
)

logger = logging.getLogger(__name__)


async def process_and_queue_discovered_contacts(
    session_maker: Any,
    parent_username: str,
    account_category: str,
    profile_biography: str | None,
    profile_external_url: str | None,
    items_data: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregated_emails: list[str] = []
    aggregated_telegram_handles: list[str] = []
    aggregated_external_links: list[str] = []
    aggregated_external_platforms: dict[str, str] = {}
    aggregated_mentions: set[str] = set()

    bio_contacts = parse_profile_contacts(profile_biography, profile_external_url)
    for email in bio_contacts.get("emails", []):
        if email and email not in aggregated_emails:
            aggregated_emails.append(email)
    for handle in bio_contacts.get("telegram_handles", []):
        if handle and handle not in aggregated_telegram_handles:
            aggregated_telegram_handles.append(handle)
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

        contacts_dict = parse_profile_contacts(content_text)
        mentions = extract_mentions(content_text)

        for email in contacts_dict.get("emails", []):
            if email and email not in aggregated_emails:
                aggregated_emails.append(email)

        for handle in contacts_dict.get("telegram_handles", []):
            if handle and handle not in aggregated_telegram_handles:
                aggregated_telegram_handles.append(handle)

        for link in contacts_dict.get("external_links", []):
            if link and link not in aggregated_external_links:
                aggregated_external_links.append(link)

        for platform_slug, handle in contacts_dict.get("external_platforms", {}).items():
            if handle and platform_slug not in aggregated_external_platforms:
                aggregated_external_platforms[platform_slug] = handle

        aggregated_mentions.update(mentions)

    aggregated_contacts: dict[str, Any] = {
        "emails": aggregated_emails,
        "telegram_handles": aggregated_telegram_handles,
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

    if has_contacts or aggregated_mentions:
        try:
            async with session_maker() as session:
                if has_contacts:
                    await queue_discovered_accounts(
                        session=session,
                        metadata=aggregated_contacts,
                        parent_handle=parent_username,
                        status="pending",
                        category=account_category,
                    )
                if aggregated_mentions:
                    await queue_discovered_mentions(
                        session=session,
                        platform="INSTAGRAM",
                        mentions=list(aggregated_mentions),
                        parent_handle=parent_username,
                        status="pending",
                        category=account_category,
                    )
                await session.commit()
        except Exception as e:
            logger.warning(
                "Failed to queue discovered contacts from Instagram content for %s: %s",
                parent_username,
                e,
            )

    return aggregated_contacts
