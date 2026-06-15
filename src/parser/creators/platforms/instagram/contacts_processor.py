import logging
from typing import Any

from src.parser.creators.core.contacts import parse_profile_contacts, extract_mentions, compile_author_metadata
from src.parser.creators.core.db.discovery_repo import (
    queue_discovered_accounts,
    queue_discovered_mentions,
)

logger = logging.getLogger(__name__)


def _extract_username_str(element: str | dict[str, Any]) -> str | None:
    if isinstance(element, str):
        return element.strip() or None
    if isinstance(element, dict):
        username = element.get("username")
        if username:
            return str(username).strip() or None
        user = element.get("user")
        if isinstance(user, dict):
            username = user.get("username")
            if username:
                return str(username).strip() or None
    return None


async def process_and_queue_discovered_contacts(
    session_maker: Any,
    parent_username: str,
    account_category: str,
    profile_biography: str | None,
    profile_external_url: str | None,
    items_data: list[dict[str, Any]],
) -> dict[str, Any]:
    parent_lower = parent_username.lower()
    aggregated_emails: list[str] = []
    aggregated_advertising_emails: list[str] = []
    aggregated_telegram_handles: list[str] = []
    aggregated_external_links: list[str] = []
    aggregated_external_platforms: dict[str, str] = {}
    aggregated_mentions: set[str] = set()

    bio_contacts = parse_profile_contacts(profile_biography, profile_external_url)
    for email in bio_contacts.get("emails", []):
        if email and email not in aggregated_emails:
            aggregated_emails.append(email)
    for email in bio_contacts.get("advertising_emails", []):
        if email and email not in aggregated_advertising_emails:
            aggregated_advertising_emails.append(email)
    for handle in bio_contacts.get("telegram_handles", []):
        if (
            handle
            and handle.lower() != parent_lower
            and handle not in aggregated_telegram_handles
        ):
            aggregated_telegram_handles.append(handle)
    for link in bio_contacts.get("external_links", []):
        if link and link not in aggregated_external_links:
            aggregated_external_links.append(link)
    for platform_slug, handle in bio_contacts.get("external_platforms", {}).items():
        if handle and platform_slug not in aggregated_external_platforms:
            aggregated_external_platforms[platform_slug] = handle

    if profile_biography:
        bio_mentions = extract_mentions(profile_biography)
        for mention in bio_mentions:
            if mention.lower() != parent_lower:
                aggregated_mentions.add(mention)

    for item_data in items_data:
        content_text = item_data.get("content_text")

        if content_text:
            contacts_dict = parse_profile_contacts(content_text)
            mentions = extract_mentions(content_text)

            for email in contacts_dict.get("emails", []):
                if email and email not in aggregated_emails:
                    aggregated_emails.append(email)

            for email in contacts_dict.get("advertising_emails", []):
                if email and email not in aggregated_advertising_emails:
                    aggregated_advertising_emails.append(email)

            for handle in contacts_dict.get("telegram_handles", []):
                if (
                    handle
                    and handle.lower() != parent_lower
                    and handle not in aggregated_telegram_handles
                ):
                    aggregated_telegram_handles.append(handle)

            for link in contacts_dict.get("external_links", []):
                if link and link not in aggregated_external_links:
                    aggregated_external_links.append(link)

            for platform_slug, handle in contacts_dict.get("external_platforms", {}).items():
                if handle and platform_slug not in aggregated_external_platforms:
                    aggregated_external_platforms[platform_slug] = handle

            for mention in mentions:
                if mention.lower() != parent_lower:
                    aggregated_mentions.add(mention)

        coauthors = item_data.get("coauthors")
        if isinstance(coauthors, list):
            for coauthor in coauthors:
                username = _extract_username_str(coauthor)
                if username and username.lower() != parent_lower:
                    aggregated_mentions.add(username.lower())

        tagged_users = item_data.get("tagged_users")
        if isinstance(tagged_users, list):
            for tagged in tagged_users:
                username = _extract_username_str(tagged)
                if username and username.lower() != parent_lower:
                    aggregated_mentions.add(username.lower())

    context_parts: list[str] = []
    if profile_biography:
        context_parts.append(profile_biography)
    for item_data in items_data:
        item_content = item_data.get("content_text")
        if item_content:
            context_parts.append(item_content)
    context_text: str | None = "\n".join(context_parts) if context_parts else None

    aggregated_contacts: dict[str, Any] = {
        "emails": aggregated_emails,
        "advertising_emails": aggregated_advertising_emails,
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
                    compiled_meta = compile_author_metadata(
                        platform="INSTAGRAM",
                        username=parent_username,
                        biography=profile_biography,
                        contacts_dict=aggregated_contacts,
                        category=account_category,
                        context_text=context_text,
                    )
                    await queue_discovered_accounts(
                        session=session,
                        metadata=compiled_meta,
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
