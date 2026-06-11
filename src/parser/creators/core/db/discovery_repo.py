import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import DatabaseError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Account
from src.parser.creators.core.contacts import URL_PATTERN
from src.parser.creators.core.db.helpers import (
    generate_deterministic_id,
    extract_platform_info,
    convert_dict_to_account_metadata,
)
from src.parser.creators.core.schemas import AccountMetadata

logger = logging.getLogger(__name__)


async def queue_discovered_accounts(
    session: AsyncSession,
    metadata: AccountMetadata | dict[str, Any],
    parent_handle: str,
    status: str = "pending",
    category: str | None = None,
) -> None:
    if isinstance(metadata, dict):
        metadata = convert_dict_to_account_metadata(metadata)

    telegram_channels: list[str] = []
    external_platforms = metadata.external_platforms
    biography: str | None = metadata.biography
    website: str | None = metadata.website
    link_in_bio: str | None = metadata.link_in_bio

    if metadata.contacts:
        telegram_channels = metadata.contacts.telegram_channels

    queued_accounts: set[tuple[str, str]] = set()

    async def _queue_if_new(platform: str, platform_id: str) -> None:
        key = (platform, platform_id)
        if key not in queued_accounts:
            queued_accounts.add(key)
            await queue_single_account(
                session, platform, platform_id, parent_handle, status, category
            )

    for handle in telegram_channels:
        if not handle:
            continue
        if handle.startswith("+"):
            continue
        clean_handle = handle.lstrip("@")
        await _queue_if_new("TELEGRAM", clean_handle)

    if external_platforms:
        platform_mapping = {
            "vk": "VK",
            "youtube": "YOUTUBE",
            "threads": "THREADS",
            "tiktok": "TIKTOK",
            "rutube": "RUTUBE",
            "yandex_dzen": "YANDEX_DZEN",
            "ok": "OK",
        }

        for platform_slug, platform_name in platform_mapping.items():
            handle = getattr(external_platforms, platform_slug, None)
            if handle:
                await _queue_if_new(platform_name, handle)

    urls_to_scan: list[str] = []

    if biography:
        bio_urls = URL_PATTERN.findall(biography)
        urls_to_scan.extend(bio_urls)

    if website:
        urls_to_scan.append(website)

    if link_in_bio:
        urls_to_scan.append(link_in_bio)

    seen_urls: set[str] = set()
    unique_urls: list[str] = []
    for url in urls_to_scan:
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_urls.append(url)

    for url in unique_urls:
        platform, platform_id = extract_platform_info(url)
        if platform and platform_id:
            if platform in ("WEBSITE", "LINK_IN_BIO"):
                continue
            await _queue_if_new(platform, platform_id)

    if not queued_accounts:
        logger.debug(
            "[SPIDER] No external social accounts discovered in bio of parent account %s.",
            parent_handle,
        )


async def queue_discovered_mentions(
    session: AsyncSession,
    platform: str,
    mentions: list[str],
    parent_handle: str,
    status: str = "pending",
    category: str | None = None,
) -> None:
    for username in mentions:
        if not username or len(username) < 3:
            continue
        if username.startswith("+"):
            continue
        await queue_single_account(session, platform, username, parent_handle, status, category)


async def queue_single_account(
    session: AsyncSession,
    platform: str,
    platform_id: str,
    parent_handle: str,
    status: str = "pending",
    category: str | None = None,
) -> None:
    stmt = select(Account).where(
        Account.platform == platform,
        Account.platform_id == platform_id,
    )
    result = await session.execute(stmt)
    existing = result.scalar_one_or_none()

    if not existing:
        generated_id = generate_deterministic_id(platform, platform_id)
        raw_metadata = {"category": category} if category else None

        insert_stmt = insert(Account).values(
            id=generated_id,
            platform=platform,
            platform_id=platform_id,
            username=platform_id,
            title=platform_id,
            status=status,
            raw_metadata=raw_metadata,
        )
        insert_stmt = insert_stmt.on_conflict_do_nothing(index_elements=["id"])

        try:
            await session.execute(insert_stmt)
            await session.flush()
            logger.info(
                "[SPIDER] Queued discovered %s account: %s from bio of parent account %s (category: %s).",
                platform,
                platform_id,
                parent_handle,
                category or "none",
            )
        except DatabaseError as e:
            await session.rollback()
            logger.warning(
                "Database error while queuing %s account %s: %s",
                platform,
                platform_id,
                e,
            )
