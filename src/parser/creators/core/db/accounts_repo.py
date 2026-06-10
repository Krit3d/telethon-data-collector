import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update, delete, or_
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Account, Content, Comment
from src.parser.creators.core.contacts import parse_profile_contacts, compile_author_metadata
from src.parser.creators.core.db.helpers import (
    generate_deterministic_id,
    SUPPORTED_PLATFORMS,
)
from src.parser.creators.core.db.discovery_repo import queue_discovered_accounts
from src.parser.creators.core.schemas import MetricsEntry

logger = logging.getLogger(__name__)


async def upsert_and_deduplicate_account(
    session: AsyncSession,
    platform: str,
    platform_id: str,
    username: str | None,
    title: str,
    description: str | None,
    subscribers_count: int | None,
    status: str,
) -> int:
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"Unsupported platform: {platform}. Must be one of {SUPPORTED_PLATFORMS}")

    conditions = []
    if platform_id:
        conditions.append(
            (Account.platform == platform) & (Account.platform_id == platform_id)
        )
    if username:
        conditions.append(
            (Account.platform == platform) & (Account.username == username)
        )

    if not conditions:
        generated_id = generate_deterministic_id(platform, platform_id or username or title)
        new_account = Account(
            id=generated_id,
            platform=platform,
            platform_id=platform_id or "",
            username=username,
            title=title,
            description=description,
            subscribers_count=subscribers_count,
            status=status,
        )
        session.add(new_account)
        await session.flush()
        return generated_id

    stmt = select(Account).where(
        Account.platform == platform,
        or_(*conditions),
    )
    result = await session.execute(stmt)
    existing_accounts = list(result.scalars().all())

    if not existing_accounts:
        generated_id = generate_deterministic_id(platform, platform_id or username or title)
        new_account = Account(
            id=generated_id,
            platform=platform,
            platform_id=platform_id or "",
            username=username,
            title=title,
            description=description,
            subscribers_count=subscribers_count,
            status=status,
        )
        session.add(new_account)
        await session.flush()
        logger.info(
            "Created new account: platform=%s, platform_id=%s, username=%s, id=%d",
            platform,
            platform_id,
            username,
            generated_id,
        )
        return generated_id

    if len(existing_accounts) == 1:
        account = existing_accounts[0]
        account.platform_id = platform_id or account.platform_id
        account.username = username or account.username
        account.title = title
        account.description = description if description is not None else account.description
        account.subscribers_count = (
            subscribers_count if subscribers_count is not None else account.subscribers_count
        )
        account.status = status
        await session.flush()
        logger.info(
            "Updated existing account: platform=%s, platform_id=%s, id=%d",
            platform,
            platform_id,
            account.id,
        )
        return account.id

    primary_account = None
    for account in existing_accounts:
        if account.platform_id and account.platform_id.isdigit():
            primary_account = account
            break

    if primary_account is None:
        primary_account = existing_accounts[0]

    primary_id = primary_account.id

    primary_account.platform_id = platform_id or primary_account.platform_id
    primary_account.username = username or primary_account.username
    primary_account.title = title
    primary_account.description = description if description is not None else primary_account.description
    primary_account.subscribers_count = (
        subscribers_count if subscribers_count is not None else primary_account.subscribers_count
    )
    primary_account.status = status

    duplicate_ids = [acc.id for acc in existing_accounts if acc.id != primary_id]
    if duplicate_ids:
        await session.execute(
            update(Content)
            .where(Content.account_id.in_(duplicate_ids))
            .values(account_id=primary_id)
        )

        await session.execute(
            update(Comment)
            .where(Comment.account_id.in_(duplicate_ids))
            .values(account_id=primary_id)
        )

        await session.execute(
            delete(Account).where(Account.id.in_(duplicate_ids))
        )

        logger.info(
            "Merged %d duplicate accounts into primary account %d for platform %s",
            len(duplicate_ids),
            primary_id,
            platform,
        )

    await session.flush()
    return primary_id


async def update_account_profile_metadata(
    session: AsyncSession,
    account_id: int,
    platform: str,
    biography: str | None,
    external_url: str | None = None,
    location: str | None = None,
    language: str | None = None,
    geo_data: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
    category: str | None = None,
    raw_profile_payload: dict[str, Any] | None = None,
    subscribers_count: int | None = None,
) -> dict[str, Any]:
    contacts: dict[str, Any] = {}
    if biography or external_url:
        contacts = parse_profile_contacts(biography, external_url)

    stmt = select(Account).where(Account.id == account_id)
    result = await session.execute(stmt)
    account = result.scalar_one_or_none()

    if not account:
        logger.warning("Account with id %d not found for metadata update", account_id)
        return {}

    username = account.username or account.platform_id

    resolved_category = category
    if resolved_category is None:
        if account.raw_metadata and isinstance(account.raw_metadata, dict):
            resolved_category = account.raw_metadata.get("category")
        if resolved_category is None:
            resolved_category = "unknown"

    compiled_metadata = compile_author_metadata(
        platform=platform,
        username=username,
        biography=biography,
        contacts_dict=contacts,
        extra_links=contacts.get("external_links", []),
        location=location,
        language=language,
        geo_data=geo_data,
        category=resolved_category,
        raw_profile_payload=raw_profile_payload,
    )

    if subscribers_count is not None:
        metrics_entry = MetricsEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            subscribers_count=subscribers_count,
            posts_count=None,
        )
        compiled_metadata.metrics_history.append(metrics_entry)

    if extra_meta:
        for key, value in extra_meta.items():
            if hasattr(compiled_metadata, key):
                current_value = getattr(compiled_metadata, key)
                if not current_value:
                    setattr(compiled_metadata, key, value)
            else:
                logger.debug(
                    "Extra meta key '%s' not found in AccountMetadata model, skipping",
                    key,
                )

    account.description = biography if biography is not None else account.description
    account.raw_metadata = compiled_metadata.model_dump(mode="json", exclude_none=False)

    if subscribers_count is not None:
        account.subscribers_count = subscribers_count
    await session.flush()

    logger.info("Updated profile metadata for account_id: %d", account_id)

    if contacts:
        parent_handle = account.username or account.platform_id or str(account_id)
        await queue_discovered_accounts(
            session, compiled_metadata, parent_handle, status="pending", category=resolved_category
        )

    return compiled_metadata.model_dump(mode="json", exclude_none=False)
