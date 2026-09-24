import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update, delete, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Account, Content, Comment
from src.parser.creators.core.contacts import parse_profile_contacts, compile_author_metadata
from src.parser.creators.core.db.helpers import (
    generate_deterministic_id,
    SUPPORTED_PLATFORMS,
)
from src.parser.creators.core.db.discovery_repo import queue_discovered_accounts
from src.parser.creators.core.schemas import AccountMetadata, MetricsEntry
from src.parser.creators.core.text import normalize_title, normalize_description

logger = logging.getLogger(__name__)

FINALIZED_STATUSES: frozenset[str] = frozenset({"parsed", "rejected", "verified"})
NON_FINALIZED_STATUSES: frozenset[str] = frozenset({"pending", "processing"})


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

    title = normalize_title(title)
    description = normalize_description(description)

    platform_id = platform_id.strip()
    if " " in platform_id or "\n" in platform_id or len(platform_id) > 100:
        logger.warning(
            "Invalid platform_id for platform=%s, first 50 chars: %r",
            platform,
            platform_id[:50],
        )
        platform_id = ""

    clean_platform_id = platform_id.strip().lower()
    clean_username = username.strip().lower() if username else None

    conditions = []
    if clean_platform_id:
        conditions.append(
            (Account.platform == platform) & (func.lower(Account.platform_id) == clean_platform_id)
        )
    if clean_username:
        conditions.append(
            (Account.platform == platform) & (func.lower(Account.username) == clean_username)
        )

    if not conditions:
        generated_id = generate_deterministic_id(platform, platform_id or username or title)
        new_account = Account(
            id=generated_id,
            platform=platform,
            platform_id=platform_id or "",
            username=clean_username,
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
            username=clean_username,
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
        if account.status == "verified":
            logger.info(
                "Skipping update for verified account: id=%d",
                account.id,
            )
            return account.id
        account.platform_id = platform_id or account.platform_id
        account.username = clean_username or account.username
        account.title = title
        account.description = description if description is not None else account.description
        account.subscribers_count = (
            subscribers_count if subscribers_count is not None else account.subscribers_count
        )
        if account.status not in FINALIZED_STATUSES or status not in NON_FINALIZED_STATUSES:
            account.status = status
        await session.flush()
        logger.info(
            "Updated existing account: platform=%s, platform_id=%s, id=%d",
            platform,
            platform_id,
            account.id,
        )
        return account.id

    primary_account: Account | None = None
    for account in existing_accounts:
        if account.status == "verified":
            primary_account = account
            break
    if primary_account is None:
        for account in existing_accounts:
            if account.platform_id and account.platform_id.isdigit():
                primary_account = account
                break
    if primary_account is None:
        primary_account = existing_accounts[0]

    primary_id = primary_account.id

    if primary_account.status != "verified":
        primary_account.platform_id = platform_id or primary_account.platform_id
        primary_account.username = clean_username or primary_account.username
        primary_account.title = title
        primary_account.description = description if description is not None else primary_account.description
        primary_account.subscribers_count = (
            subscribers_count if subscribers_count is not None else primary_account.subscribers_count
        )
        if primary_account.status not in FINALIZED_STATUSES or status not in NON_FINALIZED_STATUSES:
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


def _normalize_email(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if "@" in cleaned and "." in cleaned:
        return cleaned
    return None


def _normalize_phone(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = "".join(ch for ch in value if ch.isdigit() or ch == "+")
    if len(cleaned) >= 7:
        return cleaned
    return None


def _enrich_contacts_from_payload(
    contacts: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    existing_emails: set[str] = set(e.lower() for e in contacts.get("emails", []) if isinstance(e, str))
    existing_phones: set[str] = set(
        "".join(ch for ch in p if ch.isdigit() or ch == "+")
        for p in contacts.get("phones", [])
        if isinstance(p, str)
    )

    email_keys = ("public_email", "business_email", "email")
    for key in email_keys:
        raw_value = payload.get(key)
        normalized = _normalize_email(raw_value if isinstance(raw_value, str) else None)
        if normalized and normalized not in existing_emails:
            existing_emails.add(normalized)
            contacts.setdefault("emails", []).append(normalized)

    phone_keys = (
        "contact_phone_number",
        "business_phone_number",
        "public_phone_number",
        "phone_number",
    )
    for key in phone_keys:
        raw_value = payload.get(key)
        normalized = _normalize_phone(raw_value if isinstance(raw_value, str) else None)
        if normalized and normalized not in existing_phones:
            existing_phones.add(normalized)
            contacts.setdefault("phones", []).append(normalized)

    return contacts


def _extract_external_url_from_payload(
    payload: dict[str, Any],
) -> str | None:
    direct = payload.get("external_url")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    bio_links = payload.get("bio_links")
    if isinstance(bio_links, list):
        for entry in bio_links:
            if isinstance(entry, str) and entry.strip():
                return entry.strip()
            if isinstance(entry, dict):
                url_val = entry.get("url") or entry.get("link") or entry.get("href")
                if isinstance(url_val, str) and url_val.strip():
                    return url_val.strip()

    return None


def _load_existing_metrics_history(
    raw_metadata: dict[str, Any] | None,
) -> list[MetricsEntry]:
    if not raw_metadata or not isinstance(raw_metadata, dict):
        return []

    history_raw = raw_metadata.get("metrics_history")
    if not isinstance(history_raw, list):
        return []

    parsed: list[MetricsEntry] = []
    for entry in history_raw:
        if isinstance(entry, dict):
            try:
                parsed.append(MetricsEntry(**entry))
            except Exception:
                logger.debug("Skipping malformed metrics_history entry: %s", entry)
        elif isinstance(entry, MetricsEntry):
            parsed.append(entry)

    return parsed


def _metrics_entry_matches(a: MetricsEntry, b: MetricsEntry) -> bool:
    return (
        a.subscribers_count == b.subscribers_count
        and a.posts_count == b.posts_count
    )


def _deduplicate_metrics(history: list[MetricsEntry]) -> list[MetricsEntry]:
    unique: list[MetricsEntry] = []
    for entry in history:
        if unique and _metrics_entry_matches(unique[-1], entry):
            continue
        unique.append(entry)
    return unique


async def update_account_profile_metadata(
    session: AsyncSession,
    account_id: int,
    platform: str,
    biography: str | None,
    external_url: str | None = None,
    extra_meta: dict[str, Any] | None = None,
    raw_profile_payload: dict[str, Any] | None = None,
    subscribers_count: int | None = None,
    posts_count: int | None = None,
    account_metadata: AccountMetadata | None = None,
) -> dict[str, Any]:
    stmt = select(Account).where(Account.id == account_id)
    result = await session.execute(stmt)
    account = result.scalar_one_or_none()

    if not account:
        logger.warning("Account with id %d not found for metadata update", account_id)
        return {}

    if account.status == "verified":
        logger.info(
            "Skipping metadata update for verified account: id=%d",
            account_id,
        )
        return account.raw_metadata if isinstance(account.raw_metadata, dict) else {}

    raw_metadata_dict: dict[str, Any] = {}
    if account.raw_metadata and isinstance(account.raw_metadata, dict):
        raw_metadata_dict = account.raw_metadata

    payload: dict[str, Any] = raw_profile_payload if isinstance(raw_profile_payload, dict) else {}

    if account_metadata is not None:
        compiled_metadata = account_metadata
        contacts: dict[str, Any] = {}
    else:
        if external_url is None and payload:
            external_url = _extract_external_url_from_payload(payload)

        contacts = {}
        if biography or external_url:
            contacts = parse_profile_contacts(biography, external_url)

        if payload:
            contacts = _enrich_contacts_from_payload(contacts, payload)

        username = account.username or account.platform_id

        compiled_metadata = compile_author_metadata(
            platform=platform,
            username=username,
            biography=biography,
            contacts_dict=contacts,
            extra_links=contacts.get("external_links", []),
            raw_profile_payload=raw_profile_payload,
        )

    compiled_metadata.metrics_history = _load_existing_metrics_history(raw_metadata_dict)

    if subscribers_count is not None or posts_count is not None:
        now_iso = datetime.now(timezone.utc).isoformat()
        new_entry = MetricsEntry(
            timestamp=now_iso,
            subscribers_count=subscribers_count,
            posts_count=posts_count,
        )
        if compiled_metadata.metrics_history:
            last = compiled_metadata.metrics_history[-1]
            if not (_metrics_entry_matches(last, new_entry) and last.timestamp == now_iso):
                compiled_metadata.metrics_history.append(new_entry)
        else:
            compiled_metadata.metrics_history.append(new_entry)

    compiled_metadata.metrics_history = _deduplicate_metrics(compiled_metadata.metrics_history)

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

    normalized_biography = normalize_description(biography)
    account.description = normalized_biography if normalized_biography is not None else account.description
    account.raw_metadata = compiled_metadata.model_dump(mode="json", exclude_none=False)

    if subscribers_count is not None:
        account.subscribers_count = subscribers_count
    await session.flush()

    logger.info("Updated profile metadata for account_id: %d", account_id)

    if contacts:
        parent_handle = account.username or account.platform_id or str(account_id)
        await queue_discovered_accounts(
            session, compiled_metadata, parent_handle, status="pending"
        )

    return compiled_metadata.model_dump(mode="json", exclude_none=False)
