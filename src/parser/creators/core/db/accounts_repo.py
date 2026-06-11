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


def _extract_raw_field(raw: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = raw.get(key)
        if value is not None:
            return value
    return None


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


def _extract_geo_from_payload(
    payload: dict[str, Any],
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    city: str | None = None
    country: str | None = None
    coords: list[float] | None = None

    city_raw = _extract_raw_field(payload, "city_name", "city")
    if isinstance(city_raw, str) and city_raw.strip():
        city = city_raw.strip()

    address_fields = ("address", "location", "place")
    if city is None:
        for af in address_fields:
            addr = payload.get(af)
            if isinstance(addr, dict):
                inner_city = addr.get("city_name") or addr.get("city") or addr.get("name")
                if isinstance(inner_city, str) and inner_city.strip():
                    city = inner_city.strip()
                    break

    country_raw = _extract_raw_field(payload, "country_code", "country", "country_name")
    if isinstance(country_raw, str) and country_raw.strip():
        country = country_raw.strip()

    lat_raw = _extract_raw_field(payload, "latitude", "lat")
    lng_raw = _extract_raw_field(payload, "longitude", "lng", "lon")
    if isinstance(lat_raw, (int, float)) and isinstance(lng_raw, (int, float)):
        coords = [float(lat_raw), float(lng_raw)]

    geo_data: dict[str, Any] | None = None
    if city or country or coords:
        geo_data = {}
        if city:
            geo_data["city"] = city
        if country:
            geo_data["country"] = country
        if coords:
            geo_data["coordinates"] = coords

    return city, country, geo_data


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
    location: str | None = None,
    language: str | None = None,
    geo_data: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
    category: str | None = None,
    raw_profile_payload: dict[str, Any] | None = None,
    subscribers_count: int | None = None,
    posts_count: int | None = None,
) -> dict[str, Any]:
    stmt = select(Account).where(Account.id == account_id)
    result = await session.execute(stmt)
    account = result.scalar_one_or_none()

    if not account:
        logger.warning("Account with id %d not found for metadata update", account_id)
        return {}

    raw_metadata_dict: dict[str, Any] = {}
    if account.raw_metadata and isinstance(account.raw_metadata, dict):
        raw_metadata_dict = account.raw_metadata

    payload: dict[str, Any] = raw_profile_payload if isinstance(raw_profile_payload, dict) else {}

    if external_url is None and payload:
        external_url = _extract_external_url_from_payload(payload)

    contacts: dict[str, Any] = {}
    if biography or external_url:
        contacts = parse_profile_contacts(biography, external_url)

    if payload:
        contacts = _enrich_contacts_from_payload(contacts, payload)

    found_city: str | None = None
    found_country: str | None = None
    extracted_geo: dict[str, Any] | None = None

    if payload:
        found_city, found_country, extracted_geo = _extract_geo_from_payload(payload)

    if extracted_geo is not None:
        if geo_data is None:
            geo_data = extracted_geo
        else:
            for key in ("city", "country", "coordinates"):
                if key not in geo_data or geo_data[key] is None:
                    geo_data[key] = extracted_geo.get(key)

    if location is None and payload:
        parts = [p for p in (found_city, found_country) if p]
        if parts:
            location = ", ".join(parts)

    username = account.username or account.platform_id

    resolved_category = category
    if resolved_category is None:
        resolved_category = raw_metadata_dict.get("category")
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
