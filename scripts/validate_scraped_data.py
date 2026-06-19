import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import Database
from src.db.models import Account
from src.parser.creators.core.schemas import AccountMetadata
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

BATCH_SIZE: int = 1000
REPORT_PATH: str = "scripts/invalid_accounts_report.json"


def _has_non_empty_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _validate_columns(account: Account) -> str | None:
    if not account.id:
        return "id is null or empty"
    if not _has_non_empty_value(account.username):
        return "username is null or empty"
    if not _has_non_empty_value(account.title):
        return "title is null or empty"
    if not _has_non_empty_value(account.description):
        return "description is null or empty"
    if account.subscribers_count is None:
        return "subscribers_count is null"
    if not _has_non_empty_value(account.platform_id):
        return "platform_id is null or empty"
    return None


def _validate_subscribers_range(count: int) -> str | None:
    if not (5000 <= count <= 1000000):
        return f"subscribers_out_of_range ({count} not in [5000, 1000000])"
    return None


def _validate_metadata(raw_metadata: dict | None) -> tuple[AccountMetadata | None, str | None]:
    if not raw_metadata:
        return None, "raw_metadata is null or empty"
    if "extracted_at" not in raw_metadata:
        raw_metadata["extracted_at"] = datetime.now(timezone.utc).isoformat()
    try:
        model = AccountMetadata.model_validate(raw_metadata)
        return model, None
    except Exception as exc:
        return None, f"pydantic_validation_failed: {exc}"


def _is_geo_data_valid(model: AccountMetadata) -> bool:
    if model.geo_data is None:
        return False
    geo = model.geo_data
    if geo.city and geo.city.strip():
        return True
    if geo.country and geo.country.strip():
        return True
    if geo.coordinates and len(geo.coordinates) >= 2:
        return True
    return False


def _collect_audit_flags(model: AccountMetadata) -> dict[str, bool]:
    contacts = model.contacts
    has_website = bool(model.website and model.website.strip())
    has_emails = bool(contacts.emails)
    has_phones = bool(contacts.phones)
    has_telegram_channels = bool(contacts.telegram_channels)
    has_telegram_personal = bool(contacts.telegram_personal)
    has_advertising_emails = bool(contacts.advertising_emails)
    has_advertising_telegrams = bool(contacts.advertising_telegrams)
    has_geo_data = _is_geo_data_valid(model)
    has_location = bool(model.location and model.location.strip())
    has_external_platforms = bool(model.external_platforms)
    all_in_one = all([
        has_website,
        has_emails,
        has_phones,
        has_telegram_channels,
        has_telegram_personal,
        has_advertising_emails,
        has_advertising_telegrams,
        has_geo_data,
        has_location,
        has_external_platforms,
    ])
    return {
        "website": has_website,
        "emails": has_emails,
        "phones": has_phones,
        "telegram_channels": has_telegram_channels,
        "telegram_personal": has_telegram_personal,
        "advertising_emails": has_advertising_emails,
        "advertising_telegrams": has_advertising_telegrams,
        "geo_data": has_geo_data,
        "location": has_location,
        "external_platforms": has_external_platforms,
        "all_in_one": all_in_one,
    }


def _determine_failure_reason(
    column_error: str | None,
    metadata_error: str | None,
) -> tuple[str, str]:
    if column_error:
        if column_error == "subscribers_count is null":
            return "subscribers_out_of_range", column_error
        if "subscribers_out_of_range" in column_error:
            return "subscribers_out_of_range", column_error
        return "missing_columns", column_error
    if metadata_error:
        return "pydantic_validation_failed", metadata_error
    return "pydantic_validation_failed", "unknown failure"


def _append_invalid_entry(
    invalid_list: list[dict[str, str | int]],
    account: Account,
    reason: str,
    detail: str,
) -> None:
    invalid_list.append({
        "id": account.id,
        "username": account.username or "",
        "title": account.title or "",
        "reason": reason,
        "detail": detail,
    })


async def _count_pending_accounts(session: AsyncSession) -> int:
    stmt = select(func.count(Account.id)).where(
        Account.platform.ilike("INSTAGRAM"),
        Account.status == "parsed",
    )
    result = await session.execute(stmt)
    return result.scalar_one()


async def _process_batch(
    session: AsyncSession,
    invalid_list: list[dict[str, str | int]],
    dry_run: bool,
    offset: int = 0,
) -> tuple[int, int, dict[str, int]]:
    verified: int = 0
    invalid: int = 0
    audit_counts: dict[str, int] = {
        "website": 0,
        "emails": 0,
        "phones": 0,
        "telegram_channels": 0,
        "telegram_personal": 0,
        "advertising_emails": 0,
        "advertising_telegrams": 0,
        "geo_data": 0,
        "location": 0,
        "external_platforms": 0,
        "all_in_one": 0,
    }

    stmt = (
        select(Account)
        .where(
            Account.platform.ilike("INSTAGRAM"),
            Account.status == "parsed",
        )
        .limit(BATCH_SIZE)
    )
    if dry_run:
        stmt = stmt.offset(offset)
    else:
        stmt = stmt.with_for_update(skip_locked=True)
    result = await session.execute(stmt)
    accounts = list(result.scalars().all())

    if not accounts:
        return 0, 0, audit_counts

    for account in accounts:
        column_error = _validate_columns(account)
        if column_error:
            reason, detail = _determine_failure_reason(column_error, None)
            account.status = "invalid"
            invalid += 1
            _append_invalid_entry(invalid_list, account, reason, detail)
            logger.warning(
                "Account %s (id=%s) marked invalid: %s",
                account.username,
                account.id,
                detail,
            )
            continue

        assert account.subscribers_count is not None
        subs_error = _validate_subscribers_range(account.subscribers_count)
        if subs_error:
            reason, detail = _determine_failure_reason(subs_error, None)
            account.status = "invalid"
            invalid += 1
            _append_invalid_entry(invalid_list, account, reason, detail)
            logger.warning(
                "Account %s (id=%s) marked invalid: %s",
                account.username,
                account.id,
                detail,
            )
            continue

        model, metadata_error = _validate_metadata(account.raw_metadata)
        if metadata_error or model is None:
            reason, detail = _determine_failure_reason(None, metadata_error)
            account.status = "invalid"
            invalid += 1
            _append_invalid_entry(invalid_list, account, reason, detail)
            logger.warning(
                "Account %s (id=%s) marked invalid: %s",
                account.username,
                account.id,
                detail,
            )
            continue

        account.status = "verified"
        account.raw_metadata = model.model_dump(mode="json")
        verified += 1

        audit = _collect_audit_flags(model)
        for key in audit_counts:
            if key == "all_in_one":
                continue
            if audit[key]:
                audit_counts[key] += 1
        if audit["all_in_one"]:
            audit_counts["all_in_one"] += 1

    if dry_run:
        await session.rollback()
        logger.info("Dry-run mode: batch rolled back, no changes persisted.")
    else:
        await session.commit()

    return verified, invalid, audit_counts


def _print_statistics(total_verified: int, audit_counts: dict[str, int]) -> None:
    logger.info("=" * 70)
    logger.info("AUDIT STATISTICS (verified accounts: %d)", total_verified)
    logger.info("=" * 70)

    if total_verified == 0:
        logger.info("No verified accounts to report.")
        return

    labels: list[tuple[str, str]] = [
        ("website", "Website"),
        ("emails", "Emails (contacts)"),
        ("phones", "Phones (contacts)"),
        ("telegram_channels", "Telegram Channels (contacts)"),
        ("telegram_personal", "Telegram Personal (contacts)"),
        ("advertising_emails", "Advertising Emails (contacts)"),
        ("advertising_telegrams", "Advertising Telegrams (contacts)"),
        ("geo_data", "Geo Data"),
        ("location", "Location"),
        ("external_platforms", "External Platforms"),
        ("all_in_one", "All-in-One (all 10 fields)"),
    ]

    for key, label in labels:
        count = audit_counts[key]
        pct = (count / total_verified) * 100
        logger.info("  %-40s %d (%.2f%%)", label + ":", count, pct)

    logger.info("=" * 70)


def _build_invalid_report(invalid_list: list[dict[str, str | int]]) -> dict[str, object]:
    counts: dict[str, int] = {}
    accounts_by_reason: dict[str, list[dict[str, str | int]]] = {}
    for entry in invalid_list:
        reason = str(entry["reason"])
        counts[reason] = counts.get(reason, 0) + 1
        accounts_by_reason.setdefault(reason, []).append(entry)

    reasons_payload: dict[str, dict[str, object]] = {}
    for reason in counts:
        reasons_payload[reason] = {
            "count": counts[reason],
            "accounts": accounts_by_reason[reason],
        }

    report: dict[str, object] = {
        "total_errors": len(invalid_list),
        "reasons": reasons_payload,
    }
    return report


def _save_invalid_report(report: dict[str, object]) -> None:
    report_dir = os.path.dirname(REPORT_PATH)
    if report_dir and not os.path.exists(report_dir):
        os.makedirs(report_dir, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Invalid accounts report saved to %s", REPORT_PATH)


async def run() -> None:
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))

    parser = argparse.ArgumentParser(description="Validate scraped Instagram accounts.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Execute all logic but rollback each batch instead of committing.",
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY-RUN mode enabled. No data will be persisted.")

    db_url = os.getenv("DB_URL")
    if not db_url:
        logger.critical("DB_URL environment variable is not set.")
        sys.exit(1)

    logger.info("Starting validation of INSTAGRAM accounts with status 'parsed'...")

    db = Database(db_url)

    total_verified: int = 0
    total_invalid: int = 0
    invalid_list: list[dict[str, str | int]] = []
    combined_audit: dict[str, int] = {
        "website": 0,
        "emails": 0,
        "phones": 0,
        "telegram_channels": 0,
        "telegram_personal": 0,
        "advertising_emails": 0,
        "advertising_telegrams": 0,
        "geo_data": 0,
        "location": 0,
        "external_platforms": 0,
        "all_in_one": 0,
    }

    start_time: float = time.monotonic()

    async with db.async_session() as session:
        total_accounts: int = await _count_pending_accounts(session)
        logger.info("Found %d accounts to process.", total_accounts)

    processed: int = 0
    offset: int = 0
    while True:
        async with db.async_session() as session:
            verified, invalid, batch_audit = await _process_batch(
                session, invalid_list, args.dry_run, offset,
            )

        if verified == 0 and invalid == 0:
            break

        batch_count = verified + invalid
        total_verified += verified
        total_invalid += invalid
        processed += batch_count

        if args.dry_run:
            offset += batch_count

        for key in combined_audit:
            combined_audit[key] += batch_audit[key]

        elapsed: float = time.monotonic() - start_time
        rate: float = processed / elapsed if elapsed > 0 else 0.0
        logger.info(
            "Progress: %d/%d processed | verified=%d invalid=%d | %.1f accounts/s",
            processed,
            total_accounts,
            total_verified,
            total_invalid,
            rate,
        )

    elapsed_total: float = time.monotonic() - start_time
    logger.info("Processing complete in %.2f seconds.", elapsed_total)
    logger.info("Total verified: %d | Total invalid: %d", total_verified, total_invalid)

    _print_statistics(total_verified, combined_audit)

    if invalid_list:
        report = _build_invalid_report(invalid_list)
        _save_invalid_report(report)
        reasons_obj = report["reasons"]
        reasons_count = len(reasons_obj) if isinstance(reasons_obj, dict) else 0
        logger.info(
            "Invalid accounts report: %d total errors across %d reason categories.",
            report["total_errors"],
            reasons_count,
        )
    else:
        logger.info("No invalid accounts found. Skipping report generation.")

    await db.engine.dispose()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
