import argparse
import asyncio
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text

from src.config.config import load_settings
from src.db.database import Database
from src.parser.creators.core.contacts import (
    ContactEngine,
    ExtractionPayload,
    extract_bio_links,
    extract_external_links,
    extract_external_platforms,
    is_valid_email,
    is_valid_telegram_handle,
    normalize_phone,
    normalize_telegram_handle,
)
from src.parser.creators.core.schemas import AccountMetadata, Contacts

ACCOUNTS_QUERY = text(
    "SELECT id, username, platform, description, raw_metadata, country "
    "FROM accounts "
    "WHERE status = 'verified' AND id > :last_id "
    "ORDER BY id ASC "
    "LIMIT :batch_size"
)

CONTENT_QUERY = text(
    "SELECT account_id, content FROM content "
    "WHERE account_id = ANY(:batch_ids) "
    "AND content IS NOT NULL AND length(trim(content)) > 0 "
    "ORDER BY account_id"
)

UPDATE_QUERY = text(
    "UPDATE accounts AS a "
    "SET raw_metadata = v.raw_metadata::jsonb "
    "FROM ("
    "    SELECT unnest(:ids::bigint[]) AS id, "
    "           unnest(:raw_metadatas::text[]) AS raw_metadata"
    ") AS v "
    "WHERE a.id = v.id"
)

COUNT_QUERY = text(
    "SELECT COUNT(*) FROM accounts WHERE status = 'verified' AND id > :last_id"
)

CONTACT_KEYS = (
    "emails",
    "phones",
    "telegram_handles",
    "telegram_channels",
    "telegram_personal",
    "advertising_emails",
    "advertising_telegrams",
)

MAX_POSTS_PER_AUTHOR = 12
LOG_INTERVAL = 10_000
MAX_BATCH_RETRIES = 3


@dataclass
class Stats:
    authors_processed: int = 0
    posts_processed: int = 0
    authors_with_contacts_before: int = 0
    authors_with_contacts_after: int = 0
    authors_updated: int = 0
    dropped: int = 0
    new: int = 0
    refined: int = 0
    advertising_telegrams: int = 0
    personal_telegrams: int = 0
    telegram_channels: int = 0
    emails: int = 0
    phones: int = 0


@dataclass
class DiffRecord:
    account_id: int
    category: str
    old_contacts: dict[str, Any] | None
    new_contacts: dict[str, Any] | None
    description: str | None = None


class ProgressBar:
    def __init__(
        self,
        total: int,
        desc: str = "",
        unit: str = "",
        dynamic_ncols: bool = False,
    ) -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.n = 0
        self._start = time.monotonic()

    def update(self, n: int) -> None:
        self.n += n
        elapsed = time.monotonic() - self._start
        speed = self.n / elapsed if elapsed > 0 else 0.0
        percent = (self.n / self.total * 100) if self.total else 0.0
        print(
            f"\r{self.desc}: {percent:.1f}% | {self.n}/{self.total} {self.unit} | "
            f"{speed:.1f} {self.unit}/s",
            end="",
            flush=True,
        )

    def close(self) -> None:
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich verified authors with extracted contacts")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint-file", type=str, default="checkpoint_contacts_migration.json")
    parser.add_argument("--reset-checkpoint", action="store_true")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def save_checkpoint(path: Path, last_id: int, stats: Stats) -> None:
    payload = {
        "last_id": last_id,
        "stats": {
            "authors_processed": stats.authors_processed,
            "posts_processed": stats.posts_processed,
            "authors_with_contacts_before": stats.authors_with_contacts_before,
            "authors_with_contacts_after": stats.authors_with_contacts_after,
            "authors_updated": stats.authors_updated,
            "dropped": stats.dropped,
            "new": stats.new,
            "refined": stats.refined,
            "advertising_telegrams": stats.advertising_telegrams,
            "personal_telegrams": stats.personal_telegrams,
            "telegram_channels": stats.telegram_channels,
            "emails": stats.emails,
            "phones": stats.phones,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _non_empty_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip():
            result.append(value)
    return result


def _extract_bio_links(raw_metadata: dict[str, Any]) -> list[dict[str, Any]] | None:
    raw_profile = raw_metadata.get("raw_profile_payload")
    if not isinstance(raw_profile, dict):
        raw_profile = {}
    result: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(url: Any) -> None:
        if isinstance(url, str) and url.strip() and url.strip() not in seen:
            seen.add(url.strip())
            result.append({"url": url.strip(), "title": "bio_link"})

    _add(raw_profile.get("external_url"))
    _add(raw_metadata.get("website"))
    _add(raw_metadata.get("link_in_bio"))
    external_links = raw_metadata.get("external_links")
    if isinstance(external_links, list):
        for item in external_links:
            if isinstance(item, str):
                _add(item)
            elif isinstance(item, dict):
                _add(item.get("url"))
    bio_links = raw_profile.get("bio_links")
    if isinstance(bio_links, list):
        for item in bio_links:
            if isinstance(item, str):
                _add(item)
            elif isinstance(item, dict):
                _add(item.get("url"))
    return result or None


def _has_contacts(contacts: dict[str, Any] | None) -> bool:
    if not contacts:
        return False
    return any(contacts.get(key) for key in CONTACT_KEYS)


def _merge_old_contacts(
    contacts: Contacts,
    old_contacts: dict[str, Any] | None,
    default_region: str | None,
) -> None:
    if not old_contacts:
        return
    old_phones = old_contacts.get("phones")
    if isinstance(old_phones, list):
        for phone in old_phones:
            if isinstance(phone, str) and phone.strip():
                normalized = normalize_phone(phone.strip(), default_region or "RU")
                if normalized and normalized not in contacts.phones:
                    contacts.phones.append(normalized)
    old_emails = old_contacts.get("emails")
    if isinstance(old_emails, list):
        for email in old_emails:
            if isinstance(email, str) and email.strip() and is_valid_email(email.strip()):
                normalized = email.strip().lower()
                if normalized not in contacts.emails:
                    contacts.emails.append(normalized)


def _collect_bio_link_candidates(
    biography: str | None,
    raw_bio_links: list[dict[str, Any]] | None,
) -> list[Any]:
    candidates: list[Any] = []
    if biography:
        candidates.extend(extract_external_links(biography))
    if raw_bio_links:
        candidates.extend(raw_bio_links)
    return candidates


def _detect_link_in_bio(
    biography: str | None,
    raw_bio_links: list[dict[str, Any]] | None,
) -> str | None:
    for link in extract_bio_links(_collect_bio_link_candidates(biography, raw_bio_links)):
        if link.is_bio_link:
            return link.url
    return None


def _detect_website(
    biography: str | None,
    raw_bio_links: list[dict[str, Any]] | None,
) -> str | None:
    for link in extract_bio_links(_collect_bio_link_candidates(biography, raw_bio_links)):
        if not link.is_bio_link and link.platform is None:
            return link.url
    return None


def _merge_external_platforms(merged: dict[str, Any], platforms: dict[str, str]) -> None:
    if not platforms:
        return
    existing = merged.get("external_platforms")
    if not isinstance(existing, dict):
        existing = {}
    for key, value in platforms.items():
        if key not in existing or existing[key] is None:
            existing[key] = value
    merged["external_platforms"] = existing


def _merge_metadata(
    raw_metadata: dict[str, Any] | None,
    contacts: Any,
    biography: str | None,
    raw_bio_links: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    merged = dict(raw_metadata) if raw_metadata else {}
    merged["contacts"] = contacts.model_dump(exclude_none=True)

    if biography:
        _merge_external_platforms(merged, extract_external_platforms(biography))

    if raw_bio_links:
        for item in raw_bio_links:
            url = item.get("url")
            if isinstance(url, str) and url.strip():
                _merge_external_platforms(merged, extract_external_platforms(url))

    link_in_bio = _detect_link_in_bio(biography, raw_bio_links)
    if link_in_bio and not merged.get("link_in_bio"):
        merged["link_in_bio"] = link_in_bio

    website = _detect_website(biography, raw_bio_links)
    if website and not merged.get("website"):
        merged["website"] = website

    return merged


def _build_payload(
    account: dict[str, Any],
    posts: list[dict[str, Any]],
) -> ExtractionPayload:
    raw_metadata = account["raw_metadata"] if isinstance(account["raw_metadata"], dict) else {}
    metadata = AccountMetadata.create_with_timestamp(**raw_metadata) if raw_metadata else None
    raw_profile = metadata.raw_profile_payload if metadata and isinstance(metadata.raw_profile_payload, dict) else {}
    if not isinstance(raw_profile, dict):
        raw_profile = {}
    biography = raw_profile.get("biography") or raw_metadata.get("biography") or account["description"]

    posts_content: list[str] = []
    for post in posts:
        content_text = post.get("content")
        if isinstance(content_text, str) and content_text.strip():
            stripped = content_text.strip()
            if stripped not in posts_content:
                posts_content.append(stripped)
        if len(posts_content) >= MAX_POSTS_PER_AUTHOR:
            break

    trusted_emails: list[str] = []
    trusted_phones: list[str] = []
    country = account.get("country")
    default_region = country if isinstance(country, str) and len(country) == 2 else None
    for key in ("public_email", "business_email"):
        email = raw_profile.get(key)
        if isinstance(email, str) and email.strip() and is_valid_email(email.strip()):
            trusted_emails.append(email.strip())
    for key in ("public_phone_number", "business_phone_number", "contact_phone_number"):
        phone = raw_profile.get(key)
        if isinstance(phone, str) and phone.strip():
            normalized = normalize_phone(phone.strip(), default_region)
            if normalized:
                trusted_phones.append(normalized)

    return ExtractionPayload(
        biography=biography,
        author_username=account["username"],
        platform=account["platform"] or "INSTAGRAM",
        raw_bio_links=_extract_bio_links(raw_metadata),
        posts_content=posts_content or None,
        trusted_emails=trusted_emails or None,
        trusted_phones=trusted_phones or None,
        external_url=raw_profile.get("external_url") or raw_metadata.get("external_url"),
        default_region=default_region,
    )


def _format_contacts(contacts: dict[str, Any] | None) -> str:
    if not contacts:
        return "{}"
    parts = [f"{key}={contacts.get(key)}" for key in CONTACT_KEYS if contacts.get(key)]
    return ", ".join(parts) if parts else "{}"


def _print_diffs(diffs: list[DiffRecord]) -> None:
    if not diffs:
        print("\nNo contact changes detected.")
        return
    print(f"\n=== CONTACT CHANGES ({len(diffs)} authors) ===")
    for record in diffs:
        print(f"[{record.category}] account_id={record.account_id}")
        if record.category == "DROPPED" and record.description:
            print(f"  description: {record.description}")
        print(f"  old: {_format_contacts(record.old_contacts)}")
        print(f"  new: {_format_contacts(record.new_contacts)}")


def _print_summary(stats: Stats, dry_run: bool) -> None:
    mode = "DRY-RUN" if dry_run else "PRODUCTION"
    print(f"\n=== SUMMARY ({mode}) ===")
    print(f"Authors processed: {stats.authors_processed}")
    print(f"Posts processed: {stats.posts_processed}")
    print(f"Authors with contacts (before): {stats.authors_with_contacts_before}")
    print(f"Authors with contacts (after): {stats.authors_with_contacts_after}")
    print(f"Authors updated: {stats.authors_updated}")
    print(f"Changes: DROPPED={stats.dropped} NEW={stats.new} REFINED={stats.refined}")
    print(f"Advertising Telegram contacts: {stats.advertising_telegrams}")
    print(f"Personal Telegram contacts: {stats.personal_telegrams}")
    print(f"Telegram channels: {stats.telegram_channels}")
    print(f"Email addresses: {stats.emails}")
    print(f"WhatsApp/phones: {stats.phones}")


def _print_intermediate(stats: Stats) -> None:
    print(
        f"[{stats.authors_processed}] authors processed | "
        f"posts: {stats.posts_processed} | "
        f"with contacts: {stats.authors_with_contacts_after} | "
        f"emails: {stats.emails} | phones: {stats.phones} | "
        f"telegrams: {stats.advertising_telegrams + stats.personal_telegrams + stats.telegram_channels}"
    )


async def process_batch(
    db: Database,
    engine: ContactEngine,
    batch: list[Any],
    dry_run: bool,
    stats: Stats,
    diffs: list[DiffRecord],
) -> int:
    batch_ids = [account["id"] for account in batch]
    async with db.async_session() as session:
        try:
            content_rows = await session.execute(CONTENT_QUERY, {"batch_ids": batch_ids})
            posts_by_account: dict[int, list[dict[str, Any]]] = {}
            for row in content_rows.mappings():
                posts_by_account.setdefault(row["account_id"], []).append(
                    {"content": row["content"]}
                )

            updates: list[dict[str, Any]] = []
            for account in batch:
                raw_metadata = account["raw_metadata"] if isinstance(account["raw_metadata"], dict) else {}
                old_contacts = raw_metadata.get("contacts")
                old_has = _has_contacts(old_contacts)

                payload = _build_payload(account, posts_by_account.get(account["id"], []))
                contacts = engine.extract_contacts(payload)
                _merge_old_contacts(contacts, old_contacts, account.get("country"))
                new_metadata = _merge_metadata(raw_metadata, contacts, payload.biography, payload.raw_bio_links)
                new_contacts = new_metadata.get("contacts")
                new_has = _has_contacts(new_contacts)

                stats.authors_processed += 1
                stats.posts_processed += len(posts_by_account.get(account["id"], []))
                if old_has:
                    stats.authors_with_contacts_before += 1
                if new_has:
                    stats.authors_with_contacts_after += 1
                stats.advertising_telegrams += len(contacts.advertising_telegrams)
                stats.personal_telegrams += len(contacts.telegram_personal)
                stats.telegram_channels += len(contacts.telegram_channels)
                stats.emails += len(contacts.emails) + len(contacts.advertising_emails)
                stats.phones += len(contacts.phones)

                if old_contacts != new_contacts:
                    if old_has and not new_has:
                        category = "DROPPED"
                    elif not old_has and new_has:
                        category = "NEW"
                    elif old_has and new_has:
                        category = "REFINED"
                    else:
                        category = None
                    if category:
                        setattr(stats, category.lower(), getattr(stats, category.lower()) + 1)
                        diffs.append(
                            DiffRecord(
                                account_id=account["id"],
                                category=category,
                                old_contacts=old_contacts,
                                new_contacts=new_contacts,
                                description=account.get("description"),
                            )
                        )

                if new_metadata != raw_metadata:
                    stats.authors_updated += 1
                    updates.append(
                        {
                            "id": account["id"],
                            "raw_metadata": json.dumps(new_metadata, ensure_ascii=False),
                        }
                    )

            if updates and not dry_run:
                await session.execute(
                    UPDATE_QUERY,
                    {
                        "ids": [update["id"] for update in updates],
                        "raw_metadatas": [update["raw_metadata"] for update in updates],
                    },
                )
                await session.commit()
        except Exception:
            await session.rollback()
            raise

    return batch[-1]["id"]


async def _process_batch_with_retry(
    db: Database,
    engine: ContactEngine,
    batch: list[Any],
    dry_run: bool,
    stats: Stats,
    diffs: list[DiffRecord],
) -> int:
    for attempt in range(MAX_BATCH_RETRIES):
        stats_before = copy.deepcopy(stats)
        diffs_len = len(diffs)
        try:
            return await process_batch(db, engine, batch, dry_run, stats, diffs)
        except Exception:
            if attempt == MAX_BATCH_RETRIES - 1:
                raise
            del diffs[diffs_len:]
            stats.__dict__.update(stats_before.__dict__)
            await asyncio.sleep(1 + attempt)
    raise RuntimeError("unreachable")


async def main() -> None:
    args = parse_args()
    settings = load_settings()
    db = Database(settings.db_url)

    checkpoint_path = Path(args.checkpoint_file)
    if args.reset_checkpoint:
        checkpoint_path.unlink(missing_ok=True)

    checkpoint = load_checkpoint(checkpoint_path)
    last_id = int(checkpoint.get("last_id", 0))
    stats_dict = checkpoint.get("stats")
    stats = Stats(**stats_dict) if isinstance(stats_dict, dict) else Stats()
    diffs: list[DiffRecord] = []
    engine = ContactEngine()
    processed = 0

    try:
        async with db.async_session() as session:
            total_count = (
                await session.execute(COUNT_QUERY, {"last_id": last_id})
            ).scalar() or 0

        progress_total = min(total_count, args.limit) if args.limit is not None else total_count
        progress = ProgressBar(
            total=progress_total,
            desc="Enriching contacts",
            unit="author",
            dynamic_ncols=True,
        )
        while True:
            if args.limit is not None and processed >= args.limit:
                break

            async with db.async_session() as session:
                rows = await session.execute(
                    ACCOUNTS_QUERY,
                    {"last_id": last_id, "batch_size": args.batch_size},
                )
                batch = list(rows.mappings())
            if not batch:
                break

            if args.limit is not None:
                remaining = args.limit - processed
                if len(batch) > remaining:
                    batch = batch[:remaining]

            last_id = await _process_batch_with_retry(
                db,
                engine,
                batch,
                args.dry_run,
                stats,
                diffs,
            )
            processed += len(batch)
            progress.update(len(batch))

            if not args.dry_run:
                save_checkpoint(checkpoint_path, last_id, stats)

            if stats.authors_processed // LOG_INTERVAL > (stats.authors_processed - len(batch)) // LOG_INTERVAL:
                _print_intermediate(stats)

        progress.close()

        await db.engine.dispose()

        _print_diffs(diffs)

        _print_summary(stats, args.dry_run)

    except KeyboardInterrupt:
        if not args.dry_run:
            save_checkpoint(checkpoint_path, last_id, stats)
        await db.engine.dispose()
        print("\nInterrupted. Checkpoint saved.")
        _print_summary(stats, args.dry_run)

    except Exception:
        if not args.dry_run:
            save_checkpoint(checkpoint_path, last_id, stats)
        await db.engine.dispose()
        raise


if __name__ == "__main__":
    asyncio.run(main())
