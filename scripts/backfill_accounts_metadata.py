"""
Backfill script to populate raw_metadata for existing accounts.

This script performs a safe local-only batch migration that:
1. Fetches unprocessed accounts (raw_metadata is NULL) using keyset pagination
2. Parses contact information from account descriptions
3. Compiles OpenSPG-compliant metadata dictionaries
4. Queues discovered external accounts with 'pending' status
5. Updates account raw_metadata fields in batches

CRITICAL: This script makes NO external API requests. All processing is local.

Usage:
    python -m scripts.backfill_accounts_metadata
"""

import asyncio
import logging
import sys
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config.config import Settings, load_settings
from src.db.models import Account
from src.parser.creators.core.utils import (
    parse_profile_contacts,
    compile_author_metadata,
    queue_discovered_accounts,
)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 100
LOG_LEVEL: str = "INFO"

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main backfill logic
# ---------------------------------------------------------------------------

async def backfill_accounts_metadata() -> None:
    """
    Main entry point for the backfill migration.

    Connects to the database, queries unprocessed accounts in batches using
    keyset pagination, processes their descriptions to extract contacts and
    compile metadata, queues discovered external accounts, and commits updates
    in batches.
    """
    settings: Settings = load_settings()
    engine = create_async_engine(
        settings.db_url,
        echo=False,
        pool_size=5,
        max_overflow=2,
        pool_timeout=30.0,
        pool_pre_ping=False,
    )
    session_maker: async_sessionmaker[AsyncSession] = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    logger.info("Starting accounts metadata backfill migration...")
    logger.info("Batch size: %d", BATCH_SIZE)
    logger.info("Using keyset pagination (Account.id > last_id)")

    total_processed: int = 0
    total_updated: int = 0
    total_queued: int = 0
    total_errors: int = 0
    last_id: int = 0  # Initialize keyset pagination cursor

    try:
        while True:
            async with session_maker() as session:
                # Fetch batch of unprocessed accounts using keyset pagination
                stmt = (
                    select(Account)
                    .where(
                        Account.raw_metadata.is_(None),  # Only unprocessed accounts
                        Account.id > last_id  # Keyset pagination: fetch after last_id
                    )
                    .order_by(Account.id.asc())  # Order by id ascending for keyset
                    .limit(BATCH_SIZE)
                )
                result = await session.execute(stmt)
                accounts: list[Account] = list(result.scalars().all())

                if not accounts:
                    logger.info("No more accounts to process. Migration complete.")
                    break

                logger.info(
                    "Processing batch: %d accounts (last_id=%d)",
                    len(accounts),
                    last_id,
                )

                batch_updated: int = 0
                batch_queued: int = 0
                batch_errors: int = 0

                for account in accounts:
                    # Cache critical identifier fields before try block
                    # to avoid lazy-load queries on expired instances
                    acc_id: int = account.id
                    acc_platform: str = account.platform
                    acc_username: str | None = account.username

                    try:
                        # Use nested transaction (savepoint) to isolate each account's processing
                        async with session.begin_nested():
                            updated, queued = await process_single_account(
                                session, account
                            )
                            if updated:
                                batch_updated += 1
                                total_updated += 1
                            if queued > 0:
                                batch_queued += queued
                                total_queued += queued
                            total_processed += 1
                    except Exception as e:
                        # begin_nested() context manager automatically rolls back the savepoint on exception
                        logger.error(
                            "Error processing account id=%s, platform=%s, username=%s: %s",
                            acc_id,
                            acc_platform,
                            acc_username,
                            e,
                            exc_info=True,
                        )
                        batch_errors += 1
                        total_errors += 1
                        continue

                # Commit the entire batch once after processing all accounts
                await session.commit()

                logger.info(
                    "Batch complete: processed=%d, updated=%d, queued=%d, errors=%d",
                    len(accounts),
                    batch_updated,
                    batch_queued,
                    batch_errors,
                )

                # Update keyset pagination cursor to last account's id
                last_id = accounts[-1].id

        logger.info("=" * 60)
        logger.info("BACKFILL MIGRATION COMPLETED")
        logger.info("Total accounts processed: %d", total_processed)
        logger.info("Total accounts updated with metadata: %d", total_updated)
        logger.info("Total new pending accounts queued: %d", total_queued)
        logger.info("Total errors: %d", total_errors)
        logger.info("=" * 60)

    except Exception as e:
        logger.critical(
            "Fatal error during backfill migration: %s", e, exc_info=True
        )
        raise
    finally:
        await engine.dispose()
        logger.info("Database engine disposed.")


async def process_single_account(
    session: AsyncSession, account: Account
) -> tuple[bool, int]:
    """
    Process a single account to extract contacts and compile metadata.

    Args:
        session: SQLAlchemy async session for database operations.
        account: Account model instance to process.

    Returns:
        Tuple of (was_updated, num_queued_accounts).
    """
    # Cache account properties to avoid lazy-load queries on expired instances
    acc_id: int = account.id
    acc_platform: str = account.platform
    acc_username: str | None = account.username
    acc_description: str | None = account.description
    acc_raw_metadata: dict[str, Any] | None = account.raw_metadata
    acc_platform_id: str | None = account.platform_id

    # Skip if raw_metadata is already populated
    if acc_raw_metadata is not None:
        logger.debug(
            "Skipping account id=%s (already has raw_metadata)", acc_id
        )
        return False, 0

    if not acc_description:
        logger.debug(
            "Skipping account id=%s (no description)", acc_id
        )
        return False, 0

    # Parse contacts from description
    contacts_dict: dict[str, list[str] | str] = parse_profile_contacts(
        biography=acc_description,
        external_url=None,
    )

    # Compile OpenSPG-compliant metadata
    metadata: dict[str, Any] = compile_author_metadata(
        platform=acc_platform,
        username=acc_username,
        biography=acc_description,
        contacts_dict=contacts_dict,
    )

    queued_count: int = 0

    # Queue discovered external accounts BEFORE updating raw_metadata to avoid premature autoflush
    has_contacts: bool = (
        bool(contacts_dict.get("emails"))
        or bool(contacts_dict.get("telegram_handles"))
        or bool(contacts_dict.get("external_links"))
    )

    if has_contacts:
        parent_handle: str = acc_username or acc_platform_id or str(acc_id)
        try:
            await queue_discovered_accounts(
                session=session,
                contacts_dict=contacts_dict,
                parent_handle=parent_handle,
                status="pending",
            )
            # Count queued accounts (best-effort, non-critical)
            queued_count = (
                len(contacts_dict.get("telegram_handles", []))
                + len(contacts_dict.get("external_links", []))
            )
        except Exception as e:
            logger.warning(
                "Failed to queue discovered accounts for account id=%s: %s",
                acc_id,
                e,
                exc_info=True,
            )

    # Update account's raw_metadata and add to session at the very end to prevent autoflush issues
    account.raw_metadata = metadata
    session.add(account)

    logger.debug(
        "Processed account id=%s, platform=%s, username=%s: "
        "emails=%d, telegram=%d, links=%d",
        acc_id,
        acc_platform,
        acc_username,
        len(contacts_dict.get("emails", [])),
        len(contacts_dict.get("telegram_handles", [])),
        len(contacts_dict.get("external_links", [])),
    )

    return True, queued_count


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Check for --yes flag for non-interactive / Docker runs
    non_interactive: bool = "--yes" in sys.argv

    print("=" * 60)
    print("ACCOUNTS METADATA BACKFILL MIGRATION")
    print("=" * 60)
    print()
    print("This script will:")
    print("  1. Fetch unprocessed accounts (raw_metadata is NULL) using keyset pagination")
    print("  2. Parse contact info from account descriptions (emails, Telegram, links)")
    print("  3. Compile OpenSPG-compliant metadata into raw_metadata field")
    print("  4. Queue discovered external accounts with 'pending' status")
    print(f"  5. Process in batches of {BATCH_SIZE} for safety")
    print()
    print("CRITICAL: This script makes NO external API calls.")
    print("All processing is done locally from existing database data.")
    print("=" * 60)
    print()

    if not non_interactive:
        confirm: str = input("Proceed with migration? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Migration cancelled by user.")
            sys.exit(0)

    print()
    print("Starting migration...")
    print()

    asyncio.run(backfill_accounts_metadata())

    print()
    print("Migration script execution completed.")
