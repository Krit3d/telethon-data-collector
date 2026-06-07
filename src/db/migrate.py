"""
Database migration utility for Instagram metadata restructuring.

This script safely migrates Instagram accounts and content raw_metadata
from legacy/semi-structured formats to strictly validated Pydantic schemas (AccountMetadata,
ContentMetadata/InstagramContentMetadata) without losing existing data.

Target records:
    - Only processes accounts where platform == "INSTAGRAM"
    - Only processes content associated with these Instagram accounts
    - Does NOT query, modify, or lock any rows where platform == "TELEGRAM"

Usage:
    docker compose -f docker-compose.scraper.yml run --rm parser python -m src.db.migrate
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.config import load_settings
from src.db.database import Database
from src.db.models import Account, Content
from src.parser.creators.core.schemas import (
    AccountMetadata,
    ContentMetadata,
    InstagramContentMetadata,
    PlatformMetrics,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 50  # Process 50 Instagram accounts at a time
DEFAULT_CATEGORY: str = "unknown"
DEFAULT_LANGUAGE: str = "ru"  # Targeting Russian creators
REEL_DURATION_THRESHOLD: float = 90.0  # seconds - typical Reel duration


# ---------------------------------------------------------------------------
# Helper functions for metadata extraction and transformation
# ---------------------------------------------------------------------------


def extract_account_legacy_fields(
    raw_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Extract legacy fields from existing raw_metadata dictionary.

    Args:
        raw_metadata: Existing raw_metadata dictionary or None.

    Returns:
        Dictionary with extracted fields mapped to AccountMetadata structure.
    """
    if not raw_metadata or not isinstance(raw_metadata, dict):
        return {}

    extracted: dict[str, Any] = {}

    # Extract biography (may be under different keys)
    biography = (
        raw_metadata.get("biography")
        or raw_metadata.get("bio")
        or raw_metadata.get("description")
    )
    if biography and isinstance(biography, str):
        extracted["biography"] = biography

    # Extract location (may be a string or dict)
    location = raw_metadata.get("location")
    if location:
        if isinstance(location, str):
            extracted["location"] = location
        elif isinstance(location, dict):
            # Handle nested location object
            city = location.get("city")
            country = location.get("country")
            location_str = ", ".join(filter(None, [city, country]))
            if location_str:
                extracted["location"] = location_str
            # Also populate geo_data if coordinates available
            coords = location.get("coordinates") or location.get("coords")
            if coords and isinstance(coords, list) and len(coords) == 2:
                extracted["geo_data"] = {
                    "city": city,
                    "country": country,
                    "coordinates": coords,
                }

    # Extract followers count (may be under different keys)
    followers = (
        raw_metadata.get("followers")
        or raw_metadata.get("followers_count")
        or raw_metadata.get("subscribers_count")
    )
    if followers is not None:
        try:
            followers_int = int(followers)
            extracted["metrics_history"] = [
                {
                    "subscribers_count": followers_int,
                    "posts_count": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ]
        except (ValueError, TypeError):
            pass

    # Extract profile URL
    profile_url = raw_metadata.get("profile_url") or raw_metadata.get("url")
    if profile_url and isinstance(profile_url, str):
        extracted["profile_url"] = profile_url

    # Extract website/link in bio
    website = raw_metadata.get("website") or raw_metadata.get("link")
    if website and isinstance(website, str):
        extracted["website"] = website

    link_in_bio = raw_metadata.get("link_in_bio")
    if link_in_bio and isinstance(link_in_bio, str):
        extracted["link_in_bio"] = link_in_bio

    # Extract contacts if present
    contacts = raw_metadata.get("contacts")
    if contacts and isinstance(contacts, dict):
        extracted["contacts"] = contacts
    elif any(key in raw_metadata for key in ["emails", "phones", "telegram"]):
        # Build contacts from top-level keys
        contacts_dict: dict[str, Any] = {}
        if "emails" in raw_metadata:
            contacts_dict["emails"] = raw_metadata["emails"]
        if "phones" in raw_metadata:
            contacts_dict["phones"] = raw_metadata["phones"]
        if "telegram_channels" in raw_metadata:
            contacts_dict["telegram_channels"] = raw_metadata["telegram_channels"]
        if "telegram_personal" in raw_metadata:
            contacts_dict["telegram_personal"] = raw_metadata["telegram_personal"]
        if contacts_dict:
            extracted["contacts"] = contacts_dict

    # Extract external platforms if present
    external = raw_metadata.get("external_platforms")
    if external and isinstance(external, dict):
        extracted["external_platforms"] = external

    # Resolve raw profile payload with fallback mechanism
    raw_payload = raw_metadata.get("raw_profile_payload") or raw_metadata.get(
        "raw_api_response"
    )

    # Check if raw_payload was found in nested keys
    has_nested_payload = raw_payload is not None and isinstance(raw_payload, dict)

    if has_nested_payload:
        # Use the existing nested payload
        extracted["raw_profile_payload"] = raw_payload
    else:
        # Critical fallback: check if raw_metadata looks like a migrated schema
        # Migrated schemas typically have 'extracted_at' at top-level
        # If raw_metadata lacks these migrated schema markers, treat it as raw payload
        is_migrated_schema = (
            "extracted_at" in raw_metadata or "raw_profile_payload" in raw_metadata
        )

        if not is_migrated_schema:
            # raw_metadata is likely a flat, unstructured raw API response
            extracted["raw_profile_payload"] = raw_metadata

    # Preserve metrics_history if already present (append, don't replace)
    existing_metrics = raw_metadata.get("metrics_history")
    if existing_metrics and isinstance(existing_metrics, list):
        if "metrics_history" in extracted:
            # Merge: prepend existing metrics to new metrics
            extracted["metrics_history"] = (
                existing_metrics + extracted["metrics_history"]
            )
        else:
            extracted["metrics_history"] = existing_metrics

    return extracted


def migrate_account_metadata(
    account: Account,
) -> dict[str, Any] | None:
    """
    Migrate account raw_metadata to match AccountMetadata schema.

    Args:
        account: Account model instance with raw_metadata.

    Returns:
        Migrated metadata dictionary or None if migration fails.
    """
    try:
        raw_data: dict[str, Any] = {}

        # Get existing raw_metadata
        existing_metadata = account.raw_metadata

        # Extract legacy fields from existing metadata
        if existing_metadata and isinstance(existing_metadata, dict):
            legacy_fields = extract_account_legacy_fields(existing_metadata)
            raw_data.update(legacy_fields)

        # Set category (preserve if exists, otherwise default to "unknown")
        if (
            existing_metadata
            and isinstance(existing_metadata, dict)
            and existing_metadata.get("category")
        ):
            raw_data["category"] = existing_metadata["category"]
        else:
            raw_data["category"] = DEFAULT_CATEGORY

        # Set language (default to "ru" for Russian creators)
        if (
            existing_metadata
            and isinstance(existing_metadata, dict)
            and existing_metadata.get("language")
        ):
            raw_data["language"] = existing_metadata["language"]
        else:
            raw_data["language"] = DEFAULT_LANGUAGE

        # Ensure extracted_at timestamp exists
        if (
            existing_metadata
            and isinstance(existing_metadata, dict)
            and existing_metadata.get("extracted_at")
        ):
            raw_data["extracted_at"] = existing_metadata["extracted_at"]
        else:
            raw_data["extracted_at"] = datetime.now(timezone.utc).isoformat()

        # Validate and dump using AccountMetadata schema
        validated = AccountMetadata.model_validate(raw_data)
        return validated.model_dump(exclude_none=True)

    except Exception as e:
        logger.error(
            "Failed to migrate account metadata for account_id=%d: %s",
            account.id,
            e,
            exc_info=True,
        )
        return None


def extract_content_legacy_fields(
    raw_metadata: dict[str, Any] | None,
    raw_item_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Extract legacy fields from content raw_metadata.

    Args:
        raw_metadata: Existing content raw_metadata dictionary or None.
        raw_item_payload: Raw API response payload or None.

    Returns:
        Dictionary with extracted fields mapped to ContentMetadata structure.
    """
    if not raw_metadata and not raw_item_payload:
        return {}

    extracted: dict[str, Any] = {}

    # Resolve raw item payload with fallback mechanism
    resolved_payload: dict[str, Any] | None = None

    # Priority 1: Use explicit raw_item_payload argument if provided
    if raw_item_payload and isinstance(raw_item_payload, dict):
        resolved_payload = raw_item_payload
        extracted["raw_item_payload"] = raw_item_payload

    # Priority 2: Check if raw_metadata contains nested payload keys
    if resolved_payload is None and raw_metadata and isinstance(raw_metadata, dict):
        nested_payload = raw_metadata.get("raw_item_payload") or raw_metadata.get(
            "raw_item"
        )
        if nested_payload and isinstance(nested_payload, dict):
            resolved_payload = nested_payload
            extracted["raw_item_payload"] = nested_payload

    # Priority 3: Critical fallback - treat entire raw_metadata as raw payload
    if resolved_payload is None and raw_metadata and isinstance(raw_metadata, dict):
        # Check if raw_metadata looks like a migrated schema
        # Migrated schemas typically have 'extracted_at' or 'post_type' at top-level
        is_migrated_schema = (
            "extracted_at" in raw_metadata or "post_type" in raw_metadata
        )

        if not is_migrated_schema:
            # raw_metadata is likely a flat, unstructured raw API response
            resolved_payload = raw_metadata
            extracted["raw_item_payload"] = raw_metadata

    # Use resolved_payload for extraction (fall back to raw_item_payload argument if resolution failed)
    payload_for_extraction = resolved_payload or raw_item_payload

    # Merge with raw_metadata (raw_metadata may have processed fields)
    if raw_metadata and isinstance(raw_metadata, dict):
        # Extract video_url (may be under different keys)
        video_url = (
            raw_metadata.get("video_url")
            or raw_metadata.get("video")
            or raw_metadata.get("video_url_hd")
        )
        if not video_url and payload_for_extraction:
            # Try to extract from resolved payload
            video_url = (
                payload_for_extraction.get("video_url")
                or payload_for_extraction.get("video", {}).get("url")
                if isinstance(payload_for_extraction.get("video"), dict)
                else None
            )
        if video_url and isinstance(video_url, str):
            extracted["video_url"] = video_url

        # Extract platform_metrics using resolved payload
        platform_metrics: dict[str, Any] = {}

        # Likes
        likes = (
            raw_metadata.get("likes")
            or raw_metadata.get("like_count")
            or (payload_for_extraction or {}).get("like_count")
        )
        if likes is not None:
            try:
                platform_metrics["likes"] = int(likes)
            except (ValueError, TypeError):
                pass

        # Comments
        comments = (
            raw_metadata.get("comments_count")
            or raw_metadata.get("comment_count")
            or (payload_for_extraction or {}).get("comment_count")
        )
        if comments is not None:
            try:
                platform_metrics["comments_count"] = int(comments)
            except (ValueError, TypeError):
                pass

        # Views
        views = (
            raw_metadata.get("views")
            or raw_metadata.get("view_count")
            or (payload_for_extraction or {}).get("view_count")
        )
        if views is not None:
            try:
                platform_metrics["views"] = int(views)
            except (ValueError, TypeError):
                pass

        # Plays (for videos)
        plays = (
            raw_metadata.get("plays")
            or raw_metadata.get("play_count")
            or (payload_for_extraction or {}).get("play_count")
        )
        if plays is not None:
            try:
                platform_metrics["plays"] = int(plays)
            except (ValueError, TypeError):
                pass

        # Shares
        shares = (
            raw_metadata.get("shares")
            or raw_metadata.get("share_count")
            or (payload_for_extraction or {}).get("share_count")
        )
        if shares is not None:
            try:
                platform_metrics["shares"] = int(shares)
            except (ValueError, TypeError):
                pass

        if platform_metrics:
            extracted["platform_metrics"] = platform_metrics

        # Extract post_type
        post_type = raw_metadata.get("post_type")
        if post_type and isinstance(post_type, str):
            extracted["post_type"] = post_type
        else:
            # Try to infer from video duration or media type
            duration = raw_metadata.get("duration") or (
                payload_for_extraction or {}
            ).get("duration")
            if duration and isinstance(duration, (int, float)):
                # If duration < 90 seconds, likely a reel
                if duration < REEL_DURATION_THRESHOLD:
                    extracted["post_type"] = "reel"
                else:
                    extracted["post_type"] = "post"
            elif video_url or (payload_for_extraction or {}).get("is_video"):
                extracted["post_type"] = "reel"
            else:
                extracted["post_type"] = "post"

        # Extract geo_data if present
        geo_data = raw_metadata.get("geo_data")
        if geo_data and isinstance(geo_data, dict):
            extracted["geo_data"] = geo_data
        elif raw_metadata.get("location"):
            location = raw_metadata["location"]
            if isinstance(location, dict):
                extracted["geo_data"] = {
                    "location_id": location.get("id"),
                    "name": location.get("name"),
                    "lat": location.get("lat"),
                    "lng": location.get("lng"),
                }

        # Extract author_profile_snapshot if present
        author_snapshot = raw_metadata.get("author_profile_snapshot")
        if author_snapshot and isinstance(author_snapshot, dict):
            extracted["author_profile_snapshot"] = author_snapshot

        # Preserve extracted_at if present
        if raw_metadata.get("extracted_at"):
            extracted["extracted_at"] = raw_metadata["extracted_at"]

    return extracted


def migrate_content_metadata(
    content: Content,
    account_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Migrate content raw_metadata to match ContentMetadata schema.

    Args:
        content: Content model instance with raw_metadata.
        account_metadata: Migrated account metadata (to inherit category).

    Returns:
        Migrated metadata dictionary or None if migration fails.
    """
    try:
        raw_data: dict[str, Any] = {}

        # Get existing raw_metadata and raw_item_payload
        existing_metadata = content.raw_metadata
        raw_item_payload = None
        if existing_metadata and isinstance(existing_metadata, dict):
            raw_item_payload = existing_metadata.get("raw_item_payload")

        # Extract legacy fields
        legacy_fields = extract_content_legacy_fields(
            existing_metadata, raw_item_payload
        )
        raw_data.update(legacy_fields)

        # Inherit category from account metadata
        if account_metadata and account_metadata.get("category"):
            raw_data["category"] = account_metadata["category"]
        else:
            raw_data["category"] = DEFAULT_CATEGORY

        # Set language (default to "ru" for Russian creators)
        if (
            existing_metadata
            and isinstance(existing_metadata, dict)
            and existing_metadata.get("language")
        ):
            raw_data["language"] = existing_metadata["language"]
        else:
            raw_data["language"] = DEFAULT_LANGUAGE

        # Ensure post_type is set
        if "post_type" not in raw_data:
            # Check if it's a reel based on video duration
            duration = None
            if existing_metadata and isinstance(existing_metadata, dict):
                duration = existing_metadata.get("duration")
            if not duration and raw_item_payload:
                duration = raw_item_payload.get("duration")
            if duration and isinstance(duration, (int, float)):
                raw_data["post_type"] = (
                    "reel" if duration < REEL_DURATION_THRESHOLD else "post"
                )
            else:
                raw_data["post_type"] = "post"

        # Ensure extracted_at timestamp exists
        if (
            existing_metadata
            and isinstance(existing_metadata, dict)
            and existing_metadata.get("extracted_at")
        ):
            raw_data["extracted_at"] = existing_metadata["extracted_at"]
        else:
            raw_data["extracted_at"] = datetime.now(timezone.utc).isoformat()

        # Validate and dump using appropriate schema
        # Use InstagramContentMetadata if we have Instagram-specific fields
        is_reel = raw_data.get("post_type") == "reel"
        if is_reel:
            # Add is_reel field for InstagramContentMetadata
            raw_data["is_reel"] = True
            validated = InstagramContentMetadata.model_validate(raw_data)
        else:
            validated = ContentMetadata.model_validate(raw_data)

        return validated.model_dump(exclude_none=True)

    except Exception as e:
        logger.error(
            "Failed to migrate content metadata for content_id=%d: %s",
            content.id,
            e,
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Main migration logic
# ---------------------------------------------------------------------------


async def migrate_instagram_accounts_batch(
    session: AsyncSession,
    accounts: list[Account],
) -> dict[int, dict[str, Any]]:
    """
    Migrate a batch of Instagram accounts' metadata.

    Args:
        session: Async SQLAlchemy session.
        accounts: List of Account model instances to migrate.

    Returns:
        Dictionary mapping account ID to migrated metadata.
    """
    migrated_metadata: dict[int, dict[str, Any]] = {}

    for account in accounts:
        logger.debug("Migrating account id=%d, username=%s", account.id, account.username)

        new_metadata = migrate_account_metadata(account)
        if new_metadata:
            migrated_metadata[account.id] = new_metadata

            # Update the account object
            account.raw_metadata = new_metadata

            logger.info(
                "Migrated account id=%d, username=%s, category=%s",
                account.id,
                account.username,
                new_metadata.get("category", "unknown"),
            )
        else:
            logger.warning(
                "Skipped account id=%d, username=%s (migration failed)",
                account.id,
                account.username,
            )

    return migrated_metadata


async def migrate_instagram_content_batch(
    session: AsyncSession,
    account_ids: list[int],
    account_metadata_map: dict[int, dict[str, Any]],
) -> int:
    """
    Migrate content metadata for a batch of Instagram accounts.

    Args:
        session: Async SQLAlchemy session.
        account_ids: List of account IDs to process content for.
        account_metadata_map: Dictionary mapping account ID to migrated metadata.

    Returns:
        Number of content rows successfully migrated.
    """
    total_migrated = 0

    # Fetch all content for these accounts
    stmt = (
        select(Content)
        .where(Content.account_id.in_(account_ids))
        .order_by(Content.id)
    )
    result = await session.execute(stmt)
    content_items = list(result.scalars().all())

    logger.info(
        "Found %d content items for %d Instagram accounts",
        len(content_items),
        len(account_ids),
    )

    for content in content_items:
        account_metadata = account_metadata_map.get(content.account_id)
        new_metadata = migrate_content_metadata(content, account_metadata)

        if new_metadata:
            content.raw_metadata = new_metadata
            total_migrated += 1

            logger.debug(
                "Migrated content id=%d, account_id=%d, post_type=%s",
                content.id,
                content.account_id,
                new_metadata.get("post_type", "unknown"),
            )
        else:
            logger.warning(
                "Skipped content id=%d, account_id=%d (migration failed)",
                content.id,
                content.account_id,
            )

    logger.info(
        "Migrated %d content items for %d accounts",
        total_migrated,
        len(account_ids),
    )

    return total_migrated


async def migrate_instagram_data(db: Database) -> dict[str, int]:
    """
    Main migration function for Instagram accounts and content.

    Processes Instagram records in batches of BATCH_SIZE (50) accounts.
    Each batch is committed separately to ensure partial progress is saved.

    Args:
        db: Initialized Database instance.

    Returns:
        Dictionary with counts of migrated records.
    """
    results: dict[str, int] = {
        "accounts_migrated": 0,
        "content_migrated": 0,
        "accounts_failed": 0,
        "content_failed": 0,
    }

    last_id = 0
    batch_number = 0

    logger.info(
        "Starting Instagram data migration (batch_size=%d)...",
        BATCH_SIZE,
    )

    while True:
        batch_number += 1
        logger.info("Processing batch #%d (last_id=%d)...", batch_number, last_id)

        async with db.async_session() as session:
            # Fetch batch of Instagram accounts using keyset pagination
            stmt = (
                select(Account)
                .where(
                    Account.platform == "INSTAGRAM",
                    Account.id > last_id,
                )
                .order_by(Account.id)
                .limit(BATCH_SIZE)
            )

            result = await session.execute(stmt)
            accounts = list(result.scalars().all())

            if not accounts:
                logger.info("No more Instagram accounts to process.")
                break

            logger.info(
                "Fetched %d Instagram accounts for batch #%d",
                len(accounts),
                batch_number,
            )

            # Migrate accounts in this batch
            try:
                # Adaptive Transaction Management: check if transaction already active
                if session.in_transaction():
                    # Transaction already active (e.g., from db.async_session() context manager)
                    # Execute batch operations directly without opening new transaction
                    migrated_metadata = await migrate_instagram_accounts_batch(
                        session, accounts
                    )

                    results["accounts_migrated"] += len(migrated_metadata)
                    results["accounts_failed"] += len(accounts) - len(
                        migrated_metadata
                    )

                    # Migrate associated content
                    if migrated_metadata:
                        account_ids = list(migrated_metadata.keys())
                        content_migrated = await migrate_instagram_content_batch(
                            session, account_ids, migrated_metadata
                        )
                        results["content_migrated"] += content_migrated

                    # Flush changes to database; outer context manager handles commit
                    await session.flush()

                else:
                    # No active transaction; safely wrap batch in explicit transaction
                    async with session.begin():
                        migrated_metadata = await migrate_instagram_accounts_batch(
                            session, accounts
                        )

                        results["accounts_migrated"] += len(migrated_metadata)
                        results["accounts_failed"] += len(accounts) - len(
                            migrated_metadata
                        )

                        # Migrate associated content
                        if migrated_metadata:
                            account_ids = list(migrated_metadata.keys())
                            content_migrated = await migrate_instagram_content_batch(
                                session, account_ids, migrated_metadata
                            )
                            results["content_migrated"] += content_migrated

                logger.info(
                    "Batch #%d committed successfully (%d accounts, %d content items)",
                    batch_number,
                    len(migrated_metadata),
                    results["content_migrated"],
                )

            except Exception as e:
                logger.error(
                    "Batch #%d failed: %s",
                    batch_number,
                    e,
                    exc_info=True,
                )
                results["accounts_failed"] += len(accounts)
                # Continue to next batch (don't abort entire migration)

            # Update last_id for keyset pagination
            if accounts:
                last_id = max(account.id for account in accounts)

    logger.info(
        "Instagram data migration completed. "
        "Accounts: %d migrated, %d failed. "
        "Content: %d migrated, %d failed.",
        results["accounts_migrated"],
        results["accounts_failed"],
        results["content_migrated"],
        results["content_failed"],
    )

    return results


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


async def main() -> None:
    """
    Run Instagram metadata migration.

    Initializes database connection and executes the migration process.
    """
    logger.info("Starting Instagram metadata migration utility...")

    db: Database | None = None

    try:
        # Load settings from environment
        settings = load_settings()

        # Mask credentials in logs
        db_url_safe = (
            settings.db_url.split("@")[-1]
            if "@" in settings.db_url
            else settings.db_url
        )
        logger.info(
            "Configuration loaded. Target database: %s",
            db_url_safe,
        )

        # Initialize database connection
        db = Database(settings.db_url, echo=False)
        logger.info("Database connection established.")

        # Run migration
        migration_results = await migrate_instagram_data(db)

        logger.info(
            "Migration completed successfully. Results: %s",
            migration_results,
        )

    except Exception as e:
        logger.error(
            "Migration failed with error: %s",
            e,
            exc_info=True,
        )
        raise

    finally:
        if db is not None:
            await db.close()
            logger.info("Database connection closed.")


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Run async main function
    asyncio.run(main())
