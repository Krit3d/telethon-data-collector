"""
Script to harvest potential creator handles from existing Telegram messages in the database.

This script queries the `content` table for messages containing social media links,
extracts the usernames, and inserts them as `pending` entries into the `accounts` table.
"""

import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

# Import async_sessionmaker and engine setup from src.db.database as per task requirements
from src.db.database import async_sessionmaker, create_async_engine
from src.db.models import Account, Content

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Regex patterns for extracting usernames from social media URLs

# Instagram: matches instagram.com/username
# Excludes common non-username paths like /p/, /reel/, /tv/, /stories/, etc.
# Username can contain letters, numbers, periods, underscores (1-30 chars)
INSTAGRAM_PATTERN = re.compile(
    r'instagram\.com/(?!p/|reel/|tv/|stories/|explore/|accounts/|about/)([A-Za-z0-9_.]{1,30})',
    re.IGNORECASE,
)

# TikTok: matches tiktok.com/@username
# Username after @ can contain letters, numbers, underscores, periods (2-24 chars)
TIKTOK_PATTERN = re.compile(
    r'tiktok\.com/@([A-Za-z0-9_.]{2,24})',
    re.IGNORECASE,
)

# YouTube: matches youtube.com/@username (new handle format)
# Username after @ can contain letters, numbers, underscores, hyphens (3-30 chars)
YOUTUBE_HANDLE_PATTERN = re.compile(
    r'youtube\.com/@([A-Za-z0-9_.-]{3,30})',
    re.IGNORECASE,
)

# YouTube: matches youtube.com/c/channelname (old custom URL format)
YOUTUBE_C_PATTERN = re.compile(
    r'youtube\.com/c/([A-Za-z0-9_.-]{3,30})',
    re.IGNORECASE,
)

# YouTube: matches youtube.com/user/username (older format)
YOUTUBE_USER_PATTERN = re.compile(
    r'youtube\.com/user/([A-Za-z0-9_.-]{3,30})',
    re.IGNORECASE,
)


def extract_social_media_handles(text: str | None) -> list[dict[str, str]]:
    """
    Extract social media handles from text containing URLs.

    Parses the text for Instagram, TikTok, and YouTube URLs and extracts
    the usernames/handles from them.

    Args:
        text: The text to search for social media URLs. Can be None.

    Returns:
        A list of dictionaries with 'platform' and 'username' keys.
        Returns empty list if text is None or no handles are found.
    """
    if not text:
        return []

    handles: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    # Extract Instagram handles
    for match in INSTAGRAM_PATTERN.finditer(text):
        username = match.group(1).lower()
        # Filter out numeric-only matches (likely post IDs)
        if not username.isdigit():
            key = ("INSTAGRAM", username)
            if key not in seen:
                seen.add(key)
                handles.append({"platform": "INSTAGRAM", "username": username})

    # Extract TikTok handles
    for match in TIKTOK_PATTERN.finditer(text):
        username = match.group(1).lower()
        key = ("TIKTOK", username)
        if key not in seen:
            seen.add(key)
            handles.append({"platform": "TIKTOK", "username": username})

    # Extract YouTube handles (new @username format)
    for match in YOUTUBE_HANDLE_PATTERN.finditer(text):
        username = match.group(1).lower()
        key = ("YOUTUBE", username)
        if key not in seen:
            seen.add(key)
            handles.append({"platform": "YOUTUBE", "username": username})

    # Extract YouTube handles (old /c/ format)
    for match in YOUTUBE_C_PATTERN.finditer(text):
        username = match.group(1).lower()
        key = ("YOUTUBE", username)
        if key not in seen:
            seen.add(key)
            handles.append({"platform": "YOUTUBE", "username": username})

    # Extract YouTube handles (older /user/ format)
    for match in YOUTUBE_USER_PATTERN.finditer(text):
        username = match.group(1).lower()
        key = ("YOUTUBE", username)
        if key not in seen:
            seen.add(key)
            handles.append({"platform": "YOUTUBE", "username": username})

    return handles


async def main() -> None:
    """
    Main function to harvest creator handles from Telegram messages.

    Queries the content table for messages with social media links,
    extracts usernames, and inserts them as pending accounts.
    """
    # Get database URL from environment variable
    # Can be set via DB_URL in .env file (see .env.example)
    db_url = os.getenv("DB_URL")
    if not db_url:
        logger.error("DB_URL environment variable is not set. Please set it in .env file.")
        return

    # Create async engine and sessionmaker
    # Using imports from src.db.database as specified in task requirements
    engine = create_async_engine(db_url, echo=False)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    total_links_analyzed = 0
    new_accounts_added = 0
    batch_size = 100

    async with async_session() as session:
        # Query content table for records containing social media links
        # Look for content with instagram.com, tiktok.com, youtube.com/, or youtu.be/
        query = (
            select(Content)
            .where(Content.content.isnot(None))
            .where(
                Content.content.ilike("%instagram.com%")
                | Content.content.ilike("%tiktok.com%")
                | Content.content.ilike("%youtube.com/%")
                | Content.content.ilike("%youtu.be/%")
            )
        )

        result = await session.execute(query)
        content_records = result.scalars().all()

        logger.info(
            f"Found {len(content_records)} content records with potential social media links"
        )

        # Extract handles from all content records
        all_handles: list[dict[str, str]] = []
        for content in content_records:
            if content.content:
                handles = extract_social_media_handles(content.content)
                all_handles.extend(handles)
                total_links_analyzed += len(handles)

        logger.info(f"Extracted {len(all_handles)} total handle occurrences from content")

        # Deduplicate by platform and username
        unique_handles: list[dict[str, str]] = []
        seen_keys: set[tuple[str, str]] = set()
        for handle in all_handles:
            key = (handle["platform"], handle["username"])
            if key not in seen_keys:
                seen_keys.add(key)
                unique_handles.append(handle)

        logger.info(
            f"Found {len(unique_handles)} unique handles after deduplication"
        )

        # Fetch all existing (platform, platform_id) tuples from the accounts table
        # This avoids using PostgreSQL-specific ON CONFLICT clauses
        existing_result = await session.execute(
            select(Account.platform, Account.platform_id)
        )
        existing_accounts: set[tuple[str, str]] = {
            (row[0], row[1]) for row in existing_result.all()
        }
        logger.info(
            f"Fetched {len(existing_accounts)} existing accounts from database"
        )

        # Filter out handles that already exist in the accounts table
        # Match by (platform, platform_id) where platform_id is the username for social media accounts
        new_handles: list[dict[str, str]] = []
        for handle in unique_handles:
            key = (handle["platform"], handle["username"])
            if key not in existing_accounts:
                new_handles.append(handle)

        logger.info(
            f"Found {len(new_handles)} new handles to insert "
            f"(skipping {len(unique_handles) - len(new_handles)} existing)"
        )

        # Batch insert new accounts using session.add_all() without ON CONFLICT clauses
        now = datetime.now(timezone.utc)
        for i in range(0, len(new_handles), batch_size):
            batch = new_handles[i : i + batch_size]

            # Create Account objects for insertion
            accounts_to_add = [
                Account(
                    platform=h["platform"],
                    platform_id=h["username"],  # Use username as temporary platform_id
                    username=h["username"],
                    title="Extracted from Telegram Source",
                    status="pending",
                    created_at=now,
                    updated_at=now,
                )
                for h in batch
            ]

            session.add_all(accounts_to_add)
            await session.commit()

            new_accounts_added += len(batch)
            logger.info(
                f"Inserted batch {i // batch_size + 1}: {len(batch)} accounts"
            )

        logger.info(
            f"Completed! Analyzed {total_links_analyzed} links, "
            f"added {new_accounts_added} new pending accounts to the database."
        )

    # Clean up
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
