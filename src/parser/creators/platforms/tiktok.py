"""
TikTok platform parser using Scrape Creators API.

Implements TikTok-specific profile parsing and content ingestion into PostgreSQL.
Extracts author profile metadata (external links, contacts, location, language)
and stores it inside Content.raw_metadata under the "author_profile_metadata" key.

Features:
    - One-request cache optimization for profile data
    - Profile parsing with account upsert to accounts table
    - Minimum and maximum follower threshold enforcement (3k-150k, micro-influencers)
    - Russian language (Cyrillic) biography check
    - Virtual profile post creation for semantic search over biographies
    - Content fetching from itemList with deduplication
    - PostgreSQL ON CONFLICT DO UPDATE for high-throughput concurrency
    - Raw metadata preservation for OpenSPG processing
    - Extraction of external links, contact info from biography via shared utils
    - Video download URL extraction for GPU worker processing
    - Transcription support for video content
    - Cross-platform spidering queue for discovered accounts
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config.config import Settings
from src.db.models import Account, Content
from src.parser.creators.core.utils import parse_profile_contacts
from src.parser.creators.platforms.base import BasePlatformParser
from src.parser.creators.sc_client import ScrapeCreatorsClient

logger = logging.getLogger(__name__)

# Minimum and maximum follower thresholds for TikTok accounts (micro-influencers: 3k-150k)
MIN_FOLLOWERS: int = 3000
MAX_FOLLOWERS: int = 150000


def is_russian_text(text: str | None) -> bool:
    """Check if text contains Russian Cyrillic characters.

    Args:
        text: Text to check, or None.

    Returns:
        True if text contains at least one Russian Cyrillic character, False otherwise.
    """
    if not text:
        return False
    return bool(re.search(r'[а-яА-ЯёЁ]', text))


class TikTokPlatformParser(BasePlatformParser):
    """TikTok platform parser for profile and content ingestion.

    Inherits from BasePlatformParser and implements TikTok-specific
    profile parsing and content upsert logic using the Scrape Creators API v2.

    Features instance-level caching for the profile endpoint to avoid
    duplicate API calls when fetching profile and content data.

    All author-level metadata (external links, contacts, location, language)
    is stored inside each Content.raw_metadata JSONB column under the key
    "author_profile_metadata" since Account has no raw_metadata column.

    After each successful profile parse, a virtual profile post is upserted
    into the content table to enable semantic search over creator biographies.

    Attributes:
        session_maker: SQLAlchemy async session maker for database operations.
        client: ScrapeCreatorsClient instance for API requests.
        settings: Application settings containing configuration values.
        _cached_profile: Instance-level cache for profile response data.
        _cached_handle: The handle for which profile data is cached.
    """

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        client: ScrapeCreatorsClient,
        settings: Settings,
    ) -> None:
        """Initialize TikTok parser with configuration.

        Args:
            session_maker: SQLAlchemy async session maker for database operations.
            client: ScrapeCreatorsClient instance for API requests.
            settings: Application settings containing configuration values.
        """
        super().__init__(session_maker, client, settings)
        self._cached_profile: dict[str, Any] | None = None
        self._cached_handle: str | None = None

    async def parse_profile(self, handle: str) -> int | None:
        """Fetch TikTok profile, upsert account to database, return account ID.

        Parses the TikTok profile for the given handle, checks if the account
        meets the follower threshold (3k-150k), verifies Russian Cyrillic
        in biography, extracts author profile metadata, upserts the account
        information to the accounts table, creates a virtual profile post
        for semantic search, queues discovered cross-platform accounts,
        and returns the database ID.

        Uses instance-level caching to avoid duplicate API calls when subsequently
        calling parse_content() for the same handle.

        Args:
            handle: TikTok username (without @ prefix).

        Returns:
            Database ID of the upserted account record, or None if the profile
            could not be parsed, doesn't meet follower thresholds, or is non-Russian.
        """
        logger.info("Starting TikTok profile parse for handle: %s", handle)

        try:
            # Check cache first
            if self._cached_handle == handle and self._cached_profile:
                logger.debug("Using cached profile data for handle: %s", handle)
                response = self._cached_profile
            else:
                # Fetch profile data from Scrape Creators API v2
                try:
                    response = await self.client.get(
                        endpoint="/v2/tiktok/profile",
                        params={"handle": handle, "count": 100},
                    )
                    logger.info(
                        "API response status for profile %s: success, credits consumed: %s",
                        handle,
                        response.get("credits", "N/A"),
                    )
                    # Cache the response for subsequent content parsing
                    self._cached_profile = response
                    self._cached_handle = handle
                except Exception as e:
                    logger.error(
                        "API request failed for TikTok profile %s: %s",
                        handle,
                        e,
                        exc_info=True,
                    )
                    return None

            # Validate response structure
            data = response.get("data")
            if not data:
                logger.error("Missing 'data' in API response for TikTok handle %s", handle)
                return None

            user = data.get("user")
            if not user:
                logger.error("Missing 'user' in data for TikTok handle %s", handle)
                return None

            # Extract user ID, follower count, and biography
            user_id: str = str(user.get("id") or user.get("userId", ""))
            if not user_id:
                logger.error("Could not extract user ID for TikTok handle %s", handle)
                return None

            follower_count: int = self._extract_follower_count(user)
            biography: str | None = user.get("signature") or user.get("bio")

            # Check if account meets follower range (micro-influencer: 3k-150k)
            if follower_count < MIN_FOLLOWERS or follower_count > MAX_FOLLOWERS:
                logger.info(
                    "TikTok handle %s has %d followers, outside range [%d, %d]. Rejecting.",
                    handle,
                    follower_count,
                    MIN_FOLLOWERS,
                    MAX_FOLLOWERS,
                )
                await self._upsert_account(user, status="rejected")
                return None

            # Check if biography contains Russian Cyrillic characters
            if not is_russian_text(biography):
                logger.info(
                    "TikTok handle %s has non-Russian biography. Rejecting.",
                    handle,
                )
                await self._upsert_account(user, status="rejected")
                return None

            # Upsert account with 'parsed' status
            account_id: int = await self._upsert_account(user, status="parsed")
            logger.info(
                "Successfully parsed TikTok profile %s, account ID: %d, followers: %d",
                handle,
                account_id,
                follower_count,
            )

            # Parse contacts from biography for cross-platform spidering
            contacts_dict: dict[str, Any] = parse_profile_contacts(biography, None)

            # Queue discovered accounts from contacts
            async with self.session_maker() as session:
                await self._queue_discovered_accounts(session, contacts_dict)

            # Upsert virtual profile post for semantic search over biography
            await self._upsert_virtual_profile_post(account_id, user)

            return account_id

        except Exception as e:
            logger.error(
                "Failed to parse TikTok profile %s: %s",
                handle,
                e,
                exc_info=True,
            )
            raise

    async def parse_content(
        self,
        account_id: int,
        platform_id: str,
        max_items: int = 50,
    ) -> None:
        """Fetch TikTok content and bulk upsert to content table.

        Retrieves content items from the cached profile response itemList
        for the given account using the Scrape Creators API v2, parses the data,
        and performs a bulk upsert into the content table using PostgreSQL
        ON CONFLICT DO UPDATE.

        The raw_metadata field contains:
            - "author_profile_metadata": Profile-level data (contacts, links, location, etc.)
            - "platform_metrics": Platform-specific engagement metrics
            - "video_download_url": Direct MP4 URL for GPU worker processing

        Args:
            account_id: Database ID of the parent account record.
            platform_id: TikTok username/handle used in API calls.
            max_items: Maximum number of content items to fetch (default: 50).
        """
        logger.info(
            "Starting TikTok content parse for account_id: %d, platform_id: %s, max_items: %d",
            account_id,
            platform_id,
            max_items,
        )

        try:
            # Get profile data from cache or fetch if needed
            if self._cached_handle == platform_id and self._cached_profile:
                response = self._cached_profile
            else:
                try:
                    response = await self.client.get(
                        endpoint="/v2/tiktok/profile",
                        params={"handle": platform_id, "count": 100},
                    )
                    self._cached_profile = response
                    self._cached_handle = platform_id
                except Exception as e:
                    logger.error(
                        "API request failed for TikTok content, platform_id %s: %s",
                        platform_id,
                        e,
                        exc_info=True,
                    )
                    return

            # Validate response structure
            data = response.get("data")
            if not data:
                logger.error(
                    "Missing 'data' in API response for TikTok content, platform_id %s",
                    platform_id,
                )
                return

            # Extract itemList from cached profile response
            item_list: list[dict[str, Any]] = data.get("itemList", [])
            if not isinstance(item_list, list):
                logger.error(
                    "Invalid itemList in API response for TikTok content, platform_id %s",
                    platform_id,
                )
                return

            if not item_list:
                logger.info("No TikTok content found for account_id: %d", account_id)
                return

            # Limit to max_items
            item_list = item_list[:max_items]

            # Build author profile metadata once (reused for all content items)
            user = data.get("user", {})
            author_metadata = self._build_author_profile_metadata(user)

            # Process and upsert content items
            await self._upsert_content(item_list, account_id, author_metadata)
            logger.info(
                "Successfully upserted %d TikTok content items for account_id: %d",
                len(item_list),
                account_id,
            )

        except Exception as e:
            logger.error(
                "Failed to parse TikTok content for account_id %d: %s",
                account_id,
                e,
                exc_info=True,
            )
            raise

    async def _upsert_account(self, user: dict[str, Any], status: str = "parsed") -> int:
        """Upsert TikTok account record using select-then-insert/update pattern.

        Uses a select-then-upsert transaction pattern to avoid InvalidColumnReferenceError
        caused by missing unique constraint on (platform, platform_id) in the accounts table.

        Args:
            user: User object from Scrape Creators API response.
            status: Account status ('parsed', 'rejected', etc.).

        Returns:
            ID of the account record (auto-generated by PostgreSQL for new records).
        """
        platform_id: str = str(user.get("id") or user.get("userId", ""))
        username: str | None = user.get("uniqueId") or user.get("handle")
        full_name: str | None = user.get("nickname") or user.get("displayName")
        biography: str | None = user.get("signature") or user.get("bio")
        follower_count: int = self._extract_follower_count(user)

        async with self.session_maker() as session:
            # Select existing account by platform and platform_id
            stmt = select(Account).where(
                Account.platform == "TIKTOK",
                Account.platform_id == platform_id,
            )
            result = await session.execute(stmt)
            db_account: Account | None = result.scalar_one_or_none()

            if db_account:
                # Update existing record
                db_account.username = username
                db_account.title = full_name or username or "Unknown"
                db_account.description = biography
                db_account.subscribers_count = follower_count
                db_account.status = status
                db_account.updated_at = datetime.now(timezone.utc)
                logger.debug(
                    "Updated existing TikTok account %s (ID: %d, status: %s)",
                    username,
                    db_account.id,
                    status,
                )
            else:
                # Create new record (let PostgreSQL generate ID)
                db_account = Account(
                    platform="TIKTOK",
                    platform_id=platform_id,
                    username=username,
                    title=full_name or username or "Unknown",
                    description=biography,
                    subscribers_count=follower_count,
                    status=status,
                )
                session.add(db_account)
                logger.debug(
                    "Created new TikTok account %s (status: %s)",
                    username,
                    status,
                )

            await session.commit()
            await session.refresh(db_account)
            return db_account.id

    async def _queue_discovered_accounts(self, session: AsyncSession, contacts_dict: dict[str, Any]) -> None:
        """Queue discovered accounts from contacts for cross-platform spidering.

        Processes Telegram handles and external links from parsed contacts,
        inserting new accounts into the accounts table with status 'pending'
        if they do not already exist.

        Args:
            session: Active SQLAlchemy async session for database operations.
            contacts_dict: Dictionary of parsed contacts from parse_profile_contacts.
        """
        # Process Telegram handles
        telegram_handles: list[str] = contacts_dict.get("telegram_handles", [])
        for handle in telegram_handles:
            if not handle:
                continue
            # Check if account already exists
            stmt = select(Account).where(
                Account.platform == "TELEGRAM",
                Account.platform_id == handle,
            )
            result = await session.execute(stmt)
            if result.scalar_one_or_none():
                logger.debug("Telegram account @%s already exists, skipping", handle)
                continue
            # Insert new Telegram account
            new_account = Account(
                platform="TELEGRAM",
                platform_id=handle,
                username=handle,
                title=handle,
                status="pending",
            )
            session.add(new_account)
            logger.info("Queued Telegram account @%s for spidering", handle)

        # Process external links
        external_links: list[str] = contacts_dict.get("external_links", [])
        for link in external_links:
            if not link:
                continue
            link_lower = link.lower()

            # Instagram
            if "instagram.com/" in link_lower:
                parsed = urlparse(link)
                path_parts = [p for p in parsed.path.split("/") if p]
                if not path_parts:
                    continue
                handle = path_parts[0].split("?")[0]
                if not handle:
                    continue
                # Check if exists
                stmt = select(Account).where(
                    Account.platform == "INSTAGRAM",
                    Account.platform_id == handle,
                )
                result = await session.execute(stmt)
                if result.scalar_one_or_none():
                    logger.debug("Instagram account %s already exists, skipping", handle)
                    continue
                # Insert
                new_account = Account(
                    platform="INSTAGRAM",
                    platform_id=handle,
                    username=handle,
                    title=handle,
                    status="pending",
                )
                session.add(new_account)
                logger.info("Queued Instagram account %s for spidering", handle)
                continue

            # YouTube
            if "youtube.com/" in link_lower or "youtu.be/" in link_lower:
                parsed = urlparse(link)
                path_parts = [p for p in parsed.path.split("/") if p]
                if not path_parts:
                    continue
                channel_id = path_parts[-1].split("?")[0]
                if not channel_id:
                    continue
                # Check if exists
                stmt = select(Account).where(
                    Account.platform == "YOUTUBE",
                    Account.platform_id == channel_id,
                )
                result = await session.execute(stmt)
                if result.scalar_one_or_none():
                    logger.debug("YouTube account %s already exists, skipping", channel_id)
                    continue
                # Insert
                new_account = Account(
                    platform="YOUTUBE",
                    platform_id=channel_id,
                    username=channel_id,
                    title=channel_id,
                    status="pending",
                )
                session.add(new_account)
                logger.info("Queued YouTube account %s for spidering", channel_id)
                continue

            # Threads
            if "threads.net/" in link_lower:
                parsed = urlparse(link)
                path_parts = [p for p in parsed.path.split("/") if p]
                if not path_parts:
                    continue
                handle = path_parts[0].split("?")[0]
                if not handle:
                    continue
                # Check if exists
                stmt = select(Account).where(
                    Account.platform == "THREADS",
                    Account.platform_id == handle,
                )
                result = await session.execute(stmt)
                if result.scalar_one_or_none():
                    logger.debug("Threads account %s already exists, skipping", handle)
                    continue
                # Insert
                new_account = Account(
                    platform="THREADS",
                    platform_id=handle,
                    username=handle,
                    title=handle,
                    status="pending",
                )
                session.add(new_account)
                logger.info("Queued Threads account %s for spidering", handle)
                continue

        # Commit all queued accounts
        await session.commit()

    async def _upsert_virtual_profile_post(
        self, account_id: int, user: dict[str, Any]
    ) -> None:
        """Upsert a virtual profile post into the content table for semantic search.

        Creates a synthetic content record containing the creator's biography and
        profile metadata so the embedding worker can index it into Qdrant for
        semantic search over creator biographies.

        The virtual post has:
            - platform_content_id = "profile_bio_{platform_id}"
            - content = compiled profile metadata string
            - is_embedded = False (picked up by embedding worker)
            - has_media = False

        Args:
            account_id: Database ID of the parent account record.
            user: User object from Scrape Creators API response.
        """
        platform_id: str = str(user.get("id") or user.get("userId", ""))
        username: str | None = user.get("uniqueId") or user.get("handle")
        full_name: str | None = user.get("nickname") or user.get("displayName")
        biography: str | None = user.get("signature") or user.get("bio")
        follower_count: int = self._extract_follower_count(user)

        virtual_content_id: str = f"profile_bio_{platform_id}"
        compiled_text: str = (
            f"[PROFILE METADATA]\n"
            f"Platform: TikTok\n"
            f"Username: @{username or 'unknown'}\n"
            f"Title: {full_name or 'Unknown'}\n"
            f"Subscribers: {follower_count}\n"
            f"Bio: {biography or 'N/A'}"
        )

        now = datetime.now(timezone.utc)

        async with self.session_maker() as session:
            async with session.begin():
                stmt = pg_insert(Content).values(
                    account_id=account_id,
                    platform_content_id=virtual_content_id,
                    content=compiled_text,
                    transcription=None,
                    published_at=now,
                    views=None,
                    reactions_count=None,
                    comments_count=None,
                    shares_count=None,
                    has_media=False,
                    is_embedded=False,
                    is_graph_extracted=False,
                    raw_metadata=None,
                    updated_at=now,
                )
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_content_account_platform_id",
                    set_=dict(
                        content=stmt.excluded.content,
                        updated_at=stmt.excluded.updated_at,
                    ),
                )
                await session.execute(stmt)
                logger.debug(
                    "Upserted virtual profile post for TikTok account_id: %d (platform_content_id: %s)",
                    account_id,
                    virtual_content_id,
                )

    async def _upsert_content(
        self,
        items: list[dict[str, Any]],
        account_id: int,
        author_metadata: dict[str, Any],
    ) -> None:
        """Bulk upsert TikTok content records to database.

        Args:
            items: List of video item dictionaries from API responses.
            account_id: ID of the parent Account record.
            author_metadata: Author profile metadata to embed in each content record.
        """
        content_values: list[dict[str, Any]] = []

        for item in items:
            try:
                # Extract platform_content_id from video id
                platform_content_id: str = str(item.get("id") or "")
                if not platform_content_id:
                    logger.warning("Skipping content item with no ID")
                    continue

                # Extract content text (video description)
                content_text: str | None = item.get("desc") or item.get("description")

                # Extract published timestamp
                create_time = item.get("createTime") or item.get("createdAt")
                published_at: datetime
                if create_time:
                    try:
                        # Handle both Unix timestamp (int) and ISO string
                        if isinstance(create_time, (int, float)):
                            published_at = datetime.fromtimestamp(
                                float(create_time), tz=timezone.utc
                            )
                        else:
                            published_at = datetime.fromisoformat(
                                str(create_time).replace("Z", "+00:00")
                            )
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            "Failed to parse createTime for item %s: %s",
                            platform_content_id,
                            e,
                        )
                        published_at = datetime.now(timezone.utc)
                else:
                    published_at = datetime.now(timezone.utc)

                # Extract engagement metrics
                stats = item.get("stats") or item.get("statistics") or {}
                views: int | None = (
                    item.get("playCount")
                    or item.get("viewCount")
                    or stats.get("playCount")
                    or stats.get("viewCount")
                )
                reactions_count: int | None = (
                    item.get("diggCount")
                    or item.get("likeCount")
                    or stats.get("diggCount")
                    or stats.get("likeCount")
                )
                comments_count: int | None = (
                    item.get("commentCount")
                    or stats.get("commentCount")
                )
                shares_count: int | None = (
                    item.get("shareCount")
                    or stats.get("shareCount")
                )

                # Extract video download URL from video.playAddr or video.downloadAddr
                video_download_url: str | None = self._extract_video_download_url(item)

                # Extract transcription if returned by Scrape Creators API
                transcription: str | None = self._extract_transcription(item)

                # Build platform metrics
                platform_metrics: dict[str, Any] = {
                    "views": views,
                    "likes": reactions_count,
                    "comments": comments_count,
                    "shares": shares_count,
                    "collectCount": (item.get("collectCount")
                                     or stats.get("collectCount")),
                }

                # Build raw_metadata with author_profile_metadata and platform_metrics
                raw_metadata: dict[str, Any] = {
                    "author_profile_metadata": author_metadata,
                    "platform_metrics": platform_metrics,
                    "video_download_url": video_download_url,
                }

                content_values.append(
                    {
                        "account_id": account_id,
                        "platform_content_id": platform_content_id,
                        "content": content_text,
                        "transcription": transcription,
                        "published_at": published_at,
                        "views": views,
                        "reactions_count": reactions_count,
                        "comments_count": comments_count,
                        "shares_count": shares_count,
                        "has_media": True,  # TikTok items are videos
                        "raw_metadata": raw_metadata,
                        "is_embedded": False,
                        "is_graph_extracted": False,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )

            except Exception as e:
                logger.error(
                    "Failed to process TikTok content item: %s",
                    e,
                    exc_info=True,
                )
                continue

        if not content_values:
            logger.info("No valid TikTok content items to upsert")
            return

        # Bulk upsert using PostgreSQL ON CONFLICT
        async with self.session_maker() as session:
            async with session.begin():
                stmt = pg_insert(Content).values(content_values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["account_id", "platform_content_id"],
                    set_=dict(
                        content=stmt.excluded.content,
                        transcription=func.coalesce(stmt.excluded.transcription, Content.transcription),
                        views=stmt.excluded.views,
                        reactions_count=stmt.excluded.reactions_count,
                        comments_count=stmt.excluded.comments_count,
                        shares_count=stmt.excluded.shares_count,
                        raw_metadata=stmt.excluded.raw_metadata,
                        updated_at=stmt.excluded.updated_at,
                    ),
                )
                await session.execute(stmt)
                logger.debug(
                    "Bulk upserted %d TikTok content items for account_id: %d",
                    len(content_values),
                    account_id,
                )

    def _extract_follower_count(self, user: dict[str, Any]) -> int:
        """Extract follower count from TikTok user object.

        Args:
            user: User object from Scrape Creators API response.

        Returns:
            Follower count as integer, defaults to 0 if not found.
        """
        follower_count = (
            user.get("followerCount")
            or user.get("followers")
            or user.get("follower_count")
            or 0
        )
        try:
            return int(follower_count)
        except (ValueError, TypeError):
            return 0

    def _extract_video_download_url(self, item: dict[str, Any]) -> str | None:
        """Extract direct video download URL from TikTok video item.

        Tries video.playAddr first, then video.downloadAddr as per specification.
        Stores the URL in raw_metadata["video_download_url"].

        Args:
            item: Video item dictionary from API response.

        Returns:
            Direct video download URL string, or None if not found.
        """
        # Try video.playAddr first (higher quality), then video.downloadAddr
        video = item.get("video")
        if isinstance(video, dict):
            # Try playAddr first (higher quality)
            play_addr = video.get("playAddr") or video.get("playAddress")
            if play_addr and isinstance(play_addr, str):
                return play_addr
            # Fall back to downloadAddr
            download_addr = video.get("downloadAddr") or video.get("downloadAddress")
            if download_addr and isinstance(download_addr, str):
                return download_addr

        # Some API responses may have direct fields
        direct_url = item.get("videoUrl") or item.get("downloadUrl")
        if direct_url and isinstance(direct_url, str):
            return direct_url

        return None

    def _extract_transcription(self, item: dict[str, Any]) -> str | None:
        """Extract transcription from TikTok video item if returned by API.

        Scrape Creators API may return transcription for video content.
        This method attempts to extract it from various possible fields.

        Args:
            item: Video item dictionary from API response.

        Returns:
            Transcription text if available from API, else None.
        """
        # Check for transcription in various possible locations
        # Scrape Creators API may return transcription in different fields
        transcription = (
            item.get("transcription")
            or item.get("transcript")
            or item.get("subtitle")
            or item.get("captions")
        )
        if transcription and isinstance(transcription, str):
            return transcription.strip()
        return None

    def _build_author_profile_metadata(self, user: dict[str, Any]) -> dict[str, Any]:
        """Build author profile metadata dictionary from TikTok user object.

        Extracts profile link, parsed bio contacts, language, location,
        and geo_data from the user object using shared parse_profile_contacts
        utility for extracting emails, Telegram handles, and external links.

        Args:
            user: User object from Scrape Creators API response.

        Returns:
            Dictionary containing author profile metadata.
        """
        username: str | None = user.get("uniqueId") or user.get("handle")
        biography: str | None = user.get("signature") or user.get("bio")

        # Build profile link
        profile_link: str | None = None
        if username:
            profile_link = f"https://www.tiktok.com/@{username}"

        # Use shared utility to parse contacts from biography
        contacts_dict: dict[str, Any] = parse_profile_contacts(biography, None)
        # parse_profile_contacts returns: {emails, telegram_handles, external_links, raw_bio}

        # Build contacts list in the format expected by OpenSPG
        contacts: list[str] = []
        for email in contacts_dict.get("emails", []):
            contacts.append(f"email:{email}")
        for handle in contacts_dict.get("telegram_handles", []):
            contacts.append(f"telegram:@{handle}")
        # Add external links as contact entries
        external_links: list[str] = contacts_dict.get("external_links", [])

        # Also check for official website link in user object
        website = user.get("websiteUrl") or user.get("website")
        if website and website not in external_links:
            external_links.append(str(website))

        # Extract location from user object
        location: str | None = user.get("location")

        # Language from user object
        language: str | None = user.get("language")

        # Geo-data (region/country) from user object
        geo_data: dict[str, str] | None = None
        region = user.get("region")
        city = user.get("city")
        country = user.get("country")
        if region or city or country:
            geo_data = {}
            if region:
                geo_data["region"] = str(region)
            if city:
                geo_data["city"] = str(city)
            if country:
                geo_data["country"] = str(country)

        author_metadata: dict[str, Any] = {
            "profile_link": profile_link,
            "bio_description": biography,
            "external_links": external_links if external_links else None,
            "contacts": contacts if contacts else None,
            "advertising_contacts": contacts if contacts else None,
            "language": language,
            "location": location,
            "geo_data": geo_data,
        }

        # Remove None values for cleaner JSON
        return {k: v for k, v in author_metadata.items() if v is not None}
