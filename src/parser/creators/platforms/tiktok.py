"""
TikTok platform parser using Scrape Creators API.

Implements TikTok-specific profile parsing and content ingestion into PostgreSQL.
Extracts author profile metadata (external links, contacts, location, language)
and stores it inside Content.raw_metadata under the "author_profile_metadata" key.

Features:
    - Profile parsing with account upsert to accounts table
    - Follower threshold enforcement (3,000 to 150,000 for micro-influencers)
    - Russian language (Cyrillic) biography check
    - AI-slop / theme-page detection
    - Female creator detection with virtual profile post creation
    - Content fetching and bulk upsert to content table
    - PostgreSQL ON CONFLICT DO UPDATE for high-throughput concurrency
    - Raw metadata preservation for OpenSPG processing
    - Extraction of external links, contact info from biography via shared utils
    - Video download URL extraction for GPU worker processing
    - Transcription support via Scrape Creators API transcript endpoint
    - Cross-platform spidering queue for discovered accounts
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config.config import Settings
from src.db.models import Account, Content
from src.parser.creators.core.utils import (
    is_russian_text,
    is_slop_or_theme_page,
    detect_female_creator,
    upsert_virtual_bio_post,
    queue_discovered_accounts,
    parse_profile_contacts,
    parse_published_at,
    compile_author_metadata,
)
from src.parser.creators.platforms.base import BasePlatformParser
from src.parser.creators.sc_client import ScrapeCreatorsClient

logger = logging.getLogger(__name__)

# Follower thresholds for micro-influencers
MIN_FOLLOWERS: int = 3000
MAX_FOLLOWERS: int = 150000

# Semaphore limit for concurrent transcript API calls
TRANSCRIPT_SEMAPHORE_LIMIT: int = 5


class TikTokPlatformParser(BasePlatformParser):
    """TikTok platform parser for profile and content ingestion.

    Inherits from BasePlatformParser and implements TikTok-specific
    profile parsing and content upsert logic using the Scrape Creators API.

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
        _transcript_semaphore: Semaphore to limit concurrent transcript API calls.
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
        self._transcript_semaphore = asyncio.Semaphore(TRANSCRIPT_SEMAPHORE_LIMIT)

    async def parse_profile(self, handle: str) -> int | None:
        """Fetch TikTok profile, apply filters, upsert account.

        Parses the TikTok profile for the given handle, checks follower
        thresholds (3k-150k), verifies Russian Cyrillic in biography,
        checks for AI-slop/theme-page content, detects female creators,
        and upserts the account information to the accounts table.

        Args:
            handle: TikTok username (without @ prefix).

        Returns:
            Database ID of the upserted account record, or None if the profile
            could not be parsed, doesn't meet criteria, or is rejected.
        """
        logger.info("Starting TikTok profile parse for handle: %s", handle)

        profile = await self._get_cached_or_fetch_profile(handle)
        if not profile:
            return None

        # Follower threshold filter (micro-influencers: 3k-150k)
        followers = self._extract_follower_count(profile)
        if not (MIN_FOLLOWERS <= followers <= MAX_FOLLOWERS):
            logger.info(
                "TikTok handle %s has %d followers, outside range [%d, %d]. Rejecting.",
                handle, followers, MIN_FOLLOWERS, MAX_FOLLOWERS,
            )
            await self._upsert_account(profile, "rejected")
            return None

        # Content quality filters
        username = profile.get("uniqueId") or profile.get("handle", "")
        biography = profile.get("signature") or profile.get("bio", "")
        if not is_russian_text(biography) or is_slop_or_theme_page(username, biography):
            logger.info("TikTok handle %s failed content filters. Rejecting.", handle)
            await self._upsert_account(profile, "rejected")
            return None

        # Female creator detection
        is_female = detect_female_creator(biography)
        if is_female:
            logger.info("Female creator detected: %s", handle)

        # Upsert parsed account
        account_id = await self._upsert_account(profile, "parsed")
        logger.info(
            "Successfully parsed TikTok profile %s, account ID: %d, followers: %d",
            handle, account_id, followers,
        )

        # Post-parse actions: upsert virtual profile post for female creators
        if is_female:
            async with self.session_maker() as session:
                await upsert_virtual_bio_post(
                    session=session,
                    account_id=account_id,
                    platform="TIKTOK",
                    platform_id=str(profile.get("id") or profile.get("userId", "")),
                    username=username,
                    full_name=profile.get("nickname") or profile.get("displayName"),
                    biography=biography,
                    subscribers_count=followers,
                    raw_metadata={"female_heuristic": True},
                )

        # Queue discovered accounts from contacts (spidering)
        contacts_dict = parse_profile_contacts(biography, None)
        async with self.session_maker() as session:
            await queue_discovered_accounts(
                session=session,
                contacts_dict=contacts_dict,
                parent_handle=handle,
            )

        return account_id

    async def parse_content(
        self,
        account_id: int,
        platform_id: str,
        max_items: int = 50,
    ) -> None:
        """Fetch TikTok content and bulk upsert to content table.

        Retrieves content items (videos) for the given account using the
        Scrape Creators API, parses the data, and performs a bulk upsert
        into the content table using PostgreSQL ON CONFLICT DO UPDATE.

        The database stores numerical platform_id (e.g., '7123456789') for
        TikTok accounts, but the Scrape Creators API requires the textual
        handle (username, e.g., 'khaby.lame'). This method resolves the
        handle by querying the database for the account's username.

        If the API response does not contain transcripts, this method will
        attempt to fetch them via the /v2/tiktok/media/transcript endpoint
        with a semaphore limit of 5 concurrent requests.

        The raw_metadata field contains:
            - "author_profile_metadata": Profile-level data (contacts, links, etc.)
            - "platform_metrics": Platform-specific engagement metrics including video_url

        Args:
            account_id: Database ID of the parent account record.
            platform_id: TikTok numerical platform ID (e.g., '7123456789').
            max_items: Maximum number of content items to fetch (default: 50).
        """
        # Resolve the textual username (handle) from the database
        # The database stores numerical platform_id, but API needs the username
        handle: str | None = None
        async with self.session_maker() as session:
            stmt = select(Account.username).where(Account.id == account_id)
            result = await session.execute(stmt)
            handle = result.scalar_one_or_none()

        # Fallback to platform_id if username not found in database
        if not handle:
            logger.warning(
                "Username not found in database for account_id: %d, "
                "falling back to platform_id: %s",
                account_id,
                platform_id,
            )
            handle = platform_id

        logger.info(
            "Starting TikTok content parse for account_id: %d, handle: %s",
            account_id,
            handle,
        )

        profile = await self._get_cached_or_fetch_profile(handle)
        if not profile:
            return

        # Validate response structure
        data = profile.get("data") if isinstance(profile, dict) else None
        if not data:
            logger.error(
                "Missing 'data' in API response for TikTok content, handle %s",
                handle,
            )
            return

        # Extract itemList from profile response
        item_list: list[dict[str, Any]] = data.get("itemList", [])
        if not isinstance(item_list, list):
            logger.error(
                "Invalid itemList in API response for TikTok content, handle %s",
                handle,
            )
            return

        if not item_list:
            logger.info("No TikTok content found for account_id: %d", account_id)
            return

        # Limit to max_items
        item_list = item_list[:max_items]

        # Build author profile metadata using core helper
        user = data.get("user") or data
        username: str | None = user.get("uniqueId") or user.get("handle")
        biography: str | None = user.get("signature") or user.get("bio")

        # Parse contacts from biography
        contacts_dict: dict[str, Any] = parse_profile_contacts(biography, None)

        # Extract geo data (region/city/country) from user object
        geo_data: dict[str, Any] | None = None
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

        # Also check for official website link in user object
        extra_links: list[str] = []
        website = user.get("websiteUrl") or user.get("website")
        if website:
            extra_links.append(str(website))

        # Use core helper to compile author metadata
        author_metadata = compile_author_metadata(
            platform="TIKTOK",
            username=username,
            biography=biography,
            contacts_dict=contacts_dict,
            extra_links=extra_links if extra_links else None,
            language=user.get("language"),
            location=user.get("location"),
            geo_data=geo_data,
        )

        # Fetch transcripts concurrently with semaphore limit
        await self._fetch_transcripts_for_videos(item_list)

        # Process and upsert content items
        await self._upsert_content(item_list, account_id, author_metadata)
        logger.info(
            "Successfully upserted %d TikTok content items for account_id: %d (handle: %s)",
            len(item_list),
            account_id,
            handle,
        )

    async def _get_cached_or_fetch_profile(self, handle: str) -> dict[str, Any] | None:
        """Get profile from cache or fetch from API.

        Uses instance-level caching to avoid duplicate API calls when
        subsequently calling parse_content() for the same handle.

        Args:
            handle: TikTok username to fetch.

        Returns:
            Profile data dictionary, or None if fetch failed.
        """
        if self._cached_handle == handle and self._cached_profile:
            logger.debug("Using cached profile data for handle: %s", handle)
            return self._cached_profile

        try:
            response = await self.client.get(
                endpoint="/v2/tiktok/profile",
                params={"handle": handle, "count": 100},
            )
            data = response.get("data")
            if not data:
                logger.error("Missing 'data' in API response for TikTok handle %s", handle)
                return None

            user = data.get("user") or data
            if not user:
                logger.error("Missing user data for TikTok handle %s", handle)
                return None

            # Cache the response
            self._cached_profile = response
            self._cached_handle = handle
            return response

        except Exception as e:
            logger.error("API request failed for TikTok profile %s: %s", handle, e, exc_info=True)
            return None

    async def _upsert_account(
        self, user: dict[str, Any], status: str = "parsed",
    ) -> int:
        """Upsert TikTok account record using select-then-insert/update pattern.

        Args:
            user: User object from Scrape Creators API response.
            status: Account status ('parsed', 'rejected', etc.).

        Returns:
            ID of the account record.
        """
        platform_id: str = str(user.get("id") or user.get("userId", ""))
        username: str | None = user.get("uniqueId") or user.get("handle")
        full_name: str | None = user.get("nickname") or user.get("displayName")
        biography: str | None = user.get("signature") or user.get("bio")
        followers: int = self._extract_follower_count(user)

        async with self.session_maker() as session:
            stmt = select(Account).where(
                Account.platform == "TIKTOK",
                Account.platform_id == platform_id,
            )
            result = await session.execute(stmt)
            db_account = result.scalar_one_or_none()

            if db_account:
                db_account.username = username
                db_account.title = full_name or username or "Unknown"
                db_account.description = biography
                db_account.subscribers_count = followers
                db_account.status = status
                db_account.updated_at = datetime.now(timezone.utc)
            else:
                db_account = Account(
                    platform="TIKTOK",
                    platform_id=platform_id,
                    username=username,
                    title=full_name or username or "Unknown",
                    description=biography,
                    subscribers_count=followers,
                    status=status,
                )
                session.add(db_account)

            await session.commit()
            await session.refresh(db_account)
            return db_account.id

    async def _fetch_transcripts_for_videos(
        self, items: list[dict[str, Any]],
    ) -> None:
        """Fetch transcripts for videos that don't already have them.

        Uses the Scrape Creators API endpoint /v2/tiktok/media/transcript
        to fetch transcripts for videos that don't already have transcription
        in the response. Limits concurrent requests to 5 using a semaphore.

        After fetching, stores the transcript in the item's "transcription" field.
        Checks for the plural "transcripts" key in the JSON response.

        Args:
            items: List of video item dictionaries to fetch transcripts for.
        """
        tasks = []
        for item in items:
            # Skip if already has transcription
            if item.get("transcription") or item.get("transcript"):
                continue

            video_id = item.get("id")
            if not video_id:
                continue

            tasks.append(self._fetch_single_transcript(item, str(video_id)))

        if tasks:
            logger.info("Fetching transcripts for %d TikTok videos", len(tasks))
            await asyncio.gather(*tasks)

    async def _fetch_single_transcript(
        self, item: dict[str, Any], video_id: str,
    ) -> None:
        """Fetch transcript for a single video with semaphore limiting.

        Args:
            item: Video item dictionary to store the transcript in.
            video_id: TikTok video ID for the API call.
        """
        async with self._transcript_semaphore:
            try:
                response = await self.client.get(
                    endpoint="/v2/tiktok/media/transcript",
                    params={"video_id": video_id},
                )
                data = response.get("data")
                if not data:
                    return

                # Check for "transcripts" (plural) key in response
                transcripts = data.get("transcripts") or data.get("transcript")
                if isinstance(transcripts, list) and transcripts:
                    # Combine all transcript entries
                    text_parts = []
                    for entry in transcripts:
                        if isinstance(entry, dict):
                            text_parts.append(entry.get("text", ""))
                        elif isinstance(entry, str):
                            text_parts.append(entry)
                    item["transcription"] = " ".join(text_parts).strip()
                elif isinstance(transcripts, str):
                    item["transcription"] = transcripts.strip()

            except Exception as e:
                logger.warning(
                    "Failed to fetch transcript for TikTok video %s: %s",
                    video_id, e,
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

                # Extract published timestamp using core helper
                create_time = item.get("createTime") or item.get("createdAt")
                published_at: datetime = parse_published_at(create_time)

                # Extract engagement metrics
                views: int | None = self._extract_metric(item, "views")
                reactions_count: int | None = self._extract_metric(item, "likes")
                comments_count: int | None = self._extract_metric(item, "comments")
                shares_count: int | None = self._extract_metric(item, "shares")

                # Extract video download URL for GPU worker
                video_url: str | None = self._extract_video_download_url(item)

                # Extract transcription
                transcription: str | None = self._extract_transcription(item)

                # Build platform metrics with video_url for GPU worker
                platform_metrics: dict[str, Any] = {
                    "views": views,
                    "likes": reactions_count,
                    "comments": comments_count,
                    "shares": shares_count,
                    "video_url": video_url,
                }

                # Build raw_metadata with author_profile_metadata and platform_metrics
                raw_metadata: dict[str, Any] = {
                    "author_profile_metadata": author_metadata,
                    "platform_metrics": platform_metrics,
                }

                content_values.append({
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
                    "is_embedded": False,
                    "is_graph_extracted": False,
                    "raw_metadata": raw_metadata,
                    "updated_at": datetime.now(timezone.utc),
                })

            except Exception as e:
                logger.error("Failed to parse TikTok content item: %s", e, exc_info=True)
                continue

        if not content_values:
            logger.warning("No valid content items to upsert for account_id: %d", account_id)
            return

        async with self.session_maker() as session:
            async with session.begin():
                stmt = pg_insert(Content).values(content_values)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_content_account_platform_id",
                    set_=dict(
                        content=stmt.excluded.content,
                        transcription=stmt.excluded.transcription,
                        views=stmt.excluded.views,
                        reactions_count=stmt.excluded.reactions_count,
                        comments_count=stmt.excluded.comments_count,
                        shares_count=stmt.excluded.shares_count,
                        has_media=stmt.excluded.has_media,
                        raw_metadata=stmt.excluded.raw_metadata,
                        updated_at=stmt.excluded.updated_at,
                    ),
                )
                await session.execute(stmt)
                logger.debug(
                    "Upserted %d TikTok content records for account ID %d",
                    len(content_values),
                    account_id,
                )

    def _extract_follower_count(self, user: dict[str, Any]) -> int:
        """Extract follower count from TikTok user object.

        Args:
            user: User object from Scrape Creators API response.

        Returns:
            Follower count as integer, or 0 if not found.
        """
        follower_count = (
            user.get("followerCount")
            or user.get("followers")
            or user.get("follower_count")
            or user.get("stats", {}).get("followerCount")
            or 0
        )
        try:
            return int(follower_count)
        except (ValueError, TypeError):
            return 0

    def _extract_metric(self, item: dict[str, Any], metric: str) -> int | None:
        """Extract engagement metric from video item.

        Args:
            item: Video item dictionary from API response.
            metric: Metric name ('views', 'likes', 'comments', 'shares').

        Returns:
            Metric value as integer, or None if not found.
        """
        stats = item.get("stats") or item.get("statistics") or {}

        metric_map = {
            "views": ["playCount", "viewCount", "views"],
            "likes": ["diggCount", "likeCount", "likes"],
            "comments": ["commentCount", "commentsCount", "comments"],
            "shares": ["shareCount", "sharesCount", "shares"],
        }

        for key in metric_map.get(metric, []):
            value = item.get(key) or stats.get(key)
            if value is not None:
                try:
                    return int(value)
                except (ValueError, TypeError):
                    pass
        return None

    def _extract_video_download_url(self, item: dict[str, Any]) -> str | None:
        """Extract direct video download URL from TikTok video item.

        Tries video.playAddr first, then video.downloadAddr as per specification.
        The URL is stored in raw_metadata["platform_metrics"]["video_url"]
        for the GPU embedding worker.

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
        """Extract transcription from TikTok video item.

        Checks for transcription in various possible fields in the item.
        If the Scrape Creators API returns transcription in the video payload,
        it will be extracted here. Otherwise, _fetch_transcripts_for_videos
        will attempt to fetch it from the transcript endpoint.

        Args:
            item: Video item dictionary from API response.

        Returns:
            Transcription text if available, else None.
        """
        transcription = (
            item.get("transcription")
            or item.get("transcript")
            or item.get("subtitle")
            or item.get("captions")
        )
        if transcription and isinstance(transcription, str):
            return transcription.strip()
        return None
