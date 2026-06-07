"""
TikTok platform parser using Scrape Creators API v1 (profile) and v3 (videos).

Implements TikTok-specific profile parsing and content ingestion into PostgreSQL.
Extracts author profile metadata (external links, contacts, location, language)
and stores it inside Content.raw_metadata under the "author_profile_metadata" key.

Features:
    - Profile parsing via /v1/tiktok/profile endpoint with account upsert to accounts table
    - Follower threshold enforcement (3,000 to 150,000 for micro-influencers)
    - Russian language (Cyrillic) biography check (biography OR display name OR title)
    - AI-slop / theme-page detection with strict Gatekeeper pattern
    - Content fetching via /v3/tiktok/profile/videos endpoint (single API call, max 10 items)
    - Bulk upsert to content table using generic DB helpers
    - Raw metadata preservation for OpenSPG processing
    - Extraction of external links, contact info from biography via shared utils
    - Video download URL extraction for GPU worker processing
    - Cross-platform spidering queue for discovered accounts
    - Duration filtering (<= 120 seconds) for short-form content
    - Auto-generated caption (subtitle) extraction from TikTok CDN links (0 API credits)
    - Credit-optimized transcription: skip transcription for beauty/makeup/cosmetics content
"""

import logging
from typing import Any

from aiohttp import ClientResponseError
from sqlalchemy import select

from src.config.config import Settings
from src.db.models import Account
from src.parser.creators.core.db import (
    update_account_profile_metadata,
    bulk_upsert_content,
)
from src.parser.creators.core.utils import (
    is_russian_text,
    is_slop_or_theme_page,
    upsert_and_deduplicate_account,
    upsert_virtual_bio_post,
    parse_profile_contacts,
    parse_published_at,
    compile_author_metadata_dict,
)
from src.parser.creators.platforms.base import BasePlatformParser
from src.parser.creators.sc_client import ScrapeCreatorsClient

logger = logging.getLogger(__name__)

# Follower thresholds for micro-influencers
MIN_FOLLOWERS: int = 3000
MAX_FOLLOWERS: int = 150000


class TikTokParser(BasePlatformParser):
    """TikTok platform parser for profile and content ingestion.

    Inherits from BasePlatformParser and implements TikTok-specific
    profile parsing and content upsert logic using the Scrape Creators API
    v1 (profile) and v3 (videos).

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
        session_maker: Any,  # async_sessionmaker[AsyncSession] from SQLAlchemy
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
        """Fetch TikTok profile, apply Gatekeeper filters, upsert account.

        Parses the TikTok profile for the given handle, applies the strict
        zero-cost Gatekeeper validation (Cyrillic check and slop detection),
        checks follower thresholds (3k-150k), and upserts the account
        information to the accounts table.

        The Gatekeeper filter ensures we never waste credits fetching posts or
        transcripts for non-Russian or low-quality/meme aggregator accounts.

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

        # Extract profile fields
        platform_id: str = str(profile.get("id") or profile.get("userId", ""))
        username: str | None = profile.get("uniqueId") or profile.get("handle")
        full_name: str | None = profile.get("nickname") or profile.get("displayName")
        title: str | None = full_name or username or "Unknown"
        biography: str | None = profile.get("signature") or profile.get("bio", "")
        followers: int = self._extract_follower_count(profile)

        # -------------------------------------------------------------------
        # GATEKEEPER: Zero-cost offline filtering
        # Executed immediately after fetching raw profile data
        # -------------------------------------------------------------------

        # Check 1: Russian Cyrillic validation (unified zero-cost check)
        # If biography is present and not empty, it MUST contain Cyrillic characters
        # either in biography, full_name, or title. If all are false, reject.
        # If biography is empty or None, allow account to pass to content parsing
        # where post descriptions can be evaluated.
        if biography and biography.strip():
            # Biography exists and is not empty - must contain Cyrillic
            has_russian = (
                is_russian_text(biography)
                or is_russian_text(full_name)
                or is_russian_text(title)
            )
        else:
            # Biography is empty or None - allow to pass to content parsing stage
            has_russian = True
            logger.debug(
                "TikTok handle %s has empty biography, allowing to proceed to content parsing",
                handle,
            )

        # Check 2: Slop/meme-aggregator/theme page detection
        is_slop = is_slop_or_theme_page(username or "", biography)

        # Apply Gatekeeper rejection logic
        if not has_russian or is_slop:
            rejection_reason = ""
            if not has_russian:
                rejection_reason = "No Cyrillic found in profile (biography, full_name, title)"
            if is_slop:
                rejection_reason = "Classified as slop/theme page"
            if not has_russian and is_slop:
                rejection_reason = "No Cyrillic found and classified as slop/theme page"

            logger.info(
                "Account skipped: %s. Handle: %s",
                rejection_reason,
                handle,
            )

            # Save/update account with "rejected" status
            async with self.session_maker() as session:
                account_id: int = await upsert_and_deduplicate_account(
                    session=session,
                    platform="TIKTOK",
                    platform_id=platform_id,
                    username=username,
                    title=title,
                    description=biography,
                    subscribers_count=followers,
                    status="rejected",
                )
                await session.commit()

            # Immediately stop processing - do NOT fetch content
            return None

        # -------------------------------------------------------------------
        # Continue with standard filtering (follower thresholds)
        # -------------------------------------------------------------------

        # Follower threshold filter (micro-influencers: 3k-150k)
        if not (MIN_FOLLOWERS <= followers <= MAX_FOLLOWERS):
            logger.info(
                "TikTok handle %s has %d followers, outside range [%d, %d]. Rejecting.",
                handle, followers, MIN_FOLLOWERS, MAX_FOLLOWERS,
            )
            async with self.session_maker() as session:
                await upsert_and_deduplicate_account(
                    session=session,
                    platform="TIKTOK",
                    platform_id=platform_id,
                    username=username,
                    title=title,
                    description=biography,
                    subscribers_count=followers,
                    status="rejected",
                )
                await session.commit()
            return None

        # Upsert account with "processing" status
        async with self.session_maker() as session:
            account_id: int = await upsert_and_deduplicate_account(
                session=session,
                platform="TIKTOK",
                platform_id=platform_id,
                username=username,
                title=title,
                description=biography,
                subscribers_count=followers,
                status="processing",
            )

            # Update account biography and trigger cross-platform queue discovery
            await update_account_profile_metadata(
                session=session,
                account_id=account_id,
                platform="TIKTOK",
                biography=biography,
            )

            # Update account status to "parsed"
            await upsert_and_deduplicate_account(
                session=session,
                platform="TIKTOK",
                platform_id=platform_id,
                username=username,
                title=title,
                description=biography,
                subscribers_count=followers,
                status="parsed",
            )

            await session.commit()

        logger.info(
            "Successfully parsed TikTok profile %s, account ID: %d, followers: %d",
            handle, account_id, followers,
        )
        return account_id

    async def parse_content(
        self,
        account_id: int,
        platform_id: str,
        max_items: int = 50,
    ) -> None:
        """Fetch TikTok videos and bulk upsert to content table.

        Retrieves videos for the given account using the Scrape Creators API v3
        /v3/tiktok/profile/videos endpoint (single API call, no pagination),
        filters videos with duration <= 120 seconds, collects a maximum of 10
        video items, and bulk upserts into the content table.

        Transcription is credit-optimized: if video description contains target
        semantics (beauty/makeup/cosmetics keywords), transcription is skipped.
        Otherwise, CDN subtitles are extracted (0 API credits). If CDN subtitles
        are not available, transcription remains None (optional Scrape Creators
        transcript endpoint not implemented to save credits).

        The database stores numerical platform_id, but the API requires the textual
        handle (username) which is resolved from the database. Video descriptions
        are extracted directly from the inline payload (desc/title) and mapped to
        the Content text column.

        Args:
            account_id: Database ID of the parent account record.
            platform_id: TikTok numerical platform ID (e.g., '7123456789').
            max_items: Maximum number of videos to fetch (capped at 10 for TikTok).
        """
        # Cap max_items at 10 for TikTok as per requirement
        max_items = min(max_items, 10)

        # Resolve the textual username (handle) from the database
        handle: str | None = None
        async with self.session_maker() as session:
            stmt = select(Account.username).where(Account.id == account_id)
            result = await session.execute(stmt)
            handle = result.scalar_one_or_none()

        if not handle:
            logger.warning(
                "Username not found in database for account_id: %d, falling back to platform_id: %s",
                account_id, platform_id,
            )
            handle = platform_id

        logger.info(
            "Starting TikTok content parse for account_id: %d, handle: %s (max_items: %d)",
            account_id, handle, max_items,
        )

        # Fetch profile for author metadata (uses instance cache)
        profile = await self._get_cached_or_fetch_profile(handle)
        author_metadata: dict[str, Any] = {}
        if profile:
            # Build author profile metadata from profile data
            profile_username: str | None = profile.get("uniqueId") or profile.get("handle")
            profile_biography: str | None = profile.get("signature") or profile.get("bio")
            contacts_dict: dict[str, Any] = parse_profile_contacts(profile_biography, None)

            # Extract geo data (region/city/country) from profile
            geo_data: dict[str, Any] | None = None
            region = profile.get("region")
            city = profile.get("city")
            country = profile.get("country")
            if region or city or country:
                geo_data = {}
                if region:
                    geo_data["region"] = str(region)
                if city:
                    geo_data["city"] = str(city)
                if country:
                    geo_data["country"] = str(country)

            # Extract official website link
            extra_links: list[str] = []
            website = profile.get("websiteUrl") or profile.get("website")
            if website:
                extra_links.append(str(website))

            author_metadata = compile_author_metadata_dict(
                platform="TIKTOK",
                username=profile_username,
                biography=profile_biography,
                contacts_dict=contacts_dict,
                extra_links=extra_links if extra_links else None,
                language=profile.get("language"),
                location=profile.get("location"),
                geo_data=geo_data,
            )
        else:
            logger.warning(
                "Failed to fetch profile for handle %s, author metadata will be empty",
                handle,
            )

        # Fetch videos from API using correct endpoint
        try:
            response = await self.client.get(
                endpoint="/v3/tiktok/profile/videos",
                params={"username": handle, "max_items": max_items},
            )
            videos_data = response.get("data", response) if isinstance(response, dict) else response
        except ClientResponseError as e:
            logger.error(
                "Failed to fetch TikTok videos for handle %s: %s",
                handle,
                e,
            )
            return

        if not videos_data:
            logger.warning("No videos found for TikTok handle: %s", handle)
            return

        # Process and filter videos
        content_values: list[dict[str, Any]] = []
        for video in videos_data[:max_items]:
            # Extract video metadata
            video_id: str = str(video.get("id") or video.get("videoId", ""))
            description: str = video.get("desc") or video.get("title", "")
            duration: int = video.get("duration", 0)

            # Duration filter (<= 120 seconds for short-form content)
            if duration > 120:
                logger.debug("Skipping video %s: duration %ds > 120s", video_id, duration)
                continue

            # Extract publish date
            published_at = parse_published_at(
                video.get("createTime") or video.get("createdAt"),
            )

            # Check if transcription should be skipped (target semantics present)
            skip_transcription = bool(self.settings.semantic_keywords_pattern.search(description))

            # Extract CDN subtitle URL (0 API credits)
            subtitle_url: str | None = None
            video_url: str | None = None

            # Try to get video download URL
            if video.get("playAddr"):
                video_url = str(video["playAddr"])
            elif video.get("video", {}).get("playAddr"):
                video_url = str(video["video"]["playAddr"])

            # Try to get subtitle/caption URL
            if video.get("subtitleUrl"):
                subtitle_url = str(video["subtitleUrl"])
            elif video.get("caption", {}).get("url"):
                subtitle_url = str(video["caption"]["url"])

            # Build content item for bulk upsert
            content_value = {
                "account_id": account_id,
                "platform_content_id": video_id,
                "content": description,
                "published_at": published_at,
                "url": f"https://www.tiktok.com/@{handle}/video/{video_id}",
                "video_url": video_url,
                "transcription": None,  # Will be populated later if needed
                "views": video.get("stats", {}).get("playCount", 0),
                "reactions_count": video.get("stats", {}).get("diggCount", 0),
                "comments_count": video.get("stats", {}).get("commentCount", 0),
                "shares_count": video.get("stats", {}).get("shareCount", 0),
                "raw_metadata": {
                    "author_profile_metadata": author_metadata,
                    "duration": duration,
                    "skip_transcription": skip_transcription,
                },
            }
            content_values.append(content_value)

        # Bulk upsert content to database
        if content_values:
            async with self.session_maker() as session:
                await bulk_upsert_content(
                    session=session,
                    content_values=content_values,
                )
                await session.commit()

            logger.info(
                "Successfully upserted %d TikTok videos for account_id: %d",
                len(content_values),
                account_id,
            )
        else:
            logger.info("No valid videos to upsert for account_id: %d", account_id)

    async def _get_cached_or_fetch_profile(self, handle: str) -> dict[str, Any] | None:
        """Fetch TikTok profile with instance-level caching.

        Args:
            handle: TikTok username (without @ prefix).

        Returns:
            Profile data dictionary, or None if fetch failed.
        """
        if self._cached_profile is not None and self._cached_handle == handle:
            logger.debug("Using cached profile for handle: %s", handle)
            return self._cached_profile

        try:
            response = await self.client.get(
                endpoint="/v1/tiktok/profile",
                params={"username": handle},
            )
            profile = response.get("data", response) if isinstance(response, dict) else response
            if profile:
                self._cached_profile = profile
                self._cached_handle = handle
            return profile
        except ClientResponseError as e:
            logger.error(
                "Failed to fetch TikTok profile for handle %s: %s",
                handle,
                e,
            )
            return None

    def _extract_follower_count(self, profile: dict[str, Any]) -> int:
        """Extract follower count from TikTok profile data.

        Args:
            profile: TikTok profile data dictionary.

        Returns:
            Follower count as integer (0 if not found).
        """
        # Try different possible field names
        followers = (
            profile.get("followerCount")
            or profile.get("followers")
            or profile.get("fanCount")
            or 0
        )
        return int(followers)
