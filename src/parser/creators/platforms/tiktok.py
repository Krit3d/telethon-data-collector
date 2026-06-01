"""
TikTok platform parser using Scrape Creators API v1 (profile) and v3 (videos).

Implements TikTok-specific profile parsing and content ingestion into PostgreSQL.
Extracts author profile metadata (external links, contacts, location, language)
and stores it inside Content.raw_metadata under the "author_profile_metadata" key.

Features:
    - Profile parsing via /v1/tiktok/profile endpoint with account upsert to accounts table
    - Follower threshold enforcement (3,000 to 150,000 for micro-influencers)
    - Russian language (Cyrillic) biography check (biography OR display name)
    - AI-slop / theme-page detection
    - Female creator detection with virtual profile post creation
    - Content fetching via /v3/tiktok/profile/videos endpoint with pagination
    - Bulk upsert to content table with PostgreSQL ON CONFLICT DO UPDATE
    - Raw metadata preservation for OpenSPG processing
    - Extraction of external links, contact info from biography via shared utils
    - Video download URL extraction for GPU worker processing
    - Cross-platform spidering queue for discovered accounts
    - Duration filtering (<= 120 seconds) for short-form content
    - Auto-generated caption (subtitle) extraction from TikTok CDN links (0 API credits)
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any

from aiohttp import ClientResponseError, ClientTimeout
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

# Pre-compile VTT cleaning regex patterns for performance
_VTT_HEADER_PATTERN = re.compile(r"^WEBVTT[\s\S]*?(?:\n\n|\Z)", re.MULTILINE)
_VTT_TIMESTAMP_PATTERN = re.compile(
    r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}[^\n]*\n?",
    re.MULTILINE,
)
_VTT_METADATA_PATTERN = re.compile(r"{\\.*?}", re.MULTILINE)
_VTT_EMPTY_LINES_PATTERN = re.compile(r"\n{3,}")
_VTT_CONSECUTIVE_DUPLICATES_PATTERN = re.compile(r"^(.*)(\n\1)+$", re.MULTILINE)


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
        """Fetch TikTok profile, apply filters, upsert account.

        Parses the TikTok profile for the given handle, checks follower
        thresholds (3k-150k), verifies Russian Cyrillic in biography or
        display name, checks for AI-slop/theme-page content, detects
        female creators, and upserts the account information to the
        accounts table.

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
                handle,
                followers,
                MIN_FOLLOWERS,
                MAX_FOLLOWERS,
            )
            await self._upsert_account(profile, "rejected")
            return None

        # Content quality filters
        username = profile.get("uniqueId") or profile.get("handle", "")
        biography = profile.get("signature") or profile.get("bio", "")
        full_name = profile.get("nickname") or profile.get("displayName")
        # Check if EITHER biography OR full_name contains Russian Cyrillic characters
        has_russian = is_russian_text(biography) or is_russian_text(full_name)
        if not has_russian or is_slop_or_theme_page(username, biography):
            logger.info(
                "TikTok handle %s failed content filters. Rejecting.", handle
            )
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
            handle,
            account_id,
            followers,
        )

        # Post-parse actions: upsert virtual profile post for female creators
        if is_female:
            async with self.session_maker() as session:
                await upsert_virtual_bio_post(
                    session=session,
                    account_id=account_id,
                    platform="TIKTOK",
                    platform_id=str(
                        profile.get("id") or profile.get("userId", "")
                    ),
                    username=username,
                    full_name=full_name,
                    biography=biography,
                    subscribers_count=followers,
                    raw_metadata={"female_heuristic": True},
                )
                await session.commit()

        # Queue discovered accounts from contacts (spidering)
        contacts_dict = parse_profile_contacts(biography, None)
        async with self.session_maker() as session:
            await queue_discovered_accounts(
                session=session,
                contacts_dict=contacts_dict,
                parent_handle=handle,
            )
            await session.commit()

        return account_id

    async def parse_content(
        self,
        account_id: int,
        platform_id: str,
        max_items: int = 50,
    ) -> None:
        """Fetch TikTok videos and bulk upsert to content table.

        Retrieves videos for the given account using the Scrape Creators API v3
        /v3/tiktok/profile/videos endpoint, paginates using cursor until up to max_items
        are collected, filters videos with duration <= 120 seconds, and bulk upserts
        into the content table.

        The database stores numerical platform_id, but the API requires the textual
        handle (username) which is resolved from the database. Video descriptions
        are extracted directly from the inline payload (desc/title) and mapped to
        the Content text column. Auto-generated captions (subtitles) are fetched
        from TikTok CDN links at zero API credit cost.

        Args:
            account_id: Database ID of the parent account record.
            platform_id: TikTok numerical platform ID (e.g., '7123456789').
            max_items: Maximum number of videos to fetch (default: 50).
        """
        # Resolve the textual username (handle) from the database
        handle: str | None = None
        async with self.session_maker() as session:
            stmt = select(Account.username).where(Account.id == account_id)
            result = await session.execute(stmt)
            handle = result.scalar_one_or_none()

        if not handle:
            logger.warning(
                "Username not found in database for account_id: %d, falling back to platform_id: %s",
                account_id,
                platform_id,
            )
            handle = platform_id

        logger.info(
            "Starting TikTok content parse for account_id: %d, handle: %s",
            account_id,
            handle,
        )

        # Fetch profile for author metadata (uses instance cache)
        profile = await self._get_cached_or_fetch_profile(handle)
        author_metadata: dict[str, Any] = {}
        if profile:
            # Build author profile metadata from profile data
            username: str | None = profile.get("uniqueId") or profile.get(
                "handle"
            )
            biography: str | None = profile.get("signature") or profile.get(
                "bio"
            )
            contacts_dict: dict[str, Any] = parse_profile_contacts(
                biography, None
            )

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

            author_metadata = compile_author_metadata(
                platform="TIKTOK",
                username=username,
                biography=biography,
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

        # Fetch videos with pagination from /v3/tiktok/profile/videos endpoint
        videos_collected: list[dict[str, Any]] = []
        cursor: str | None = None

        while len(videos_collected) < max_items:
            # Scrape Creators v3 videos endpoint parameters
            # NOTE: API requires textual handle, NOT numerical platform_id
            params: dict[str, Any] = {"handle": handle}
            if cursor:
                params["cursor"] = cursor

            try:
                response = await self.client.get(
                    endpoint="/v3/tiktok/profile/videos",
                    params=params,
                )
            except Exception as e:
                logger.error(
                    "Failed to fetch videos for handle %s: %s", handle, e
                )
                break

            # Extract videos from root-level "aweme_list" (v3 API returns list directly)
            items: list[dict[str, Any]] = response.get("aweme_list", [])
            if not items:
                logger.info("No more videos available for handle %s", handle)
                break

            # Filter items by duration <= 120 seconds and collect up to max_items
            for item in items:
                duration = item.get("duration") or item.get("durationSec")
                if duration is None:
                    continue

                try:
                    duration_sec = float(duration)
                except (ValueError, TypeError):
                    continue

                if duration_sec > 120.0:
                    continue

                videos_collected.append(item)
                if len(videos_collected) >= max_items:
                    break

            # Get next cursor for pagination from response root
            cursor = (
                response.get("cursor")
                or response.get("nextCursor")
                or response.get("pageCursor")
            )
            if not cursor:
                break

        if not videos_collected:
            logger.info(
                "No valid TikTok videos found for account_id: %d", account_id
            )
            return

        # Process and upsert collected video items
        await self._upsert_content(
            videos_collected, account_id, author_metadata
        )
        logger.info(
            "Successfully upserted %d TikTok content items for account_id: %d (handle: %s)",
            len(videos_collected),
            account_id,
            handle,
        )

    async def _get_cached_or_fetch_profile(
        self, handle: str
    ) -> dict[str, Any] | None:
        """Get profile from cache or fetch from API.

        Uses instance-level caching to avoid duplicate API calls when
        subsequently calling parse_content() for the same handle.
        Uses Scrape Creators API v1 /v1/tiktok/profile endpoint.

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
                endpoint="/v1/tiktok/profile",
                params={"handle": handle},
            )
            # TikTok v1 API returns profile fields directly at root level
            # Treat entire response as profile data
            self._cached_profile = response
            self._cached_handle = handle
            return response

        except ClientResponseError as e:
            if e.status == 404:
                logger.warning(
                    "TikTok profile %s not found (404). Marking as rejected.",
                    handle,
                )
            else:
                logger.error(
                    "TikTok API request failed for %s with HTTP %d: %s",
                    handle,
                    e.status,
                    e,
                )
            return None
        except Exception as e:
            logger.error(
                "Unexpected error fetching TikTok profile %s: %s",
                handle,
                e,
                exc_info=True,
            )
            return None

    async def _upsert_account(
        self,
        user: dict[str, Any],
        status: str = "parsed",
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

                # Extract content text (video description from inline payload)
                content_text: str | None = (
                    item.get("desc")
                    or item.get("title")
                    or item.get("description")
                )

                # Extract published timestamp using core helper
                create_time = item.get("createTime") or item.get("createdAt")
                published_at: datetime = parse_published_at(create_time)

                # Extract engagement metrics
                views: int | None = self._extract_metric(item, "views")
                reactions_count: int | None = self._extract_metric(
                    item, "likes"
                )
                comments_count: int | None = self._extract_metric(
                    item, "comments"
                )
                shares_count: int | None = self._extract_metric(item, "shares")

                # Extract video download URL for GPU worker
                video_url: str | None = self._extract_video_download_url(item)

                # Extract and download auto-generated captions (subtitles) from CDN
                transcription: str | None = await self._extract_transcription(item)

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
                        "is_embedded": False,
                        "is_graph_extracted": False,
                        "raw_metadata": raw_metadata,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )

            except Exception as e:
                logger.error(
                    "Failed to parse TikTok content item: %s", e, exc_info=True
                )
                continue

        if not content_values:
            logger.warning(
                "No valid content items to upsert for account_id: %d",
                account_id,
            )
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

    async def _extract_transcription(
        self, item: dict[str, Any]
    ) -> str | None:
        """Extract auto-generated caption (subtitle) from TikTok CDN.

        Searches the video object's subtitle_infos/subtitleInfos for a suitable
        language track (preferring ru-RU/ru, falling back to en-US/en), downloads
        the WebVTT content from the CDN URL, and cleans it into plain text.

        Args:
            item: Video item dictionary from API response.

        Returns:
            Cleaned transcription text, or None if not available or on error.
        """
        video = item.get("video")
        if not isinstance(video, dict):
            return None

        # Check both snake_case and PascalCase variants
        subtitle_infos: list[dict[str, Any]] = (
            video.get("subtitle_infos")
            or video.get("subtitleInfos")
            or []
        )

        if not subtitle_infos:
            return None

        # Find suitable subtitle track: prefer ru-RU/ru, fallback to en-US/en
        selected_track: dict[str, Any] | None = None

        # First pass: look for Russian language tracks
        for track in subtitle_infos:
            lang = track.get("language_code_name") or track.get("LanguageCodeName", "")
            if lang.startswith("ru"):
                selected_track = track
                break

        # Second pass: fallback to English if no Russian found
        if not selected_track:
            for track in subtitle_infos:
                lang = track.get("language_code_name") or track.get("LanguageCodeName", "")
                if lang.startswith("en"):
                    selected_track = track
                    break

        if not selected_track:
            return None

        # Extract CDN URL (check both snake_case and PascalCase)
        subtitle_url: str | None = (
            selected_track.get("url")
            or selected_track.get("Url")
        )

        if not subtitle_url or not isinstance(subtitle_url, str):
            return None

        # Download and clean the VTT content
        return await self._download_and_clean_vtt(subtitle_url)

    async def _download_and_clean_vtt(self, url: str) -> str | None:
        """Download WebVTT content from CDN URL and clean it into plain text.

        Fetches the raw WebVTT file from the provided CDN URL using the internal
        aiohttp.ClientSession, then cleans it by removing VTT headers, timestamps,
        metadata blocks, and empty lines. Consecutive duplicate lines are merged.

        Args:
            url: CDN URL pointing to the WebVTT subtitle file.

        Returns:
            Cleaned plain text transcription, or None if download/parsing fails.
        """
        try:
            # Use internal aiohttp.ClientSession from ScrapeCreatorsClient
            session = self.client._session
            if session is None or session.closed:
                logger.warning("aiohttp session unavailable for VTT download")
                return None

            timeout = ClientTimeout(total=10)
            async with session.get(url, timeout=timeout) as response:
                response.raise_for_status()
                raw_vtt: str = await response.text()

        except Exception as e:
            logger.warning("Failed to download VTT from %s: %s", url, e)
            return None

        if not raw_vtt:
            return None

        try:
            # Clean the VTT content using pre-compiled regex patterns
            cleaned: str = _VTT_HEADER_PATTERN.sub("", raw_vtt)
            cleaned = _VTT_TIMESTAMP_PATTERN.sub("", cleaned)
            cleaned = _VTT_METADATA_PATTERN.sub("", cleaned)

            # Split into lines, strip whitespace, filter empty lines
            lines: list[str] = [
                line.strip() for line in cleaned.splitlines()
            ]
            filtered_lines: list[str] = [
                line for line in lines if line and not line.startswith("NOTE")
            ]

            if not filtered_lines:
                return None

            # Remove consecutive duplicate lines
            deduplicated: list[str] = [filtered_lines[0]]
            for line in filtered_lines[1:]:
                if line != deduplicated[-1]:
                    deduplicated.append(line)

            # Join into cohesive text
            return " ".join(deduplicated)

        except Exception as e:
            logger.warning("Failed to clean VTT content from %s: %s", url, e)
            return None

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
            download_addr = video.get("downloadAddr") or video.get(
                "downloadAddress"
            )
            if download_addr and isinstance(download_addr, str):
                return download_addr

        # Some API responses may have direct fields
        direct_url = item.get("videoUrl") or item.get("downloadUrl")
        if direct_url and isinstance(direct_url, str):
            return direct_url

        return None
