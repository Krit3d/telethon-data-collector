"""
TikTok platform parser using Scrape Creators API.

Implements TikTok-specific profile parsing and content ingestion into PostgreSQL.
Extracts author profile metadata (external links, contacts, location, language)
and stores it inside Content.raw_metadata under the "author_profile_metadata" key.

Features:
    - One-request cache optimization for profile data
    - Profile parsing with account upsert to accounts table
    - Minimum follower threshold enforcement (3000 followers)
    - Virtual profile post creation for semantic search over biographies
    - Content fetching from itemList with deduplication
    - PostgreSQL ON CONFLICT DO UPDATE for high-throughput concurrency
    - Raw metadata preservation for OpenSPG processing
    - Extraction of external links, contact info from biography via shared utils
    - Video download URL extraction for GPU worker processing
    - Transcription support for video content
"""

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config.config import Settings
from src.db.models import Account, Content
from src.parser.creators.core.utils import parse_profile_contacts
from src.parser.creators.platforms.base import BasePlatformParser
from src.parser.creators.sc_client import ScrapeCreatorsClient

logger = logging.getLogger(__name__)

# Minimum follower threshold for TikTok accounts
MIN_FOLLOWERS: int = 3000


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
        meets the minimum follower threshold (3000), extracts author profile
        metadata (external links, contacts, location, language), upserts the
        account information to the accounts table, creates a virtual profile post
        for semantic search, and returns the database ID.

        Uses instance-level caching to avoid duplicate API calls when subsequently
        calling parse_content() for the same handle.

        Args:
            handle: TikTok username (without @ prefix).

        Returns:
            Database ID of the upserted account record, or None if the profile
            could not be parsed or doesn't meet the minimum follower threshold.
        """
        logger.info("Starting TikTok profile parse for handle: %s", handle)

        try:
            # Check cache first
            if self._cached_handle == handle and self._cached_profile:
                logger.debug("Using cached profile data for handle: %s", handle)
                response = self._cached_profile
            else:
                # Fetch profile data from Scrape Creators API
                try:
                    response = await self.client.get(
                        endpoint="/v1/tiktok/profile",
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

            # Extract user ID and follower count
            user_id: str = str(user.get("id") or user.get("userId", ""))
            if not user_id:
                logger.error("Could not extract user ID for TikTok handle %s", handle)
                return None

            follower_count: int = self._extract_follower_count(user)

            # Check if account meets minimum threshold
            if follower_count < MIN_FOLLOWERS:
                logger.info(
                    "TikTok handle %s has %d followers, below minimum %d. Rejecting.",
                    handle,
                    follower_count,
                    MIN_FOLLOWERS,
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
        for the given account using the Scrape Creators API, parses the data,
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
                        endpoint="/v1/tiktok/profile",
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
        """Upsert TikTok account record to database using PostgreSQL ON CONFLICT.

        Args:
            user: User object from Scrape Creators API response.
            status: Account status ('parsed', 'rejected', etc.).

        Returns:
            ID of the upserted Account record (auto-generated by PostgreSQL).
        """
        platform_id: str = str(user.get("id") or user.get("userId", ""))
        username: str | None = user.get("uniqueId") or user.get("handle")
        full_name: str | None = user.get("nickname") or user.get("displayName")
        biography: str | None = user.get("signature") or user.get("bio")
        follower_count: int = self._extract_follower_count(user)

        async with self.session_maker() as session:
            async with session.begin():
                stmt = pg_insert(Account).values(
                    platform="TIKTOK",
                    platform_id=platform_id,
                    username=username,
                    title=full_name or username or "Unknown",
                    description=biography,
                    subscribers_count=follower_count,
                    status=status,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["platform", "platform_id"],
                    set_=dict(
                        username=username,
                        title=full_name or username,
                        description=biography,
                        subscribers_count=follower_count,
                        status=status,
                        updated_at=datetime.now(timezone.utc),
                    ),
                ).returning(Account.id)

                result = await session.execute(stmt)
                account_id: int = result.scalar_one()
                logger.debug(
                    "Upserted TikTok account %s (ID: %d, status: %s)",
                    username,
                    account_id,
                    status,
                )
                return account_id

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
                        transcription=stmt.excluded.transcription,
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
