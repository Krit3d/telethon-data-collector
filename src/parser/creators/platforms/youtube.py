"""
YouTube platform parser using Scrape Creators API.

Implements YouTube-specific profile parsing and content ingestion into PostgreSQL.
Focuses on YouTube Shorts content extraction and stores author profile metadata
(external links, contacts, location, language) inside Content.raw_metadata
under the "author_profile_metadata" key.

Features:
    - Profile parsing with account upsert to accounts table
    - Minimum subscriber threshold enforcement (3000 subscribers)
    - Virtual profile post creation for semantic search over biographies
    - YouTube Shorts content fetching and bulk upsert to content table
    - PostgreSQL ON CONFLICT DO UPDATE for high-throughput concurrency
    - Raw metadata preservation for OpenSPG processing
    - Extraction of external links, contact info from channel description via shared utils
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

# Minimum subscriber threshold for YouTube channels
MIN_SUBSCRIBERS: int = 3000


class YouTubePlatformParser(BasePlatformParser):
    """YouTube platform parser for profile and content ingestion.

    Inherits from BasePlatformParser and implements YouTube-specific
    profile parsing and content upsert logic using the Scrape Creators API.
    Focuses on YouTube Shorts content extraction.

    All author-level metadata (external links, contacts, location, language)
    is stored inside each Content.raw_metadata JSONB column under the key
    "author_profile_metadata" since Account has no raw_metadata column.

    After each successful profile parse, a virtual profile post is upserted
    into the content table to enable semantic search over creator biographies.

    Attributes:
        session_maker: SQLAlchemy async session maker for database operations.
        client: ScrapeCreatorsClient instance for API requests.
        settings: Application settings containing configuration values.
    """

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        client: ScrapeCreatorsClient,
        settings: Settings,
    ) -> None:
        """Initialize YouTube parser with configuration.

        Args:
            session_maker: SQLAlchemy async session maker for database operations.
            client: ScrapeCreatorsClient instance for API requests.
            settings: Application settings containing configuration values.
        """
        super().__init__(session_maker, client, settings)

    async def parse_profile(self, handle: str) -> int | None:
        """Fetch YouTube profile, upsert account to database, return account ID.

        Parses the YouTube channel profile for the given handle, checks if the channel
        meets the minimum subscriber threshold (3000), extracts author profile
        metadata (external links, contacts, location, language), upserts the
        account information to the accounts table, creates a virtual profile post
        for semantic search, and returns the database ID.

        Args:
            handle: YouTube channel handle (without @ prefix) or custom URL.

        Returns:
            Database ID of the upserted account record, or None if the profile
            could not be parsed or doesn't meet the minimum subscriber threshold.
        """
        logger.info("Starting YouTube profile parse for handle: %s", handle)

        try:
            # Fetch profile data from Scrape Creators API
            try:
                response: dict[str, Any] = await self.client.get(
                    endpoint="/v1/youtube/profile",
                    params={"handle": handle},
                )
                logger.info(
                    "API response status for profile %s: success, credits consumed: %s",
                    handle,
                    response.get("credits", "N/A"),
                )
            except Exception as e:
                logger.error(
                    "API request failed for YouTube profile %s: %s",
                    handle,
                    e,
                    exc_info=True,
                )
                return None

            # Validate response structure
            data = response.get("data")
            if not data:
                logger.error("Missing 'data' in API response for YouTube handle %s", handle)
                return None

            # YouTube API may return channel data at different levels
            channel = data.get("channel") or data.get("author") or data
            if not channel:
                logger.error("Missing channel data in API response for YouTube handle %s", handle)
                return None

            # Extract channel ID and subscriber count
            channel_id: str = str(
                channel.get("id")
                or channel.get("channelId")
                or channel.get("channel_id")
                or ""
            )
            if not channel_id:
                logger.error("Could not extract channel ID for YouTube handle %s", handle)
                return None

            subscriber_count: int = self._extract_subscriber_count(channel)

            # Check if account meets minimum threshold
            if subscriber_count < MIN_SUBSCRIBERS:
                logger.info(
                    "YouTube handle %s has %d subscribers, below minimum %d. Rejecting.",
                    handle,
                    subscriber_count,
                    MIN_SUBSCRIBERS,
                )
                await self._upsert_account(channel, status="rejected")
                return None

            # Upsert account with 'parsed' status
            account_id: int = await self._upsert_account(channel, status="parsed")
            logger.info(
                "Successfully parsed YouTube profile %s, account ID: %d, subscribers: %d",
                handle,
                account_id,
                subscriber_count,
            )

            # Upsert virtual profile post for semantic search over biography
            await self._upsert_virtual_profile_post(account_id, channel)

            return account_id

        except Exception as e:
            logger.error(
                "Failed to parse YouTube profile %s: %s",
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
        """Fetch YouTube Shorts content and bulk upsert to content table.

        Retrieves YouTube Shorts for the given channel using the Scrape Creators
        API, parses the data, and performs a bulk upsert into the content table
        using PostgreSQL ON CONFLICT DO UPDATE.

        The raw_metadata field contains:
            - "author_profile_metadata": Profile-level data (contacts, links, location, etc.)
            - "platform_metrics": Platform-specific engagement metrics
            - "video_download_url": Direct URL for GPU worker processing

        Args:
            account_id: Database ID of the parent account record.
            platform_id: YouTube channel handle/ID used in API calls.
            max_items: Maximum number of content items to fetch (default: 50).
        """
        logger.info(
            "Starting YouTube Shorts content parse for account_id: %d, platform_id: %s, max_items: %d",
            account_id,
            platform_id,
            max_items,
        )

        try:
            # Fetch YouTube Shorts data from Scrape Creators API
            try:
                response: dict[str, Any] = await self.client.get(
                    endpoint="/v1/youtube/shorts",
                    params={"handle": platform_id, "limit": max_items},
                )
                logger.info(
                    "API response status for YouTube Shorts, platform_id %s: success, credits consumed: %s",
                    platform_id,
                    response.get("credits", "N/A"),
                )
            except Exception as e:
                logger.error(
                    "API request failed for YouTube Shorts, platform_id %s: %s",
                    platform_id,
                    e,
                    exc_info=True,
                )
                return

            # Validate response structure
            data = response.get("data")
            if not data:
                logger.error(
                    "Missing 'data' in API response for YouTube Shorts, platform_id %s",
                    platform_id,
                )
                return

            # Extract shorts/videos from response
            shorts: list[dict[str, Any]] = self._extract_shorts_from_response(data)
            if not shorts:
                logger.info("No YouTube Shorts found for account_id: %d", account_id)
                return

            # Limit to max_items
            shorts = shorts[:max_items]

            # Build author profile metadata once (reused for all content items)
            channel_data = data.get("channel") or data.get("author") or {}
            author_metadata = self._build_author_profile_metadata(channel_data)

            # Process and upsert content items
            await self._upsert_content(shorts, account_id, author_metadata)
            logger.info(
                "Successfully upserted %d YouTube Shorts items for account_id: %d",
                len(shorts),
                account_id,
            )

        except Exception as e:
            logger.error(
                "Failed to parse YouTube Shorts for account_id %d: %s",
                account_id,
                e,
                exc_info=True,
            )
            raise

    async def _upsert_account(self, channel: dict[str, Any], status: str = "parsed") -> int:
        """Upsert YouTube account record to database using PostgreSQL ON CONFLICT.

        Args:
            channel: Channel object from Scrape Creators API response.
            status: Account status ('parsed', 'rejected', etc.).

        Returns:
            ID of the upserted Account record (auto-generated by PostgreSQL).
        """
        platform_id: str = str(
            channel.get("id")
            or channel.get("channelId")
            or channel.get("channel_id")
            or ""
        )
        username: str | None = (
            channel.get("handle")
            or channel.get("customUrl")
            or channel.get("username")
        )
        full_name: str | None = (
            channel.get("title")
            or channel.get("channelTitle")
            or channel.get("name")
        )
        description: str | None = (
            channel.get("description")
            or channel.get("bio")
            or channel.get("channelDescription")
        )
        subscriber_count: int = self._extract_subscriber_count(channel)

        async with self.session_maker() as session:
            async with session.begin():
                stmt = pg_insert(Account).values(
                    platform="YOUTUBE",
                    platform_id=platform_id,
                    username=username,
                    title=full_name or username or "Unknown",
                    description=description,
                    subscribers_count=subscriber_count,
                    status=status,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["platform", "platform_id"],
                    set_=dict(
                        username=username,
                        title=full_name or username,
                        description=description,
                        subscribers_count=subscriber_count,
                        status=status,
                        updated_at=datetime.now(timezone.utc),
                    ),
                ).returning(Account.id)

                result = await session.execute(stmt)
                account_id: int = result.scalar_one()
                logger.debug(
                    "Upserted YouTube account %s (ID: %d, status: %s)",
                    username,
                    account_id,
                    status,
                )
                return account_id

    async def _upsert_virtual_profile_post(
        self, account_id: int, channel: dict[str, Any]
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
            channel: Channel object from Scrape Creators API response.
        """
        platform_id: str = str(
            channel.get("id")
            or channel.get("channelId")
            or channel.get("channel_id")
            or ""
        )
        username: str | None = (
            channel.get("handle")
            or channel.get("customUrl")
            or channel.get("username")
        )
        full_name: str | None = (
            channel.get("title")
            or channel.get("channelTitle")
            or channel.get("name")
        )
        description: str | None = (
            channel.get("description")
            or channel.get("bio")
            or channel.get("channelDescription")
        )
        subscriber_count: int = self._extract_subscriber_count(channel)

        virtual_content_id: str = f"profile_bio_{platform_id}"
        compiled_text: str = (
            f"[PROFILE METADATA]\n"
            f"Platform: YouTube\n"
            f"Username: @{username or 'unknown'}\n"
            f"Title: {full_name or 'Unknown'}\n"
            f"Subscribers: {subscriber_count}\n"
            f"Bio: {description or 'N/A'}"
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
                    "Upserted virtual profile post for YouTube account_id: %d (platform_content_id: %s)",
                    account_id,
                    virtual_content_id,
                )

    async def _upsert_content(
        self,
        shorts: list[dict[str, Any]],
        account_id: int,
        author_metadata: dict[str, Any],
    ) -> None:
        """Bulk upsert YouTube Shorts content records to database.

        Args:
            shorts: List of YouTube Shorts video dictionaries from API responses.
            account_id: ID of the parent Account record.
            author_metadata: Author profile metadata to embed in each content record.
        """
        content_values: list[dict[str, Any]] = []

        for video in shorts:
            try:
                # Extract platform_content_id from video id
                platform_content_id: str = str(
                    video.get("id")
                    or video.get("videoId")
                    or video.get("video_id")
                    or ""
                )
                if not platform_content_id:
                    logger.warning("Skipping content item with no ID")
                    continue

                # Extract content text (video title/description)
                content_text: str | None = (
                    video.get("title")
                    or video.get("description")
                    or (video.get("snippet") or {}).get("title")
                )

                # Extract published timestamp
                published_at: datetime
                publish_time = (
                    video.get("publishedAt")
                    or video.get("publishDate")
                    or video.get("published")
                    or (video.get("snippet") or {}).get("publishedAt")
                )
                if publish_time:
                    try:
                        # Handle ISO string format
                        if isinstance(publish_time, str):
                            published_at = datetime.fromisoformat(
                                publish_time.replace("Z", "+00:00")
                            )
                        else:
                            published_at = datetime.now(timezone.utc)
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            "Failed to parse publishedAt for video %s: %s",
                            platform_content_id,
                            e,
                        )
                        published_at = datetime.now(timezone.utc)
                else:
                    published_at = datetime.now(timezone.utc)

                # Extract engagement metrics
                statistics = video.get("statistics") or video.get("stats") or {}
                views: int | None = (
                    video.get("viewCount")
                    or statistics.get("viewCount")
                    or video.get("views")
                )
                reactions_count: int | None = (
                    video.get("likeCount")
                    or statistics.get("likeCount")
                    or video.get("likes")
                )
                comments_count: int | None = (
                    video.get("commentCount")
                    or statistics.get("commentCount")
                    or video.get("comments")
                )
                shares_count: int | None = (
                    video.get("shareCount")
                    or statistics.get("shareCount")
                    or video.get("shares")
                )

                # Extract direct video download link - store in raw_metadata["video_download_url"]
                video_download_url: str | None = self._extract_video_download_url(video)

                # Extract transcription and save into Content.transcription
                transcription: str | None = self._extract_transcription(video)

                # Build platform metrics
                platform_metrics: dict[str, Any] = {
                    "views": views,
                    "likes": reactions_count,
                    "comments": comments_count,
                    "shares": shares_count,
                    "duration": video.get("duration"),
                    "definition": video.get("definition"),
                }

                # Build raw_metadata with author_profile_metadata and platform_metrics
                # Store video_download_url inside raw_metadata
                raw_metadata: dict[str, Any] = {
                    "author_profile_metadata": author_metadata,
                    "platform_metrics": platform_metrics,
                }
                if video_download_url:
                    raw_metadata["video_download_url"] = video_download_url

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
                        "has_media": True,  # YouTube Shorts are videos
                        "raw_metadata": raw_metadata,
                        "is_embedded": False,
                        "is_graph_extracted": False,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )

            except Exception as e:
                logger.error(
                    "Failed to process YouTube Shorts item: %s",
                    e,
                    exc_info=True,
                )
                continue

        if not content_values:
            logger.info("No valid YouTube Shorts items to upsert")
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
                    "Bulk upserted %d YouTube Shorts items for account_id: %d",
                    len(content_values),
                    account_id,
                )

    def _extract_shorts_from_response(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract YouTube Shorts videos from API response.

        Args:
            data: Response data dictionary from Scrape Creators API.

        Returns:
            List of video item dictionaries.
        """
        videos: list[dict[str, Any]] = []

        # Try different possible response structures
        # Direct videos array
        if "videos" in data and isinstance(data["videos"], list):
            videos = data["videos"]
        # Items array (common in YouTube API)
        elif "items" in data and isinstance(data["items"], list):
            videos = data["items"]
        # Shorts array
        elif "shorts" in data and isinstance(data["shorts"], list):
            videos = data["shorts"]
        # Nested under response
        elif "response" in data and isinstance(data["response"], dict):
            response = data["response"]
            if "videos" in response and isinstance(response["videos"], list):
                videos = response["videos"]
            elif "items" in response and isinstance(response["items"], list):
                videos = response["items"]

        return videos

    def _extract_subscriber_count(self, channel: dict[str, Any]) -> int:
        """Extract subscriber count from YouTube channel object.

        Args:
            channel: Channel object from Scrape Creators API response.

        Returns:
            Subscriber count as integer, defaults to 0 if not found.
        """
        # Try different possible field names
        subscriber_count = (
            channel.get("subscriberCount")
            or channel.get("subscribers")
            or (channel.get("statistics") or {}).get("subscriberCount")
            or (channel.get("statistics") or {}).get("subscriber_count")
            or channel.get("subscriber_count")
            or 0
        )
        try:
            return int(subscriber_count)
        except (ValueError, TypeError):
            return 0

    def _extract_video_download_url(self, video: dict[str, Any]) -> str | None:
        """Extract direct video download URL from YouTube video item.

        Stores the direct video download link in raw_metadata["video_download_url"]
        for GPU worker processing.

        Args:
            video: Video item dictionary from API response.

        Returns:
            Direct video download URL string, or None if not found.
        """
        # Check for direct download URL in various possible fields
        download_url = (
            video.get("downloadUrl")
            or video.get("download_url")
            or video.get("videoUrl")
            or video.get("video_url")
            or video.get("streamUrl")
        )
        if download_url and isinstance(download_url, str):
            return download_url

        # Check in nested objects (e.g., from YouTube API response)
        player_response = video.get("playerResponse") or video.get("player_response")
        if isinstance(player_response, dict):
            streaming_data = player_response.get("streamingData") or player_response.get("streaming_data")
            if isinstance(streaming_data, dict):
                # Try to get the highest quality format
                formats = streaming_data.get("formats") or []
                adaptive_formats = streaming_data.get("adaptiveFormats") or []
                all_formats = (formats if isinstance(formats, list) else []) + \
                              (adaptive_formats if isinstance(adaptive_formats, list) else [])
                if all_formats:
                    # Get the first format with a URL
                    for fmt in all_formats:
                        if isinstance(fmt, dict) and fmt.get("url"):
                            return str(fmt["url"])

        return None

    def _extract_transcription(self, video: dict[str, Any]) -> str | None:
        """Extract transcription from YouTube video item.

        Saves the transcription into Content.transcription column.

        Args:
            video: Video item dictionary from API response.

        Returns:
            Transcription text if available, else None.
        """
        # Check for transcription in various possible locations
        # Scrape Creators API may return transcription in different fields
        transcription = (
            video.get("transcription")
            or video.get("transcript")
            or video.get("captions")
            or video.get("subtitles")
        )

        # Handle case where transcription is a string
        if transcription and isinstance(transcription, str):
            return transcription.strip()

        # Handle case where transcription is a list of caption objects
        if isinstance(transcription, list) and len(transcription) > 0:
            # If it's a list of caption objects, try to concatenate
            try:
                return " ".join(
                    item.get("text", "") for item in transcription if isinstance(item, dict) and "text" in item
                ).strip()
            except (AttributeError, TypeError):
                pass

        # Check in nested objects (e.g., from YouTube API response)
        if isinstance(transcription, dict) and "text" in transcription:
            return str(transcription["text"]).strip()

        return None

    def _build_author_profile_metadata(self, channel: dict[str, Any]) -> dict[str, Any]:
        """Build author profile metadata dictionary from YouTube channel object.

        Extracts profile link, parsed bio contacts, language, location,
        and geo_data from the channel object using shared parse_profile_contacts
        utility for extracting emails, Telegram handles, and external links.

        Args:
            channel: Channel object from Scrape Creators API response.

        Returns:
            Dictionary containing author profile metadata.
        """
        username: str | None = (
            channel.get("handle")
            or channel.get("customUrl")
            or channel.get("username")
        )
        description: str | None = (
            channel.get("description")
            or channel.get("bio")
            or channel.get("channelDescription")
        )

        # Build profile link
        profile_link: str | None = None
        if username:
            profile_link = f"https://www.youtube.com/@{username}"
        elif channel.get("id"):
            profile_link = f"https://www.youtube.com/channel/{channel['id']}"

        # Use shared utility to parse contacts from description
        contacts_dict: dict[str, Any] = parse_profile_contacts(description, None)
        # parse_profile_contacts returns: {emails, telegram_handles, external_links, raw_bio}

        # Build contacts list in the format expected by OpenSPG
        contacts: list[str] = []
        for email in contacts_dict.get("emails", []):
            contacts.append(f"email:{email}")
        for handle in contacts_dict.get("telegram_handles", []):
            contacts.append(f"telegram:@{handle}")
        # Add external links as contact entries
        external_links: list[str] = contacts_dict.get("external_links", [])

        # Also check for official website links in channel object
        related_links = channel.get("relatedLinks") or channel.get("externalLinks") or []
        if isinstance(related_links, list):
            for link in related_links:
                if isinstance(link, dict) and link.get("url"):
                    url = str(link["url"])
                    if url not in external_links:
                        external_links.append(url)
                elif isinstance(link, str) and link not in external_links:
                    external_links.append(link)

        # Also check for direct website field
        website = channel.get("website") or channel.get("websiteUrl")
        if website and isinstance(website, str) and website not in external_links:
            external_links.append(website)

        # Extract location from channel object
        location: str | None = channel.get("country") or channel.get("region")

        # Language from channel object
        language: str | None = channel.get("defaultLanguage") or channel.get("language")

        # Geo-data (country/region) from channel object
        geo_data: dict[str, str] | None = None
        country = channel.get("country")
        region = channel.get("region")
        if country or region:
            geo_data = {}
            if country:
                geo_data["country"] = str(country)
            if region:
                geo_data["region"] = str(region)

        author_metadata: dict[str, Any] = {
            "profile_link": profile_link,
            "bio_description": description,
            "external_links": external_links if external_links else None,
            "contacts": contacts if contacts else None,
            "advertising_contacts": contacts if contacts else None,
            "language": language,
            "location": location,
            "geo_data": geo_data,
        }

        # Remove None values for cleaner JSON
        return {k: v for k, v in author_metadata.items() if v is not None}
