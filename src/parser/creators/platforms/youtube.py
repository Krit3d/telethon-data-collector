"""
YouTube platform parser using Scrape Creators API v1.

Implements YouTube-specific profile parsing and content ingestion into PostgreSQL.
Focuses on YouTube content extraction and stores author profile metadata
(external links, contacts, location, language) inside Content.raw_metadata
under the "author_profile_metadata" key.

Features:
    - Profile parsing with account upsert to accounts table
    - Minimum and maximum subscriber threshold enforcement (3k-150k)
    - Russian language content filtering
    - Virtual profile post creation for semantic search over biographies
    - YouTube content fetching via v1 API (single call, no pagination)
    - Concurrent transcript fetching using asyncio.Semaphore (limit 5)
    - PostgreSQL ON CONFLICT DO UPDATE for high-throughput concurrency
    - Raw metadata preservation for OpenSPG processing
    - Extraction of external links, contact info from channel description via shared utils
    - Cross-platform spidering queue for discovered accounts
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from aiohttp import ClientResponseError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config.config import Settings
from src.db.models import Account, Content
from src.parser.creators.core.db import (
    upsert_and_deduplicate_account,
    update_account_profile_metadata,
    bulk_upsert_content,
    upsert_virtual_bio_post,
    queue_discovered_accounts,
    queue_discovered_mentions,
)
from src.parser.creators.core.utils import (
    is_russian_text,
    is_slop_or_theme_page,
    extract_mentions,
    parse_profile_contacts,
    parse_published_at,
)
from src.parser.creators.platforms.base import BasePlatformParser
from src.parser.creators.sc_client import ScrapeCreatorsClient

logger = logging.getLogger(__name__)

# Minimum subscriber threshold for YouTube channels (micro-influencers)
MIN_SUBSCRIBERS: int = 3000

# Maximum subscriber threshold for YouTube channels (micro-influencers)
MAX_SUBSCRIBERS: int = 150000

# Semaphore limit for concurrent transcript fetching
TRANSCRIPT_SEMAPHORE_LIMIT: int = 5

# Maximum number of videos to collect
MAX_VIDEOS: int = 10


class YouTubeParser(BasePlatformParser):
    """YouTube platform parser for profile and content ingestion.

    Inherits from BasePlatformParser and implements YouTube-specific
    profile parsing and content upsert logic using the Scrape Creators API v1.

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
        self._cached_profile: dict[str, Any] | None = None
        self._cached_handle: str | None = None

    def _format_handle(self, handle: str) -> str:
        """Format YouTube handle with proper prefix for API calls.

        Handles that start with "UC" (channel IDs) are kept as-is.
        Handles that start with "@" are kept as-is.
        All other handles are prefixed with "@".

        Args:
            handle: Raw handle string from user input or database.

        Returns:
            Formatted handle ready for API calls.
        """
        if handle.startswith("UC") or handle.startswith("@"):
            return handle
        return f"@{handle}"

    def _get_api_param_key(self, formatted_handle: str) -> str:
        """Determine the correct API parameter key for the handle.

        If the handle starts with "UC", it's a channel ID and should use
        the "channelId" parameter. Otherwise, use the "handle" parameter.

        Args:
            formatted_handle: The formatted handle string.

        Returns:
            "channelId" if handle starts with "UC", otherwise "handle".
        """
        if formatted_handle.startswith("UC"):
            return "channelId"
        return "handle"

    async def _get_cached_or_fetch_profile(
        self, handle: str
    ) -> dict[str, Any] | None:
        """Get profile from cache or fetch from API.

        Args:
            handle: YouTube channel handle to fetch.

        Returns:
            Channel data dictionary, or None if not found or error.
        """
        # Format handle for API call
        formatted_handle = self._format_handle(handle)

        if self._cached_handle == formatted_handle and self._cached_profile:
            logger.debug(
                "Using cached profile data for handle: %s", formatted_handle
            )
            return self._cached_profile

        try:
            # Determine correct parameter key for profile endpoint
            param_key = self._get_api_param_key(formatted_handle)
            response: dict[str, Any] = await self.client.get(
                endpoint="/v1/youtube/channel",
                params={param_key: formatted_handle},
            )
            logger.debug(
                "API response status for profile %s: success, remaining credits: %s",
                formatted_handle,
                response.get("credits_remaining", "N/A"),
            )

            # YouTube v1 API returns profile fields directly at root level
            # Treat entire response as profile data
            self._cached_profile = response
            self._cached_handle = formatted_handle
            return response

        except ClientResponseError as e:
            if e.status == 404:
                logger.warning(
                    "YouTube channel %s not found (404). Marking as rejected.",
                    handle,
                )
            else:
                logger.error(
                    "YouTube API request failed for %s with HTTP %d: %s",
                    formatted_handle,
                    e.status,
                    e,
                )
            return None
        except Exception as e:
            logger.error(
                "Unexpected error fetching YouTube profile %s: %s",
                formatted_handle,
                e,
                exc_info=True,
            )
            return None

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

    def _build_author_profile_metadata(
        self, channel: dict[str, Any]
    ) -> dict[str, Any]:
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
        contacts_dict: dict[str, Any] = parse_profile_contacts(
            description, None
        )

        # Build contacts list in the format expected by OpenSPG
        contacts: list[str] = []
        for email in contacts_dict.get("emails", []):
            contacts.append(f"email:{email}")
        for telegram_handle in contacts_dict.get("telegram_handles", []):
            contacts.append(f"telegram:@{telegram_handle}")
        # Add external links as contact entries
        external_links: list[str] = contacts_dict.get("external_links", [])

        # Also check for official website links in channel object
        related_links = (
            channel.get("relatedLinks") or channel.get("externalLinks") or []
        )
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
        if (
            website
            and isinstance(website, str)
            and website not in external_links
        ):
            external_links.append(website)

        # Extract location from channel object
        location: str | None = channel.get("country") or channel.get("region")

        # Language from channel object
        language: str | None = channel.get("defaultLanguage") or channel.get(
            "language"
        )

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

    async def parse_profile(self, handle: str) -> int | None:
        """Fetch YouTube profile, upsert account to database, return account ID.

        Parses the YouTube channel profile for the given handle, checks if the channel
        meets the subscriber threshold (3k-150k), verifies Russian language content,
        filters out AI-slop and theme pages, upserts the account information to the
        accounts table, creates a virtual profile post for semantic search, queues
        discovered accounts for cross-platform spidering, and returns the database ID.

        Args:
            handle: YouTube channel handle (without @ prefix) or custom URL.

        Returns:
            Database ID of the upserted account record, or None if the profile
            could not be parsed or doesn't meet the filtering criteria.
        """
        logger.info("Starting YouTube profile parse for handle: %s", handle)

        channel = await self._get_cached_or_fetch_profile(handle)
        if not channel:
            return None

        # Extract channel ID and subscriber count
        channel_id: str = str(
            channel.get("id")
            or channel.get("channelId")
            or channel.get("channel_id")
            or ""
        )
        if not channel_id:
            logger.error(
                "Could not extract channel ID for YouTube handle %s", handle
            )
            return None

        subscriber_count: int = self._extract_subscriber_count(channel)

        # Check if account meets subscriber thresholds [3000, 150000]
        if (
            subscriber_count < MIN_SUBSCRIBERS
            or subscriber_count > MAX_SUBSCRIBERS
        ):
            logger.info(
                "YouTube handle %s has %d subscribers, outside threshold [%d, %d]. Rejecting.",
                handle,
                subscriber_count,
                MIN_SUBSCRIBERS,
                MAX_SUBSCRIBERS,
            )
            async with self.session_maker() as session:
                await upsert_and_deduplicate_account(
                    session=session,
                    platform="YOUTUBE",
                    platform_id=channel_id,
                    username=channel.get("handle") or channel.get("customUrl"),
                    title=channel.get("title") or channel.get("name") or "Unknown",
                    description=channel.get("description") or channel.get("bio"),
                    subscribers_count=subscriber_count,
                    status="rejected",
                )
                await session.commit()
            return None

        # Extract description and full_name for filtering
        description: str | None = (
            channel.get("description")
            or channel.get("bio")
            or channel.get("channelDescription")
        )
        full_name: str | None = (
            channel.get("title")
            or channel.get("channelTitle")
            or channel.get("name")
        )

        # Check for Russian language content in description OR full_name
        # This is our zero-cost filter against foreign spam
        if not (is_russian_text(description) or is_russian_text(full_name)):
            logger.info(
                "YouTube handle %s does not contain Russian text in description or title. Rejecting.",
                handle,
            )
            async with self.session_maker() as session:
                await upsert_and_deduplicate_account(
                    session=session,
                    platform="YOUTUBE",
                    platform_id=channel_id,
                    username=channel.get("handle") or channel.get("customUrl"),
                    title=full_name or "Unknown",
                    description=description,
                    subscribers_count=subscriber_count,
                    status="rejected",
                )
                await session.commit()
            return None

        # Check for AI-slop / theme-page / meme-page
        username: str | None = (
            channel.get("handle")
            or channel.get("customUrl")
            or channel.get("username")
        )
        if is_slop_or_theme_page(username, description):
            logger.info(
                "YouTube handle %s detected as AI-slop or theme page. Rejecting.",
                handle,
            )
            async with self.session_maker() as session:
                await upsert_and_deduplicate_account(
                    session=session,
                    platform="YOUTUBE",
                    platform_id=channel_id,
                    username=username,
                    title=full_name or "Unknown",
                    description=description,
                    subscribers_count=subscriber_count,
                    status="rejected",
                )
                await session.commit()
            return None

        # Upsert account with 'processing' status
        async with self.session_maker() as session:
            account_id: int = await upsert_and_deduplicate_account(
                session=session,
                platform="YOUTUBE",
                platform_id=channel_id,
                username=username,
                title=full_name or "Unknown",
                description=description,
                subscribers_count=subscriber_count,
                status="processing",
            )

            # Update biography and trigger cross-platform queue discovery
            await update_account_profile_metadata(
                session=session,
                account_id=account_id,
                platform="YOUTUBE",
                biography=description,
            )

            # Create a standard virtual profile post in the Content table
            await upsert_virtual_bio_post(
                session=session,
                account_id=account_id,
                platform="YOUTUBE",
                platform_id=channel_id,
                username=username,
                full_name=full_name,
                biography=description,
                subscribers_count=subscriber_count,
                raw_metadata=None,  # No female flags
            )

            await session.commit()

        # Update account status to 'parsed'
        async with self.session_maker() as session:
            await upsert_and_deduplicate_account(
                session=session,
                platform="YOUTUBE",
                platform_id=channel_id,
                username=username,
                title=full_name or "Unknown",
                description=description,
                subscribers_count=subscriber_count,
                status="parsed",
            )
            await session.commit()

        logger.info(
            "Successfully parsed YouTube profile %s, account ID: %d, subscribers: %d",
            handle,
            account_id,
            subscriber_count,
        )

        return account_id

    async def parse_content(
        self,
        account_id: int,
        platform_id: str,
        max_items: int = MAX_VIDEOS,
    ) -> None:
        """Fetch YouTube content via v1 API, fetch transcripts, and bulk upsert to content table.

        Performs exactly ONE API call to '/v1/youtube/channel-videos' (no pagination).
        Collects a MAXIMUM of 10 items.

        Retrieves YouTube videos for the given channel using the Scrape Creators
        API v1, parses the data, concurrently fetches transcripts for videos that
        don't contain target keywords (using a semaphore limit of 5),
        and performs a bulk upsert into the content table using PostgreSQL
        ON CONFLICT DO UPDATE.

        The content text is formed by concatenating title, description, and transcript
        from the video payload.

        The raw_metadata field contains:
            - "author_profile_metadata": Profile-level data (contacts, links, location, etc.)
            - "platform_metrics": Platform-specific engagement metrics including video_url

        Args:
            account_id: Database ID of the parent account record.
            platform_id: YouTube channel ID (UC...) or handle.
            max_items: Maximum number of content items to fetch (default: 10).
        """
        logger.info(
            "Starting YouTube content parse for account_id: %d, platform_id: %s, max_items: %d",
            account_id,
            platform_id,
            max_items,
        )

        # Fetch the account's username (handle) from the database
        handle: str = platform_id  # fallback to platform_id (channel ID)
        try:
            async with self.session_maker() as session:
                stmt = select(Account.username).where(Account.id == account_id)
                result = await session.execute(stmt)
                db_username: str | None = result.scalar_one_or_none()
                if db_username:
                    handle = db_username
                    logger.debug(
                        "Using username '%s' as handle for account_id: %d",
                        handle,
                        account_id,
                    )
                else:
                    logger.warning(
                        "Username not found for account_id: %d, falling back to platform_id: %s",
                        account_id,
                        platform_id,
                    )
        except Exception as e:
            logger.warning(
                "Failed to fetch username for account_id: %d: %s. Using platform_id as fallback.",
                account_id,
                e,
            )

        # Format handle for API call
        formatted_handle = self._format_handle(handle)

        # Determine correct parameter key (channelId for UC..., handle otherwise)
        param_key = self._get_api_param_key(formatted_handle)
        if param_key == "channelId":
            logger.debug(
                "Using channelId parameter for handle: %s", formatted_handle
            )
        else:
            logger.debug(
                "Using handle parameter for handle: %s", formatted_handle
            )

        try:
            # Exactly ONE API call to fetch videos (no pagination)
            params: dict[str, Any] = {param_key: formatted_handle}
            response: dict[str, Any] = await self.client.get(
                endpoint="/v1/youtube/channel-videos",
                params=params,
            )
            logger.debug(
                "API response status for YouTube videos, handle %s: success, remaining credits: %s",
                formatted_handle,
                response.get("credits_remaining", "N/A"),
            )

            # Extract videos from response (v1 API returns videos at root level)
            videos: list[dict[str, Any]] = response.get("videos", [])
            if not videos:
                logger.info(
                    "No videos found for handle %s", formatted_handle
                )
                return

            # Limit to max_items (maximum 10)
            videos = videos[:max_items]
            logger.info(
                "Fetched %d videos for handle %s",
                len(videos),
                formatted_handle,
            )

            # Build author profile metadata once (reused for all content items)
            channel_data: dict[str, Any] = (
                self._cached_profile if self._cached_profile else {}
            )
            author_metadata = self._build_author_profile_metadata(channel_data)

            # Get compiled semantic keywords pattern from settings
            keywords_pattern = self.settings.semantic_keywords_pattern

            # Process videos and fetch transcripts concurrently
            content_values: list[dict[str, Any]] = []
            transcript_tasks: list[tuple[
                dict[str, Any],
                str,
                str,
                datetime | None,
                int | None,
                int | None,
                int | None,
                int | None,
                dict[str, Any],
            ]] = []

            for video in videos:
                video_id: str = str(
                    video.get("id")
                    or video.get("videoId")
                    or video.get("video_id")
                    or ""
                )
                if not video_id:
                    continue

                # Extract title and description
                title: str = video.get("title") or ""
                description: str | None = video.get("description") or video.get(
                    "videoDescription"
                )

                # Concatenate title and description for content text
                if description:
                    content_text: str = f"{title} {description}".strip()
                else:
                    content_text: str = title.strip()

                # Check if content contains target keywords using compiled regex pattern
                has_target_keyword: bool = bool(keywords_pattern.search(content_text))

                # Build video URL for downstream audio embedding extraction
                video_url: str = f"https://www.youtube.com/watch?v={video_id}"

                # Extract published timestamp using shared utility
                publish_time = (
                    video.get("publishedAt")
                    or video.get("publishDate")
                    or video.get("published")
                )
                published_at = parse_published_at(publish_time)

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

                # Build platform metrics with video_url
                platform_metrics: dict[str, Any] = {
                    "views": views,
                    "likes": reactions_count,
                    "comments": comments_count,
                    "shares": shares_count,
                    "duration": video.get("duration"),
                    "definition": video.get("definition"),
                    "video_url": video_url,
                }

                # Build raw_metadata
                raw_metadata: dict[str, Any] = {
                    "author_profile_metadata": author_metadata,
                    "platform_metrics": platform_metrics,
                }

                # If content has target keyword, skip transcript
                if has_target_keyword:
                    logger.debug(
                        "Video %s contains target keyword, skipping transcript",
                        video_id,
                    )
                    content_values.append(
                        {
                            "account_id": account_id,
                            "platform_content_id": video_id,
                            "content": content_text,
                            "transcription": None,
                            "published_at": published_at,
                            "views": views,
                            "reactions_count": reactions_count,
                            "comments_count": comments_count,
                            "shares_count": shares_count,
                            "has_media": True,
                            "raw_metadata": raw_metadata,
                            "is_embedded": False,
                            "is_graph_extracted": False,
                            "updated_at": datetime.now(timezone.utc),
                        }
                    )
                else:
                    # Queue transcript fetching task
                    transcript_tasks.append(
                        (video, video_id, content_text, published_at, views,
                         reactions_count, comments_count, shares_count,
                         raw_metadata)
                    )

            # Fetch transcripts concurrently for videos without target keywords
            if transcript_tasks:
                logger.info(
                    "Fetching transcripts for %d videos concurrently",
                    len(transcript_tasks),
                )
                semaphore = asyncio.Semaphore(TRANSCRIPT_SEMAPHORE_LIMIT)
                tasks = [
                    self._fetch_transcript(video, video_id, semaphore)
                    for video, video_id, *_ in transcript_tasks
                ]
                transcript_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process transcript results
                for (video, video_id, content_text, published_at, views,
                     reactions_count, comments_count, shares_count,
                     raw_metadata), result in zip(transcript_tasks, transcript_results):
                    if isinstance(result, Exception):
                        logger.warning(
                            "Failed to fetch transcript for video %s: %s",
                            video_id,
                            result,
                        )
                        transcript_text = None
                    else:
                        transcript_text = result

                    # Append transcript to content text if available
                    if transcript_text:
                        if content_text:
                            content_text = f"{content_text}\n\n[TRANSCRIPT]\n{transcript_text}"
                        else:
                            content_text = f"[TRANSCRIPT]\n{transcript_text}"

                    content_values.append(
                        {
                            "account_id": account_id,
                            "platform_content_id": video_id,
                            "content": content_text,
                            "transcription": transcript_text,
                            "published_at": published_at,
                            "views": views,
                            "reactions_count": reactions_count,
                            "comments_count": comments_count,
                            "shares_count": shares_count,
                            "has_media": True,
                            "raw_metadata": raw_metadata,
                            "is_embedded": False,
                            "is_graph_extracted": False,
                            "updated_at": datetime.now(timezone.utc),
                        }
                    )

            # Content-based Discovery Spider
            # Extract cross-platform links and same-platform mentions from video text
            for video in videos:
                video_content_text = f"{video.get('title', '')} {video.get('description', '')}"
                if video_content_text:
                    try:
                        # Parse profile contacts (cross-platform links)
                        contacts_dict = parse_profile_contacts(video_content_text)
                        # Extract same-platform mentions (e.g., @otheruser)
                        mentions = extract_mentions(video_content_text)

                        # Queue discovered accounts in independent session
                        async with self.session_maker() as session:
                            # Cross-platform links
                            await queue_discovered_accounts(
                                session, contacts_dict, handle
                            )
                            # Same-platform mentions
                            await queue_discovered_mentions(
                                session, "YOUTUBE", mentions, handle
                            )
                            await session.commit()
                    except Exception as e:
                        logger.warning(
                            "Failed to queue discovered accounts from YouTube video %s: %s",
                            video.get("id") or video.get("videoId"),
                            e,
                        )

            # Bulk upsert content to database
            async with self.session_maker() as session:
                await bulk_upsert_content(
                    session=session,
                    content_values=content_values,
                )
                await session.commit()

            logger.info(
                "Successfully upserted %d YouTube content items for account_id: %d",
                len(content_values),
                account_id,
            )

        except ClientResponseError as e:
            if e.status == 404:
                logger.warning(
                    "YouTube videos for %s not found (404).", formatted_handle
                )
            else:
                logger.error(
                    "YouTube videos API request failed for %s with HTTP %d: %s",
                    formatted_handle,
                    e.status,
                    e,
                )
        except Exception as e:
            logger.error(
                "Failed to parse YouTube videos for account_id %d: %s",
                account_id,
                e,
                exc_info=True,
            )
            raise

    async def _fetch_transcript(
        self,
        video: dict[str, Any],
        video_id: str,
        semaphore: asyncio.Semaphore,
    ) -> str | None:
        """Fetch transcript for a single YouTube video using Scrape Creators API.

        Args:
            video: Video dictionary containing the video ID.
            video_id: YouTube video ID.
            semaphore: asyncio.Semaphore to limit concurrent requests.

        Returns:
            Joined transcript text, or None if fetching fails or no transcript available.
        """
        url = f"https://www.youtube.com/watch?v={video_id}"

        # Transcript request log with required format
        logger.debug("Requesting transcript for video: %s", url)

        async with semaphore:
            try:
                response: dict[str, Any] = await self.client.get(
                    endpoint="/v1/youtube/video/transcript",
                    params={"url": url},
                )

                # Extract transcript segments from response
                segments = response.get("transcript") or []
                if not segments:
                    logger.debug("No transcript available for video: %s", url[:50])
                    return None

                # Join the "text" field of each segment with a space
                transcript_text = " ".join(
                    segment.get("text", "")
                    for segment in segments
                    if isinstance(segment, dict)
                )

                if transcript_text:
                    logger.debug(
                        "Successfully retrieved transcript for video %s",
                        url[:50],
                    )
                    return transcript_text
                return None

            except ClientResponseError as e:
                if e.status == 404:
                    logger.debug(
                        "Transcript not found for video %s (404)", video_id
                    )
                else:
                    logger.warning(
                        "Failed to fetch transcript for video %s: HTTP %d: %s",
                        video_id,
                        e.status,
                        e,
                    )
                return None
            except Exception as e:
                logger.warning(
                    "Error fetching transcript for video %s: %s",
                    video_id,
                    e,
                )
                return None
