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
    - AI-slop and theme-page filtering
    - Female creator detection
    - Virtual profile post creation for semantic search over biographies
    - YouTube content fetching via v1 API with pagination using continuationToken
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

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

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
    queue_discovered_mentions,
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
            logger.info(
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

    async def parse_profile(self, handle: str) -> int | None:
        """Fetch YouTube profile, upsert account to database, return account ID.

        Parses the YouTube channel profile for the given handle, checks if the channel
        meets the subscriber threshold (3k-150k), verifies Russian language content,
        filters out AI-slop and theme pages, detects female creators, extracts
        author profile metadata (external links, contacts, location, language),
        upserts the account information to the accounts table, creates a virtual
        profile post for semantic search, queues discovered accounts for cross-platform
        spidering, and returns the database ID.

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
            await self._upsert_account(channel, status="rejected")
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
        if not (is_russian_text(description) or is_russian_text(full_name)):
            logger.info(
                "YouTube handle %s does not contain Russian text in description or title. Rejecting.",
                handle,
            )
            await self._upsert_account(channel, status="rejected")
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
            await self._upsert_account(channel, status="rejected")
            return None

        # Detect female creator
        female_heuristic = detect_female_creator(description)
        if female_heuristic:
            logger.info("YouTube handle %s detected as female creator.", handle)

        # Upsert account with 'parsed' status
        account_id: int = await self._upsert_account(channel, status="parsed")
        logger.info(
            "Successfully parsed YouTube profile %s, account ID: %d, subscribers: %d",
            handle,
            account_id,
            subscriber_count,
        )

        # Upsert virtual profile post for semantic search over biography
        raw_metadata: dict[str, Any] = {}
        if female_heuristic:
            raw_metadata["female_heuristic"] = True

        async with self.session_maker() as session:
            await upsert_virtual_bio_post(
                session=session,
                account_id=account_id,
                platform="YOUTUBE",
                platform_id=channel_id,
                username=username,
                full_name=full_name,
                biography=description,
                subscribers_count=subscriber_count,
                raw_metadata=raw_metadata if raw_metadata else None,
            )
            await session.commit()

        # Queue discovered accounts from contacts for cross-platform spidering
        contacts_dict: dict[str, Any] = parse_profile_contacts(
            description, None
        )
        async with self.session_maker() as session:
            await queue_discovered_accounts(session, contacts_dict, handle)
            await session.commit()

        return account_id

    async def parse_content(
        self,
        account_id: int,
        platform_id: str,
        max_items: int = 50,
    ) -> None:
        """Fetch YouTube content via v1 API, fetch transcripts, and bulk upsert to content table.

        Implements Smart Adaptive Pagination:
        - Early exit on page 1 if no videos found (dead channels)
        - 5-page ceiling to limit credit usage
        - Collects up to max_items (50) valid videos/Shorts

        Retrieves YouTube videos for the given channel using the Scrape Creators
        API v1 with pagination support (using continuationToken), parses the data,
        concurrently fetches transcripts for all videos using a semaphore limit of 5,
        and performs a bulk upsert into the content table using PostgreSQL
        ON CONFLICT DO UPDATE.

        The content text is formed by concatenating title, description, and transcript
        from the video payload. Unlike Instagram, YouTube videos have no duration
        limit for transcript fetching.

        The raw_metadata field contains:
            - "author_profile_metadata": Profile-level data (contacts, links, location, etc.)
            - "platform_metrics": Platform-specific engagement metrics

        Args:
            account_id: Database ID of the parent account record.
            platform_id: YouTube channel ID (UC...) or handle.
            max_items: Maximum number of content items to fetch (default: 50).
        """
        # Smart Adaptive Pagination: 5-page ceiling to limit credit usage
        MAX_PAGES_TO_CHECK: int = 5

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

        # Collect all videos using pagination with continuationToken
        all_videos: list[dict[str, Any]] = []
        continuation_token: str | None = None

        # Track page count for safety limit
        page_count = 0

        try:
            while len(all_videos) < max_items:
                page_count += 1

                # Ceiling Check: stop pagination after MAX_PAGES_TO_CHECK pages
                if page_count >= MAX_PAGES_TO_CHECK:
                    logger.info(
                        "Reached page limit (%d) for account %s, stopping pagination with %d videos collected",
                        MAX_PAGES_TO_CHECK, platform_id, len(all_videos)
                    )
                    break

                # Page fetch log with required format
                logger.info("Fetching page %d of posts for %s...", page_count, platform_id)

                # Build request parameters with correct key
                params: dict[str, Any] = {param_key: formatted_handle}
                if continuation_token:
                    params["continuationToken"] = continuation_token

                # Fetch videos from v1 API
                response: dict[str, Any] = await self.client.get(
                    endpoint="/v1/youtube/channel-videos",
                    params=params,
                )
                logger.info(
                    "API response status for YouTube videos, handle %s: success, remaining credits: %s",
                    formatted_handle,
                    response.get("credits_remaining", "N/A"),
                )

                # Extract videos from response (v1 API returns videos at root level)
                videos: list[dict[str, Any]] = response.get("videos", [])
                if not videos:
                    logger.info(
                        "No more videos found for handle %s", formatted_handle
                    )
                    break

                all_videos.extend(videos)
                logger.debug(
                    "Fetched %d videos, total collected: %d",
                    len(videos),
                    len(all_videos),
                )

                # Early Exit Check: After processing page 1, if no videos found,
                # immediately break to avoid wasting credits on dead channels
                if page_count == 1 and len(all_videos) == 0:
                    logger.info(
                        "Early exit: No videos found on page 1 for %s. Stopping to avoid credit waste.",
                        platform_id
                    )
                    break

                # Check for continuation token at root level
                continuation_token = response.get("continuationToken")
                if not continuation_token:
                    logger.debug(
                        "No more pages for handle %s", formatted_handle
                    )
                    break

            if not all_videos:
                logger.info(
                    "No YouTube videos found for account_id: %d", account_id
                )
                return

            # Limit to max_items
            all_videos = all_videos[:max_items]

            # Build author profile metadata once (reused for all content items)
            channel_data: dict[str, Any] = (
                self._cached_profile if self._cached_profile else {}
            )
            author_metadata = self._build_author_profile_metadata(channel_data)

            # Fetch transcripts concurrently with semaphore limit of 5
            transcript_map: dict[str, str | None] = {}
            if all_videos:
                logger.info(
                    "Fetching transcripts for %d videos with semaphore limit %d",
                    len(all_videos),
                    TRANSCRIPT_SEMAPHORE_LIMIT,
                )
                semaphore = asyncio.Semaphore(TRANSCRIPT_SEMAPHORE_LIMIT)
                tasks = [
                    self._fetch_transcript(video, semaphore)
                    for video in all_videos
                ]
                transcript_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Build map of video_id -> transcript text
                for video, result in zip(all_videos, transcript_results):
                    video_id = str(
                        video.get("id")
                        or video.get("videoId")
                        or video.get("video_id")
                        or ""
                    )
                    if isinstance(result, Exception):
                        logger.warning(
                            "Failed to fetch transcript for video %s: %s",
                            video_id,
                            result,
                        )
                        transcript_map[video_id] = None
                    else:
                        # result is str | None at this point (Exception already handled)
                        transcript_map[video_id] = result if isinstance(result, (str, type(None))) else None

            # Process and upsert content items
            await self._upsert_content(
                all_videos, account_id, author_metadata, transcript_map, handle
            )
            logger.info(
                "Successfully upserted %d YouTube content items for account_id: %d",
                len(all_videos),
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
        self, video: dict[str, Any], semaphore: asyncio.Semaphore
    ) -> str | None:
        """Fetch transcript for a single YouTube video.

        Implements a zero-cost hybrid pipeline:
        - Step 1: Try youtube-transcript-api library (free, no API credits).
          Runs synchronously via asyncio.to_thread to avoid blocking.
          Prioritizes Russian (ru) then English (en) transcripts.
        - Step 2: Fall back to Scrape Creators API (/v1/youtube/video/transcript)
          if the local extraction fails (TranscriptsDisabled, NoTranscriptFound,
          network errors, or HTTP 429 rate limits from Google).

        Args:
            video: Video dictionary containing the video ID.
            semaphore: asyncio.Semaphore to limit concurrent requests.

        Returns:
            Joined transcript text, or None if fetching fails or no transcript available.
        """
        video_id = str(
            video.get("id")
            or video.get("videoId")
            or video.get("video_id")
            or ""
        )
        if not video_id:
            return None

        url = f"https://www.youtube.com/watch?v={video_id}"

        # Transcript request log with required format
        logger.info("Requesting transcript for video: %s", url)

        async with semaphore:
            # Step 1: Free local extraction via youtube-transcript-api
            try:
                # Create API instance (proxies can be configured via env vars if needed)
                api = YouTubeTranscriptApi()

                # Run synchronous library call in a non-blocking thread
                # fetch() returns FetchedTranscript (iterable of segment objects)
                fetched_transcript = await asyncio.to_thread(
                    api.fetch,
                    video_id,
                    ["ru", "en"],
                )

                # Convert FetchedTranscript segments to text
                # Each segment is an object with .text, .start, .duration attributes
                transcript_text = " ".join(
                    str(segment.text) if hasattr(segment, "text") else str(segment)
                    for segment in fetched_transcript
                )

                if transcript_text.strip():
                    logger.info(
                        "Successfully retrieved transcript (free) for video %s",
                        url[:50],
                    )
                    return transcript_text.strip()

            except (TranscriptsDisabled, NoTranscriptFound) as e:
                logger.info(
                    "No transcript available for video %s: %s",
                    video_id,
                    e,
                )
                return None
            except Exception as e:
                # Catch rate limits (HTTP 429), connection errors, etc.
                logger.warning(
                    "Free transcript extraction failed for video %s: %s. "
                    "Falling back to Scrape Creators API.",
                    video_id,
                    e,
                )

            # Step 2: Defensive fallback to Scrape Creators API
            try:
                response: dict[str, Any] = await self.client.get(
                    endpoint="/v1/youtube/video/transcript",
                    params={"url": url},
                )

                # Extract transcript segments from response
                segments = response.get("transcript") or []
                if not segments:
                    logger.info("No transcript available for video: %s", url[:50])
                    return None

                # Join the "text" field of each segment with a space
                transcript_text = " ".join(
                    segment.get("text", "")
                    for segment in segments
                    if isinstance(segment, dict)
                )

                if transcript_text:
                    logger.info(
                        "Successfully retrieved transcript (Scrape Creators) for video %s",
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

    async def _upsert_content(
        self,
        videos: list[dict[str, Any]],
        account_id: int,
        author_metadata: dict[str, Any],
        transcript_map: dict[str, str | None],
        parent_handle: str,
    ) -> None:
        """Bulk upsert YouTube content records to database.

        Args:
            videos: List of YouTube video dictionaries from API responses.
            account_id: ID of the parent Account record.
            author_metadata: Author profile metadata to embed in each content record.
            transcript_map: Dictionary mapping video_id to transcript text.
            parent_handle: Channel handle for discovery spider parent reference.
        """
        content_values: list[dict[str, Any]] = []

        for video in videos:
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

                # Extract title and description, concatenate for content text
                title: str = video.get("title") or ""
                description: str | None = video.get("description") or video.get(
                    "videoDescription"
                )

                # Concatenate title and description for text column
                if description:
                    content_text: str = f"{title} {description}".strip()
                else:
                    content_text: str = title.strip()

                # Append transcript to content text if available
                transcript = transcript_map.get(platform_content_id)
                if transcript:
                    if content_text:
                        content_text = f"{content_text}\n\n[TRANSCRIPT]\n{transcript}"
                    else:
                        content_text = f"[TRANSCRIPT]\n{transcript}"

                if not content_text:
                    logger.warning(
                        "Skipping content item with no title, description, or transcript"
                    )
                    continue

                # Content-based Discovery Spider
                # Extract cross-platform links and same-platform mentions from video text
                if content_text:
                    try:
                        # Parse profile contacts (cross-platform links)
                        contacts_dict = parse_profile_contacts(content_text)
                        # Extract same-platform mentions (e.g., @otheruser)
                        mentions = extract_mentions(content_text)

                        # Queue discovered accounts in independent session
                        async with self.session_maker() as session:
                            # Cross-platform links
                            await queue_discovered_accounts(
                                session, contacts_dict, parent_handle
                            )
                            # Same-platform mentions
                            await queue_discovered_mentions(
                                session, "YOUTUBE", mentions, parent_handle
                            )
                            await session.commit()
                    except Exception as e:
                        logger.warning(
                            "Failed to queue discovered accounts from YouTube video %s: %s",
                            platform_content_id, e,
                        )

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
                raw_metadata: dict[str, Any] = {
                    "author_profile_metadata": author_metadata,
                    "platform_metrics": platform_metrics,
                }

                content_values.append(
                    {
                        "account_id": account_id,
                        "platform_content_id": platform_content_id,
                        "content": content_text,
                        "transcription": transcript,  # Store transcript in transcription column
                        "published_at": published_at,
                        "views": views,
                        "reactions_count": reactions_count,
                        "comments_count": comments_count,
                        "shares_count": shares_count,
                        "has_media": True,  # YouTube videos are media
                        "raw_metadata": raw_metadata,
                        "is_embedded": False,
                        "is_graph_extracted": False,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )

            except Exception as e:
                logger.error(
                    "Failed to process YouTube video item: %s",
                    e,
                    exc_info=True,
                )
                continue

        if not content_values:
            logger.info("No valid YouTube content items to upsert")
            return

        # Bulk upsert using PostgreSQL ON CONFLICT on uq_content_account_platform_id
        async with self.session_maker() as session:
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
                    raw_metadata=stmt.excluded.raw_metadata,
                    updated_at=stmt.excluded.updated_at,
                ),
            )
            await session.execute(stmt)
            await session.commit()
            logger.debug(
                "Bulk upserted %d YouTube content items for account_id: %d",
                len(content_values),
                account_id,
            )

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
        # parse_profile_contacts returns: {emails, telegram_handles, external_links, raw_bio}

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

    async def _upsert_account(
        self, channel: dict[str, Any], status: str = "parsed"
    ) -> int:
        """Upsert YouTube account record using select-then-insert/update pattern.

        Uses a select-then-upsert transaction pattern to avoid InvalidColumnReferenceError
        caused by missing unique constraint on (platform, platform_id) in the accounts table.

        Args:
            channel: Channel object from Scrape Creators API response.
            status: Account status ('parsed', 'rejected', etc.).

        Returns:
            ID of the account record (auto-generated by PostgreSQL for new records).
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
            # Select existing account by platform and platform_id
            stmt = select(Account).where(
                Account.platform == "YOUTUBE",
                Account.platform_id == platform_id,
            )
            result = await session.execute(stmt)
            db_account: Account | None = result.scalar_one_or_none()

            if db_account:
                # Update existing record
                db_account.username = username
                db_account.title = full_name or username or "Unknown"
                db_account.description = description
                db_account.subscribers_count = subscriber_count
                db_account.status = status
                db_account.updated_at = datetime.now(timezone.utc)
                logger.debug(
                    "Updated existing YouTube account %s (ID: %d, status: %s)",
                    username,
                    db_account.id,
                    status,
                )
            else:
                # Create new record (let PostgreSQL generate ID)
                db_account = Account(
                    platform="YOUTUBE",
                    platform_id=platform_id,
                    username=username,
                    title=full_name or username or "Unknown",
                    description=description,
                    subscribers_count=subscriber_count,
                    status=status,
                )
                session.add(db_account)
                logger.debug(
                    "Created new YouTube account %s (status: %s)",
                    username,
                    status,
                )

            await session.commit()
            await session.refresh(db_account)
            return db_account.id
