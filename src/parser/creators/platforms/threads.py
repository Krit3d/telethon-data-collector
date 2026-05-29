"""
Threads platform parser using Scrape Creators API.

Implements Threads-specific profile parsing and content ingestion into PostgreSQL.
Extracts author profile metadata and stores it inside Content.raw_metadata.

Features:
    - Profile parsing with account upsert to accounts table
    - Subscriber threshold enforcement (3,000 to 150,000 for micro-influencers)
    - Russian language (Cyrillic) biography check
    - AI-slop / theme-page detection
    - Female creator detection with virtual profile post creation
    - Content fetching and bulk upsert to content table
    - PostgreSQL ON CONFLICT DO UPDATE for high-throughput concurrency
    - Cross-platform spidering queue for discovered accounts
    - Transcription set to None (Threads is primarily text-based)
"""

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

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
from src.config.config import Settings

logger = logging.getLogger(__name__)

# Subscriber thresholds for micro-influencers
MIN_SUBSCRIBERS: int = 3000
MAX_SUBSCRIBERS: int = 150000


class ThreadsParser(BasePlatformParser):
    """Threads platform parser for profile and content ingestion.

    Inherits from BasePlatformParser and implements Threads-specific
    profile parsing and content upsert logic using the Scrape Creators API.

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
        """Initialize Threads parser with configuration."""
        super().__init__(session_maker, client, settings)
        self._cached_profile: dict[str, Any] | None = None
        self._cached_handle: str | None = None

    async def parse_profile(self, handle: str) -> int | None:
        """Fetch Threads profile, apply filters, upsert account.

        Parses the Threads profile for the given handle, checks subscriber
        thresholds (3k-150k), verifies Russian Cyrillic in biography,
        checks for AI-slop/theme-page content, detects female creators,
        and upserts the account information to the accounts table.

        Args:
            handle: Threads username (without @ prefix).

        Returns:
            Database ID of the upserted account record, or None if the profile
            could not be parsed, doesn't meet criteria, or is rejected.
        """
        logger.info("Starting Threads profile parse for handle: %s", handle)

        profile = await self._get_cached_or_fetch_profile(handle)
        if not profile:
            return None

        # Subscriber threshold filter (micro-influencers: 3k-150k)
        subscribers = self._extract_followers_count(profile)
        if not (MIN_SUBSCRIBERS <= subscribers <= MAX_SUBSCRIBERS):
            logger.info(
                "Threads handle %s has %d subscribers, outside range [%d, %d]. Rejecting.",
                handle, subscribers, MIN_SUBSCRIBERS, MAX_SUBSCRIBERS,
            )
            await self._upsert_account(profile, "rejected")
            return None

        # Content quality filters
        username = profile.get("username", "")
        biography = profile.get("biography", "") or profile.get("bio", "")
        if not is_russian_text(biography) or is_slop_or_theme_page(username, biography):
            logger.info("Threads handle %s failed content filters. Rejecting.", handle)
            await self._upsert_account(profile, "rejected")
            return None

        # Female creator detection
        is_female = detect_female_creator(biography)
        if is_female:
            logger.info("Female creator detected: %s", handle)

        # Upsert parsed account
        account_id = await self._upsert_account(profile, "parsed")
        logger.info(
            "Successfully parsed Threads profile %s, account ID: %d, subscribers: %d",
            handle, account_id, subscribers,
        )

        # Post-parse actions: upsert virtual profile post for female creators
        if is_female:
            async with self.session_maker() as session:
                await upsert_virtual_bio_post(
                    session=session,
                    account_id=account_id,
                    platform="THREADS",
                    platform_id=profile.get("id", handle),
                    username=username,
                    full_name=profile.get("full_name") or profile.get("name"),
                    biography=biography,
                    subscribers_count=subscribers,
                    raw_metadata={"female_heuristic": True},
                )

        # Queue discovered accounts from contacts (spidering)
        contacts_dict = parse_profile_contacts(biography, profile.get("external_url"))
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
        """Fetch Threads content and bulk upsert to content table.

        Retrieves content items (threads) for the given account using the
        Scrape Creators API, parses the data, and performs a bulk upsert
        into the content table using PostgreSQL ON CONFLICT DO UPDATE.

        Threads are primarily text posts. Transcription is set to None.
        The has_media flag is set based on presence of image/video URLs.

        Args:
            account_id: Database ID of the parent account record.
            platform_id: Numerical platform ID (Threads user ID) stored in the database.
            max_items: Maximum number of content items to fetch (default: 50).
        """
        logger.info(
            "Starting Threads content parse for account_id: %d, platform_id: %s",
            account_id,
            platform_id,
        )

        # Resolve handle (username) from database using account_id
        async with self.session_maker() as session:
            result = await session.execute(
                select(Account.username).where(Account.id == account_id)
            )
            db_username = result.scalar_one_or_none()

        if db_username:
            handle = db_username
            logger.debug("Resolved handle %s for account_id %d", handle, account_id)
        else:
            logger.warning(
                "Username not found for account_id %d, falling back to platform_id %s as handle",
                account_id,
                platform_id,
            )
            handle = platform_id

        # Fetch profile using resolved handle
        profile = await self._get_cached_or_fetch_profile(handle)
        if not profile:
            return

        # Fetch content data from Scrape Creators API
        try:
            response = await self.client.get(
                endpoint="/v2/threads/posts",
                params={"handle": handle, "limit": max_items},
            )
            logger.info(
                "API response status for content, handle %s: success, credits consumed: %s",
                handle,
                response.get("credits", "N/A"),
            )
        except Exception as e:
            logger.error(
                "API request failed for Threads content, handle %s: %s",
                handle,
                e,
                exc_info=True,
            )
            return

        # Validate response structure
        data = response.get("data")
        if not data:
            logger.error(
                "Missing 'data' in API response for Threads content, handle %s",
                handle,
            )
            return

        # Extract posts/threads from response
        posts = self._extract_posts_from_response(data)
        if not posts:
            logger.info("No Threads content found for account_id: %d", account_id)
            return

        # Limit to max_items
        posts = posts[:max_items]

        # Build author profile metadata using core helper
        user_data = data.get("user") or data.get("author") or profile
        username: str | None = user_data.get("username") or user_data.get("handle")
        biography: str | None = user_data.get("biography") or user_data.get("bio")

        # Parse contacts from biography and external_url
        external_url: str | None = user_data.get("external_url")
        contacts_dict: dict[str, Any] = parse_profile_contacts(biography, external_url)

        # Use core helper to compile author metadata
        author_metadata = compile_author_metadata(
            platform="THREADS",
            username=username,
            biography=biography,
            contacts_dict=contacts_dict,
            location=user_data.get("location") or user_data.get("address"),
        )

        # Process and upsert content items
        await self._upsert_content(posts, account_id, author_metadata)
        logger.info(
            "Successfully upserted %d Threads content items for account_id: %d",
            len(posts),
            account_id,
        )

    async def _get_cached_or_fetch_profile(self, handle: str) -> dict[str, Any] | None:
        """Get profile from cache or fetch from API.

        Uses instance-level caching to avoid duplicate API calls when
        subsequently calling parse_content() for the same handle.

        Args:
            handle: Threads username to fetch.

        Returns:
            Profile data dictionary, or None if fetch failed.
        """
        if self._cached_handle == handle and self._cached_profile:
            logger.debug("Using cached profile data for handle: %s", handle)
            return self._cached_profile

        try:
            response = await self.client.get(
                endpoint="/v2/threads/profile",
                params={"handle": handle},
            )
            data = response.get("data")
            if not data:
                logger.error("Missing 'data' in API response for Threads handle %s", handle)
                return None

            # Threads API may return user data at different levels
            user = data.get("user") or data.get("author") or data
            if not user:
                logger.error("Missing user data for Threads handle %s", handle)
                return None

            # Cache the profile
            self._cached_profile = user
            self._cached_handle = handle
            return user

        except Exception as e:
            logger.error("API request failed for Threads profile %s: %s", handle, e, exc_info=True)
            return None

    async def _upsert_account(
        self, user: dict[str, Any], status: str = "parsed"
    ) -> int:
        """Upsert Threads account record using select-then-insert/update pattern.

        Args:
            user: User object from Scrape Creators API response.
            status: Account status ('parsed', 'rejected', etc.).

        Returns:
            ID of the account record.
        """
        platform_id: str = str(
            user.get("id") or user.get("pk") or user.get("user_id", "")
        )
        username: str | None = (
            user.get("username")
            or user.get("handle")
            or user.get("shortname")
        )
        full_name: str | None = (
            user.get("full_name")
            or user.get("name")
            or user.get("display_name")
        )
        biography: str | None = user.get("biography") or user.get("bio")
        subscribers: int = self._extract_followers_count(user)

        async with self.session_maker() as session:
            stmt = select(Account).where(
                Account.platform == "THREADS",
                Account.platform_id == platform_id,
            )
            result = await session.execute(stmt)
            db_account = result.scalar_one_or_none()

            if db_account:
                db_account.username = username
                db_account.title = full_name or username or "Unknown"
                db_account.description = biography
                db_account.subscribers_count = subscribers
                db_account.status = status
                db_account.updated_at = datetime.now(timezone.utc)
            else:
                db_account = Account(
                    platform="THREADS",
                    platform_id=platform_id,
                    username=username,
                    title=full_name or username or "Unknown",
                    description=biography,
                    subscribers_count=subscribers,
                    status=status,
                )
                session.add(db_account)

            await session.commit()
            await session.refresh(db_account)
            return db_account.id

    async def _upsert_content(
        self,
        posts: list[dict[str, Any]],
        account_id: int,
        author_metadata: dict[str, Any],
    ) -> None:
        """Bulk upsert Threads content records to database.

        Args:
            posts: List of post dictionaries from API responses.
            account_id: ID of the parent Account record.
            author_metadata: Author profile metadata to embed in each content record.
        """
        content_values: list[dict[str, Any]] = []

        for post in posts:
            try:
                # Extract platform_content_id from post
                platform_content_id: str = str(
                    post.get("id")
                    or post.get("post_id")
                    or post.get("thread_id")
                    or post.get("shortcode", "")
                )
                if not platform_content_id:
                    logger.warning("Skipping content item with no ID")
                    continue

                # Extract content text
                content_text: str | None = self._extract_content_text(post)

                # Extract published timestamp using core helper
                timestamp = (
                    post.get("taken_at_timestamp")
                    or post.get("timestamp")
                    or post.get("created_at")
                    or post.get("published_at")
                    or post.get("created_time")
                )
                published_at: datetime = parse_published_at(timestamp)

                # Extract engagement metrics
                likes: int | None = (
                    post.get("like_count")
                    or post.get("likes")
                    or post.get("reactions_count")
                )
                replies: int | None = (
                    post.get("reply_count")
                    or post.get("replies")
                    or post.get("comments_count")
                )

                # Check if post has media (image or video)
                has_media: bool = self._has_media(post)

                # Build platform metrics for raw_metadata
                platform_metrics: dict[str, Any] = {
                    "likes": likes,
                    "replies": replies,
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
                    "transcription": None,  # Threads is primarily text-based
                    "published_at": published_at,
                    "views": None,
                    "reactions_count": likes,
                    "comments_count": replies,
                    "shares_count": None,
                    "has_media": has_media,
                    "is_embedded": False,
                    "is_graph_extracted": False,
                    "raw_metadata": raw_metadata,
                    "updated_at": datetime.now(timezone.utc),
                })

            except Exception as e:
                logger.error("Failed to parse Threads content item: %s", e, exc_info=True)
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
                    "Upserted %d Threads content records for account ID %d",
                    len(content_values),
                    account_id,
                )

    def _extract_posts_from_response(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract posts list from API response data.

        Args:
            data: Response data dictionary.

        Returns:
            List of post dictionaries.
        """
        posts: list[dict[str, Any]] = []

        # Try different possible response structures
        if "posts" in data and isinstance(data["posts"], list):
            posts = data["posts"]
        elif "threads" in data and isinstance(data["threads"], list):
            posts = data["threads"]
        elif "items" in data and isinstance(data["items"], list):
            posts = data["items"]

        return posts

    def _extract_followers_count(self, user: dict[str, Any]) -> int:
        """Extract follower count from Threads user data.

        Args:
            user: User object from Threads API response.

        Returns:
            Follower count as integer, or 0 if not found.
        """
        # Try standard follower fields
        followers = (
            user.get("followers")
            or user.get("followers_count")
            or user.get("follower_count")
        )
        if followers is not None:
            try:
                return int(followers)
            except (ValueError, TypeError):
                pass

        # Try edge_followed_by.count (GraphQL structure)
        edge_followed_by = user.get("edge_followed_by")
        if isinstance(edge_followed_by, dict):
            count = edge_followed_by.get("count")
            if count is not None:
                try:
                    return int(count)
                except (ValueError, TypeError):
                    pass

        return 0

    def _extract_content_text(self, post: dict[str, Any]) -> str | None:
        """Extract text content from Threads post.

        Args:
            post: Post dictionary from API response.

        Returns:
            Extracted text content, or None if not found.
        """
        text: str | None = (
            post.get("text")
            or post.get("content")
            or post.get("caption")
            or post.get("post_text")
        )
        if text and isinstance(text, str):
            return text.strip()
        return None

    def _has_media(self, post: dict[str, Any]) -> bool:
        """Check if the post has any media (image or video).

        Args:
            post: Post dictionary from API response.

        Returns:
            True if the post has media, False otherwise.
        """
        return bool(
            post.get("thumbnail_url")
            or post.get("display_url")
            or post.get("image_url")
            or post.get("video_url")
            or post.get("is_video")
            or post.get("media_type") in ("IMAGE", "VIDEO", 1, 2)
        )
