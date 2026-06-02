"""
Threads platform parser using Scrape Creators API v1.

Implements Threads-specific profile parsing and content ingestion into PostgreSQL.
Extracts author profile metadata and stores it inside Content.raw_metadata.

Features:
    - Profile parsing with account upsert to accounts table
    - Subscriber threshold enforcement (3,000 to 150,000 for micro-influencers)
    - Russian language (Cyrillic) biography check (only if biography is non-empty)
    - AI-slop / theme-page detection
    - Female creator detection with virtual profile post creation
    - Content fetching with pagination and bulk upsert to content table
    - PostgreSQL ON CONFLICT DO UPDATE for high-throughput concurrency
    - Cross-platform spidering queue for discovered accounts
    - Cross-platform seeding: Threads -> Instagram (identical usernames)
    - No transcript requests (Threads is text-based, content text is inline)
"""

import logging
from datetime import datetime, timezone
from typing import Any

from aiohttp import ClientResponseError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

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
    profile parsing and content upsert logic using the Scrape Creators API v1.

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
        thresholds (3k-150k), verifies Russian Cyrillic in biography (only if
        biography is non-empty), checks for AI-slop/theme-page content,
        detects female creators, and upserts the account information to the
        accounts table. After successful parse, seeds an Instagram account
        with the same username for cross-platform parsing.

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

        # Only reject on language if biography is non-empty
        # Empty biographies are allowed to pass - language will be evaluated
        # during parse_content when actual posts are analyzed
        if biography and not is_russian_text(biography):
            logger.info(
                "Threads handle %s has non-Russian biography. Rejecting.",
                handle,
            )
            await self._upsert_account(profile, "rejected")
            return None

        # Check for AI-slop/theme-page (always apply, even with empty biography)
        if is_slop_or_theme_page(username, biography):
            logger.info(
                "Threads handle %s appears to be AI-slop/theme-page. Rejecting.",
                handle,
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
            "Successfully parsed Threads profile %s, account ID: %d, subscribers: %d",
            handle, account_id, subscribers,
        )

        # Cross-platform seeding: Threads -> Instagram
        # Since usernames are often identical across Threads and Instagram,
        # seed an Instagram account for the same username
        if username:
            await self._seed_instagram_from_threads(username)

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
                await session.commit()  # Explicit commit after virtual bio post upsert

        # Queue discovered accounts from contacts (spidering)
        contacts_dict = parse_profile_contacts(biography, profile.get("external_url"))
        async with self.session_maker() as session:
            await queue_discovered_accounts(
                session=session,
                contacts_dict=contacts_dict,
                parent_handle=handle,
            )
            await session.commit()  # Explicit commit after discovered accounts queue

        return account_id

    async def parse_content(
        self,
        account_id: int,
        platform_id: str,
        max_items: int = 50,
    ) -> None:
        """Fetch Threads content with pagination and bulk upsert to content table.

        Retrieves content items (threads) for the given account using the
        Scrape Creators API v1 /v1/threads/user/posts endpoint, paginates using
        cursor until max_items are collected, and performs a bulk upsert
        into the content table using PostgreSQL ON CONFLICT DO UPDATE.

        Threads are text-based: content text is extracted directly from the
        post payload (text or caption.text) with no transcript requests.

        Args:
            account_id: Database ID of the parent account record.
            platform_id: Threads user ID (numerical platform ID) stored in the database.
            max_items: Maximum number of content items to fetch (default: 50).
        """
        logger.info(
            "Starting Threads content parse for account_id: %d, platform_id: %s",
            account_id,
            platform_id,
        )

        collected_posts: list[dict[str, Any]] = []
        cursor: str | None = None
        last_response: dict[str, Any] = {}

        # Paginate through API responses until max_items collected
        while len(collected_posts) < max_items:
            params: dict[str, Any] = {"handle": platform_id}
            if cursor:
                params["cursor"] = cursor
            # Request remaining items needed, capped at reasonable page size
            remaining = max_items - len(collected_posts)
            params["limit"] = min(remaining, 100)  # API page size limit if any

            try:
                response = await self.client.get(
                    endpoint="/v1/threads/user/posts",
                    params=params,
                )
                logger.info(
                    "API response for content, platform_id %s: success, remaining credits: %s",
                    platform_id,
                    response.get("credits_remaining", "N/A"),
                )
            except Exception as e:
                logger.error(
                    "API request failed for Threads content, platform_id %s: %s",
                    platform_id,
                    e,
                    exc_info=True,
                )
                break

            # Store last successful response for author metadata extraction
            last_response = response

            # Extract posts directly from root response (no "data" wrapper per v1 API spec)
            posts = self._extract_posts_from_response(response)
            if not posts:
                logger.info("No more posts in current page for platform_id %s", platform_id)
                break

            collected_posts.extend(posts)
            logger.debug(
                "Collected %d posts so far for account_id %d",
                len(collected_posts),
                account_id,
            )

            # Check if we have enough posts
            if len(collected_posts) >= max_items:
                collected_posts = collected_posts[:max_items]
                break

            # Get next cursor for pagination directly from root response
            cursor = response.get("next_cursor") or response.get("cursor")
            if not cursor:
                logger.debug("No more pages for platform_id %s", platform_id)
                break

        if not collected_posts:
            logger.info("No Threads content found for account_id: %d", account_id)
            return

        # Limit to max_items (redundant but safe)
        collected_posts = collected_posts[:max_items]

        # Build author profile metadata from last response's user/author data
        user_data: dict[str, Any] = last_response.get("user") or last_response.get("author", {})
        username: str | None = user_data.get("username") or user_data.get("handle")
        biography: str | None = user_data.get("biography") or user_data.get("bio")

        # Parse contacts from biography and external_url
        external_url: str | None = user_data.get("external_url")
        contacts_dict: dict[str, Any] = parse_profile_contacts(biography, external_url)

        # Compile author metadata using core helper
        author_metadata = compile_author_metadata(
            platform="THREADS",
            username=username,
            biography=biography,
            contacts_dict=contacts_dict,
            location=user_data.get("location") or user_data.get("address"),
        )

        # Process and upsert content items
        await self._upsert_content(collected_posts, account_id, author_metadata, platform_id)
        logger.info(
            "Successfully upserted %d Threads content items for account_id: %d",
            len(collected_posts),
            account_id,
        )

    async def _get_cached_or_fetch_profile(self, handle: str) -> dict[str, Any] | None:
        """Get profile from cache or fetch from API.

        Uses instance-level caching to avoid duplicate API calls when
        subsequently calling parse_content() for the same handle.

        Catches ClientResponseError and logs a quiet warning for 404s
        without traceback (exc_info=True is not used for 404s).

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
                endpoint="/v1/threads/profile",
                params={"handle": handle},
            )
            # Threads v1 API returns profile fields directly at root level
            # Treat entire response as profile data
            self._cached_profile = response
            self._cached_handle = handle
            return response

        except ClientResponseError as e:
            if e.status == 404:
                logger.warning("Threads profile %s not found (404). Marking as rejected.", handle)
            else:
                logger.error(
                    "Threads API request failed for %s with HTTP %d: %s",
                    handle, e.status, e,
                )
            return None
        except Exception as e:
            logger.error(
                "Unexpected error fetching Threads profile %s: %s",
                handle, e, exc_info=True,
            )
            return None

    async def _seed_instagram_from_threads(self, username: str) -> None:
        """Seed an Instagram account from a successfully parsed Threads profile.

        Performs a select-then-insert database transaction to check if an Account
        on platform "INSTAGRAM" with the given username already exists. If it does
        not exist, inserts a new Account row with status "pending".

        This is wrapped in a try-except block to ensure that database lockups or
        unique constraint violations never interrupt the primary Threads parsing.

        Args:
            username: The username to seed on Instagram platform.
        """
        try:
            async with self.session_maker() as session:
                # Select: check if Instagram account already exists
                stmt = select(Account).where(
                    Account.platform == "INSTAGRAM",
                    Account.username == username,
                )
                result = await session.execute(stmt)
                existing_account = result.scalar_one_or_none()

                if existing_account:
                    logger.debug(
                        "Instagram account already exists for username: %s. Skipping seed.",
                        username,
                    )
                    return

                # Insert: create new Instagram account seeded from Threads
                instagram_account = Account(
                    platform="INSTAGRAM",
                    platform_id=username,  # Use username as platform_id for Instagram
                    username=username,
                    title=username,
                    description=None,
                    subscribers_count=0,
                    status="pending",
                )
                session.add(instagram_account)
                await session.commit()
                logger.info(
                    "Seeded Instagram account for username: %s with status 'pending'",
                    username,
                )

        except Exception as e:
            # Catch all exceptions to prevent interrupting primary Threads parsing
            # This includes database lockups, unique constraint violations, etc.
            logger.warning(
                "Failed to seed Instagram account for username %s: %s. "
                "Continuing with Threads parsing.",
                username,
                e,
                exc_info=True,
            )

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
        parent_handle: str,
    ) -> None:
        """Bulk upsert Threads content records to database.

        Uses PostgreSQL-specific ON CONFLICT DO UPDATE on constraint
        uq_content_account_platform_id for high-throughput concurrency.

        Args:
            posts: List of post dictionaries from API responses.
            account_id: ID of the parent Account record.
            author_metadata: Author profile metadata to embed in each content record.
            parent_handle: Threads user ID for discovery spider parent reference.
        """
        content_values: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)

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

                # Extract content text (inline, no transcript)
                content_text: str | None = self._extract_content_text(post)

                # Content-based Discovery Spider
                # Extract cross-platform links and same-platform mentions from post text
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
                                session, "THREADS", mentions, parent_handle
                            )
                            await session.commit()
                    except Exception as e:
                        logger.warning(
                            "Failed to queue discovered accounts from Threads post %s: %s",
                            platform_content_id, e,
                        )

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
                    "created_at": now,  # Set created_at for new records
                    "views": None,
                    "reactions_count": likes,
                    "comments_count": replies,
                    "shares_count": None,
                    "has_media": has_media,
                    "is_embedded": False,
                    "is_graph_extracted": False,
                    "raw_metadata": raw_metadata,
                    "updated_at": now,
                })

            except Exception as e:
                logger.error("Failed to parse Threads content item: %s", e, exc_info=True)
                continue

        if not content_values:
            logger.warning("No valid content items to upsert for account_id: %d", account_id)
            return

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
                    has_media=stmt.excluded.has_media,
                    is_embedded=stmt.excluded.is_embedded,
                    is_graph_extracted=stmt.excluded.is_graph_extracted,
                    raw_metadata=stmt.excluded.raw_metadata,
                    updated_at=stmt.excluded.updated_at,
                ),
            )
            await session.execute(stmt)
            await session.commit()
            logger.debug(
                "Upserted %d Threads content records for account ID %d",
                len(content_values),
                account_id,
            )

    def _extract_posts_from_response(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract posts list from root API response.

        Searches for posts in top-level keys: "posts", "threads", "items"
        to handle potential API response variations.

        Args:
            response: Root response dictionary from Scrape Creators API.

        Returns:
            List of post dictionaries, empty if no valid posts found.
        """
        posts: list[dict[str, Any]] = []

        # Try different possible top-level response structures
        if "posts" in response and isinstance(response["posts"], list):
            posts = response["posts"]
        elif "threads" in response and isinstance(response["threads"], list):
            posts = response["threads"]
        elif "items" in response and isinstance(response["items"], list):
            posts = response["items"]

        return posts

    def _extract_followers_count(self, user: dict[str, Any]) -> int:
        """Extract follower count from Threads user data.

        Args:
            user: User object from Threads API response.

        Returns:
            Follower count as integer, or 0 if not found.
        """
        # Try standard follower fields at root level
        followers = (
            user.get("follower_count")
            or user.get("followerCount")
            or user.get("followers")
            or user.get("followers_count")
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

        Prioritizes top-level "text" field, then nested "caption.text"
        (where caption is a dictionary), then falls back to other text fields.

        Args:
            post: Post dictionary from API response.

        Returns:
            Extracted text content, or None if not found.
        """
        # Check top-level "text" field first
        text: str | None = post.get("text")
        if text and isinstance(text, str):
            return text.strip()

        # Check nested caption.text (caption is a dict with "text" key)
        caption: dict[str, Any] | None = post.get("caption")
        if isinstance(caption, dict):
            caption_text: str | None = caption.get("text")
            if caption_text and isinstance(caption_text, str):
                return caption_text.strip()

        # Fallback to other common text fields
        text = post.get("content") or post.get("post_text")
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
