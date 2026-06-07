"""
Threads platform parser using Scrape Creators API v1.

Implements Threads-specific profile parsing and content ingestion into PostgreSQL.
Extracts author profile metadata and stores it inside Content.raw_metadata.

Features:
    - Profile parsing with account upsert to accounts table (using centralized db helpers)
    - Subscriber threshold enforcement (3,000 to 150,000 for micro-influencers)
    - Russian language (Cyrillic) biography check (only if biography is non-empty)
    - AI-slop / theme-page detection
    - Single-shot content fetching (no pagination loops, exactly one API call)
    - PostgreSQL ON CONFLICT DO UPDATE for high-throughput concurrency
    - Cross-platform spidering queue for discovered accounts
    - Cross-platform seeding: Threads -> Instagram (identical usernames)
    - No transcript requests (Threads is text-based, content text is inline)
"""

import logging
from datetime import datetime, timezone
from typing import Any

from aiohttp import ClientResponseError

from src.parser.creators.core.db import (
    upsert_and_deduplicate_account,
    update_account_profile_metadata,
    upsert_virtual_bio_post,
    bulk_upsert_content,
    queue_discovered_accounts,
    queue_discovered_mentions,
)
from src.parser.creators.core.utils import (
    is_russian_text,
    is_slop_or_theme_page,
    parse_profile_contacts,
    parse_published_at,
    extract_mentions,
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
    Uses centralized database helpers from src.parser.creators.core.db.

    Attributes:
        session_maker: SQLAlchemy async session maker for database operations.
        client: ScrapeCreatorsClient instance for API requests.
        settings: Application settings containing configuration values.
        _cached_profile: Instance-level cache for profile response data.
        _cached_handle: The handle for which profile data is cached.
    """

    def __init__(
        self,
        session_maker: Any,
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
        and upserts the account information to the accounts table using
        centralized db helpers. After successful parse, seeds an Instagram
        account with the same username for cross-platform parsing.

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
            # Update database with rejected status using centralized helper
            async with self.session_maker() as session:
                await upsert_and_deduplicate_account(
                    session=session,
                    platform="THREADS",
                    platform_id=str(profile.get("id", handle)),
                    username=profile.get("username", ""),
                    title=profile.get("full_name") or profile.get("username", "Unknown"),
                    description=profile.get("biography") or profile.get("bio"),
                    subscribers_count=subscribers,
                    status="rejected",
                )
                await session.commit()
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
            # Update database with rejected status using centralized helper
            async with self.session_maker() as session:
                await upsert_and_deduplicate_account(
                    session=session,
                    platform="THREADS",
                    platform_id=str(profile.get("id", handle)),
                    username=username,
                    title=profile.get("full_name") or username or "Unknown",
                    description=biography,
                    subscribers_count=subscribers,
                    status="rejected",
                )
                await session.commit()
            return None

        # Check for AI-slop/theme-page (always apply, even with empty biography)
        if is_slop_or_theme_page(username, biography):
            logger.info(
                "Threads handle %s appears to be AI-slop/theme-page. Rejecting.",
                handle,
            )
            # Update database with rejected status using centralized helper
            async with self.session_maker() as session:
                await upsert_and_deduplicate_account(
                    session=session,
                    platform="THREADS",
                    platform_id=str(profile.get("id", handle)),
                    username=username,
                    title=profile.get("full_name") or username or "Unknown",
                    description=biography,
                    subscribers_count=subscribers,
                    status="rejected",
                )
                await session.commit()
            return None

        # Upsert account with "processing" status using centralized helper
        async with self.session_maker() as session:
            account_id = await upsert_and_deduplicate_account(
                session=session,
                platform="THREADS",
                platform_id=str(profile.get("id", handle)),
                username=username,
                title=profile.get("full_name") or username or "Unknown",
                description=biography,
                subscribers_count=subscribers,
                status="processing",
            )
            await session.commit()

        logger.info(
            "Successfully parsed Threads profile %s, account ID: %d, subscribers: %d",
            handle,
            account_id,
            subscribers,
        )

        # Update biography and trigger cross-platform queue discovery
        async with self.session_maker() as session:
            await update_account_profile_metadata(
                session=session,
                account_id=account_id,
                platform="THREADS",
                biography=biography,
                external_url=profile.get("external_url"),
            )
            # Upsert virtual bio post for semantic search
            await upsert_virtual_bio_post(
                session=session,
                account_id=account_id,
                platform="THREADS",
                platform_id=str(profile.get("id", handle)),
                username=username,
                full_name=profile.get("full_name"),
                biography=biography,
                subscribers_count=subscribers,
                raw_metadata=None,
            )
            await session.commit()

        # Cross-platform seeding: Threads -> Instagram
        # Since usernames are often identical across Threads and Instagram,
        # seed an Instagram account for the same username
        if username:
            await self._seed_instagram_from_threads(username)

        # Update account status to "parsed" using centralized helper
        async with self.session_maker() as session:
            await upsert_and_deduplicate_account(
                session=session,
                platform="THREADS",
                platform_id=str(profile.get("id", handle)),
                username=username,
                title=profile.get("full_name") or username or "Unknown",
                description=biography,
                subscribers_count=subscribers,
                status="parsed",
            )
            await session.commit()

        return account_id

    async def parse_content(
        self,
        account_id: int,
        platform_id: str,
        max_items: int = 50,
    ) -> None:
        """Fetch Threads content with single API call and bulk upsert to content table.

        Retrieves content items (threads) for the given account using the
        Scrape Creators API v1 /v1/threads/user/posts endpoint. Performs exactly
        ONE API call with limit=50 (no pagination loops). Processes and bulk-upserts
        the content using centralized db helpers.

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

        # Single API call with limit=50 - no pagination loops
        params: dict[str, Any] = {"handle": platform_id, "limit": min(max_items, 50)}

        try:
            response = await self.client.get(
                endpoint="/v1/threads/user/posts",
                params=params,
            )
            logger.debug(
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
            return

        # Extract posts directly from root response (no "data" wrapper per v1 API spec)
        posts = self._extract_posts_from_response(response)
        if not posts:
            logger.info("No posts found for platform_id %s", platform_id)
            return

        logger.info(
            "Fetched %d posts for platform_id %s (single API call, no pagination)",
            len(posts),
            platform_id,
        )

        # Process posts and prepare for bulk upsert
        content_values: list[dict[str, Any]] = []
        aggregated_contacts: dict[str, list[str]] = {}
        aggregated_mentions: set[str] = set()

        for post in posts[:max_items]:
            post_id: str = str(
                post.get("id") or post.get("post_id") or post.get("pk") or ""
            )
            if not post_id:
                continue

            # Extract content text
            content_text: str = self._extract_post_text(post)
            if not content_text:
                continue

            # Extract published timestamp
            published_at = parse_published_at(
                post.get("timestamp") or post.get("created_at") or post.get("taken_at")
            )

            # Content-based discovery spider
            if content_text:
                # Parse profile contacts (cross-platform links)
                contacts_dict = parse_profile_contacts(content_text)
                # Extract same-platform mentions (e.g., @otheruser)
                mentions = extract_mentions(content_text)

                # Accumulate contacts
                for platform, handles in contacts_dict.items():
                    if platform not in aggregated_contacts:
                        aggregated_contacts[platform] = []
                    for handle in handles:
                        if handle not in aggregated_contacts[platform]:
                            aggregated_contacts[platform].append(handle)

                # Accumulate mentions
                aggregated_mentions.update(mentions)

            # Build raw_metadata
            raw_metadata: dict[str, Any] = {
                "platform_metrics": {
                    "likes": post.get("like_count") or post.get("likes"),
                    "replies": post.get("reply_count") or post.get("replies"),
                    "reposts": post.get("repost_count") or post.get("reposts"),
                    "quotes": post.get("quote_count") or post.get("quotes"),
                },
            }

            content_values.append(
                {
                    "account_id": account_id,
                    "platform_content_id": post_id,
                    "content": content_text,
                    "published_at": published_at,
                    "views": post.get("view_count") or post.get("views"),
                    "reactions_count": post.get("like_count") or post.get("likes"),
                    "comments_count": post.get("reply_count") or post.get("replies"),
                    "shares_count": post.get("repost_count") or post.get("reposts"),
                    "has_media": bool(post.get("media") or post.get("media_url")),
                    "is_embedded": False,
                    "is_graph_extracted": False,
                    "raw_metadata": raw_metadata,
                    "updated_at": datetime.now(timezone.utc),
                }
            )

        # Bulk write all accumulated discovery data
        if aggregated_contacts or aggregated_mentions:
            try:
                async with self.session_maker() as session:
                    if aggregated_contacts:
                        await queue_discovered_accounts(
                            session=session,
                            metadata=aggregated_contacts,
                            parent_handle=platform_id,
                            status="pending",
                        )
                    if aggregated_mentions:
                        await queue_discovered_mentions(
                            session=session,
                            platform="THREADS",
                            mentions=list(aggregated_mentions),
                            parent_handle=platform_id,
                            status="pending",
                        )
                    await session.commit()
            except Exception as e:
                logger.warning(
                    "Failed to queue discovered accounts from Threads content: %s",
                    e,
                )

        # Bulk upsert content using centralized function
        if content_values:
            async with self.session_maker() as session:
                await bulk_upsert_content(
                    session=session,
                    content_values=content_values,
                )
                await session.commit()
                logger.info(
                    "Bulk upserted %d Threads content items for account_id: %d",
                    len(content_values),
                    account_id,
                )

    def _extract_posts_from_response(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract posts list from API response with multiple fallback strategies.

        Args:
            response: API response dictionary.

        Returns:
            List of post dictionaries.
        """
        # Try different possible response structures
        posts = (
            response.get("data") or
            response.get("posts") or
            response.get("items") or
            []
        )
        if isinstance(posts, list):
            return posts
        return []

    def _extract_post_text(self, post: dict[str, Any]) -> str:
        """Extract text content from a Threads post.

        Args:
            post: Threads post dictionary.

        Returns:
            Extracted text content as string.
        """
        # Try different possible field names for post text
        text = (
            post.get("text") or
            post.get("caption") or
            post.get("content") or
            ""
        )
        if isinstance(text, dict):
            # Handle nested caption object
            return text.get("text", "")
        return str(text) if text else ""

    async def _get_cached_or_fetch_profile(
        self, handle: str
    ) -> dict[str, Any] | None:
        """Get profile from cache or fetch from API.

        Args:
            handle: Threads username to fetch.

        Returns:
            Profile data dictionary, or None if fetch fails.
        """
        if self._cached_handle == handle and self._cached_profile:
            logger.debug("Using cached profile data for handle: %s", handle)
            return self._cached_profile

        try:
            response = await self.client.get(
                endpoint="/v1/threads/user/profile",
                params={"handle": handle},
            )
            data = response.get("data") or response
            if not data:
                logger.error(
                    "Missing data in API response for Threads handle %s",
                    handle,
                )
                return None

            user = data.get("user") or data
            if not user:
                logger.error(
                    "Missing user data for Threads handle %s", handle
                )
                return None

            # Cache the profile
            self._cached_profile = user
            self._cached_handle = handle
            return user

        except ClientResponseError as e:
            if e.status == 404:
                logger.warning(
                    "Threads profile %s not found (404). Marking as rejected.",
                    handle,
                )
            else:
                logger.error(
                    "Threads API request failed for %s with HTTP %d: %s",
                    handle,
                    e.status,
                    e,
                )
            return None
        except Exception as e:
            logger.error(
                "Unexpected error fetching Threads profile %s: %s",
                handle,
                e,
                exc_info=True,
            )
            return None

    def _extract_followers_count(self, profile: dict[str, Any]) -> int:
        """Extract follower count from Threads profile data.

        Args:
            profile: Threads profile data dictionary.

        Returns:
            Follower count as integer (0 if not found).
        """
        followers = (
            profile.get("follower_count") or
            profile.get("followers") or
            profile.get("followers_count") or
            0
        )
        try:
            return int(followers)
        except (ValueError, TypeError):
            return 0

    async def _seed_instagram_from_threads(self, username: str) -> None:
        """Seed an Instagram account with the same username for cross-platform parsing.

        Args:
            username: Username to seed on Instagram platform.
        """
        if not username:
            return

        try:
            async with self.session_maker() as session:
                await upsert_and_deduplicate_account(
                    session=session,
                    platform="INSTAGRAM",
                    platform_id=username,
                    username=username,
                    title=username,
                    description=None,
                    subscribers_count=None,
                    status="pending",
                )
                await session.commit()
                logger.info(
                    "Cross-platform seed: Threads username %s -> Instagram (pending)",
                    username,
                )
        except Exception as e:
            logger.warning(
                "Failed to seed Instagram account from Threads username %s: %s",
                username,
                e,
            )
