"""
TikTok platform parser using Scrape Creators API.

Handles TikTok profile parsing and video content ingestion into PostgreSQL.
Implements the BasePlatformParser interface for TikTok-specific data extraction.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config.config import Settings
from src.db.models import Account, Content
from src.parser.creators.platforms.base import BasePlatformParser
from src.parser.creators.sc_client import ScrapeCreatorsClient

logger = logging.getLogger(__name__)


class TikTokParser(BasePlatformParser):
    """TikTok platform parser for profile and content ingestion.

    Inherits from BasePlatformParser and implements TikTok-specific
    profile parsing and video content upsert logic using the Scrape Creators API.

    Features:
        - Profile parsing with account upsert to accounts table
        - Video content pagination and bulk upsert to content table
        - PostgreSQL ON CONFLICT DO UPDATE for high-throughput concurrency
        - Configurable minimum follower threshold
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
        self.max_videos: int = 50  # Maximum number of videos to collect per profile
        # Minimum followers threshold - can be configured via settings
        self.min_followers: int = getattr(settings, "TIKTOK_MIN_FOLLOWERS", 1000)

    async def parse_profile(self, handle: str) -> int | None:
        """Fetch TikTok profile, upsert account to database, return account ID.

        Parses the TikTok profile for the given handle, upserts the account
        information to the accounts table, and returns the database ID.
        Also fetches and upserts video content if the account meets the
        minimum follower threshold.

        Args:
            handle: TikTok username (without @ prefix).

        Returns:
            Database ID of the upserted account record, or None if the profile
            could not be parsed or doesn't meet the minimum follower threshold.
        """
        logger.info(f"Starting TikTok profile parse for handle: {handle}")
        account_id: int | None = None

        try:
            # Fetch profile data from Scrape Creators API
            try:
                response: dict[str, Any] = await self.client.get(
                    endpoint="/v1/tiktok/profile",
                    params={"handle": handle},
                )
            except Exception as e:
                logger.error(
                    f"API request failed for TikTok profile {handle}: {e}",
                    exc_info=True,
                )
                return None

            # Validate response structure
            # TikTok API response structure may vary - adjust based on actual API response
            data = response.get("data")
            if not data:
                logger.error(f"Missing 'data' in API response for TikTok handle {handle}")
                return None

            # Extract user info - structure based on typical TikTok API responses
            # Adjust field names based on actual Scrape Creators TikTok API response
            user = data.get("user") or data.get("author") or data
            if not user:
                logger.error(f"Missing user data in API response for TikTok handle {handle}")
                return None

            # Upsert account to database
            account_id = await self._upsert_account(user)

            # Check follower count against minimum threshold
            followers_count = self._extract_followers_count(user)
            if followers_count < self.min_followers:
                logger.info(
                    f"TikTok handle {handle} has {followers_count} followers, "
                    f"below minimum {self.min_followers}. Skipping content import."
                )
                return account_id

            # Fetch and upsert video content
            platform_id = str(user.get("id") or user.get("secUid", ""))
            if platform_id:
                await self.parse_content(account_id, platform_id, self.max_videos)
            else:
                logger.warning(f"Could not extract platform_id for TikTok handle {handle}")

            return account_id

        except Exception as e:
            logger.error(f"Failed to parse TikTok profile {handle}: {e}", exc_info=True)
            raise

    async def parse_content(
        self,
        account_id: int,
        platform_id: str,
        max_items: int = 50,
    ) -> None:
        """Fetch TikTok video content and bulk upsert to content table.

        Retrieves video content for the given account using pagination through
        the TikTok posts/feed endpoint. Upserts the collected videos to the
        content table using PostgreSQL ON CONFLICT DO UPDATE.

        Args:
            account_id: Database ID of the parent account record.
            platform_id: TikTok user ID (secUid or userId from API).
            max_items: Maximum number of content items to fetch (default: 50).
        """
        logger.info(
            f"Starting TikTok content parse for account_id: {account_id}, "
            f"platform_id: {platform_id}, max_items: {max_items}"
        )
        video_items: list[dict[str, Any]] = []
        cursor: str | None = None
        has_more: bool = True

        try:
            while len(video_items) < max_items and has_more:
                # Build API request parameters for TikTok posts endpoint
                params: dict[str, Any] = {"secUid": platform_id}
                if cursor:
                    params["cursor"] = cursor

                # Fetch data from Scrape Creators API
                # Endpoint may vary - adjust based on actual API documentation
                try:
                    response: dict[str, Any] = await self.client.get(
                        endpoint="/v1/tiktok/posts",
                        params=params,
                    )
                except Exception as e:
                    logger.error(
                        f"API request failed for TikTok content, platform_id {platform_id}: {e}",
                        exc_info=True,
                    )
                    break

                # Validate response structure
                data = response.get("data")
                if not data:
                    logger.error(
                        f"Missing 'data' in API response for TikTok content, "
                        f"platform_id {platform_id}"
                    )
                    break

                # Extract video items - adjust based on actual API response structure
                items = data.get("videos") or data.get("posts") or data.get("items", [])
                page_info = data.get("pageInfo") or data.get("cursor_info", {})
                has_more = page_info.get("hasMore", False) if page_info else False
                cursor = page_info.get("cursor") if page_info else None

                # Collect video items up to max_items
                for item in items:
                    video_items.append(item)
                    if len(video_items) >= max_items:
                        break

            # Upsert collected video content to database
            if video_items:
                await self._upsert_content(video_items, account_id)
                logger.info(
                    f"Successfully upserted {len(video_items)} TikTok videos for "
                    f"account_id: {account_id}"
                )
            else:
                logger.info(f"No TikTok video content found for account_id: {account_id}")

        except Exception as e:
            logger.error(
                f"Failed to parse TikTok content for account_id {account_id}: {e}",
                exc_info=True,
            )
            raise

    async def _upsert_account(self, user: dict[str, Any]) -> int:
        """Upsert TikTok account record to database using PostgreSQL ON CONFLICT.

        Args:
            user: User object from Scrape Creators API response.

        Returns:
            ID of the upserted Account record.
        """
        # Extract fields from TikTok API response
        # Adjust field names based on actual Scrape Creators TikTok API response
        platform_id: str = str(
            user.get("id") or user.get("secUid") or user.get("userId", "")
        )
        username: str | None = user.get("uniqueId") or user.get("username")
        full_name: str | None = user.get("nickname") or user.get("displayName")
        biography: str | None = user.get("bio") or user.get("signature")
        followers_count: int = self._extract_followers_count(user)

        async with self.session_maker() as session:
            async with session.begin():
                # Upsert using PostgreSQL ON CONFLICT DO UPDATE
                # Assumes unique constraint on (platform, platform_id)
                stmt = pg_insert(Account).values(
                    platform="TIKTOK",
                    platform_id=platform_id,
                    username=username,
                    title=full_name,
                    description=biography,
                    subscribers_count=followers_count,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["platform", "platform_id"],
                    set_=dict(
                        username=username,
                        title=full_name,
                        description=biography,
                        subscribers_count=followers_count,
                        updated_at=datetime.now(timezone.utc),
                    ),
                ).returning(Account.id)

                result = await session.execute(stmt)
                account_id: int = result.scalar_one()
                logger.debug(f"Upserted TikTok account {username} (ID: {account_id})")
                return account_id

    async def _upsert_content(
        self,
        video_items: list[dict[str, Any]],
        account_id: int,
    ) -> None:
        """Bulk upsert TikTok video content records to database.

        Args:
            video_items: List of video item dictionaries from API responses.
            account_id: ID of the parent Account record.
        """
        content_values: list[dict[str, Any]] = []

        for item in video_items:
            # Extract core fields from TikTok video item
            # Adjust field names based on actual Scrape Creators TikTok API response
            platform_content_id: str = str(
                item.get("id") or item.get("videoId") or item.get("awemeId", "")
            )

            # Extract content text (description)
            content_text: str | None = (
                item.get("desc") or item.get("description") or item.get("text")
            )

            # Convert timestamp to timezone-aware datetime
            # TikTok may use 'createTime' or 'timestamp' in seconds
            create_time = item.get("createTime") or item.get("timestamp")
            published_at: datetime | None = None
            if create_time:
                try:
                    published_at = datetime.fromtimestamp(
                        int(create_time), tz=timezone.utc
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse timestamp {create_time}: {e}"
                    )

            # Extract engagement metrics
            # Adjust field names based on actual API response
            stats = item.get("stats") or item.get("statistics") or {}
            views: int | None = (
                item.get("playCount")
                or stats.get("playCount")
                or stats.get("views")
            )
            comments_count: int | None = (
                item.get("commentCount")
                or stats.get("commentCount")
                or stats.get("comments")
            )
            shares_count: int | None = (
                item.get("shareCount")
                or stats.get("shareCount")
                or stats.get("shares")
            )
            reactions_count: int | None = (
                item.get("diggCount")
                or item.get("likeCount")
                or stats.get("diggCount")
                or stats.get("likes")
            )

            # Prepare raw metadata with non-mapped fields
            raw_metadata: dict[str, Any] = {
                "video_url": item.get("video", {}).get("playAddr") if item.get("video") else None,
                "cover_url": item.get("video", {}).get("cover") if item.get("video") else None,
                "duration": item.get("video", {}).get("duration") if item.get("video") else None,
                "music": item.get("music"),
                "author": item.get("author"),
                "hashtags": self._extract_hashtags(content_text),
            }

            content_values.append({
                "platform_content_id": platform_content_id,
                "account_id": account_id,
                "content": content_text,
                "published_at": published_at,
                "views": views,
                "comments_count": comments_count,
                "shares_count": shares_count,
                "reactions_count": reactions_count,
                "raw_metadata": raw_metadata,
                "is_embedded": False,
                "is_graph_extracted": False,
            })

        async with self.session_maker() as session:
            async with session.begin():
                # Bulk upsert with ON CONFLICT DO UPDATE
                # Uses the unique constraint on (account_id, platform_content_id)
                stmt = pg_insert(Content).values(content_values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["account_id", "platform_content_id"],
                    set_=dict(
                        content=stmt.excluded.content,
                        published_at=stmt.excluded.published_at,
                        views=stmt.excluded.views,
                        comments_count=stmt.excluded.comments_count,
                        shares_count=stmt.excluded.shares_count,
                        reactions_count=stmt.excluded.reactions_count,
                        raw_metadata=stmt.excluded.raw_metadata,
                        is_embedded=stmt.excluded.is_embedded,
                        is_graph_extracted=stmt.excluded.is_graph_extracted,
                        updated_at=datetime.now(timezone.utc),
                    ),
                )
                await session.execute(stmt)
                logger.debug(
                    f"Upserted {len(content_values)} TikTok content records for "
                    f"account ID {account_id}"
                )

    def _extract_followers_count(self, user: dict[str, Any]) -> int:
        """Extract follower count from TikTok user data.

        Args:
            user: User object from TikTok API response.

        Returns:
            Follower count as integer, or 0 if not found.
        """
        # Try different possible field names for follower count
        stats = user.get("stats") or user.get("statistics") or {}
        followers = (
            user.get("followerCount")
            or user.get("followers")
            or stats.get("followerCount")
            or stats.get("followers")
        )
        if followers is not None:
            try:
                return int(followers)
            except (ValueError, TypeError):
                pass
        return 0

    def _extract_hashtags(self, text: str | None) -> list[str]:
        """Extract hashtags from content text.

        Args:
            text: Content text to extract hashtags from.

        Returns:
            List of hashtags (without # symbol).
        """
        if not text:
            return []
        hashtags = []
        for word in text.split():
            if word.startswith("#"):
                hashtags.append(word[1:])  # Remove # symbol
        return hashtags
