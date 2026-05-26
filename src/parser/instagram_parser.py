"""
Instagram profile parser using Scrape Creators API.
Imports Instagram profile and video content into PostgreSQL database.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config.config import Settings
from src.db.models import Account, Content
from src.parser.sc_client import ScrapeCreatorsClient

logger = logging.getLogger(__name__)


class InstagramParser:
    """Parser for Instagram profiles via Scrape Creators API, upserts data to PostgreSQL."""

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        client: ScrapeCreatorsClient,
        settings: Settings | None = None,
    ) -> None:
        """
        Initialize Instagram parser.

        Args:
            session_maker: SQLAlchemy async session maker.
            client: ScrapeCreatorsClient instance for API requests.
            settings: Optional Settings instance for configuration.
        """
        self.session_maker = session_maker
        self.client = client
        self.settings = settings
        self.max_videos: int = 50  # Maximum number of videos to collect per profile
        # Minimum subscribers threshold - can be configured via settings
        self.min_subscribers: int = 3000
        if settings:
            self.min_subscribers = getattr(settings, "INSTAGRAM_MIN_SUBSCRIBERS", 3000)

    async def parse_profile(self, handle: str) -> None:
        """
        Parse Instagram profile by handle, upsert account and video content to database.

        Args:
            handle: Instagram username (without @ prefix).
        """
        logger.info(f"Starting Instagram profile parse for handle: {handle}")
        account_id: int | None = None
        video_nodes: list[dict[str, Any]] = []
        cursor: str | None = None
        has_next_page: bool = True

        try:
            while len(video_nodes) < self.max_videos and has_next_page:
                # Build API request parameters
                params: dict[str, Any] = {"handle": handle}
                if cursor:
                    params["cursor"] = cursor

                # Fetch data from Scrape Creators API
                try:
                    response: dict[str, Any] = await self.client.raw_get(
                        endpoint="/v1/instagram/profile",
                        params=params,
                    )
                except Exception as e:
                    logger.error(
                        f"API request failed for {handle} (cursor: {cursor}): {e}",
                        exc_info=True,
                    )
                    break

                # Validate response structure
                data = response.get("data")
                if not data:
                    logger.error(f"Missing 'data' in API response for {handle}")
                    break

                user = data.get("user")
                if not user:
                    logger.error(f"Missing 'user' in API data for {handle}")
                    break

                # Process account on first iteration (first page)
                if account_id is None:
                    account_id = await self._upsert_account(user)
                    subscribers_count = user.get("edge_followed_by", {}).get("count", 0)
                    if subscribers_count < self.min_subscribers:
                        logger.info(
                            f"Handle {handle} has {subscribers_count} subscribers, "
                            f"below minimum {self.min_subscribers}. Skipping content import."
                        )
                        return

                # Extract media edges and pagination info
                media = user.get("edge_owner_to_timeline_media", {})
                edges = media.get("edges", [])
                page_info = media.get("page_info", {})
                has_next_page = page_info.get("has_next_page", False)
                cursor = page_info.get("end_cursor")

                # Filter video nodes from current page edges
                for edge in edges:
                    node = edge.get("node", {})
                    if node.get("is_video"):
                        video_nodes.append(node)
                        if len(video_nodes) >= self.max_videos:
                            break

            # Upsert collected video content to database
            if video_nodes and account_id is not None:
                await self._upsert_content(video_nodes, account_id)
                logger.info(f"Successfully upserted {len(video_nodes)} videos for {handle}")
            elif not video_nodes:
                logger.info(f"No video content found for {handle}")
            else:
                logger.error(f"Missing account ID for {handle}, cannot upsert content")

        except Exception as e:
            logger.error(f"Failed to parse profile {handle}: {e}", exc_info=True)
            raise

    async def _upsert_account(self, user: dict[str, Any]) -> int:
        """
        Upsert Instagram account record to database using PostgreSQL ON CONFLICT.

        Args:
            user: User object from Scrape Creators API response.

        Returns:
            ID of the upserted Account record.
        """
        platform_id: str = str(user.get("id"))
        username: str | None = user.get("username")
        full_name: str | None = user.get("full_name")
        biography: str | None = user.get("biography")
        subscribers_count: int = user.get("edge_followed_by", {}).get("count", 0)

        async with self.session_maker() as session:
            async with session.begin():
                # Upsert using PostgreSQL ON CONFLICT DO UPDATE
                # Assumes unique constraint on (platform, platform_id)
                stmt = pg_insert(Account).values(
                    platform="INSTAGRAM",
                    platform_id=platform_id,
                    username=username,
                    title=full_name,
                    description=biography,
                    subscribers_count=subscribers_count,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["platform", "platform_id"],
                    set_=dict(
                        username=username,
                        title=full_name,
                        description=biography,
                        subscribers_count=subscribers_count,
                        updated_at=datetime.now(timezone.utc),
                    ),
                ).returning(Account.id)

                result = await session.execute(stmt)
                account_id: int = result.scalar_one()
                logger.debug(f"Upserted Instagram account {username} (ID: {account_id})")
                return account_id

    async def _upsert_content(
        self,
        video_nodes: list[dict[str, Any]],
        account_id: int,
    ) -> None:
        """
        Bulk upsert Instagram video content records to database.

        Args:
            video_nodes: List of video node dictionaries from API responses.
            account_id: ID of the parent Account record.
        """
        content_values: list[dict[str, Any]] = []

        for node in video_nodes:
            # Extract core fields from node
            platform_content_id: str = str(node.get("id"))

            # Safely extract caption text
            caption_edges = node.get("edge_media_to_caption", {}).get("edges", [])
            content_text: str | None = None
            if caption_edges:
                content_text = caption_edges[0].get("node", {}).get("text")

            # Convert UNIX timestamp to timezone-aware datetime
            taken_at_timestamp = node.get("taken_at_timestamp")
            published_at: datetime | None = None
            if taken_at_timestamp:
                published_at = datetime.fromtimestamp(taken_at_timestamp, tz=timezone.utc)

            # Extract engagement metrics (adjust field names based on actual API response)
            views: int | None = node.get("video_view_count")
            comments_count: int | None = node.get("edge_media_to_comment", {}).get("count")
            shares_count: int | None = node.get("shares_count")  # Update if API uses different field
            reactions_count: int | None = node.get("edge_media_preview_like", {}).get("count")

            # Prepare raw metadata with non-mapped fields
            raw_metadata: dict[str, Any] = {
                "external_url": node.get("external_url"),
                "thumbnail_src": node.get("thumbnail_src"),
                "dimensions": node.get("dimensions"),
                "accessibility_caption": node.get("accessibility_caption"),
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
                logger.debug(f"Upserted {len(content_values)} content records for account ID {account_id}")
