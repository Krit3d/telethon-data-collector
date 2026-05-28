"""
Threads platform parser using Scrape Creators API.

Implements Threads-specific profile parsing and content ingestion into PostgreSQL.
Extracts author profile metadata (external links, contacts, location, language)
and stores it inside Content.raw_metadata under the "author_profile_metadata" key.

Features:
    - Profile parsing with account upsert to accounts table
    - Minimum subscriber threshold enforcement (3000 followers)
    - Virtual profile post creation for semantic search over biographies
    - Content fetching and bulk upsert to content table
    - PostgreSQL ON CONFLICT DO UPDATE for high-throughput concurrency
    - Raw metadata preservation for OpenSPG processing
    - Extraction of external links, contact info from biography via shared utils
    - Video download URL extraction for GPU worker processing
    - Transcription support for video content
"""

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config.config import Settings
from src.db.models import Account, Content
from src.parser.creators.core.utils import parse_profile_contacts
from src.parser.creators.platforms.base import BasePlatformParser
from src.parser.creators.sc_client import ScrapeCreatorsClient

logger = logging.getLogger(__name__)

# Minimum subscriber threshold for Threads accounts
MIN_SUBSCRIBERS: int = 3000


class ThreadsPlatformParser(BasePlatformParser):
    """Threads platform parser for profile and content ingestion.

    Inherits from BasePlatformParser and implements Threads-specific
    profile parsing and content upsert logic using the Scrape Creators API.

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
        """Initialize Threads parser with configuration.

        Args:
            session_maker: SQLAlchemy async session maker for database operations.
            client: ScrapeCreatorsClient instance for API requests.
            settings: Application settings containing configuration values.
        """
        super().__init__(session_maker, client, settings)

    async def parse_profile(self, handle: str) -> int | None:
        """Fetch Threads profile, upsert account to database, return account ID.

        Parses the Threads profile for the given handle, checks if the account
        meets the minimum subscriber threshold (3000), extracts author profile
        metadata (external links, contacts, location, language), upserts the
        account information to the accounts table, creates a virtual profile post
        for semantic search, and returns the database ID.

        Args:
            handle: Threads username (without @ prefix).

        Returns:
            Database ID of the upserted account record, or None if the profile
            could not be parsed or doesn't meet the minimum subscriber threshold.
        """
        logger.info("Starting Threads profile parse for handle: %s", handle)

        try:
            # Fetch profile data from Scrape Creators API
            try:
                response: dict[str, Any] = await self.client.get(
                    endpoint="/v1/threads/profile",
                    params={"handle": handle},
                )
                logger.info(
                    "API response status for profile %s: success, credits consumed: %s",
                    handle,
                    response.get("credits", "N/A"),
                )
            except Exception as e:
                logger.error(
                    "API request failed for Threads profile %s: %s",
                    handle,
                    e,
                    exc_info=True,
                )
                return None

            # Validate response structure
            data = response.get("data")
            if not data:
                logger.error("Missing 'data' in API response for Threads handle %s", handle)
                return None

            # Threads API may return user data at different levels
            user = data.get("user") or data.get("author") or data
            if not user:
                logger.error("Missing user data in API response for Threads handle %s", handle)
                return None

            # Extract user ID and follower count
            user_id: str = str(
                user.get("id") or user.get("pk") or user.get("user_id", "")
            )
            if not user_id:
                logger.error("Could not extract user ID for Threads handle %s", handle)
                return None

            followers_count: int = self._extract_followers_count(user)

            # Check if account meets minimum threshold
            if followers_count < MIN_SUBSCRIBERS:
                logger.info(
                    "Threads handle %s has %d followers, below minimum %d. Rejecting.",
                    handle,
                    followers_count,
                    MIN_SUBSCRIBERS,
                )
                await self._upsert_account(user, status="rejected")
                return None

            # Upsert account with 'parsed' status
            account_id: int = await self._upsert_account(user, status="parsed")
            logger.info(
                "Successfully parsed Threads profile %s, account ID: %d, followers: %d",
                handle,
                account_id,
                followers_count,
            )

            # Upsert virtual profile post for semantic search over biography
            await self._upsert_virtual_profile_post(account_id, user)

            return account_id

        except Exception as e:
            logger.error(
                "Failed to parse Threads profile %s: %s",
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
        """Fetch Threads content and bulk upsert to content table.

        Retrieves content items for the given account using the Scrape Creators
        API, parses the data, and performs a bulk upsert into the content table
        using PostgreSQL ON CONFLICT DO UPDATE.

        The raw_metadata field contains:
            - "author_profile_metadata": Profile-level data (contacts, links, location, etc.)
            - "platform_metrics": Platform-specific engagement metrics
            - "video_download_url": Direct URL for GPU worker processing

        Args:
            account_id: Database ID of the parent account record.
            platform_id: Threads username/handle used in API calls.
            max_items: Maximum number of content items to fetch (default: 50).
        """
        logger.info(
            "Starting Threads content parse for account_id: %d, platform_id: %s, max_items: %d",
            account_id,
            platform_id,
            max_items,
        )

        try:
            # Fetch content data from Scrape Creators API
            try:
                response: dict[str, Any] = await self.client.get(
                    endpoint="/v1/threads/posts",
                    params={"handle": platform_id, "limit": max_items},
                )
                logger.info(
                    "API response status for content, platform_id %s: success, credits consumed: %s",
                    platform_id,
                    response.get("credits", "N/A"),
                )
            except Exception as e:
                logger.error(
                    "API request failed for Threads content, platform_id %s: %s",
                    platform_id,
                    e,
                    exc_info=True,
                )
                return

            # Validate response structure
            data = response.get("data")
            if not data:
                logger.error(
                    "Missing 'data' in API response for Threads content, platform_id %s",
                    platform_id,
                )
                return

            # Extract posts/threads from response
            posts: list[dict[str, Any]] = self._extract_posts_from_response(data)
            if not posts:
                logger.info("No Threads content found for account_id: %d", account_id)
                return

            # Limit to max_items
            posts = posts[:max_items]

            # Build author profile metadata once (reused for all content items)
            # Try to get user data from response for metadata
            user_data = data.get("user") or data.get("author") or {}
            author_metadata = self._build_author_profile_metadata(user_data)

            # Process and upsert content items
            await self._upsert_content(posts, account_id, author_metadata)
            logger.info(
                "Successfully upserted %d Threads content items for account_id: %d",
                len(posts),
                account_id,
            )

        except Exception as e:
            logger.error(
                "Failed to parse Threads content for account_id %d: %s",
                account_id,
                e,
                exc_info=True,
            )
            raise

    async def _upsert_account(self, user: dict[str, Any], status: str = "parsed") -> int:
        """Upsert Threads account record using select-then-insert/update pattern.

        Uses a select-then-upsert transaction pattern to avoid InvalidColumnReferenceError
        caused by missing unique constraint on (platform, platform_id) in the accounts table.

        Args:
            user: User object from Scrape Creators API response.
            status: Account status ('parsed', 'rejected', etc.).

        Returns:
            ID of the account record (auto-generated by PostgreSQL for new records).
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
        followers_count: int = self._extract_followers_count(user)

        async with self.session_maker() as session:
            # Select existing account by platform and platform_id
            stmt = select(Account).where(
                Account.platform == "THREADS",
                Account.platform_id == platform_id,
            )
            result = await session.execute(stmt)
            db_account: Account | None = result.scalar_one_or_none()

            if db_account:
                # Update existing record
                db_account.username = username
                db_account.title = full_name or username or "Unknown"
                db_account.description = biography
                db_account.subscribers_count = followers_count
                db_account.status = status
                db_account.updated_at = datetime.now(timezone.utc)
                logger.debug(
                    "Updated existing Threads account %s (ID: %d, status: %s)",
                    username,
                    db_account.id,
                    status,
                )
            else:
                # Create new record (let PostgreSQL generate ID)
                db_account = Account(
                    platform="THREADS",
                    platform_id=platform_id,
                    username=username,
                    title=full_name or username or "Unknown",
                    description=biography,
                    subscribers_count=followers_count,
                    status=status,
                )
                session.add(db_account)
                logger.debug(
                    "Created new Threads account %s (status: %s)",
                    username,
                    status,
                )

            await session.commit()
            await session.refresh(db_account)
            return db_account.id

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
        followers_count: int = self._extract_followers_count(user)

        virtual_content_id: str = f"profile_bio_{platform_id}"
        compiled_text: str = (
            f"[PROFILE METADATA]\n"
            f"Platform: Threads\n"
            f"Username: @{username or 'unknown'}\n"
            f"Title: {full_name or 'Unknown'}\n"
            f"Subscribers: {followers_count}\n"
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
                    "Upserted virtual profile post for Threads account_id: %d (platform_content_id: %s)",
                    account_id,
                    virtual_content_id,
                )

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

                # Extract published timestamp
                published_at: datetime | None = self._extract_published_at(post)

                # Extract engagement metrics
                views: int | None = (
                    post.get("view_count")
                    or post.get("views")
                    or post.get("video_view_count")
                )
                reactions_count: int | None = (
                    post.get("like_count")
                    or post.get("likes")
                    or post.get("reactions_count")
                )
                comments_count: int | None = (
                    post.get("comment_count")
                    or post.get("comments")
                    or post.get("comments_count")
                )
                shares_count: int | None = (
                    post.get("share_count")
                    or post.get("shares")
                    or post.get("reposts")
                )

                # Check if video post and extract video URL
                is_video: bool = self._is_video_post(post)
                has_media: bool = is_video or self._has_media(post)

                # Extract high-quality direct URL for GPU worker
                video_download_url: str | None = None
                if is_video:
                    video_download_url = self._extract_video_download_url(post)

                # Extract transcription if available
                transcription: str | None = self._extract_transcription(post)

                # Build platform metrics
                platform_metrics: dict[str, Any] = {
                    "views": views,
                    "likes": reactions_count,
                    "comments": comments_count,
                    "shares": shares_count,
                    "is_video": is_video,
                }

                # Build raw_metadata with author_profile_metadata and platform_metrics
                raw_metadata: dict[str, Any] = {
                    "author_profile_metadata": author_metadata,
                    "platform_metrics": platform_metrics,
                }
                if video_download_url:
                    raw_metadata["video_download_url"] = video_download_url

                content_values.append({
                    "account_id": account_id,
                    "platform_content_id": platform_content_id,
                    "content": content_text,
                    "transcription": transcription,
                    "published_at": published_at or datetime.now(timezone.utc),
                    "views": views,
                    "comments_count": comments_count,
                    "shares_count": shares_count,
                    "reactions_count": reactions_count,
                    "has_media": has_media,
                    "raw_metadata": raw_metadata,
                    "is_embedded": False,
                    "is_graph_extracted": False,
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
                        shares_count=stmt.excluded.shares_count,
                        comments_count=stmt.excluded.comments_count,
                        has_media=stmt.excluded.has_media,
                        raw_metadata=stmt.excluded.raw_metadata,
                        updated_at=datetime.now(timezone.utc),
                    ),
                )
                await session.execute(stmt)
                logger.debug(
                    "Upserted %d Threads content records for account ID %d",
                    len(content_values),
                    account_id,
                )

    def _build_author_profile_metadata(self, user: dict[str, Any]) -> dict[str, Any]:
        """Build author profile metadata dictionary from user data.

        Extracts external links, contact information, location, language,
        and other profile-level data. This metadata is stored inside each
        Content.raw_metadata under the "author_profile_metadata" key.

        Uses shared parse_profile_contacts from core.utils to extract
        emails, Telegram handles, and external links from the biography.

        Args:
            user: User object from Threads API response.

        Returns:
            Dictionary containing author profile metadata.
        """
        biography: str | None = user.get("biography") or user.get("bio")
        username: str | None = user.get("username") or user.get("handle")
        full_name: str | None = user.get("full_name") or user.get("name")

        # Use shared utility to parse contacts from biography and external_url
        external_url: str | None = user.get("external_url")
        contacts_dict: dict[str, Any] = parse_profile_contacts(biography, external_url)
        # parse_profile_contacts returns: {emails, telegram_handles, external_links, raw_bio}

        # Build profile link
        profile_link: str | None = None
        if username:
            profile_link = f"https://threads.net/@{username}"

        # Build contacts list in the format expected by OpenSPG
        contacts: list[str] = []
        for email in contacts_dict.get("emails", []):
            contacts.append(f"email:{email}")
        for handle in contacts_dict.get("telegram_handles", []):
            contacts.append(f"telegram:@{handle}")
        # Add external links as contact entries
        external_links: list[str] = contacts_dict.get("external_links", [])

        # Extract location
        location: str | None = user.get("location") or user.get("address")

        # Language is not directly available from Threads API
        language: str | None = None

        # Geo-data
        geo_data: dict[str, float] | None = None
        latitude = user.get("latitude") or user.get("lat")
        longitude = user.get("longitude") or user.get("lng")
        if latitude is not None and longitude is not None:
            try:
                geo_data = {"lat": float(latitude), "lng": float(longitude)}
            except (ValueError, TypeError):
                pass

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

    def _extract_posts_from_response(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract posts list from API response data.

        Threads API response may have different structures for posts.

        Args:
            data: Response data dictionary.

        Returns:
            List of post dictionaries.
        """
        posts: list[dict[str, Any]] = []

        # Try different possible response structures
        # Structure 1: data.posts (array of posts)
        if "posts" in data and isinstance(data["posts"], list):
            posts = data["posts"]

        # Structure 2: data.threads (array of threads)
        elif "threads" in data and isinstance(data["threads"], list):
            posts = data["threads"]

        # Structure 3: data.user.edge_owner_to_timeline_media.edges (similar to Instagram)
        elif "user" in data:
            user = data["user"]
            timeline = user.get("edge_owner_to_timeline_media", {})
            if isinstance(timeline, dict):
                edges = timeline.get("edges", [])
                if isinstance(edges, list):
                    posts = [edge.get("node", edge) for edge in edges]

        # Structure 4: data.items (generic array)
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
        # Try edge_followed_by.count first (GraphQL structure similar to Instagram)
        edge_followed_by = user.get("edge_followed_by")
        if isinstance(edge_followed_by, dict):
            count = edge_followed_by.get("count")
            if count is not None:
                try:
                    return int(count)
                except (ValueError, TypeError):
                    pass

        # Try followers field
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

        return 0

    def _extract_content_text(self, post: dict[str, Any]) -> str | None:
        """Extract text content from Threads post.

        Args:
            post: Post dictionary from API response.

        Returns:
            Extracted text content, or None if not found.
        """
        # Try direct text field
        text: str | None = (
            post.get("text")
            or post.get("content")
            or post.get("caption")
            or post.get("post_text")
        )
        if text and isinstance(text, str):
            return text

        # Try caption edges structure (similar to Instagram)
        caption_edges = post.get("edge_media_to_caption")
        if isinstance(caption_edges, dict):
            edges = caption_edges.get("edges", [])
            if isinstance(edges, list) and edges:
                first_edge = edges[0]
                if isinstance(first_edge, dict):
                    node = first_edge.get("node", {})
                    if isinstance(node, dict) and node.get("text"):
                        return str(node["text"])

        return None

    def _extract_published_at(self, post: dict[str, Any]) -> datetime | None:
        """Extract and convert published timestamp to timezone-aware datetime.

        Args:
            post: Post dictionary from API response.

        Returns:
            Timezone-aware datetime, or None if timestamp is invalid.
        """
        timestamp = (
            post.get("taken_at_timestamp")
            or post.get("timestamp")
            or post.get("created_at")
            or post.get("published_at")
            or post.get("created_time")
        )

        if timestamp:
            try:
                if isinstance(timestamp, (int, float)):
                    return datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
                elif isinstance(timestamp, str):
                    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except (ValueError, TypeError, OSError) as e:
                logger.warning("Failed to parse timestamp %s: %s", timestamp, e)
        return None

    def _is_video_post(self, post: dict[str, Any]) -> bool:
        """Check if the post is a video.

        Args:
            post: Post dictionary from API response.

        Returns:
            True if the post is a video, False otherwise.
        """
        return bool(
            post.get("is_video")
            or post.get("video")
            or post.get("media_type") == "VIDEO"
            or post.get("media_type") == 2  # Instagram/Threads media type for video
        )

    def _has_media(self, post: dict[str, Any]) -> bool:
        """Check if the post has any media (image or video).

        Args:
            post: Post dictionary from API response.

        Returns:
            True if the post has media, False otherwise.
        """
        return bool(
            self._is_video_post(post)
            or post.get("thumbnail_url")
            or post.get("display_url")
            or post.get("image_url")
            or post.get("media_type") in ("IMAGE", "VIDEO", 1, 2)
        )

    def _extract_video_download_url(self, post: dict[str, Any]) -> str | None:
        """Extract high-quality direct URL from video post.

        Args:
            post: Post dictionary from API response.

        Returns:
            Direct URL string, or None if not found.
        """
        # Try video_url field first
        video_url: str | None = post.get("video_url")
        if video_url and isinstance(video_url, str):
            return video_url

        # Try video resources
        video_resources = post.get("video_resources")
        if isinstance(video_resources, list) and video_resources:
            sorted_resources = sorted(
                video_resources,
                key=lambda r: r.get("profile", 0) if isinstance(r, dict) else 0,
                reverse=True,
            )
            best_resource = sorted_resources[0]
            if isinstance(best_resource, dict):
                return best_resource.get("src")

        # Try display_url as fallback
        display_url: str | None = post.get("display_url")
        if display_url and isinstance(display_url, str):
            return display_url

        return None

    def _extract_transcription(self, post: dict[str, Any]) -> str | None:
        """Extract transcription from post if available.

        Args:
            post: Post dictionary from API response.

        Returns:
            Transcription text, or None if not available.
        """
        transcription: str | None = (
            post.get("transcription")
            or post.get("transcript")
            or post.get("video_transcription")
        )
        if transcription and isinstance(transcription, str):
            return transcription.strip()
        return None
