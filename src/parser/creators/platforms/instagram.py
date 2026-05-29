"""
Instagram platform parser using Scrape Creators API.

Implements Instagram-specific profile parsing and content ingestion into PostgreSQL.
Extracts author profile metadata (external links, contacts, location, language)
and stores it inside Content.raw_metadata under the "author_profile_metadata" key.

Features:
    - Single API request for both profile and posts (saves credits)
    - Profile data caching to avoid redundant API calls
    - Minimum subscriber threshold enforcement (3000 followers)
    - Virtual profile post creation for semantic search over biographies
    - Content fetching from timeline and video edges with deduplication
    - PostgreSQL ON CONFLICT DO UPDATE for high-throughput concurrency
    - Raw metadata preservation for OpenSPG processing
    - Extraction of external links, contact info from biography via shared utils
    - Video download URL extraction for GPU worker processing
    - Transcription support for video content
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any

import aiohttp

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config.config import Settings
from src.db.models import Account, Content
from src.parser.creators.core.utils import parse_profile_contacts
from src.parser.creators.platforms.base import BasePlatformParser
from src.parser.creators.sc_client import ScrapeCreatorsClient

logger = logging.getLogger(__name__)

# Minimum subscriber threshold for Instagram accounts
MIN_SUBSCRIBERS: int = 3000

# Maximum subscriber threshold for Instagram accounts (micro-influencers)
MAX_SUBSCRIBERS: int = 150000

# Maximum video duration for transcription API (in seconds)
MAX_TRANSCRIPTION_DURATION: float = 120.0

# Email extraction pattern
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Stop-words to identify AI slop, meme pages, and compilation accounts
SLOP_STOP_WORDS: frozenset[str] = frozenset({
    # AI Slop / Generation terms
    "ai art", "midjourney", "stable diffusion", "нейросеть", "нейросети",
    "генерация", "chatgpt", "dall-e",
    # Compilations / Meme pages / Theme channels
    "нарезки", "нарезка", "мемы", "мем", "юмор", "приколы", "прикол",
    "смешно", "смешное", "гороскоп", "гороскопы",
    "анекдоты", "анекдот", "фильмы на вечер", "лучшие фильмы",
    "кино на вечер", "треки", "музыка", "сохрани",
    # Aggregators & Spam terms
    "подпишись", "взаимно", "паблик", "админ", "по вопросам рекламы",
    "сливы", "слив", "shina", "compilation",
    "gaming clips", "funny moments", "pubg", "fortnite", "dota", "csgo",
})


def is_russian_text(text: str | None) -> bool:
    """Check if text contains any Cyrillic (Russian) characters.

    Args:
        text: Text to check, or None.

    Returns:
        True if text contains at least one Cyrillic character, False otherwise.
    """
    if not text:
        return False
    return bool(re.search(r"[а-яА-ЯёЁ]", text))


class InstagramPlatformParser(BasePlatformParser):
    """Instagram platform parser for profile and content ingestion.

    Inherits from BasePlatformParser and implements Instagram-specific
    profile parsing and content upsert logic using the Scrape Creators API.

    Uses a single API endpoint (/v1/instagram/profile) to fetch both
    profile data and posts, caching the response to avoid redundant
    API calls and credit consumption.

    All author-level metadata (external links, contacts, location, language)
    is stored inside each Content.raw_metadata JSONB column under the key
    "author_profile_metadata" since Account has no raw_metadata column.

    After each successful profile parse, a virtual profile post is upserted
    into the content table to enable semantic search over creator biographies.

    Attributes:
        session_maker: SQLAlchemy async session maker for database operations.
        client: ScrapeCreatorsClient instance for API requests.
        settings: Application settings containing configuration values.
        _cached_profile_data: Cached raw profile API response.
        _cached_handle: Handle associated with the cached profile data.
    """

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        client: ScrapeCreatorsClient,
        settings: Settings,
    ) -> None:
        """Initialize Instagram parser with configuration.

        Args:
            session_maker: SQLAlchemy async session maker for database operations.
            client: ScrapeCreatorsClient instance for API requests.
            settings: Application settings containing configuration values.
        """
        super().__init__(session_maker, client, settings)
        self._cached_profile_data: dict[str, Any] | None = None
        self._cached_handle: str | None = None

    def _is_slop_or_theme_page(self, username: str | None, biography: str | None) -> bool:
        """Check if account is an AI slop, meme page, or compilation channel.

        Scans the username and biography for stop-words that indicate
        non-author accounts such as AI generation pages, meme aggregators,
        video compilation channels, and spam accounts.

        Args:
            username: Instagram username (without @ prefix), or None.
            biography: User biography text, or None.

        Returns:
            True if any stop-word from SLOP_STOP_WORDS is found in either
            the username or biography (case-insensitive), False otherwise.
        """
        search_text: str = ""

        if username:
            search_text += username.lower() + " "

        if biography:
            search_text += biography.lower()

        return any(stop_word in search_text for stop_word in SLOP_STOP_WORDS)

    async def parse_profile(self, handle: str) -> int | None:
        """Fetch Instagram profile, upsert account to database, return account ID.

        Parses the Instagram profile for the given handle using a single API call,
        checks if the account meets the minimum subscriber threshold (3000),
        extracts author profile metadata (external links, contacts, location, language),
        upserts the account information to the accounts table, creates a virtual
        profile post for semantic search, and returns the database ID.

        Caches the API response in self._cached_profile_data to avoid redundant
        API calls when parse_content() is called subsequently.

        Args:
            handle: Instagram username (without @ prefix).

        Returns:
            Database ID of the upserted account record, or None if the profile
            could not be parsed or doesn't meet the minimum subscriber threshold.
        """
        logger.info("Starting Instagram profile parse for handle: %s", handle)

        try:
            # Fetch profile data from Scrape Creators API (single call for profile + posts)
            response: dict[str, Any] | None = await self._fetch_profile_data(
                handle
            )

            if response is None:
                logger.error(
                    "Failed to fetch profile data for handle: %s", handle
                )
                return None

            # Validate response structure
            data = response.get("data")
            if not data:
                logger.info(
                    "Raw failed API response for %s: %s", handle, response
                )
                logger.error(
                    "Missing 'data' in API response for Instagram handle %s",
                    handle,
                )
                return None

            user = data.get("user")
            if not user:
                logger.error(
                    "Missing 'user' in data for Instagram handle %s", handle
                )
                return None

            # Extract user ID and subscriber count
            user_id: str = str(user.get("id", ""))
            if not user_id:
                logger.error(
                    "Could not extract user ID for Instagram handle %s", handle
                )
                return None

            subscribers_count: int = self._extract_subscribers_count(user)

            # Extract biography for Russian text check
            biography: str | None = user.get("biography")

            # Check if account meets subscriber thresholds
            if subscribers_count < MIN_SUBSCRIBERS:
                logger.info(
                    "Instagram handle %s rejected: subscribers count %d below minimum %d",
                    handle,
                    subscribers_count,
                    MIN_SUBSCRIBERS,
                )
                await self._upsert_account(user, status="rejected")
                return None

            if subscribers_count > MAX_SUBSCRIBERS:
                logger.info(
                    "Instagram handle %s rejected: subscribers count %d above maximum %d",
                    handle,
                    subscribers_count,
                    MAX_SUBSCRIBERS,
                )
                await self._upsert_account(user, status="rejected")
                return None

            # Check if profile is Russian (contains Cyrillic characters in bio)
            if not is_russian_text(biography):
                logger.info(
                    "Instagram handle %s rejected: non-Russian profile",
                    handle,
                )
                await self._upsert_account(user, status="rejected")
                return None

            # Check if account is AI slop, meme page, or compilation channel
            username: str | None = user.get("username")
            if self._is_slop_or_theme_page(username, biography):
                logger.info(
                    "Instagram handle %s rejected: identified as AI slop or theme/meme page.",
                    handle,
                )
                await self._upsert_account(user, status="rejected")
                return None

            # Upsert account with 'parsed' status
            account_id: int = await self._upsert_account(user, status="parsed")
            logger.info(
                "Successfully parsed Instagram profile %s, account ID: %d, subscribers: %d",
                handle,
                account_id,
                subscribers_count,
            )

            # Upsert virtual profile post for semantic search over biography
            await self._upsert_virtual_profile_post(account_id, user)

            # Build contacts dictionary for queue expansion
            external_url: str | None = user.get("external_url")
            contacts_dict: dict[str, Any] = parse_profile_contacts(
                biography, external_url
            )

            # Queue discovered accounts from contacts (cross-platform expansion)
            async with self.session_maker() as session:
                await self._queue_discovered_accounts(session, contacts_dict, handle)

            return account_id

        except Exception as e:
            logger.error(
                "Failed to parse Instagram profile %s: %s",
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
        """Fetch Instagram content and bulk upsert to content table.

        Retrieves content items from timeline and video edges for the given account
        using cached profile data (from parse_profile) or a new API call if cache
        is missing. Parses the data and performs a bulk upsert into the content
        table using PostgreSQL ON CONFLICT DO UPDATE.

        The raw_metadata field contains:
            - "author_profile_metadata": Profile-level data (contacts, links, location, etc.)
            - "platform_metrics": Platform-specific engagement metrics
            - "video_download_url": Direct MP4 URL for GPU worker processing
            - "raw_post_node": Original raw JSON of the post node

        Args:
            account_id: Database ID of the parent account record.
            platform_id: Instagram username/handle used in API calls.
            max_items: Maximum number of content items to fetch (default: 50).
        """
        logger.info(
            "Starting Instagram content parse for account_id: %d, platform_id: %s, max_items: %d",
            account_id,
            platform_id,
            max_items,
        )

        try:
            # Get profile data (use cache if available, otherwise fetch)
            response: dict[str, Any] | None = (
                await self._get_cached_or_fetch_profile(platform_id)
            )

            if response is None:
                logger.error(
                    "Failed to get profile data for content parsing, platform_id: %s",
                    platform_id,
                )
                return

            # Validate response structure
            data = response.get("data")
            if not data:
                logger.info(
                    "Raw failed API response for %s: %s", platform_id, response
                )
                logger.error(
                    "Missing 'data' in API response for Instagram content, platform_id %s",
                    platform_id,
                )
                return

            user = data.get("user")
            if not user:
                logger.error(
                    "Missing 'user' in data for Instagram content, platform_id %s",
                    platform_id,
                )
                return

            # Extract edges from both timeline and video timeline
            all_edges: list[dict[str, Any]] = []

            # Extract from edge_owner_to_timeline_media
            timeline_data = user.get("edge_owner_to_timeline_media", {})
            if isinstance(timeline_data, dict):
                timeline_edges = timeline_data.get("edges", [])
                if isinstance(timeline_edges, list):
                    all_edges.extend(timeline_edges)

            # Extract from edge_felix_video_timeline
            video_data = user.get("edge_felix_video_timeline", {})
            if isinstance(video_data, dict):
                video_edges = video_data.get("edges", [])
                if isinstance(video_edges, list):
                    all_edges.extend(video_edges)

            # Deduplicate by post ID (using node -> id)
            seen_ids: set[str] = set()
            unique_edges: list[dict[str, Any]] = []
            for edge in all_edges:
                node = edge.get("node", {})
                if not isinstance(node, dict):
                    continue
                post_id: str = str(node.get("id", ""))
                if post_id and post_id not in seen_ids:
                    seen_ids.add(post_id)
                    unique_edges.append(edge)

            # Sort by taken_at_timestamp descending (newest first)
            def get_timestamp(edge: dict[str, Any]) -> int:
                node = edge.get("node", {})
                timestamp = node.get("taken_at_timestamp", 0)
                return int(timestamp) if timestamp else 0

            unique_edges.sort(key=get_timestamp, reverse=True)

            # Slice the list to max_items
            unique_edges = unique_edges[:max_items]

            if not unique_edges:
                logger.info(
                    "No Instagram content found for account_id: %d", account_id
                )
                return

            # Fetch transcriptions for video posts concurrently
            video_edges_with_shortcode: list[tuple[dict[str, Any], str]] = []
            for edge in unique_edges:
                node = edge.get("node", {})
                if not isinstance(node, dict):
                    continue
                is_video = node.get("is_video")
                shortcode = node.get("shortcode")
                if (
                    is_video is True
                    and isinstance(shortcode, str)
                    and shortcode
                ):
                    # Check video duration limit (120 seconds max for transcription API)
                    duration: float | None = node.get("video_duration")
                    if duration is not None and float(duration) > MAX_TRANSCRIPTION_DURATION:
                        logger.info(
                            "Skipping transcription for shortcode %s: duration %ss exceeds the %ss API limit.",
                            shortcode,
                            duration,
                            MAX_TRANSCRIPTION_DURATION,
                        )
                        continue
                    video_edges_with_shortcode.append((edge, shortcode))

            if video_edges_with_shortcode:
                logger.info(
                    "Fetching transcriptions for %d video posts",
                    len(video_edges_with_shortcode),
                )
                semaphore = asyncio.Semaphore(5)
                tasks = [
                    self._fetch_transcription_api(shortcode, semaphore)
                    for _, shortcode in video_edges_with_shortcode
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for (edge, shortcode), result in zip(
                    video_edges_with_shortcode, results
                ):
                    if isinstance(result, Exception):
                        logger.warning(
                            "Transcription task failed for shortcode %s: %s",
                            shortcode,
                            result,
                            exc_info=result,
                        )
                        continue
                    if isinstance(result, str) and result:
                        node = edge.get("node", {})
                        if isinstance(node, dict):
                            node["api_transcription"] = result

            # Build author profile metadata once (reused for all content items)
            author_metadata = self._build_author_profile_metadata(user)

            # Process and upsert content items
            await self._upsert_content(
                unique_edges, account_id, author_metadata
            )
            logger.info(
                "Successfully upserted %d Instagram content items for account_id: %d",
                len(unique_edges),
                account_id,
            )

        except Exception as e:
            logger.error(
                "Failed to parse Instagram content for account_id %d: %s",
                account_id,
                e,
                exc_info=True,
            )
            raise

    async def _fetch_profile_data(self, handle: str) -> dict[str, Any] | None:
        """Fetch profile data from Scrape Creators API and cache it.

        Args:
            handle: Instagram username (without @ prefix).

        Returns:
            API response dictionary, or None if the request failed.
        """
        try:
            response: dict[str, Any] = await self.client.get(
                endpoint="/v1/instagram/profile",
                params={"handle": handle},
            )
            credits_remaining = response.get("credits_remaining", "N/A")
            logger.info(
                "API response success. Remaining credits: %s",
                credits_remaining,
            )

            # Cache the response
            self._cached_profile_data = response
            self._cached_handle = handle

            return response

        except Exception as e:
            logger.error(
                "API request failed for Instagram profile %s: %s",
                handle,
                e,
                exc_info=True,
            )
            return None

    async def _get_cached_or_fetch_profile(
        self, handle: str
    ) -> dict[str, Any] | None:
        """Get profile data from cache or fetch from API if cache is missing/stale.

        Args:
            handle: Instagram username (without @ prefix).

        Returns:
            API response dictionary, or None if the request failed.
        """
        # Check if cache is valid
        if (
            self._cached_profile_data is not None
            and self._cached_handle == handle
        ):
            logger.debug("Using cached profile data for handle: %s", handle)
            return self._cached_profile_data

        # Cache miss or stale, fetch new data
        logger.info("Cache miss for handle: %s, fetching from API", handle)
        return await self._fetch_profile_data(handle)

    async def _fetch_transcription_api(
        self, shortcode: str, semaphore: asyncio.Semaphore
    ) -> str | None:
        """Fetch transcription for a video post via Scrape Creators transcript endpoint.

        Uses the verified winning format (https://instagram.com/p/{shortcode})
        to fetch transcription. Logs 4xx/5xx errors with logger.warning
        but returns None gracefully on failure.

        Args:
            shortcode: Instagram post shortcode.
            semaphore: Semaphore to limit concurrent requests.

        Returns:
            Transcription text if available, None otherwise.
        """
        endpoint = "/v2/instagram/media/transcript"
        # Use the verified winning format
        url_param: str = f"https://instagram.com/p/{shortcode}"

        async with semaphore:
            try:
                response: dict[str, Any] = await self.client.get(
                    endpoint=endpoint,
                    params={"url": url_param},
                )

                # Parse response JSON safely matching the verified structure
                if "transcripts" in response:
                    transcripts = response["transcripts"]
                    if isinstance(transcripts, list) and transcripts:
                        first_item = transcripts[0]
                        if isinstance(first_item, dict):
                            text: str | None = first_item.get("text")
                            if text and isinstance(text, str):
                                return text.strip()

                # No valid transcription found in response
                logger.debug(
                    "No transcription data found in response for shortcode %s",
                    shortcode,
                )
                return None

            except Exception as e:
                # Suppress tracebacks for aiohttp.ClientError (e.g., server errors, connection issues)
                if isinstance(e, aiohttp.ClientError):
                    if isinstance(e, aiohttp.ClientResponseError) and e.status == 500:
                        logger.warning(
                            "Transcription API failed for shortcode %s due to Scrape Creators server error (500).",
                            shortcode,
                        )
                    else:
                        # Other aiohttp errors (connection errors, timeouts, non-500 responses)
                        logger.warning(
                            "Transcription API failed for shortcode %s: %s",
                            shortcode,
                            e,
                        )
                else:
                    # Unexpected non-aiohttp exceptions: log full traceback
                    logger.error(
                        "Unexpected error fetching transcription for shortcode %s: %s",
                        shortcode,
                        e,
                        exc_info=True,
                    )
                return None

    async def _upsert_account(
        self, user: dict[str, Any], status: str = "parsed"
    ) -> int:
        """Upsert Instagram account record using select-then-insert/update pattern.

        Uses a select-then-upsert transaction pattern to avoid InvalidColumnReferenceError
        caused by missing unique constraint on (platform, platform_id) in the accounts table.

        Args:
            user: User object from Scrape Creators API response.
            status: Account status ('parsed', 'rejected', etc.).

        Returns:
            ID of the account record (auto-generated by PostgreSQL for new records).
        """
        platform_id: str = str(user.get("id", ""))
        username: str | None = user.get("username")
        full_name: str | None = user.get("full_name")
        biography: str | None = user.get("biography")
        subscribers_count: int = self._extract_subscribers_count(user)

        async with self.session_maker() as session:
            # Select existing account by platform and platform_id
            stmt = select(Account).where(
                Account.platform == "INSTAGRAM",
                Account.platform_id == platform_id,
            )
            result = await session.execute(stmt)
            db_account: Account | None = result.scalar_one_or_none()

            if db_account:
                # Update existing record
                db_account.username = username
                db_account.title = full_name or username or "Unknown"
                db_account.description = biography
                db_account.subscribers_count = subscribers_count
                db_account.status = status
                db_account.updated_at = datetime.now(timezone.utc)
                logger.debug(
                    "Updated existing Instagram account %s (ID: %d, status: %s)",
                    username,
                    db_account.id,
                    status,
                )
            else:
                # Create new record (let PostgreSQL generate ID)
                db_account = Account(
                    platform="INSTAGRAM",
                    platform_id=platform_id,
                    username=username,
                    title=full_name or username or "Unknown",
                    description=biography,
                    subscribers_count=subscribers_count,
                    status=status,
                )
                session.add(db_account)
                logger.debug(
                    "Created new Instagram account %s (status: %s)",
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
        platform_id: str = str(user.get("id", ""))
        username: str | None = user.get("username")
        full_name: str | None = user.get("full_name")
        biography: str | None = user.get("biography")
        subscribers_count: int = self._extract_subscribers_count(user)

        virtual_content_id: str = f"profile_bio_{platform_id}"
        compiled_text: str = (
            f"[PROFILE METADATA]\n"
            f"Platform: INSTAGRAM\n"
            f"Username: @{username or 'unknown'}\n"
            f"Title: {full_name or 'Unknown'}\n"
            f"Subscribers: {subscribers_count}\n"
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
                    raw_metadata={},
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
                    "Upserted virtual profile post for Instagram account_id: %d (platform_content_id: %s)",
                    account_id,
                    virtual_content_id,
                )

    async def _upsert_content(
        self,
        edges: list[dict[str, Any]],
        account_id: int,
        author_metadata: dict[str, Any],
    ) -> None:
        """Bulk upsert Instagram content records to database.

        Args:
            edges: List of edge dictionaries containing node data from API responses.
            account_id: ID of the parent Account record.
            author_metadata: Author profile metadata to embed in each content record.
        """
        content_values: list[dict[str, Any]] = []

        for edge in edges:
            try:
                node: dict[str, Any] = edge.get("node", {})
                if not isinstance(node, dict):
                    continue

                # Extract platform_content_id from shortcode or id
                platform_content_id: str = str(
                    node.get("shortcode") or node.get("id", "")
                )
                if not platform_content_id:
                    logger.warning("Skipping content item with no ID")
                    continue

                # Extract content text
                content_text: str | None = self._extract_content_text(node)

                # Extract published timestamp
                published_at: datetime | None = self._extract_published_at(node)

                # Extract engagement metrics
                views: int | None = node.get("video_view_count")
                reactions_count: int | None = self._extract_reactions_count(
                    node
                )
                comments_count: int | None = self._extract_comments_count(node)
                shares_count: int | None = None

                # Check if video post and extract video URL
                is_video: bool = node.get("is_video", False)
                has_media: bool = is_video

                # Extract direct MP4 URL for GPU worker
                video_download_url: str | None = None
                if is_video:
                    video_download_url = self._extract_video_download_url(node)

                # Extract transcription if available
                transcription: str | None = self._extract_transcription(node)

                # Build platform metrics
                platform_metrics: dict[str, Any] = {
                    "views": views,
                    "likes": reactions_count,
                    "comments": comments_count,
                    "shares": shares_count,
                    "is_video": is_video,
                }

                # Build raw_metadata with all required fields
                raw_metadata: dict[str, Any] = {
                    "author_profile_metadata": author_metadata,
                    "platform_metrics": platform_metrics,
                    "raw_post_node": node,
                }
                if video_download_url:
                    raw_metadata["video_download_url"] = video_download_url

                content_values.append(
                    {
                        "account_id": account_id,
                        "platform_content_id": platform_content_id,
                        "content": content_text,
                        "transcription": transcription,
                        "published_at": published_at
                        or datetime.now(timezone.utc),
                        "views": views,
                        "comments_count": comments_count,
                        "shares_count": shares_count,
                        "reactions_count": reactions_count,
                        "has_media": has_media,
                        "raw_metadata": raw_metadata,
                        "is_embedded": False,
                        "is_graph_extracted": False,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )

            except Exception as e:
                logger.error(
                    "Failed to parse Instagram content item: %s",
                    e,
                    exc_info=True,
                )
                continue

        if not content_values:
            logger.warning(
                "No valid content items to upsert for account_id: %d",
                account_id,
            )
            return

        async with self.session_maker() as session:
            async with session.begin():
                stmt = pg_insert(Content).values(content_values)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_content_account_platform_id",
                    set_=dict(
                        content=stmt.excluded.content,
                        transcription=func.coalesce(
                            stmt.excluded.transcription, Content.transcription
                        ),
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
                    "Upserted %d Instagram content records for account ID %d",
                    len(content_values),
                    account_id,
                )

    def _build_author_profile_metadata(
        self, user: dict[str, Any]
    ) -> dict[str, Any]:
        """Build author profile metadata dictionary from user data.

        Extracts external links, contact information, location, language,
        and other profile-level data. This metadata is stored inside each
        Content.raw_metadata under the "author_profile_metadata" key.

        Uses shared parse_profile_contacts from core.utils to extract
        emails, Telegram handles, and external links from the biography.

        Args:
            user: User object from Instagram API response.

        Returns:
            Dictionary containing author profile metadata.
        """
        biography: str | None = user.get("biography")
        username: str | None = user.get("username")
        full_name: str | None = user.get("full_name")

        # Use shared utility to parse contacts from biography and external_url
        external_url: str | None = user.get("external_url")
        contacts_dict: dict[str, Any] = parse_profile_contacts(
            biography, external_url
        )
        # parse_profile_contacts returns: {emails, telegram_handles, external_links, raw_bio}

        # Build profile link
        profile_link: str | None = None
        if username:
            profile_link = f"https://instagram.com/{username}"

        # Build contacts list in the format expected by OpenSPG
        contacts: list[str] = []
        for email in contacts_dict.get("emails", []):
            contacts.append(f"email:{email}")
        for handle in contacts_dict.get("telegram_handles", []):
            contacts.append(f"telegram:@{handle}")
        # Add external links as contact entries
        external_links: list[str] = contacts_dict.get("external_links", [])

        # Extract location from business address JSON if available
        location: str | None = None
        business_address = user.get("business_address_json")
        if business_address and isinstance(business_address, dict):
            location = business_address.get(
                "street_address"
            ) or business_address.get("city")
        elif user.get("location"):
            location = user.get("location")

        # Language is not directly available from Instagram API
        language: str | None = None

        # Geo-data (latitude/longitude) is not typically available from basic API
        geo_data: dict[str, float] | None = None

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

    async def _queue_discovered_accounts(
        self, session: AsyncSession, contacts_dict: dict[str, Any], parent_handle: str
    ) -> None:
        """Queue discovered accounts from contacts for cross-platform expansion.
    
        Processes emails, telegram handles, and external links from the contacts
        dictionary and inserts new account records into the accounts table with
        status "pending" for platforms YOUTUBE, TIKTOK, THREADS, and TELEGRAM.
    
        Uses on_conflict_do_nothing to avoid duplicate entries.
    
        Args:
            session: SQLAlchemy async session for database operations.
            contacts_dict: Dictionary containing emails, telegram_handles, and external_links.
            parent_handle: Instagram handle of the parent account whose bio was scanned.
        """
        # Extract contacts from contacts_dict
        telegram_handles: list[str] = contacts_dict.get("telegram_handles", [])
        external_links: list[str] = contacts_dict.get("external_links", [])

        # Check if there are any contacts to process
        if not telegram_handles and not external_links:
            logger.debug(
                "[SPIDER] No external social accounts discovered in bio of parent account %s.",
                parent_handle,
            )
            return

        # Process external links for YouTube, TikTok, and Threads
        for link in external_links:
            if not isinstance(link, str):
                continue

            link_lower = link.lower()

            # YouTube: youtube.com/ or youtu.be/
            if "youtube.com/" in link_lower or "youtu.be/" in link_lower:
                # Extract handle or channel ID
                platform_id: str | None = None
                if "/@" in link:
                    # Format: youtube.com/@handle
                    platform_id = (
                        link.split("/@")[-1].split("?")[0].split("/")[0]
                    )
                elif "youtube.com/channel/" in link_lower:
                    # Format: youtube.com/channel/UC...
                    platform_id = (
                        link.split("/channel/")[-1].split("?")[0].split("/")[0]
                    )
                elif "youtu.be/" in link_lower:
                    # Format: youtu.be/VIDEO_ID (not a channel, skip)
                    continue

                if platform_id:
                    inserted: bool = await self._insert_pending_account(
                        session, "YOUTUBE", platform_id
                    )
                    if inserted:
                        logger.info(
                            "[SPIDER] Queued discovered YOUTUBE account: %s from bio of parent account %s.",
                            platform_id,
                            parent_handle,
                        )

            # TikTok: tiktok.com/@
            elif "tiktok.com/@" in link_lower:
                # Extract username
                platform_id = link.split("/@")[-1].split("?")[0].split("/")[0]
                if platform_id:
                    inserted: bool = await self._insert_pending_account(
                        session, "TIKTOK", platform_id
                    )
                    if inserted:
                        logger.info(
                            "[SPIDER] Queued discovered TIKTOK account: %s from bio of parent account %s.",
                            platform_id,
                            parent_handle,
                        )

            # Threads: threads.net/@
            elif "threads.net/@" in link_lower or "threads.net/" in link_lower:
                # Extract username
                if "/@" in link:
                    platform_id = (
                        link.split("/@")[-1].split("?")[0].split("/")[0]
                    )
                else:
                    platform_id = (
                        link.split("/")[-1].split("?")[0].split("/")[0]
                    )
                if platform_id:
                    inserted: bool = await self._insert_pending_account(
                        session, "THREADS", platform_id
                    )
                    if inserted:
                        logger.info(
                            "[SPIDER] Queued discovered THREADS account: %s from bio of parent account %s.",
                            platform_id,
                            parent_handle,
                        )

        # Process Telegram handles
        for handle in telegram_handles:
            if not isinstance(handle, str) or not handle:
                continue
            # Remove @ if present
            platform_id = handle.lstrip("@")
            if platform_id:
                inserted: bool = await self._insert_pending_account(
                    session, "TELEGRAM", platform_id
                )
                if inserted:
                    logger.info(
                        "[SPIDER] Queued discovered TELEGRAM account: @%s from bio of parent account %s.",
                        platform_id,
                        parent_handle,
                    )

    async def _insert_pending_account(
        self, session: AsyncSession, platform: str, platform_id: str
    ) -> bool:
        """Insert a pending account record if it doesn't already exist.
    
        Args:
            session: SQLAlchemy async session for database operations.
            platform: Platform name (YOUTUBE, TIKTOK, THREADS, TELEGRAM).
            platform_id: Platform-specific account ID or handle.
    
        Returns:
            True if a new account was inserted, False if it already existed.
        """
        # Check if account already exists
        stmt = select(Account).where(
            Account.platform == platform,
            Account.platform_id == platform_id,
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()
    
        if existing:
            logger.debug(
                "Account already exists: platform=%s, platform_id=%s",
                platform,
                platform_id,
            )
            return False
    
        # Insert new pending account
        new_account = Account(
            platform=platform,
            platform_id=platform_id,
            username=None,
            title=platform_id,
            description=None,
            subscribers_count=None,
            status="pending",
        )
        session.add(new_account)
        logger.debug(
            "Queued new account: platform=%s, platform_id=%s",
            platform,
            platform_id,
        )
        return True

    def _extract_video_download_url(self, node: dict[str, Any]) -> str | None:
        """Extract high-quality direct MP4 URL from video node.

        Tries multiple fields to find the best quality video URL for
        downstream GPU worker processing.

        Args:
            node: Content node dictionary from API response.

        Returns:
            Direct MP4 URL string, or None if not found.
        """
        # Try video_url field first (highest quality)
        video_url: str | None = node.get("video_url")
        if video_url and isinstance(video_url, str):
            return video_url

        # Try display_url as fallback (may be image, check)
        display_url: str | None = node.get("display_url")
        if (
            display_url
            and isinstance(display_url, str)
            and "cdninstagram" in display_url
        ):
            return display_url

        # Try video_resources for highest quality
        video_resources = node.get("video_resources")
        if isinstance(video_resources, list) and video_resources:
            # Sort by profile (higher is better quality) and get the best
            sorted_resources = sorted(
                video_resources,
                key=lambda r: r.get("profile", 0) if isinstance(r, dict) else 0,
                reverse=True,
            )
            best_resource = sorted_resources[0]
            if isinstance(best_resource, dict):
                return best_resource.get("src")

        return None

    def _extract_transcription(self, node: dict[str, Any]) -> str | None:
        """Extract transcription from content node if available.

        Scrape Creators API may provide transcriptions for video content.
        This method attempts to extract it from various possible fields.

        Args:
            node: Content node dictionary from API response.

        Returns:
            Transcription text, or None if not available.
        """
        transcription: str | None = (
            node.get("api_transcription")
            or node.get("transcription")
            or node.get("transcript")
            or node.get("video_transcription")
        )
        if transcription and isinstance(transcription, str):
            return transcription.strip()
        return None

    def _extract_subscribers_count(self, user: dict[str, Any]) -> int:
        """Extract subscriber count from Instagram user data.

        Args:
            user: User object from Instagram API response.

        Returns:
            Subscriber count as integer, or 0 if not found.
        """
        # Try edge_followed_by.count first (GraphQL structure)
        edge_followed_by = user.get("edge_followed_by")
        if isinstance(edge_followed_by, dict):
            count = edge_followed_by.get("count")
            if count is not None:
                try:
                    return int(count)
                except (ValueError, TypeError):
                    pass

        # Try followers field
        followers = user.get("followers") or user.get("followers_count")
        if followers is not None:
            try:
                return int(followers)
            except (ValueError, TypeError):
                pass

        return 0

    def _extract_content_text(self, node: dict[str, Any]) -> str | None:
        """Extract text content from Instagram content node.

        Args:
            node: Content node dictionary from API response.

        Returns:
            Extracted text content, or None if not found.
        """
        # Try edge_media_to_caption.edges[0].node.text
        edge_media_to_caption = node.get("edge_media_to_caption")
        if isinstance(edge_media_to_caption, dict):
            edges = edge_media_to_caption.get("edges", [])
            if isinstance(edges, list) and edges:
                first_edge = edges[0]
                if isinstance(first_edge, dict):
                    text_node = first_edge.get("node", {})
                    if isinstance(text_node, dict) and text_node.get("text"):
                        return str(text_node["text"])

        # Fallback to accessibility_caption
        accessibility_caption = node.get("accessibility_caption")
        if accessibility_caption and isinstance(accessibility_caption, str):
            return accessibility_caption

        return None

    def _extract_published_at(self, node: dict[str, Any]) -> datetime | None:
        """Extract and convert published timestamp to timezone-aware datetime.

        Args:
            node: Content node dictionary from API response.

        Returns:
            Timezone-aware datetime, or None if timestamp is invalid.
        """
        timestamp = node.get("taken_at_timestamp")

        if timestamp:
            try:
                if isinstance(timestamp, (int, float)):
                    return datetime.fromtimestamp(
                        int(timestamp), tz=timezone.utc
                    )
            except (ValueError, TypeError, OSError) as e:
                logger.warning("Failed to parse timestamp %s: %s", timestamp, e)
        return None

    def _extract_reactions_count(self, node: dict[str, Any]) -> int | None:
        """Extract reactions (likes) count from content node.

        Args:
            node: Content node dictionary from API response.

        Returns:
            Reactions count as integer, or None if not found.
        """
        edge_media_preview_like = node.get("edge_media_preview_like")
        if isinstance(edge_media_preview_like, dict):
            count = edge_media_preview_like.get("count")
            if count is not None:
                try:
                    return int(count)
                except (ValueError, TypeError):
                    pass
        return None

    def _extract_comments_count(self, node: dict[str, Any]) -> int | None:
        """Extract comments count from content node.

        Args:
            node: Content node dictionary from API response.

        Returns:
            Comments count as integer, or None if not found.
        """
        edge_media_to_comment = node.get("edge_media_to_comment")
        if isinstance(edge_media_to_comment, dict):
            count = edge_media_to_comment.get("count")
            if count is not None:
                try:
                    return int(count)
                except (ValueError, TypeError):
                    pass
        return None
