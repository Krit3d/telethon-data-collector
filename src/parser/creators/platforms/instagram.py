"""
Instagram platform parser using Scrape Creators API.

Implements Instagram-specific profile parsing and content ingestion into PostgreSQL.
Uses v1 API for profile endpoint and v2 API for posts and transcripts.
Extracts author profile metadata and stores it inside Content.raw_metadata.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from aiohttp import ClientResponseError
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models import Account, Content
from src.parser.creators.core.utils import (
    extract_instagram_subscribers,
    is_russian_text,
    is_slop_or_theme_page,
    detect_female_creator,
    upsert_virtual_bio_post,
    queue_discovered_accounts,
    queue_discovered_mentions,
    extract_mentions,
    extract_instagram_content_text,
    extract_instagram_published_at,
    extract_instagram_video_url,
    extract_instagram_metrics,
    build_instagram_author_metadata,
    parse_profile_contacts,
)
from src.parser.creators.platforms.base import BasePlatformParser

logger = logging.getLogger(__name__)

# Subscriber thresholds for micro-influencers
MIN_SUBSCRIBERS: int = 3000
MAX_SUBSCRIBERS: int = 150000


class InstagramParser(BasePlatformParser):
    """Instagram platform parser for profile and content ingestion."""

    def __init__(
        self,
        session_maker,
        client,
        settings,
    ) -> None:
        """Initialize Instagram parser with configuration."""
        super().__init__(session_maker, client, settings)
        self._cached_profile: dict[str, Any] | None = None
        self._cached_handle: str | None = None

    async def parse_profile(self, handle: str) -> int | None:
        """Fetch Instagram profile, apply filters, upsert account.

        Args:
            handle: Instagram username (without @ prefix).

        Returns:
            Database ID of the upserted account record, or None if the profile
            could not be parsed, doesn't meet criteria, or is rejected.
        """
        logger.info("Starting Instagram profile parse for handle: %s", handle)

        profile = await self._get_cached_or_fetch_profile(handle)
        if not profile:
            return None

        # Subscriber threshold filter
        subscribers = extract_instagram_subscribers(profile)

        if subscribers == 0:
            logger.warning(
                "Instagram handle %s parsed 0 subscribers. Profile dict keys: %s",
                handle,
                list(profile.keys()),
            )

        if not (MIN_SUBSCRIBERS <= subscribers <= MAX_SUBSCRIBERS):
            logger.warning(
                "Instagram handle %s REJECTED: subscriber count %d is outside range [%d, %d].",
                handle,
                subscribers,
                MIN_SUBSCRIBERS,
                MAX_SUBSCRIBERS,
            )
            await self._upsert_account(profile, "rejected")
            return None

        # Content quality filters
        username = profile.get("username", "")
        biography = profile.get("biography", "")
        full_name = profile.get("full_name", "")

        # Check for Cyrillic characters in biography or full_name
        has_cyrillic_bio = is_russian_text(biography)
        has_cyrillic_name = is_russian_text(full_name)

        if not (has_cyrillic_bio or has_cyrillic_name):
            logger.warning(
                "Instagram handle %s REJECTED: No Cyrillic characters found in biography or full_name.",
                handle,
            )
            await self._upsert_account(profile, "rejected")
            return None

        # Check for slop/theme stop-words
        if is_slop_or_theme_page(username, biography):
            logger.warning(
                "Instagram handle %s REJECTED: Matched slop/theme stop-words.",
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
            "Successfully parsed Instagram profile %s, account ID: %d, subscribers: %d",
            handle,
            account_id,
            subscribers,
        )

        # Post-parse actions for female creators
        if is_female:
            async with self.session_maker() as session:
                await upsert_virtual_bio_post(
                    session=session,
                    account_id=account_id,
                    platform="INSTAGRAM",
                    platform_id=profile.get("id", handle),
                    username=username,
                    full_name=profile.get("full_name"),
                    biography=biography,
                    subscribers_count=subscribers,
                    raw_metadata={"female_heuristic": True},
                )
                await session.commit()

            # Cross-platform username parity seeding
            try:
                async with self.session_maker() as session:
                    # Import _queue_single_account dynamically
                    from src.parser.creators.core.db import _queue_single_account

                    # Queue identical handle for TikTok
                    await _queue_single_account(session, "TIKTOK", username, f"instagram_sync_{username}")
                    # Queue identical handle for Threads
                    await _queue_single_account(session, "THREADS", username, f"instagram_sync_{username}")

                    await session.commit()
                    logger.info("Cross-platform seeds queued for TIKTOK and THREADS from Instagram: %s", username)
            except Exception as e:
                logger.warning("Failed to queue cross-platform seeds for %s: %s", username, e)

        # Queue discovered accounts from contacts
        contacts_dict = parse_profile_contacts(
            biography, profile.get("external_url")
        )
        async with self.session_maker() as session:
            await queue_discovered_accounts(
                session=session,
                contacts_dict=contacts_dict,
                parent_handle=handle,
            )
            await session.commit()

        # Profile Chaining Discovery (Infinite Spidering)
        # Extract related accounts to prevent running out of seed accounts
        try:
            related_edges = profile.get("edge_related_profiles", {}).get(
                "edges", []
            )
            if related_edges:
                logger.info(
                    "Found %d related profiles for %s, processing chaining discovery",
                    len(related_edges),
                    handle,
                )
                async with self.session_maker() as session:
                    for edge in related_edges:
                        node = edge.get("node", {})
                        related_username = node.get("username")
                        if not related_username or not isinstance(
                            related_username, str
                        ):
                            continue

                        # Select-then-insert: check if account already exists
                        stmt = (
                            select(Account.id)
                            .where(
                                Account.platform == "INSTAGRAM",
                                Account.username == related_username,
                            )
                            .limit(1)
                        )
                        result = await session.execute(stmt)
                        existing = result.scalar()

                        if not existing:
                            # Insert new Account for chaining
                            new_account = Account(
                                platform="INSTAGRAM",
                                platform_id=related_username,
                                username=related_username,
                                title=related_username,
                                status="pending",
                            )
                            session.add(new_account)
                            logger.debug(
                                "Chained new INSTAGRAM account: %s from seed: %s",
                                related_username,
                                handle,
                            )

                    await session.commit()
                    logger.info(
                        "Profile chaining discovery completed for %s", handle
                    )
        except Exception as e:
            logger.error(
                "Profile chaining discovery failed for %s: %s",
                handle,
                e,
                exc_info=True,
            )

        return account_id

    async def _get_cached_or_fetch_profile(
        self, handle: str
    ) -> dict[str, Any] | None:
        """Get profile from cache or fetch from API.

        Args:
            handle: Instagram username to fetch.

        Returns:
            Profile data dictionary, or None if fetch fails.
        """
        if self._cached_handle == handle and self._cached_profile:
            logger.debug("Using cached profile data for handle: %s", handle)
            return self._cached_profile

        try:
            response = await self.client.get(
                endpoint="/v1/instagram/profile",
                params={"handle": handle},
            )
            data = response.get("data")
            if not data:
                logger.error(
                    "Missing 'data' in API response for Instagram handle %s",
                    handle,
                )
                return None

            user = data.get("user") or data
            if not user:
                logger.error(
                    "Missing user data for Instagram handle %s", handle
                )
                return None

            # Cache the profile
            self._cached_profile = user
            self._cached_handle = handle
            return user

        except ClientResponseError as e:
            if e.status == 404:
                logger.warning(
                    "Instagram profile %s not found (404). Marking as rejected.",
                    handle,
                )
            else:
                logger.error(
                    "Instagram API request failed for %s with HTTP %d: %s",
                    handle,
                    e.status,
                    e,
                )
            return None
        except Exception as e:
            logger.error(
                "Unexpected error fetching Instagram profile %s: %s",
                handle,
                e,
                exc_info=True,
            )
            return None

    async def _upsert_account(
        self, profile: dict[str, Any], status: str
    ) -> int:
        """Select-then-upsert account to avoid database conflicts.

        Args:
            profile: Instagram profile data dictionary.
            status: Account status (e.g., "parsed", "rejected").

        Returns:
            Database ID of the upserted account.
        """
        platform_id = str(profile.get("id") or profile.get("username", ""))
        username = profile.get("username")
        full_name = profile.get("full_name")
        biography = profile.get("biography")
        subscribers = extract_instagram_subscribers(profile)

        async with self.session_maker() as session:
            stmt = select(Account).where(
                Account.platform == "INSTAGRAM",
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
                    platform="INSTAGRAM",
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

    async def parse_content(
        self, account_id: int, platform_id: str, max_items: int = 50
    ) -> None:
        """Parse timeline content via paginated posts endpoint, fetch transcripts, and bulk upsert.

        Implements Smart Adaptive Pagination:
        - Early exit on page 1 if no valid videos found (photo-only accounts)
        - 5-page ceiling to limit credit usage
        - Collects up to max_items (50) valid videos

        Args:
            account_id: Database ID of the associated account.
            platform_id: Instagram platform ID (username or ID).
            max_items: Maximum number of valid items to collect (default: 50).
        """
        # Smart Adaptive Pagination: 5-page ceiling to limit credit usage
        MAX_PAGES_TO_CHECK: int = 5

        logger.info(
            "Starting Instagram content parse for account_id: %d, platform_id: %s",
            account_id,
            platform_id,
        )

        # Fetch profile for author metadata
        profile = await self._get_cached_or_fetch_profile(platform_id)
        if not profile:
            return

        author_metadata = build_instagram_author_metadata(profile)

        # Paginated fetch from /v2/instagram/user/posts
        valid_items: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        next_max_id: str | None = None
        page_count = 0

        while len(valid_items) < max_items:
            page_count += 1

            # Ceiling Check: stop pagination after MAX_PAGES_TO_CHECK pages
            if page_count >= MAX_PAGES_TO_CHECK:
                logger.info(
                    "Reached page limit (%d) for account %s, stopping pagination with %d videos collected",
                    MAX_PAGES_TO_CHECK,
                    platform_id,
                    len(valid_items),
                )
                break

            # Page fetch log with required format
            logger.debug(
                "Fetching page %d of posts for %s...", page_count, platform_id
            )

            params: dict[str, Any] = {"handle": platform_id}
            if next_max_id:
                params["next_max_id"] = next_max_id

            try:
                response = await self.client.get(
                    endpoint="/v2/instagram/user/posts",
                    params=params,
                )
            except Exception as e:
                logger.error(
                    "Failed to fetch posts page %d for %s: %s",
                    page_count,
                    platform_id,
                    e,
                )
                break

            # Extract top-level fields from Scrape Creators response
            items: list[dict[str, Any]] = response.get("items", [])
            next_max_id = response.get("next_max_id")
            more_available: bool = response.get("more_available", False)

            if not items:
                logger.info(
                    "No more items found for %s after %d pages",
                    platform_id,
                    page_count,
                )
                break

            # Process items - filter for videos under 120 seconds
            for item in items:
                item_id = (
                    item.get("id") or item.get("media_id") or item.get("pk")
                )
                if not item_id:
                    continue

                # Deduplicate
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)

                # Check if it's a valid video under 120 seconds
                # Strict video detection: exclude static photos and carousels
                # A valid Instagram video must satisfy one of these criteria:
                # 1. media_type == 2 (standard Instagram code for video/reels)
                # 2. is_video is explicitly True
                # 3. video_versions is present, is a list, and is NOT empty
                is_video = (
                    item.get("media_type") == 2
                    or item.get("is_video") is True
                    or (
                        isinstance(item.get("video_versions"), list)
                        and bool(item.get("video_versions"))
                    )
                )
                if not is_video:
                    continue

                # Extract duration safely
                # Static images may return 0.0 or have missing duration
                duration = item.get("video_duration") or item.get("duration", 0.0)
                # Valid video must have duration strictly between 0 and 120 seconds
                if not (0.0 < duration <= 120.0):
                    continue

                valid_items.append(item)

                if len(valid_items) >= max_items:
                    break

            # Early Exit Check: After processing page 1, if no valid videos found,
            # immediately break to avoid wasting credits on photo-only or dead accounts
            if page_count == 1 and len(valid_items) == 0:
                logger.info(
                    "Early exit: No valid videos found on page 1 for Instagram handle '%s'. Stopping to avoid credit waste.",
                    platform_id,
                )
                break

            # Check if we should continue pagination
            if not more_available or next_max_id is None:
                break

        if not valid_items:
            logger.info(
                "No valid Instagram content found for account_id: %d",
                account_id,
            )
            return

        logger.info(
            "Fetched %d valid video items for account_id: %d after %d pages",
            len(valid_items),
            account_id,
            page_count,
        )

        # Get existing transcripts to skip already-processed content
        platform_ids: list[str] = [
            item_id
            for item in valid_items
            if (
                item_id := (
                    item.get("id") or item.get("media_id") or item.get("pk")
                )
            )
            is not None
        ]
        existing_transcripts = await self._get_existing_transcripts(
            account_id, platform_ids
        )

        # Fetch transcripts concurrently for eligible videos
        semaphore = asyncio.Semaphore(5)
        transcript_tasks = []
        eligible_items = []

        for item in valid_items:
            item_id = item.get("id") or item.get("media_id") or item.get("pk")
            if not item_id or item_id in existing_transcripts:
                continue

            # Extract shortcode to construct post permalink for transcript API
            # Note: extract_instagram_video_url() is still used later (line ~615)
            # to save the CDN video URL in raw_metadata
            shortcode = item.get("code") or item.get("shortcode")
            if not shortcode:
                continue

            # Construct Instagram post permalink (no 'www', no trailing slash)
            post_url = f"https://instagram.com/p/{shortcode}"

            eligible_items.append(item)
            transcript_tasks.append(
                self._fetch_transcript(semaphore, post_url)
            )

        transcripts: dict[str, str | None] = {}
        if transcript_tasks:
            results = await asyncio.gather(
                *transcript_tasks, return_exceptions=True
            )
            for item, result in zip(eligible_items, results, strict=False):
                item_id = (
                    item.get("id") or item.get("media_id") or item.get("pk")
                )
                if isinstance(result, Exception):
                    logger.error(
                        "Transcript fetch failed for %s: %s", item_id, result
                    )
                elif isinstance(result, str) or result is None:
                    transcripts[item_id] = result

        # Build and upsert content items
        content_values = []
        for item in valid_items:
            item_id = item.get("id") or item.get("media_id") or item.get("pk")
            if not item_id:
                continue

            likes, comments = extract_instagram_metrics(item)

            # Extract content text for discovery spider
            content_text: str | None = transcripts.get(
                item_id
            ) or extract_instagram_content_text(item)

            # Content-based Discovery Spider
            # Extract cross-platform links and same-platform mentions from post caption
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
                            session, contacts_dict, profile.get("username", "")
                        )
                        # Same-platform mentions
                        await queue_discovered_mentions(
                            session,
                            "INSTAGRAM",
                            mentions,
                            profile.get("username", ""),
                        )
                        await session.commit()
                except Exception as e:
                    logger.warning(
                        "Failed to queue discovered accounts from Instagram post %s: %s",
                        item_id,
                        e,
                    )

            content_values.append(
                {
                    "account_id": account_id,
                    "platform_content_id": item_id,
                    "content": content_text,
                    "published_at": extract_instagram_published_at(item),
                    "transcription": transcripts.get(item_id),
                    "views": item.get("video_view_count")
                    or item.get("play_count"),
                    "reactions_count": likes,
                    "comments_count": comments,
                    "shares_count": None,
                    "has_media": True,
                    "is_embedded": False,
                    "is_graph_extracted": False,
                    "raw_metadata": {
                        "platform_metrics": {
                            "likes": likes,
                            "comments": comments,
                            "video_url": self._extract_video_url(item),
                        },
                        "author_profile_metadata": author_metadata,
                        "raw_item_payload": item,
                    },
                    "updated_at": datetime.now(timezone.utc),
                }
            )

        if content_values:
            async with self.session_maker() as session:
                async with session.begin():
                    stmt = pg_insert(Content).values(content_values)
                    stmt = stmt.on_conflict_do_update(
                        constraint="uq_content_account_platform_id",
                        set_=dict(
                            content=stmt.excluded.content,
                            transcription=func.coalesce(
                                stmt.excluded.transcription,
                                Content.transcription,
                            ),
                            views=stmt.excluded.views,
                            reactions_count=stmt.excluded.reactions_count,
                            comments_count=stmt.excluded.comments_count,
                            raw_metadata=stmt.excluded.raw_metadata,
                            updated_at=stmt.excluded.updated_at,
                        ),
                    )
                    await session.execute(stmt)
                    logger.info(
                        "Bulk upserted %d Instagram content items for account_id: %d",
                        len(content_values),
                        account_id,
                    )

    def _extract_video_url(self, item: dict[str, Any]) -> str | None:
        """Extract video URL from Instagram post item with robust fallback logic.

        Implements a three-tier extraction strategy:
        1. Check video_versions array (most reliable, used by Instagram internally)
        2. Fallback to video_url field
        3. Final fallback to extract_instagram_video_url helper

        Args:
            item: Instagram post dictionary.

        Returns:
            Video URL string if found, otherwise None.
        """
        # Tier 1: Check video_versions array (most reliable source)
        video_versions = item.get("video_versions")
        if isinstance(video_versions, list) and len(video_versions) > 0:
            first_video = video_versions[0]
            if isinstance(first_video, dict):
                url = first_video.get("url")
                if isinstance(url, str) and url:
                    return url

        # Tier 2: Fallback to video_url field
        video_url = item.get("video_url")
        if isinstance(video_url, str) and video_url:
            return video_url

        # Tier 3: Final fallback to extract_instagram_video_url
        return extract_instagram_video_url(item)

    async def _fetch_transcript(
        self, semaphore: asyncio.Semaphore, post_url: str
    ) -> str | None:
        """Fetch media transcript with rate limiting using semaphore.

        Args:
            semaphore: Asyncio semaphore for rate limiting.
            post_url: Instagram post permalink (e.g., https://instagram.com/p/SHORTCODE).

        Returns:
            Transcript text if available, otherwise None.
        """
        # Transcript request log with required format
        logger.debug("Requesting transcript for post: %s", post_url)

        async with semaphore:
            try:
                response = await self.client.get(
                    endpoint="/v2/instagram/media/transcript",
                    params={"url": post_url},
                )
                # Parse the transcripts list from the API response
                # Scrape Creators API returns: [{"id": "...", "shortcode": "...", "text": "..."}]
                # or ["transcript text"] (list of strings)
                # or [null] / [] if empty
                transcripts = response.get("transcripts")
                if isinstance(transcripts, list) and len(transcripts) > 0:
                    first_item = transcripts[0]

                    # Skip None values in the list
                    if first_item is None:
                        logger.debug(
                            "No transcript available for post: %s", post_url[:50]
                        )
                        return None

                    transcript_text: str | None = None

                    if isinstance(first_item, str) and first_item:
                        # Handle case where API returns list of strings
                        transcript_text = first_item
                    elif isinstance(first_item, dict):
                        # Handle case where API returns list of dictionaries
                        text_value = first_item.get("text")
                        if isinstance(text_value, str) and text_value:
                            transcript_text = text_value

                    if transcript_text:
                        # Transcript success log with required format, handling None safely
                        logger.debug(
                            "Successfully retrieved transcript for post %s",
                            post_url[:50],
                        )
                        return transcript_text

                logger.debug(
                    "No transcript available for post: %s", post_url[:50]
                )
                return None
            except Exception as e:
                logger.error("Transcript fetch failed for %s: %s", post_url, e)
                return None

    async def _get_existing_transcripts(
        self, account_id: int, platform_ids: list[str]
    ) -> set[str]:
        """Get set of platform_content_ids that already have transcripts.

        Args:
            account_id: Database ID of the associated account.
            platform_ids: List of platform content IDs to check.

        Returns:
            Set of platform content IDs that already have transcripts.
        """
        if not platform_ids:
            return set()

        async with self.session_maker() as session:
            stmt = select(Content.platform_content_id).where(
                Content.account_id == account_id,
                Content.platform_content_id.in_(platform_ids),
                Content.transcription.isnot(None),
            )
            result = await session.execute(stmt)
            return {row[0] for row in result}
