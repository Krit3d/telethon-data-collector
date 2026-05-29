"""
Instagram platform parser using Scrape Creators API v2.

Implements Instagram-specific profile parsing and content ingestion into PostgreSQL.
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
                handle, list(profile.keys())
            )

        if not (MIN_SUBSCRIBERS <= subscribers <= MAX_SUBSCRIBERS):
            logger.warning(
                "Instagram handle %s REJECTED: subscriber count %d is outside range [%d, %d].",
                handle, subscribers, MIN_SUBSCRIBERS, MAX_SUBSCRIBERS
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
                handle
            )
            await self._upsert_account(profile, "rejected")
            return None

        # Check for slop/theme stop-words
        if is_slop_or_theme_page(username, biography):
            logger.warning(
                "Instagram handle %s REJECTED: Matched slop/theme stop-words.",
                handle
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
            handle, account_id, subscribers
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

        # Queue discovered accounts from contacts
        contacts_dict = parse_profile_contacts(biography, profile.get("external_url"))
        async with self.session_maker() as session:
            await queue_discovered_accounts(
                session=session,
                contacts_dict=contacts_dict,
                parent_handle=handle,
            )
            await session.commit()

        return account_id

    async def _get_cached_or_fetch_profile(self, handle: str) -> dict[str, Any] | None:
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
                endpoint="/v2/instagram/profile",
                params={"handle": handle},
            )
            data = response.get("data")
            if not data:
                logger.error("Missing 'data' in API response for Instagram handle %s", handle)
                return None

            user = data.get("user") or data
            if not user:
                logger.error("Missing user data for Instagram handle %s", handle)
                return None

            # Cache the profile
            self._cached_profile = user
            self._cached_handle = handle
            return user

        except ClientResponseError as e:
            if e.status == 404:
                logger.warning("Instagram profile %s not found (404). Marking as rejected.", handle)
            else:
                logger.error(
                    "Instagram API request failed for %s with HTTP %d: %s",
                    handle, e.status, e
                )
            return None
        except Exception as e:
            logger.error(
                "Unexpected error fetching Instagram profile %s: %s",
                handle, e, exc_info=True
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

        Args:
            account_id: Database ID of the associated account.
            platform_id: Instagram platform ID (username or ID).
            max_items: Maximum number of valid items to collect (default: 50).
        """
        logger.info(
            "Starting Instagram content parse for account_id: %d, platform_id: %s",
            account_id, platform_id
        )

        # Fetch profile for author metadata
        profile = await self._get_cached_or_fetch_profile(platform_id)
        if not profile:
            return

        author_metadata = build_instagram_author_metadata(profile)

        # Paginated fetch from /v2/instagram/posts
        valid_items: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        cursor: str | None = None
        page_count = 0

        while len(valid_items) < max_items:
            page_count += 1
            params: dict[str, Any] = {"handle": platform_id}
            if cursor:
                params["cursor"] = cursor

            try:
                response = await self.client.get(
                    endpoint="/v2/instagram/posts",
                    params=params,
                )
            except Exception as e:
                logger.error("Failed to fetch posts page %d for %s: %s", page_count, platform_id, e)
                break

            # Extract items from response
            data = response.get("data") or response
            items: list[dict[str, Any]] = []
            has_next = False

            # Handle different response structures
            if isinstance(data, dict):
                items = data.get("items") or data.get("posts") or []
                page_info = data.get("page_info") or {}
                cursor = page_info.get("end_cursor") or response.get("next_cursor")
                has_next = page_info.get("has_next_page", False)
            elif isinstance(data, list):
                items = data
                cursor = response.get("next_cursor") or response.get("end_cursor")
                has_next = bool(cursor)
            else:
                items = []

            if not items:
                logger.info("No more items found for %s after %d pages", platform_id, page_count)
                break

            # Process items - filter for videos under 120 seconds
            for item in items:
                item_id = item.get("id") or item.get("media_id") or item.get("pk")
                if not item_id:
                    continue

                # Deduplicate
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)

                # Check if it's a valid video under 120 seconds
                is_video = item.get("is_video", False)
                if not is_video:
                    continue

                duration = item.get("video_duration") or item.get("duration", 0.0)
                if duration > 120.0:
                    continue

                valid_items.append(item)

                if len(valid_items) >= max_items:
                    break

            # Check if we should continue pagination
            if not has_next or not cursor:
                break

        if not valid_items:
            logger.info("No valid Instagram content found for account_id: %d", account_id)
            return

        logger.info(
            "Fetched %d valid video items for account_id: %d after %d pages",
            len(valid_items), account_id, page_count
        )

        # Get existing transcripts to skip already-processed content
        platform_ids: list[str] = [
            item_id for item in valid_items
            if (item_id := (item.get("id") or item.get("media_id") or item.get("pk"))) is not None
        ]
        existing_transcripts = await self._get_existing_transcripts(account_id, platform_ids)

        # Fetch transcripts concurrently for eligible videos
        semaphore = asyncio.Semaphore(5)
        transcript_tasks = []
        eligible_items = []

        for item in valid_items:
            item_id = item.get("id") or item.get("media_id") or item.get("pk")
            if not item_id or item_id in existing_transcripts:
                continue

            # Get video URL for transcript endpoint
            video_url = extract_instagram_video_url(item)
            if not video_url:
                continue

            eligible_items.append(item)
            transcript_tasks.append(self._fetch_transcript(semaphore, video_url))

        transcripts: dict[str, str | None] = {}
        if transcript_tasks:
            results = await asyncio.gather(*transcript_tasks, return_exceptions=True)
            for item, result in zip(eligible_items, results, strict=False):
                item_id = item.get("id") or item.get("media_id") or item.get("pk")
                if isinstance(result, Exception):
                    logger.error("Transcript fetch failed for %s: %s", item_id, result)
                elif isinstance(result, str) or result is None:
                    transcripts[item_id] = result

        # Build and upsert content items
        content_values = []
        for item in valid_items:
            item_id = item.get("id") or item.get("media_id") or item.get("pk")
            if not item_id:
                continue

            likes, comments = extract_instagram_metrics(item)
            content_values.append({
                "account_id": account_id,
                "platform_content_id": item_id,
                "content": transcripts.get(item_id) or extract_instagram_content_text(item),
                "published_at": extract_instagram_published_at(item),
                "transcription": transcripts.get(item_id),
                "views": item.get("video_view_count") or item.get("play_count"),
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
                        "video_url": extract_instagram_video_url(item),
                    },
                    "author_profile_metadata": author_metadata,
                },
                "updated_at": datetime.now(timezone.utc),
            })

        if content_values:
            async with self.session_maker() as session:
                async with session.begin():
                    stmt = pg_insert(Content).values(content_values)
                    stmt = stmt.on_conflict_do_update(
                        constraint="uq_content_account_platform_id",
                        set_=dict(
                            content=stmt.excluded.content,
                            transcription=func.coalesce(stmt.excluded.transcription, Content.transcription),
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
                        len(content_values), account_id
                    )

    async def _fetch_transcript(self, semaphore: asyncio.Semaphore, video_url: str) -> str | None:
        """Fetch media transcript with rate limiting using semaphore.

        Args:
            semaphore: Asyncio semaphore for rate limiting.
            video_url: URL of the video to fetch transcript for.

        Returns:
            Transcript text if available, otherwise None.
        """
        async with semaphore:
            try:
                response = await self.client.get(
                    endpoint="/v2/instagram/media/transcript",
                    params={"url": video_url},
                )
                # Parse the transcripts list from the API response
                transcripts = response.get("transcripts")
                if (
                    isinstance(transcripts, list)
                    and len(transcripts) > 0
                    and isinstance(transcripts[0], str)
                    and transcripts[0]
                ):
                    return transcripts[0]
                return None
            except Exception as e:
                logger.error("Transcript fetch failed for %s: %s", video_url, e)
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
