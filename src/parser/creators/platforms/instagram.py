"""
Instagram platform parser using Scrape Creators API.

Implements Instagram-specific profile parsing and content ingestion into PostgreSQL.
Uses v1 API for profile endpoint and v2 API for posts and transcripts.
Extracts author profile metadata and stores it inside Content.raw_metadata.

Refactored to use centralized db.py architecture:
- Uses upsert_and_deduplicate_account for account operations
- Uses update_account_profile_metadata for profile metadata
- Uses bulk_upsert_content for content operations
- Implements conditional two-page pagination to guarantee 10 valid videos
- Limits to max 10 videos to reduce credit consumption
- Implements conditional transcript fetching based on semantic keywords

Implements Two-Stage Russian Language Check:
- Stage1 (parse_profile): Check biography/full_name for Cyrillic
- Stage2 (parse_content): Check post captions/hashtags for Cyrillic
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any

from aiohttp import ClientResponseError
from sqlalchemy import update, select
from src.db.models import Account

from src.parser.creators.core.queries import SearchQueriesManager
from src.parser.creators.core.schemas import (
    InstagramContentMetadata,
    PlatformMetrics,
    AuthorProfileSnapshot,
)
from src.parser.creators.core.utils import (
    upsert_and_deduplicate_account,
    update_account_profile_metadata,
    bulk_upsert_content,
    extract_instagram_subscribers,
    is_russian_text,
    is_slop_or_theme_page,
    queue_discovered_accounts,
    queue_discovered_mentions,
    extract_mentions,
    extract_instagram_content_text,
    extract_instagram_published_at,
    extract_instagram_metrics,
    parse_profile_contacts,
    extract_instagram_video_url,
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
        self._queries_manager = SearchQueriesManager(settings.search_queries_path)

    async def parse_profile(self, handle: str) -> int | None:
        """Fetch Instagram profile, apply Stage1 Russian language check, upsert account.

        Stage1 Russian Language Check:
        - Extract biography, full_name, and username
        - If biography is empty/None after stripping: pass to Stage2 with "processing" status
        - If biography is NOT empty: check Cyrillic in biography OR full_name
        - If no Cyrillic found: reject account immediately

        Args:
            handle: Instagram username (without @ prefix).

        Returns:
            Database ID of the upserted account record, or None if the profile
            could not be fetched from the API.
            Note: Returns the account_id even when the profile is rejected,
            so the coordinator can use the correct merged account ID.
        """
        logger.info("Starting Instagram profile parse for handle: %s", handle)

        profile = await self._get_cached_or_fetch_profile(handle)
        if not profile:
            return None

        # Extract profile fields for Stage1 validation
        username = profile.get("username", "")
        biography = profile.get("biography")  # May be None from API
        full_name = profile.get("full_name", "")

        # Subscriber threshold filter
        subscribers = extract_instagram_subscribers(profile)

        if subscribers == 0:
            logger.warning(
                "Instagram handle %s parsed 0 subscribers. Profile dict keys: %s",
                handle,
                list(profile.keys()),
            )

        # Apply subscriber range filter
        if not (MIN_SUBSCRIBERS <= subscribers <= MAX_SUBSCRIBERS):
            logger.warning(
                "Instagram handle %s REJECTED: subscriber count %d is outside range [%d, %d].",
                handle,
                subscribers,
                MIN_SUBSCRIBERS,
                MAX_SUBSCRIBERS,
            )
            async with self.session_maker() as session:
                account_id = await upsert_and_deduplicate_account(
                    session=session,
                    platform="INSTAGRAM",
                    platform_id=str(profile.get("id") or username),
                    username=username,
                    title=full_name or username or "Unknown",
                    description=biography or "",
                    subscribers_count=subscribers,
                    status="rejected",
                )
                await session.commit()
                return account_id

        # Check for slop/theme stop-words
        if is_slop_or_theme_page(username, biography or ""):
            logger.warning(
                "Instagram handle %s REJECTED: Matched slop/theme stop-words.",
                handle,
            )
            async with self.session_maker() as session:
                account_id = await upsert_and_deduplicate_account(
                    session=session,
                    platform="INSTAGRAM",
                    platform_id=str(profile.get("id") or username),
                    username=username,
                    title=full_name or username or "Unknown",
                    description=biography or "",
                    subscribers_count=subscribers,
                    status="rejected",
                )
                await session.commit()
                return account_id

        # ===================================================================
        # STAGE1: RUSSIAN LANGUAGE CHECK (BIOGRAPHY)
        # ===================================================================
        biography_stripped = biography.strip() if biography else ""

        if not biography_stripped:
            # Biography is empty/None: let account pass to Stage2 for post validation
            logger.info(
                "Instagram handle %s: biography is empty, passing to Stage2 content validation.",
                handle,
            )
            async with self.session_maker() as session:
                account_id = await upsert_and_deduplicate_account(
                    session=session,
                    platform="INSTAGRAM",
                    platform_id=str(profile.get("id") or username),
                    username=username,
                    title=full_name or username or "Unknown",
                    description=biography or "",
                    subscribers_count=subscribers,
                    status="processing",
                )

                # Update profile metadata using centralized function
                await update_account_profile_metadata(
                    session=session,
                    account_id=account_id,
                    platform="INSTAGRAM",
                    biography=biography or "",
                    external_url=profile.get("external_url"),
                )

                await session.commit()

            logger.info(
                "Successfully parsed Instagram profile %s, account ID: %d, subscribers: %d",
                handle,
                account_id,
                subscribers,
            )
            return account_id

        # Biography is NOT empty: check for Cyrillic characters
        has_cyrillic_bio = is_russian_text(biography)
        has_cyrillic_name = is_russian_text(full_name)

        if not (has_cyrillic_bio or has_cyrillic_name):
            # No Cyrillic found in biography or full_name: reject account
            logger.warning(
                "Instagram handle %s REJECTED: Non-empty biography and name contain no Cyrillic characters.",
                handle,
            )
            async with self.session_maker() as session:
                account_id = await upsert_and_deduplicate_account(
                    session=session,
                    platform="INSTAGRAM",
                    platform_id=str(profile.get("id") or username),
                    username=username,
                    title=full_name or username or "Unknown",
                    description=biography or "",
                    subscribers_count=subscribers,
                    status="rejected",
                )
                await session.commit()
                return account_id

        # Cyrillic found in biography or full_name: pass to Stage2
        logger.info(
            "Instagram handle %s: Stage1 PASSED (Cyrillic detected). Passing to Stage2.",
            handle,
        )
        async with self.session_maker() as session:
            account_id = await upsert_and_deduplicate_account(
                session=session,
                platform="INSTAGRAM",
                platform_id=str(profile.get("id") or username),
                username=username,
                title=full_name or username or "Unknown",
                description=biography or "",
                subscribers_count=subscribers,
                status="processing",
            )

            # Update profile metadata using centralized function
            await update_account_profile_metadata(
                session=session,
                account_id=account_id,
                platform="INSTAGRAM",
                biography=biography or "",
                external_url=profile.get("external_url"),
            )

            await session.commit()

        logger.info(
            "Successfully parsed Instagram profile %s, account ID: %d, subscribers: %d",
            handle,
            account_id,
            subscribers,
        )
        return account_id

    async def discover_candidates(self, query: str, category: str) -> int:
        """Discover Instagram creator candidates using Scrape Creators search API.

        Queries the /v1/instagram/search/profiles endpoint to find Instagram profiles
        matching the given search query, then filters and upserts valid candidates
        into the database for further processing.

        Args:
            query: Search query string (e.g., "fitness influencer").
            category: Category name for the discovered accounts (e.g., "fitness").

        Returns:
            Number of successfully discovered and stored candidate accounts.
            Returns 0 if the API request fails or an exception occurs.
        """
        logger.info(
            "Starting Instagram candidate discovery for query: '%s', category: '%s'",
            query,
            category,
        )

        try:
            # Call Scrape Creators API to search for profiles
            response = await self.client.get(
                endpoint="/v1/instagram/search/profiles",
                params={"query": query},
            )

            if not response:
                logger.warning(
                    "Empty response from Instagram search API for query: '%s'",
                    query,
                )
                return 0

            # Robustly extract profiles from response
            # API might return them under "profiles", "data", or "items"
            profiles: list[dict[str, Any]] = []
            if isinstance(response.get("profiles"), list):
                profiles = response["profiles"]
            elif isinstance(response.get("data"), list):
                profiles = response["data"]
            elif isinstance(response.get("items"), list):
                profiles = response["items"]
            elif isinstance(response, list):
                # Response might be a list directly
                profiles = response
            else:
                logger.warning(
                    "Unexpected Instagram search API response structure for query: '%s'. "
                    "Response keys: %s",
                    query,
                    list(response.keys()) if isinstance(response, dict) else type(response),
                )
                return 0

            if not profiles:
                logger.info(
                    "No profiles found in Instagram search results for query: '%s'",
                    query,
                )
                return 0

            discovered_count = 0

            # Process each profile
            for profile in profiles:
                if not isinstance(profile, dict):
                    continue

                # Safely retrieve username (or handle)
                username = profile.get("username") or profile.get("handle")
                if not username or not isinstance(username, str):
                    logger.debug(
                        "Skipping profile with missing username in search results: %s",
                        profile.get("id", "unknown"),
                    )
                    continue

                # Safely extract subscriber/follower count
                followers = profile.get("follower_count") or profile.get("followers")
                if followers is None:
                    # Try nested paths
                    followers = (
                        profile.get("stats", {}).get("followers")
                        or profile.get("user", {}).get("follower_count")
                    )

                # Convert to int if possible, default to 0
                try:
                    followers = int(followers) if followers is not None else 0
                except (ValueError, TypeError):
                    followers = 0

                # Perform subscriber range check
                # Skip if followers == 0 (invalid data) or outside [MIN, MAX] range
                if followers == 0:
                    logger.debug(
                        "Skipping Instagram profile %s: follower count is 0",
                        username,
                    )
                    continue

                if not (MIN_SUBSCRIBERS <= followers <= MAX_SUBSCRIBERS):
                    logger.debug(
                        "Skipping Instagram profile %s: follower count %d outside range [%d, %d]",
                        username,
                        followers,
                        MIN_SUBSCRIBERS,
                        MAX_SUBSCRIBERS,
                    )
                    continue

                # Extract profile fields for upsert
                profile_id = profile.get("id")
                full_name = profile.get("full_name", "")
                biography = profile.get("biography", "") or ""

                # Upsert account using centralized helper
                async with self.session_maker() as session:
                    try:
                        account_id = await upsert_and_deduplicate_account(
                            session=session,
                            platform="INSTAGRAM",
                            platform_id=str(profile_id or username),
                            username=username,
                            title=full_name or username,
                            description=biography,
                            subscribers_count=followers,
                            status="pending",
                        )

                        # Prepare raw metadata dict with category and search metadata
                        meta: dict[str, Any] = {
                            "category": category,
                            "discovery_query": query,
                            "search_metadata": profile,
                        }

                        # Execute explicit SQLAlchemy update to store meta
                        # This guarantees Phase 2 processing (parse_content) can read the category name
                        stmt = (
                            update(Account)
                            .where(Account.id == account_id)
                            .values(raw_metadata=meta, updated_at=datetime.now(timezone.utc))
                        )
                        await session.execute(stmt)
                        await session.commit()

                        discovered_count += 1
                        logger.debug(
                            "Discovered and stored Instagram candidate: %s (account_id: %d)",
                            username,
                            account_id,
                        )

                    except Exception as e:
                        logger.error(
                            "Failed to upsert Instagram profile %s: %s",
                            username,
                            e,
                            exc_info=True,
                        )
                        # Continue processing other profiles
                        continue

            logger.info(
                "Instagram candidate discovery completed for query: '%s'. "
                "Discovered %d valid candidates.",
                query,
                discovered_count,
            )
            return discovered_count

        except Exception as e:
            logger.error(
                "Instagram candidate discovery failed for query: '%s': %s",
                query,
                e,
                exc_info=True,
            )
            return 0

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

    def _has_sufficient_semantics(
        self,
        bio: str | None,
        external_url: str | None,
        caption: str | None,
        hashtags: list[str],
        has_target_semantics: bool,
    ) -> bool:
        """Decide if a post has enough semantic content to skip transcription.

        Returns True (skip transcription) ONLY when ALL of the following hold:
        1. ``has_target_semantics`` is True (matched search query keywords).
        2. The caption is "rich enough":
           - ``caption`` length is strictly greater than 120 characters, OR
           - ``hashtags`` list contains 3 or more items.
        3. The account bio or external URL contains at least one contact/social
           indicator (email, Telegram handle/link, or any HTTP/HTTPS URL).

        Returns False in all other cases, meaning a transcript MUST be requested.
        
        Note:
            Telegram pattern matches:
            - Standard usernames: @username
            - Classic links: t.me/username, telegram.me/username, telegram.dog/username
            - Private channels: t.me/+invitehash (with + character)
            - Custom domains and slugs with hyphens
            - Text mentions: тг, тгк, телеграм, tg, telegram, канал (as distinct words)
        """
        # Condition A: must have matched target semantics
        # Return False immediately (always transcribe) if no target semantics
        if not has_target_semantics:
            return False

        # Condition B: caption must be rich enough
        # Return False immediately (always transcribe) if caption is too short AND has few hashtags
        caption_len = len(caption) if caption else 0
        if caption_len <= 120 and len(hashtags) < 3:
            return False

        # Condition C: bio or external_url must contain a contact/social indicator
        # Check case-insensitively by converting to lower case
        bio = bio or ""
        external_url = external_url or ""
        combined = (bio + " " + external_url).lower()

        # Email pattern
        has_email = bool(re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", combined))
        
        # Improved Telegram pattern (case-insensitive)
        # Matches: @username, t.me/*, telegram.me/*, telegram.dog/*, and text mentions
        telegram_pattern = r"@\w+|t\.me/[a-zA-Z0-9_\+\-]+|telegram\.(?:me|dog)/[a-zA-Z0-9_\+\-]+|\b(?:тг|тгк|телеграм|tg|telegram|канал)\b"
        has_telegram = bool(re.search(telegram_pattern, combined, re.IGNORECASE))
        
        # Any http(s) URL
        has_external_link = bool(re.search(r"https?://", combined))

        # Return False immediately (always transcribe) if no valid contact details present
        if not (has_email or has_telegram or has_external_link):
            return False

        # All hurdles passed: skip transcription
        return True

    async def parse_content(
        self, account_id: int, platform_id: str, max_items: int = 10
    ) -> None:
        """Parse timeline content via conditional two-page pagination.

        Implements efficient credit-efficient parsing:
        - First API call to /v2/instagram/user/posts (page 1)
        - Filter for valid videos under 120 seconds (Reels)
        - If fewer than 10 valid videos AND more_available is true
          AND cursor exists: fetch page 2 via cursor/max_id
        - Combine results from both pages, deduplicate, limit to 10
        - Conditional transcript fetching based on semantic keywords
        - Extracts and saves high-quality MP4 URL for GPU embedding
        - Stage2 Cyrillic validation: rejects account if no Russian text in posts

        Args:
            account_id: Database ID of the associated account.
            platform_id: Instagram platform ID (username or ID).
            max_items: Maximum number of valid items to collect (default: 10).
        """
        logger.info(
            "Starting Instagram content parse for account_id: %d, platform_id: %s",
            account_id,
            platform_id,
        )

        # Query database to get existing account category from raw_metadata
        account_category = "unknown"
        async with self.session_maker() as session:
            stmt = select(Account.raw_metadata).where(Account.id == account_id)
            result = await session.execute(stmt)
            raw_metadata = result.scalar_one_or_none()
            if raw_metadata and isinstance(raw_metadata, dict):
                category = raw_metadata.get("category")
                if category and isinstance(category, str):
                    account_category = category

        # Fetch profile for author metadata
        profile = await self._get_cached_or_fetch_profile(platform_id)
        if not profile:
            raise RuntimeError(f"Could not retrieve profile metadata for {platform_id} during content parsing.")

        # Extract bio and external_url from profile for semantics helper
        profile_biography = profile.get("biography")
        profile_external_url = profile.get("external_url")

        # Build author profile snapshot from profile data
        author_profile_snapshot = AuthorProfileSnapshot(
            username=profile.get("username", ""),
            title=profile.get("full_name") or profile.get("username", ""),
        )

        # ===================================================================
        # STEP 1: FETCH PAGE 1
        # ===================================================================
        try:
            response_page1 = await self.client.get(
                endpoint="/v2/instagram/user/posts",
                params={"handle": platform_id},
            )
        except Exception as e:
            logger.error(
                "Failed to fetch posts for %s: %s",
                platform_id,
                e,
            )
            raise

        # Extract items from page 1
        items_page1: list[dict[str, Any]] = response_page1.get("items", [])

        if not items_page1:
            logger.info("No items found for %s", platform_id)
            return

        # ===================================================================
        # STEP 2: FILTER VALID VIDEOS FROM PAGE 1
        # ===================================================================
        valid_items: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for item in items_page1:
            item_id = item.get("id") or item.get("media_id") or item.get("pk")
            if not item_id:
                continue

            # Deduplicate
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            # Check if it's a valid video under 120 seconds
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
            duration = item.get("video_duration") or item.get("duration", 0.0)
            # Valid video must have duration strictly between 0 and 120 seconds
            if not (0.0 < duration <= 120.0):
                continue

            valid_items.append(item)

            if len(valid_items) >= max_items:
                break

        # ===================================================================
        # STEP 3: CONDITIONAL PAGINATION (PAGE 2)
        # ===================================================================
        # Check if we need to fetch page 2
        if len(valid_items) < max_items:
            # Check if API response indicates more content is available
            more_available = response_page1.get("more_available", False)

            # Get cursor from profile_grid_items_cursor or next_max_id
            cursor = (
                response_page1.get("profile_grid_items_cursor")
                or response_page1.get("next_max_id")
            )

            if more_available and cursor:
                logger.info(
                    "Fetching page 2 for %s (collected %d valid videos from page 1)",
                    platform_id,
                    len(valid_items),
                )

                try:
                    response_page2 = await self.client.get(
                        endpoint="/v2/instagram/user/posts",
                        params={"handle": platform_id, "cursor": cursor},
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to fetch page 2 for %s: %s. Proceeding with page 1 results only.",
                        platform_id,
                        e,
                    )
                else:
                    # Extract items from page 2
                    items_page2: list[dict[str, Any]] = response_page2.get("items", [])

                    # Filter valid videos from page 2 and append to valid_items
                    for item in items_page2:
                        if len(valid_items) >= max_items:
                            break

                        item_id = item.get("id") or item.get("media_id") or item.get("pk")
                        if not item_id:
                            continue

                        # Deduplicate
                        if item_id in seen_ids:
                            continue
                        seen_ids.add(item_id)

                        # Check if it's a valid video under 120 seconds
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
                        duration = item.get("video_duration") or item.get("duration", 0.0)
                        # Valid video must have duration strictly between 0 and 120 seconds
                        if not (0.0 < duration <= 120.0):
                            continue

                        valid_items.append(item)

        # ===================================================================
        # HANDLE EDGE CASES
        # ===================================================================
        if not valid_items:
            # No valid content found: reject account to prevent false positives
            # This handles accounts with empty biographies that reached Stage2 with "processing" status
            logger.info(
                "No valid Instagram content found for account_id: %d. Rejecting account.",
                account_id,
            )
            async with self.session_maker() as session:
                # Direct update of account status to "rejected"
                stmt = (
                    update(Account)
                    .where(Account.id == account_id)
                    .values(status="rejected", updated_at=datetime.now(timezone.utc))
                )
                await session.execute(stmt)
                await session.commit()
            return

        logger.info(
            "Fetched %d valid video items for account_id: %d (after conditional pagination)",
            len(valid_items),
            account_id,
        )

        # ===================================================================
        # STAGE2: CYRILLIC VALIDATION (SIEVE)
        # Combine all post text and check for Russian characters
        # If no Cyrillic found, reject account and exit without DB writes
        # ===================================================================
        aggregated_text = ""
        for item in valid_items:
            # Extract description
            caption_data = item.get("caption", {})
            description = ""
            if isinstance(caption_data, dict):
                description = caption_data.get("text", "")
            elif isinstance(caption_data, str):
                description = caption_data

            # Extract hashtags
            hashtags = item.get("hashtags") or []
            if not hashtags and description:
                hashtags = [
                    tag.strip("#") for tag in description.split()
                    if tag.startswith("#")
                ]

            # Accumulate text
            aggregated_text += " " + description + " " + " ".join(hashtags)

        # Check if any Cyrillic characters exist in aggregated text
        has_cyrillic = is_russian_text(aggregated_text)

        if not has_cyrillic:
            # REJECT ACCOUNT: No Russian text found in any posts
            logger.warning(
                "Account %s (account_id: %d) REJECTED: No Cyrillic characters found in %d fetched posts. "
                "Rejecting without writing content to database.",
                platform_id,
                account_id,
                len(valid_items),
            )
            async with self.session_maker() as session:
                # Direct update of account status to "rejected"
                stmt = (
                    update(Account)
                    .where(Account.id == account_id)
                    .values(status="rejected", updated_at=datetime.now(timezone.utc))
                )
                await session.execute(stmt)
                await session.commit()
            return

        # Cyrillic detected - proceed with parsing
        logger.info(
            "Stage2 Cyrillic validation PASSED for account_id: %d. Proceeding with content parsing.",
            account_id,
        )

        # Process items: conditional transcript fetching based on semantic keywords
        # First pass: determine which items need transcripts
        items_data: list[dict[str, Any]] = []
        items_needing_transcripts: list[tuple[str, str]] = []  # (item_id, post_url)

        for item in valid_items:
            item_id = item.get("id") or item.get("media_id") or item.get("pk")
            if not item_id:
                continue

            # Extract description and hashtags for semantic check
            caption_data = item.get("caption", {})
            description = ""
            if isinstance(caption_data, dict):
                description = caption_data.get("text", "")
            elif isinstance(caption_data, str):
                description = caption_data

            # Extract hashtags
            hashtags = item.get("hashtags") or []
            if not hashtags and description:
                hashtags = [
                    tag.strip("#") for tag in description.split()
                    if tag.startswith("#")
                ]

            # Check if content matches target semantics using SearchQueriesManager
            combined_text = description + " " + " ".join(hashtags)
            keywords_pattern = self._queries_manager.get_compiled_keywords_pattern()
            has_target_semantics = (
                keywords_pattern.search(combined_text) is not None
            )

            likes, comments = extract_instagram_metrics(item)

            # Extract content text - will be updated with transcript if needed
            content_text: str | None = extract_instagram_content_text(item)

            # Extract high-quality video URL for GPU embedding
            video_url = self._extract_video_url(item)

            # Extract duration to determine post_type
            duration = item.get("video_duration") or item.get("duration", 0.0)
            post_type = "reel" if duration <= 120.0 else "post"

            # Extract views
            views = item.get("video_view_count") or item.get("play_count")

            item_data: dict[str, Any] = {
                "item": item,
                "item_id": item_id,
                "has_target_semantics": has_target_semantics,
                "content_text": content_text,
                "likes": likes,
                "comments": comments,
                "views": views,
                "video_url": video_url,
                "post_type": post_type,
                "transcript": None,
            }

            # Decide whether transcript is needed using _has_sufficient_semantics
            # Only Reels (duration <= 120s) are eligible for transcription
            needs_transcript = not self._has_sufficient_semantics(
                bio=profile_biography,
                external_url=profile_external_url,
                caption=description,
                hashtags=hashtags,
                has_target_semantics=has_target_semantics,
            ) and post_type == "reel"

            if needs_transcript:
                # Request transcript via GET method
                shortcode = item.get("code") or item.get("shortcode")
                if shortcode:
                    post_url = f"https://instagram.com/p/{shortcode}"
                    items_needing_transcripts.append((item_id, post_url))

            items_data.append(item_data)

        # Fetch transcripts concurrently for items that need them
        if items_needing_transcripts:
            semaphore = asyncio.Semaphore(5)
            transcript_fetch_tasks = [
                self._fetch_transcript(semaphore, post_url)
                for _, post_url in items_needing_transcripts
            ]

            results = await asyncio.gather(
                *transcript_fetch_tasks, return_exceptions=True
            )

            # Build transcript mapping
            transcript_map: dict[str, str | None] = {}
            for (item_id, _), result in zip(
                items_needing_transcripts, results, strict=False
            ):
                if isinstance(result, Exception):
                    logger.error(
                        "Transcript fetch failed for %s: %s", item_id, result
                    )
                    transcript_map[item_id] = None
                elif isinstance(result, str) or result is None:
                    transcript_map[item_id] = result
                else:
                    transcript_map[item_id] = None

            # Update items_data with transcripts (transcript only, NOT content_text)
            for item_data in items_data:
                item_id = item_data["item_id"]
                if item_id in transcript_map:
                    item_data["transcript"] = transcript_map[item_id]

        # Build final content values for bulk upsert
        final_content_values = []
        # Accumulated contacts structured to match parse_profile_contacts() output format
        aggregated_emails: list[str] = []
        aggregated_telegram_handles: list[str] = []
        aggregated_external_links: list[str] = []
        aggregated_external_platforms: dict[str, str] = {}
        aggregated_mentions: set[str] = set()

        # Stage2 Cyrillic validation passed, language is Russian
        language = "ru"

        for item_data in items_data:
            item = item_data["item"]
            item_id = item_data["item_id"]

            # Content-based discovery spider
            content_text = item_data["content_text"]
            if content_text:
                # Parse profile contacts (cross-platform links)
                contacts_dict = parse_profile_contacts(content_text)
                # Extract same-platform mentions (e.g., @otheruser)
                mentions = extract_mentions(content_text)

                # Accumulate emails: distinct flat list
                for email in contacts_dict.get("emails", []):
                    if email and email not in aggregated_emails:
                        aggregated_emails.append(email)

                # Accumulate telegram_handles: distinct flat list
                for handle in contacts_dict.get("telegram_handles", []):
                    if handle and handle not in aggregated_telegram_handles:
                        aggregated_telegram_handles.append(handle)

                # Accumulate external_links: distinct flat list
                for link in contacts_dict.get("external_links", []):
                    if link and link not in aggregated_external_links:
                        aggregated_external_links.append(link)

                # Accumulate external_platforms: dict mapping platform slugs to handles
                # If multiple items mention the same platform, keep the first handle discovered
                for platform_slug, handle in contacts_dict.get("external_platforms", {}).items():
                    if handle and platform_slug not in aggregated_external_platforms:
                        aggregated_external_platforms[platform_slug] = handle

                # Accumulate mentions
                aggregated_mentions.update(mentions)

            # Extract metrics
            likes = item_data["likes"]
            comments = item_data["comments"]
            views = item_data["views"]
            video_url = item_data["video_url"]
            post_type = item_data["post_type"]

            # Build raw_metadata using Pydantic models
            platform_metrics = PlatformMetrics(
                likes=likes,
                comments_count=comments,
                views=views,
                shares=None,  # Instagram API doesn't always provide shares
                plays=views,  # Use views as plays for Instagram
            )

            content_metadata = InstagramContentMetadata.create_with_timestamp(
                video_url=video_url,
                category=account_category,
                language=language,
                post_type=post_type,
                platform_metrics=platform_metrics,
                author_profile_snapshot=author_profile_snapshot,
                raw_item_payload=item,
                is_reel=post_type == "reel",
            )

            # Map to Content model columns explicitly
            # views -> views (int | None)
            # reactions_count -> likes (int | None)
            # comments_count -> comments (int | None)
            final_content_values.append(
                {
                    "account_id": account_id,
                    "platform_content_id": item_id,
                    "content": item_data["content_text"],
                    "published_at": extract_instagram_published_at(item),
                    "transcription": item_data["transcript"],
                    "views": views,
                    "reactions_count": likes,
                    "comments_count": comments,
                    "shares_count": None,
                    "has_media": True,
                    "is_embedded": False,
                    "is_graph_extracted": False,
                    "raw_metadata": content_metadata,
                    "updated_at": datetime.now(timezone.utc),
                }
            )

        # Build aggregated_contacts dict for queue_discovered_accounts
        aggregated_contacts: dict[str, Any] = {
            "emails": aggregated_emails,
            "telegram_handles": aggregated_telegram_handles,
            "external_links": aggregated_external_links,
            "external_platforms": aggregated_external_platforms,
            "raw_bio": "",
        }

        # Bulk write all accumulated discovery data
        if aggregated_contacts or aggregated_mentions:
            try:
                async with self.session_maker() as session:
                    if aggregated_contacts:
                        await queue_discovered_accounts(
                            session=session,
                            metadata=aggregated_contacts,
                            parent_handle=profile.get("username", ""),
                            status="pending",
                            category=account_category,
                        )
                    if aggregated_mentions:
                        await queue_discovered_mentions(
                            session=session,
                            platform="INSTAGRAM",
                            mentions=list(aggregated_mentions),
                            parent_handle=profile.get("username", ""),
                            status="pending",
                            category=account_category,
                        )
                    await session.commit()
            except Exception as e:
                logger.warning(
                    "Failed to queue discovered accounts from Instagram content: %s",
                    e,
                )

        # Bulk upsert content using centralized function
        if final_content_values:
            async with self.session_maker() as session:
                await bulk_upsert_content(
                    session=session,
                    content_values=final_content_values,
                )
                await session.commit()
                logger.info(
                    "Bulk upserted %d Instagram content items for account_id: %d",
                    len(final_content_values),
                    account_id,
                )

    def _extract_video_url(self, item: dict[str, Any]) -> str | None:
        """Extract video URL from Instagram post item with robust fallback logic.

        Implements a multi-tier extraction strategy to handle various Instagram
        API response structures:
        1. Check nested "media" object for video_versions
        2. Check video_versions as a list of dicts
        3. Check video_versions as a dict with numeric keys
        4. Fallback to video_url at media or item level
        5. Fallback to carousel_media list (recursive extraction)
        6. Final fallback to extract_instagram_video_url helper

        Args:
            item: Instagram post dictionary.

        Returns:
            Video URL string if found, otherwise None.
        """
        # Tier1: Check nested "media" object
        media = item.get("media")
        if isinstance(media, dict):
            # Check video_versions inside media object (list format)
            video_versions = media.get("video_versions")
            if isinstance(video_versions, list) and len(video_versions) > 0:
                first_video = video_versions[0]
                if isinstance(first_video, dict):
                    url = first_video.get("url")
                    if isinstance(url, str) and url:
                        return url

            # Check video_versions inside media object (dict format with numeric keys)
            if isinstance(video_versions, dict) and video_versions:
                # Extract URL from the first available key
                first_key = next(iter(video_versions), None)
                if first_key is not None:
                    first_item = video_versions[first_key]
                    if isinstance(first_item, dict):
                        url = first_item.get("url")
                        if isinstance(url, str) and url:
                            return url

            # Fallback to video_url at media level
            video_url = media.get("video_url")
            if isinstance(video_url, str) and video_url:
                return video_url

        # Tier2: Check video_versions at item level (list format)
        video_versions = item.get("video_versions")
        if isinstance(video_versions, list) and len(video_versions) > 0:
            first_video = video_versions[0]
            if isinstance(first_video, dict):
                url = first_video.get("url")
                if isinstance(url, str) and url:
                    return url

        # Tier3: Check video_versions at item level (dict format with numeric keys)
        if isinstance(video_versions, dict) and video_versions:
            # Extract URL from the first available key
            first_key = next(iter(video_versions), None)
            if first_key is not None:
                first_item = video_versions[first_key]
                if isinstance(first_item, dict):
                    url = first_item.get("url")
                    if isinstance(url, str) and url:
                        return url

        # Tier4: Fallback to video_url at item level
        video_url = item.get("video_url")
        if isinstance(video_url, str) and video_url:
            return video_url

        # Tier5: Fallback to carousel_media list (recursive extraction)
        carousel_media = item.get("carousel_media")
        if isinstance(carousel_media, list) and len(carousel_media) > 0:
            # Recursively extract video URL from the first carousel item
            first_carousel_item = carousel_media[0]
            if isinstance(first_carousel_item, dict):
                carousel_video_url = self._extract_video_url(
                    first_carousel_item
                )
                if isinstance(carousel_video_url, str) and carousel_video_url:
                    return carousel_video_url

        # Tier6: Final fallback to extract_instagram_video_url helper
        return extract_instagram_video_url(item)

    async def _fetch_transcript(
        self, semaphore: asyncio.Semaphore, post_url: str
    ) -> str | None:
        """Fetch media transcript using client-side retries.

        Delegates retry logic to the HTTP client's built-in exponential backoff
        configured via environment settings. Rate limiting is handled by the
        semaphore wrapper.

        Args:
            semaphore: Asyncio semaphore for rate limiting.
            post_url: Instagram post permalink (e.g., https://instagram.com/p/SHORTCODE).

        Returns:
            Transcript text if available, otherwise None.
        """
        logger.debug("Requesting transcript: %s", post_url[:50])

        async with semaphore:
            try:
                response = await self.client.get(
                    endpoint="/v2/instagram/media/transcript",
                    params={"url": post_url},
                    max_retries=3,
                )
                # Parse the transcripts list from the API response
                transcripts = response.get("transcripts")
                if isinstance(transcripts, list) and len(transcripts) > 0:
                    first_item = transcripts[0]

                    # Skip None values in the list
                    if first_item is None:
                        logger.debug("No transcript: %s", post_url[:50])
                        return None

                    transcript_text: str | None = None

                    if isinstance(first_item, str) and first_item:
                        transcript_text = first_item
                    elif isinstance(first_item, dict):
                        text_value = first_item.get("text")
                        if isinstance(text_value, str) and text_value:
                            transcript_text = text_value

                    if transcript_text:
                        logger.debug("Got transcript: %s", post_url[:50])
                        return transcript_text

                logger.debug("No transcript: %s", post_url[:50])
                return None

            except Exception as e:
                logger.warning(
                    "Transcript permanently failed for %s: %s", post_url, e
                )
                return None
