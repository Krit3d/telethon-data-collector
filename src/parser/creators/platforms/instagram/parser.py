import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update

from src.db.models import Account
from src.parser.creators.core.db.accounts_repo import (
    upsert_and_deduplicate_account,
    update_account_profile_metadata,
)
from src.parser.creators.core.db.content_repo import bulk_upsert_content
from src.parser.creators.platforms.instagram.helpers import (
    extract_instagram_subscribers,
    extract_instagram_content_text,
    extract_instagram_published_at,
    extract_instagram_metrics,
)
from src.parser.creators.core.queries import SearchQueriesManager
from src.parser.creators.core.schemas import (
    InstagramContentMetadata,
    PlatformMetrics,
    AuthorProfileSnapshot,
)
from src.parser.creators.core.text import is_slop_or_theme_page
from src.parser.creators.platforms.base import BasePlatformParser

from .client import fetch_instagram_profile, fetch_video_transcript
from .contacts_processor import process_and_queue_discovered_contacts
from .fetcher import fetch_valid_instagram_videos
from .helpers import extract_instagram_video_url, prune_instagram_payload
from .semantics import has_sufficient_semantics
from .validators import (
    check_cyrillic_stage1,
    check_cyrillic_stage2,
    validate_follower_count,
    MIN_SUBSCRIBERS,
    MAX_SUBSCRIBERS,
)

logger = logging.getLogger(__name__)


class InstagramParser(BasePlatformParser):

    def __init__(
        self,
        session_maker,
        client,
        settings,
    ) -> None:
        super().__init__(session_maker, client, settings)
        self._queries_manager = SearchQueriesManager(settings.search_queries_path)

    async def parse_profile(self, handle: str) -> int | None:
        logger.info("Starting Instagram profile parse for handle: %s", handle)

        profile = await fetch_instagram_profile(self.client, handle)
        if not profile:
            return None

        username = profile.get("username", "")
        biography = profile.get("biography")
        full_name = profile.get("full_name", "")

        subscribers = extract_instagram_subscribers(profile)

        if subscribers == 0:
            logger.warning(
                "Instagram handle %s parsed 0 subscribers. Profile dict keys: %s",
                handle,
                list(profile.keys()),
            )

        if not validate_follower_count(subscribers):
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

        biography_stripped = biography.strip() if biography else ""

        if not biography_stripped:
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

        has_cyrillic = check_cyrillic_stage1(biography, full_name)

        if not has_cyrillic:
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
        logger.info(
            "Starting Instagram candidate discovery for query: '%s', category: '%s'",
            query,
            category,
        )

        try:
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

            profiles: list[dict[str, Any]] = []
            if isinstance(response.get("profiles"), list):
                profiles = response["profiles"]
            elif isinstance(response.get("data"), list):
                profiles = response["data"]
            elif isinstance(response.get("items"), list):
                profiles = response["items"]
            elif isinstance(response, list):
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

            for profile in profiles:
                if not isinstance(profile, dict):
                    continue

                username = profile.get("username") or profile.get("handle")
                if not username or not isinstance(username, str):
                    logger.debug(
                        "Skipping profile with missing username in search results: %s",
                        profile.get("id", "unknown"),
                    )
                    continue

                followers = profile.get("follower_count") or profile.get("followers")
                if followers is None:
                    followers = (
                        profile.get("stats", {}).get("followers")
                        or profile.get("user", {}).get("follower_count")
                    )

                try:
                    followers = int(followers) if followers is not None else 0
                except (ValueError, TypeError):
                    followers = 0

                if followers == 0:
                    logger.debug(
                        "Skipping Instagram profile %s: follower count is 0",
                        username,
                    )
                    continue

                if not validate_follower_count(followers):
                    logger.debug(
                        "Skipping Instagram profile %s: follower count %d outside range [%d, %d]",
                        username,
                        followers,
                        MIN_SUBSCRIBERS,
                        MAX_SUBSCRIBERS,
                    )
                    continue

                profile_id = profile.get("id")
                full_name = profile.get("full_name", "")
                biography = profile.get("biography", "") or ""

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

                        meta: dict[str, Any] = {
                            "category": category,
                            "discovery_query": query,
                            "search_metadata": profile,
                        }

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

    async def parse_content(
        self, account_id: int, platform_id: str, max_items: int = 10
    ) -> None:
        logger.info(
            "Starting Instagram content parse for account_id: %d, platform_id: %s",
            account_id,
            platform_id,
        )

        account_category = "unknown"
        async with self.session_maker() as session:
            stmt = select(Account.raw_metadata).where(Account.id == account_id)
            result = await session.execute(stmt)
            raw_metadata = result.scalar_one_or_none()
            if raw_metadata and isinstance(raw_metadata, dict):
                category = raw_metadata.get("category")
                if category and isinstance(category, str):
                    account_category = category

        profile = await fetch_instagram_profile(self.client, platform_id)
        if not profile:
            raise RuntimeError(f"Could not retrieve profile metadata for {platform_id} during content parsing.")

        profile_biography = profile.get("biography")
        profile_external_url = profile.get("external_url")

        author_profile_snapshot = AuthorProfileSnapshot(
            username=profile.get("username", ""),
            title=profile.get("full_name") or profile.get("username", ""),
        )

        valid_items = await fetch_valid_instagram_videos(
            self.client, platform_id, max_items,
        )

        if not valid_items:
            logger.info(
                "No valid Instagram content found for account_id: %d. Rejecting account.",
                account_id,
            )
            async with self.session_maker() as session:
                stmt = (
                    update(Account)
                    .where(Account.id == account_id)
                    .values(status="rejected", updated_at=datetime.now(timezone.utc))
                )
                await session.execute(stmt)
                await session.commit()
            return

        logger.info(
            "Fetched %d valid video items for account_id: %d",
            len(valid_items),
            account_id,
        )

        aggregated_text = ""
        for item in valid_items:
            caption_data = item.get("caption", {})
            description = ""
            if isinstance(caption_data, dict):
                description = caption_data.get("text", "")
            elif isinstance(caption_data, str):
                description = caption_data

            hashtags = item.get("hashtags") or []
            if not hashtags and description:
                hashtags = [
                    tag.strip("#") for tag in description.split()
                    if tag.startswith("#")
                ]

            aggregated_text += " " + description + " " + " ".join(hashtags)

        has_cyrillic = check_cyrillic_stage2(aggregated_text)

        if not has_cyrillic:
            logger.warning(
                "Account %s (account_id: %d) REJECTED: No Cyrillic characters found in %d fetched posts. "
                "Rejecting without writing content to database.",
                platform_id,
                account_id,
                len(valid_items),
            )
            async with self.session_maker() as session:
                stmt = (
                    update(Account)
                    .where(Account.id == account_id)
                    .values(status="rejected", updated_at=datetime.now(timezone.utc))
                )
                await session.execute(stmt)
                await session.commit()
            return

        logger.info(
            "Stage2 Cyrillic validation PASSED for account_id: %d. Proceeding with content parsing.",
            account_id,
        )

        items_data: list[dict[str, Any]] = []
        items_needing_transcripts: list[tuple[str, str]] = []

        for item in valid_items:
            item_id = item.get("id") or item.get("media_id") or item.get("pk")
            if not item_id:
                continue

            caption_data = item.get("caption", {})
            description = ""
            if isinstance(caption_data, dict):
                description = caption_data.get("text", "")
            elif isinstance(caption_data, str):
                description = caption_data

            hashtags = item.get("hashtags") or []
            if not hashtags and description:
                hashtags = [
                    tag.strip("#") for tag in description.split()
                    if tag.startswith("#")
                ]

            combined_text = description + " " + " ".join(hashtags)
            keywords_pattern = self._queries_manager.get_compiled_keywords_pattern()
            has_target_semantics = (
                keywords_pattern.search(combined_text) is not None
            )

            likes, comments = extract_instagram_metrics(item)

            content_text: str | None = extract_instagram_content_text(item)

            video_url = extract_instagram_video_url(item)

            duration = item.get("video_duration") or item.get("duration", 0.0)
            post_type = "reel" if duration <= 120.0 else "post"

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

            needs_transcript = not has_sufficient_semantics(
                bio=profile_biography,
                external_url=profile_external_url,
                caption=description,
                hashtags=hashtags,
                has_target_semantics=has_target_semantics,
            ) and post_type == "reel"

            if needs_transcript:
                shortcode = item.get("code") or item.get("shortcode")
                if shortcode:
                    post_url = f"https://instagram.com/p/{shortcode}"
                    items_needing_transcripts.append((item_id, post_url))

            items_data.append(item_data)

        transcript_map: dict[str, str | None] = {}

        if items_needing_transcripts:
            semaphore = asyncio.Semaphore(5)
            transcript_fetch_tasks = [
                fetch_video_transcript(self.client, semaphore, post_url)
                for _, post_url in items_needing_transcripts
            ]

            results = await asyncio.gather(
                *transcript_fetch_tasks, return_exceptions=True
            )

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

            for item_data in items_data:
                item_id = item_data["item_id"]
                if item_id in transcript_map:
                    item_data["transcript"] = transcript_map[item_id]

        await process_and_queue_discovered_contacts(
            session_maker=self.session_maker,
            parent_username=profile.get("username", ""),
            account_category=account_category,
            profile_biography=profile_biography,
            profile_external_url=profile_external_url,
            items_data=items_data,
        )

        final_content_values = []

        for item_data in items_data:
            item = item_data["item"]
            item_id = item_data["item_id"]
            likes = item_data["likes"]
            comments = item_data["comments"]
            views = item_data["views"]
            video_url = item_data["video_url"]
            post_type = item_data["post_type"]

            platform_metrics = PlatformMetrics(
                likes=likes,
                comments_count=comments,
                views=views,
                shares=None,
                plays=views,
            )

            content_metadata = InstagramContentMetadata.create_with_timestamp(
                video_url=video_url,
                category=account_category,
                language="ru",
                post_type=post_type,
                platform_metrics=platform_metrics,
                author_profile_snapshot=author_profile_snapshot,
                raw_item_payload=prune_instagram_payload(item),
                is_reel=post_type == "reel",
            )

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
