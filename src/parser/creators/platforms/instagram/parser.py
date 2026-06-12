import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import or_, select, update

from src.db.models import Account, Content
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
from .helpers import extract_instagram_geo_data, extract_instagram_video_url, prune_instagram_payload
from .search_cursor import InstagramSearchPaginator
from .validators import (
    check_cyrillic_stage1,
    check_cyrillic_stage2,
    has_commercial_music,
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
        self.queries_manager = SearchQueriesManager(settings.search_queries_path)

    async def _fetch_account_category(self, account_id: int) -> str | None:
        async with self.session_maker() as session:
            stmt = select(Account.raw_metadata).where(Account.id == account_id)
            result = await session.execute(stmt)
            raw_metadata = result.scalar_one_or_none()
            if raw_metadata and isinstance(raw_metadata, dict):
                category = raw_metadata.get("category")
                if category and isinstance(category, str):
                    return category
        return None

    async def _persist_fallback_category(self, account_id: int, category: str) -> None:
        async with self.session_maker() as session:
            stmt = select(Account.raw_metadata).where(Account.id == account_id)
            result = await session.execute(stmt)
            raw_meta = result.scalar_one_or_none()
            if not isinstance(raw_meta, dict):
                raw_meta = {}
            raw_meta["category"] = category
            upd = (
                update(Account)
                .where(Account.id == account_id)
                .values(raw_metadata=raw_meta, updated_at=datetime.now(timezone.utc))
            )
            await session.execute(upd)
            await session.commit()

    async def _fetch_raw_profile_payload(self, account_id: int) -> dict[str, Any] | None:
        async with self.session_maker() as session:
            stmt = select(Account.raw_metadata).where(Account.id == account_id)
            result = await session.execute(stmt)
            raw_metadata = result.scalar_one_or_none()
            if not isinstance(raw_metadata, dict):
                return None
            payload = raw_metadata.get("raw_profile_payload")
            if not isinstance(payload, dict) or not payload:
                return None
            if "username" not in payload and "id" not in payload:
                return None
            return payload

    async def parse_profile(self, handle: str) -> int | None:
        logger.info("Starting Instagram profile parse for handle: %s", handle)

        profile = await fetch_instagram_profile(self.client, handle)
        if not profile:
            logger.warning(
                "Instagram handle %s: profile fetch returned None (deleted or not found). Marking as rejected.",
                handle,
            )
            async with self.session_maker() as session:
                account_id = await upsert_and_deduplicate_account(
                    session=session,
                    platform="INSTAGRAM",
                    platform_id=handle,
                    username=handle,
                    title=handle,
                    description="",
                    subscribers_count=0,
                    status="rejected",
                )
                await session.commit()
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

                existing_category = await self._fetch_account_category(account_id)

                if not existing_category or existing_category == "unknown":
                    fallback = self.queries_manager.classify_text(f"{full_name} {biography or ''}")
                    if fallback:
                        logger.info(
                            "Fallback category '%s' resolved for handle %s via keyword matching",
                            fallback,
                            handle,
                        )
                        existing_category = fallback
                        await self._persist_fallback_category(account_id, fallback)

                location_str, geo_data_dict = extract_instagram_geo_data(profile, biography, full_name)

                await update_account_profile_metadata(
                    session=session,
                    account_id=account_id,
                    platform="INSTAGRAM",
                    biography=biography or "",
                    external_url=profile.get("external_url"),
                    category=existing_category,
                    subscribers_count=subscribers,
                    raw_profile_payload=profile,
                    posts_count=profile.get("media_count") or profile.get("posts_count"),
                    language="ru",
                    location=location_str,
                    geo_data=geo_data_dict,
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

        if has_cyrillic:
            logger.info(
                "Instagram handle %s: Stage1 PASSED (Cyrillic detected). Passing to Stage2.",
                handle,
            )
        else:
            logger.info(
                "Instagram handle %s: Stage1 did not detect Cyrillic in biography/full_name. "
                "Transitioning to processing to validate via Stage2 content check.",
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

            existing_category = await self._fetch_account_category(account_id)

            if not existing_category or existing_category == "unknown":
                fallback = self.queries_manager.classify_text(f"{full_name} {biography or ''}")
                if fallback:
                    logger.info(
                        "Fallback category '%s' resolved for handle %s via keyword matching",
                        fallback,
                        handle,
                    )
                    existing_category = fallback
                    await self._persist_fallback_category(account_id, fallback)

            location_str, geo_data_dict = extract_instagram_geo_data(profile, biography, full_name)

            await update_account_profile_metadata(
                session=session,
                account_id=account_id,
                platform="INSTAGRAM",
                biography=biography or "",
                external_url=profile.get("external_url"),
                category=existing_category,
                subscribers_count=subscribers,
                raw_profile_payload=profile,
                posts_count=profile.get("media_count") or profile.get("posts_count"),
                language="ru",
                location=location_str,
                geo_data=geo_data_dict,
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

        paginator = InstagramSearchPaginator(query, max_depth=2)
        total_discovered = 0

        while paginator.should_continue():
            try:
                params = paginator.get_params()

                response = await self.client.get(
                    endpoint="/v1/instagram/search/profiles",
                    params=params,
                )

                if not response:
                    paginator.handle_empty_response()
                    break

                profiles = paginator.extract_profiles(response)
                if not profiles:
                    paginator.handle_empty_response()
                    break

                discovered_count = 0
                new_candidates_found = 0

                async with self.session_maker() as session:
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

                        exists_stmt = select(Account.id).where(
                            Account.platform == "INSTAGRAM",
                            or_(
                                Account.platform_id == str(profile_id or username),
                                Account.username == username,
                            ),
                        )
                        exists_result = await session.execute(exists_stmt)
                        already_exists = exists_result.scalar_one_or_none() is not None

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

                            if not already_exists:
                                stmt = (
                                    update(Account)
                                    .where(Account.id == account_id)
                                    .values(raw_metadata=meta, updated_at=datetime.now(timezone.utc))
                                )
                                await session.execute(stmt)

                            discovered_count += 1
                            if not already_exists:
                                new_candidates_found += 1

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

                    await session.commit()

                paginator.register_candidates(response, new_candidates_found, discovered_count)
                total_discovered = discovered_count

                if new_candidates_found > 0:
                    return discovered_count

                await paginator.sleep()

            except Exception as e:
                paginator.handle_error(e)
                return 0

        paginator.finalize_exhausted()
        return total_discovered

    async def parse_content(
        self, account_id: int, platform_id: str, max_items: int = 10
    ) -> None:
        logger.info(
            "Starting Instagram content parse for account_id: %d, platform_id: %s",
            account_id,
            platform_id,
        )

        account_category = await self._fetch_account_category(account_id) or "unknown"

        profile = await self._fetch_raw_profile_payload(account_id)
        if profile is None:
            profile = await fetch_instagram_profile(self.client, platform_id)
        if not profile:
            raise RuntimeError(f"Could not retrieve profile metadata for {platform_id} during content parsing.")

        profile_biography = profile.get("biography")
        profile_external_url = profile.get("external_url")

        author_profile_snapshot = AuthorProfileSnapshot(
            username=profile.get("username", ""),
            title=profile.get("full_name") or profile.get("username", ""),
        )

        try:
            valid_items = await fetch_valid_instagram_videos(
                self.client, platform_id, max_items,
            )
        except Exception as e:
            if getattr(e, "status", None) == 404 or "404" in str(e):
                logger.warning(
                    "Instagram account %s returned 404 when fetching posts. Marking as rejected.",
                    platform_id,
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
            raise

        def _to_utc_aware(dt: datetime) -> datetime:
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        def _published_at_key(item: dict[str, Any]) -> datetime:
            try:
                dt = extract_instagram_published_at(item)
            except Exception:
                return datetime.min.replace(tzinfo=timezone.utc)
            return _to_utc_aware(dt)

        valid_items.sort(key=_published_at_key, reverse=True)
        valid_items = valid_items[:10]

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
                hashtags = re.findall(r"#(\w+)", description)

            aggregated_text += " " + description + " " + " ".join(hashtags)

        if not account_category or account_category == "unknown":
            rich_text = f"{author_profile_snapshot.title or ''} {profile_biography or ''} {aggregated_text}"
            fallback = self.queries_manager.classify_text(rich_text)
            if fallback:
                account_category = fallback
                await self._persist_fallback_category(account_id, fallback)

        if not account_category or account_category == "unknown":
            discovery_query_text = None
            async with self.session_maker() as session:
                stmt = select(Account.raw_metadata).where(Account.id == account_id)
                result = await session.execute(stmt)
                raw_meta = result.scalar_one_or_none()
                if isinstance(raw_meta, dict):
                    discovery_query_text = raw_meta.get("discovery_query")
            if discovery_query_text and isinstance(discovery_query_text, str):
                fallback = self.queries_manager.classify_text(discovery_query_text)
                if fallback:
                    account_category = fallback
                    await self._persist_fallback_category(account_id, fallback)

        if not account_category or account_category == "unknown":
            account_category = "lifestyle"
            await self._persist_fallback_category(account_id, "lifestyle")

        has_cyrillic = (
            check_cyrillic_stage1(profile_biography, author_profile_snapshot.title)
            or check_cyrillic_stage2(aggregated_text)
        )

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
                hashtags = re.findall(r"#(\w+)", description)

            combined_text = description + " " + " ".join(hashtags)
            keywords_pattern = self.queries_manager.get_compiled_keywords_pattern()
            has_target_semantics = (
                keywords_pattern.search(combined_text) is not None
            )

            likes, comments = extract_instagram_metrics(item)

            content_text: str | None = extract_instagram_content_text(item)

            video_url = extract_instagram_video_url(item)

            try:
                duration = float(item.get("video_duration") or item.get("duration") or 0.0)
            except (ValueError, TypeError):
                duration = 0.0
            post_type = "reel" if duration <= 120.0 else "post"

            views = item.get("video_view_count") or item.get("play_count")

            shortcode = item.get("code") or item.get("shortcode")
            post_url = f"https://instagram.com/p/{shortcode}" if shortcode else None

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
                "post_url": post_url,
                "transcript": None,
                "hashtags": hashtags,
                "combined_text": combined_text,
            }

            is_valid_video = bool(video_url) and duration > 0.0

            needs_transcript = not (
                duration < 3.0
                or duration > 120.0
                or not is_valid_video
                or has_target_semantics
                or has_commercial_music(item)
            )

            if needs_transcript and post_url:
                items_needing_transcripts.append((item_id, post_url))

            items_data.append(item_data)

        items_needing_transcripts = items_needing_transcripts[:10]

        candidate_item_ids = [d["item_id"] for d in items_data]

        already_transcribed: set[str] = set()
        if candidate_item_ids:
            async with self.session_maker() as session:
                stmt = (
                    select(Content.platform_content_id)
                    .where(
                        Content.platform_content_id.in_(candidate_item_ids),
                        Content.transcription.isnot(None),
                    )
                )
                result = await session.execute(stmt)
                already_transcribed = {row[0] for row in result.all()}

        items_needing_transcripts = [
            (item_id, post_url)
            for item_id, post_url in items_needing_transcripts
            if item_id not in already_transcribed
        ]

        for t_item_id, t_post_url in items_needing_transcripts:

            async def _background_transcribe_and_update(
                item_id: str = t_item_id, post_url: str = t_post_url,
            ) -> None:
                try:
                    result = await fetch_video_transcript(
                        self.client, self.client.global_semaphore, post_url,
                    )
                    if isinstance(result, str):
                        cleaned = result.strip()
                        if not cleaned.lower().startswith("please provide") and cleaned:
                            async with self.session_maker() as session:
                                stmt = (
                                    update(Content)
                                    .where(Content.platform_content_id == item_id)
                                    .values(
                                        transcription=cleaned,
                                        updated_at=datetime.now(timezone.utc),
                                    )
                                )
                                await session.execute(stmt)
                                await session.commit()
                            logger.info(
                                "Background transcript updated for item %s",
                                item_id,
                            )
                except Exception as e:
                    logger.error(
                        "Background transcript fetch/update failed for %s: %s",
                        item_id,
                        e,
                    )

            task: asyncio.Task[None] = asyncio.create_task(
                _background_transcribe_and_update()
            )
            self.client.background_tasks.add(task)
            task.add_done_callback(self.client.background_tasks.discard)

        await process_and_queue_discovered_contacts(
            session_maker=self.session_maker,
            parent_username=profile.get("username", ""),
            account_category=account_category,
            profile_biography=profile_biography,
            profile_external_url=profile_external_url,
            items_data=items_data,
        )

        final_content_values: list[dict[str, Any]] = []

        for item_data in items_data:
            item = item_data["item"]
            item_id = item_data["item_id"]
            likes = item_data["likes"]
            comments = item_data["comments"]
            views = item_data["views"]
            video_url = item_data["video_url"]
            post_type = item_data["post_type"]
            hashtags = item_data["hashtags"]

            platform_metrics = PlatformMetrics(
                likes=likes,
                comments_count=comments,
                views=views,
                shares=None,
                plays=views,
            )

            post_category = self.queries_manager.classify_text(item_data.get("combined_text", ""))
            if not post_category or post_category == "unknown":
                post_category = account_category

            content_metadata = InstagramContentMetadata.create_with_timestamp(
                video_url=video_url,
                category=post_category,
                language="ru",
                post_type=post_type,
                platform_metrics=platform_metrics,
                author_profile_snapshot=author_profile_snapshot,
                raw_item_payload=prune_instagram_payload(item),
                is_reel=post_type == "reel",
                hashtags=hashtags,
                post_url=item_data["post_url"],
            )

            raw_meta_dict = content_metadata.model_dump(mode="json", exclude_none=False)

            final_content_values.append(
                {
                    "account_id": account_id,
                    "platform_content_id": item_id,
                    "content": item_data["content_text"],
                    "published_at": _to_utc_aware(extract_instagram_published_at(item)),
                    "transcription": item_data["transcript"],
                    "views": views,
                    "reactions_count": likes,
                    "comments_count": comments,
                    "shares_count": None,
                    "has_media": True,
                    "is_embedded": False,
                    "is_graph_extracted": False,
                    "raw_metadata": raw_meta_dict,
                    "updated_at": datetime.now(timezone.utc),
                }
            )

        if final_content_values:
            async with self.session_maker() as session:
                await bulk_upsert_content(
                    session=session,
                    content_values=final_content_values,
                )

                location_str, geo_data_dict = extract_instagram_geo_data(
                    profile, profile_biography, author_profile_snapshot.title or "",
                )

                await update_account_profile_metadata(
                    session=session,
                    account_id=account_id,
                    platform="INSTAGRAM",
                    biography=profile_biography or "",
                    external_url=profile_external_url,
                    category=account_category,
                    raw_profile_payload=profile,
                    posts_count=profile.get("media_count") or profile.get("posts_count"),
                    language="ru",
                    location=location_str,
                    geo_data=geo_data_dict,
                )

                await session.commit()
                logger.info(
                    "Bulk upserted %d Instagram content items for account_id: %d",
                    len(final_content_values),
                    account_id,
                )
