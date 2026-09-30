import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import cast, func, or_, select, update
from sqlalchemy.dialects.postgresql import JSONB

from src.db.models import Account, Content
from src.parser.creators.core.db.accounts_repo import (
    upsert_and_deduplicate_account,
    update_account_profile_metadata,
)
from src.parser.creators.core.db.content_repo import (
    bulk_upsert_content,
    clean_and_validate_transcription,
)
from src.parser.creators.platforms.instagram.helpers import (
    extract_instagram_subscribers,
    extract_instagram_content_text,
    extract_instagram_published_at,
    extract_instagram_metrics,
    extract_instagram_primary_external_url,
)
from src.parser.creators.core.schemas import (
    InstagramContentMetadata,
    PlatformMetrics,
    AuthorProfileSnapshot,
    AccountMetadata,
    Contacts,
)
from src.parser.creators.core.contacts import (
    compile_author_metadata,
    extract_structural_links,
)
from src.parser.creators.core.media_detector import detect_content_media
from src.parser.creators.core.text import is_slop_or_theme_page
from src.parser.creators.platforms.base import BasePlatformParser

from .client import fetch_instagram_profile, fetch_video_transcript
from .contacts_processor import process_and_queue_discovered_contacts
from .fetcher import fetch_recent_instagram_posts
from .helpers import (
    extract_instagram_video_url,
    normalize_instagram_hashtags,
    prune_instagram_payload,
)
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

    def _build_account_metadata(
        self,
        profile: dict[str, Any],
        biography: str | None,
        contacts_dict: dict[str, Any] | None = None,
        context_text: str | None = None,
    ) -> AccountMetadata:
        if self.settings.enable_contact_extraction:
            return compile_author_metadata(
                platform="INSTAGRAM",
                username=profile.get("username", ""),
                biography=biography,
                raw_profile_payload=profile,
                contacts_dict=contacts_dict or {},
                context_text=context_text,
            )
        link_in_bio, website, external_platforms, external_links = extract_structural_links(
            biography, profile, contacts_dict=contacts_dict
        )
        username = profile.get("username", "")
        return AccountMetadata(
            profile_url=f"https://instagram.com/{username}" if username else None,
            biography=biography or None,
            website=website,
            link_in_bio=link_in_bio,
            external_platforms=external_platforms,
            external_links=external_links,
            contacts=Contacts(),
            raw_profile_payload=profile,
            extracted_at=datetime.now(timezone.utc).isoformat(),
        )

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

    async def _update_transcription_status(self, item_id: str, status: str) -> None:
        async with self.session_maker() as session:
            stmt = (
                update(Content)
                .where(Content.platform_content_id == item_id)
                .values(
                    raw_metadata=Content.raw_metadata.concat(
                        cast({"transcription_status": status}, JSONB)
                    ),
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.execute(stmt)
            await session.commit()

    async def _transcribe_and_update_content(self, item_id: str, post_url: str) -> None:
        try:
            result = await fetch_video_transcript(
                self.client, self.client.global_semaphore, post_url,
            )
        except Exception as e:
            logger.warning(
                "Background transcript fetch failed for %s: %s. Leaving record for retry.",
                item_id,
                e,
            )
            return

        cleaned = result.strip() if isinstance(result, str) else ""

        if not cleaned:
            await self._update_transcription_status(item_id, "skipped_no_speech")
            logger.info(
                "Background transcript for %s marked as skipped_no_speech (no speech or empty result).",
                item_id,
            )
            return

        validated_text = clean_and_validate_transcription(cleaned)

        if validated_text:
            async with self.session_maker() as session:
                stmt = (
                    update(Content)
                    .where(Content.platform_content_id == item_id)
                    .values(
                        transcription=validated_text,
                        has_media=True,
                        is_embedded=False,
                        graph_status=0,
                        raw_metadata=Content.raw_metadata.concat(
                            cast(
                                {
                                    "post_type": "reel",
                                    "transcription_status": "completed",
                                },
                                JSONB,
                            )
                        ),
                        updated_at=datetime.now(timezone.utc),
                    )
                )
                await session.execute(stmt)
                await session.commit()
            logger.debug(
                "Background transcript updated for item %s",
                item_id,
            )
            return

        await self._update_transcription_status(item_id, "rejected")

    async def parse_profile(self, handle: str) -> int | None:
        async with self.session_maker() as session:
            verified_stmt = select(Account.id, Account.status).where(
                Account.platform == "INSTAGRAM",
                Account.status == "verified",
                or_(
                    func.lower(Account.username) == handle.lower(),
                    func.lower(Account.platform_id) == handle.lower(),
                ),
            )
            verified_result = await session.execute(verified_stmt)
            verified_row = verified_result.first()
            if verified_row is not None and verified_row[1] == "verified":
                return verified_row[0]

        profile = await fetch_instagram_profile(self.client, handle)
        if not profile:
            logger.info(
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
        profile_external_url = extract_instagram_primary_external_url(profile)

        subscribers = extract_instagram_subscribers(profile)

        if not subscribers:
            edge_followed_by = profile.get("edge_followed_by")
            if isinstance(edge_followed_by, dict):
                fallback_val = edge_followed_by.get("count")
                if fallback_val is not None:
                    try:
                        subscribers = int(fallback_val)
                    except (ValueError, TypeError):
                        pass

        if not subscribers:
            for key in ("follower_count", "followers", "followers_count"):
                raw_val = profile.get(key)
                if raw_val is not None:
                    try:
                        subscribers = int(raw_val)
                    except (ValueError, TypeError):
                        continue
                    if subscribers:
                        break

        if not subscribers:
            logger.warning(
                "Instagram handle %s parsed 0 subscribers. Profile dict keys: %s",
                handle,
                list(profile.keys()),
            )

        if not validate_follower_count(subscribers):
            logger.info(
                "Instagram handle %s REJECTED: subscriber count %d is outside range [%d, %d].",
                handle,
                subscribers,
                MIN_SUBSCRIBERS,
                MAX_SUBSCRIBERS,
            )
            async with self.session_maker() as session:
                existing_stmt = select(Account.id, Account.status).where(
                    Account.platform == "INSTAGRAM",
                    Account.status == "verified",
                    or_(
                        Account.platform_id == str(profile.get("id") or username),
                        func.lower(Account.username) == username.lower(),
                    ),
                )
                existing_result = await session.execute(existing_stmt)
                existing_row = existing_result.first()
                if existing_row is not None and existing_row[1] == "verified":
                    return existing_row[0]
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
            logger.info(
                "Instagram handle %s REJECTED: Matched slop/theme stop-words.",
                handle,
            )
            async with self.session_maker() as session:
                existing_stmt = select(Account.id, Account.status).where(
                    Account.platform == "INSTAGRAM",
                    Account.status == "verified",
                    or_(
                        Account.platform_id == str(profile.get("id") or username),
                        func.lower(Account.username) == username.lower(),
                    ),
                )
                existing_result = await session.execute(existing_stmt)
                existing_row = existing_result.first()
                if existing_row is not None and existing_row[1] == "verified":
                    return existing_row[0]
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
                    external_url=profile_external_url,
                    subscribers_count=subscribers,
                    raw_profile_payload=profile,
                    posts_count=profile.get("media_count") or profile.get("posts_count"),
                    account_metadata=self._build_account_metadata(profile, biography),
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
            logger.debug(
                "Instagram handle %s: Stage1 PASSED (Cyrillic detected). Passing to Stage2.",
                handle,
            )
        else:
            logger.debug(
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

            await update_account_profile_metadata(
                session=session,
                account_id=account_id,
                platform="INSTAGRAM",
                biography=biography or "",
                external_url=profile_external_url,
                subscribers_count=subscribers,
                raw_profile_payload=profile,
                posts_count=profile.get("media_count") or profile.get("posts_count"),
                account_metadata=self._build_account_metadata(profile, biography),
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

            if isinstance(response, list):
                profiles = [item for item in response if isinstance(item, dict)]
            elif isinstance(response, dict):
                profiles = []
                for key in ("profiles", "data", "items"):
                    value = response.get(key)
                    if isinstance(value, list):
                        profiles = [item for item in value if isinstance(item, dict)]
                        break
            else:
                profiles = []

            if not profiles:
                logger.info(
                    "Instagram search for query='%s' returned 0 profiles",
                    query,
                )
                return 0

            raw_count = len(profiles)
            valid_count = 0
            new_count = 0
            existing_count = 0
            filtered_count = 0

            async with self.session_maker() as session:
                for profile in profiles:
                    username = profile.get("username") or profile.get("handle")
                    if not username or not isinstance(username, str):
                        filtered_count += 1
                        continue

                    followers_raw = profile.get("follower_count")
                    if followers_raw is None:
                        followers_raw = profile.get("followers")
                    if followers_raw is None:
                        stats = profile.get("stats")
                        if isinstance(stats, dict):
                            followers_raw = stats.get("followers")
                    if followers_raw is None:
                        user = profile.get("user")
                        if isinstance(user, dict):
                            followers_raw = user.get("follower_count")

                    try:
                        followers = int(followers_raw) if followers_raw is not None else 0
                    except (ValueError, TypeError):
                        followers = 0

                    if followers == 0 or not validate_follower_count(followers):
                        filtered_count += 1
                        continue

                    valid_count += 1

                    profile_id = profile.get("id")
                    full_name = profile.get("full_name", "")
                    biography = profile.get("biography", "") or ""

                    exists_stmt = select(Account.id).where(
                        Account.platform == "INSTAGRAM",
                        or_(
                            Account.platform_id == str(profile_id or username),
                            func.lower(Account.username) == username.lower(),
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

                        if not already_exists:
                            meta: dict[str, Any] = {
                                "discovery_query": query,
                                "search_metadata": profile,
                            }
                            stmt = (
                                update(Account)
                                .where(
                                    Account.id == account_id,
                                    Account.status != "verified",
                                )
                                .values(raw_metadata=meta, updated_at=datetime.now(timezone.utc))
                            )
                            await session.execute(stmt)
                            new_count += 1
                        else:
                            existing_count += 1

                    except Exception as e:
                        logger.error(
                            "Failed to upsert Instagram profile %s: %s",
                            username,
                            e,
                            exc_info=True,
                        )
                        continue

                await session.commit()

            logger.info(
                "Discovery stats for query='%s' (category='%s'): raw=%d | valid=%d | new=%d | existing=%d | filtered=%d",
                query,
                category,
                raw_count,
                valid_count,
                new_count,
                existing_count,
                filtered_count,
            )
            return valid_count

        except Exception as e:
            logger.error(
                "Instagram candidate discovery failed for query: '%s': %s",
                query,
                e,
                exc_info=True,
            )
            return 0

    async def parse_content(self, account_id: int, platform_id: str, max_items: int = 12) -> int:
        logger.debug(
            "Starting Instagram content parse for account_id: %d, platform_id: %s",
            account_id,
            platform_id,
        )

        async with self.session_maker() as session:
            status_stmt = select(Account.status).where(Account.id == account_id)
            status_result = await session.execute(status_stmt)
            current_status = status_result.scalar_one_or_none()

        if current_status == "verified":
            logger.debug(
                "Skipping content parsing for account_id: %d because it is verified.",
                account_id,
            )
            return 0

        profile = await self._fetch_raw_profile_payload(account_id)
        if profile is None:
            profile = await fetch_instagram_profile(self.client, platform_id)
        if not profile:
            raise RuntimeError(f"Could not retrieve profile metadata for {platform_id} during content parsing.")

        profile_biography = profile.get("biography")
        profile_external_url = extract_instagram_primary_external_url(profile)

        author_profile_snapshot = AuthorProfileSnapshot(
            username=profile.get("username", ""),
            title=profile.get("full_name") or profile.get("username", ""),
        )

        target_handle = str(profile.get("username") or platform_id)

        try:
            recent_items = await fetch_recent_instagram_posts(
                client=self.client,
                handle=target_handle,
                max_total_items=max_items,
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
                        .where(Account.id == account_id, Account.status != "verified")
                        .values(status="rejected", updated_at=datetime.now(timezone.utc))
                    )
                    await session.execute(stmt)
                    await session.commit()
                return 0
            raise

        def _to_utc_aware(dt: datetime) -> datetime:
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        if not recent_items:
            logger.info(
                "No valid Instagram content found for account_id: %d. Rejecting account.",
                account_id,
            )
            async with self.session_maker() as session:
                stmt = (
                    update(Account)
                    .where(Account.id == account_id, Account.status != "verified")
                    .values(status="rejected", updated_at=datetime.now(timezone.utc))
                )
                await session.execute(stmt)
                await session.commit()
            return 0

        recent_items = recent_items[:max_items]

        logger.debug(
            "Fetched %d recent posts for account_id: %d",
            len(recent_items),
            account_id,
        )

        aggregated_text = ""
        for item in recent_items:
            description = extract_instagram_content_text(item) or ""

            hashtags = normalize_instagram_hashtags(item, description)

            aggregated_text += " " + description + " " + " ".join(hashtags)

        has_cyrillic = (
            check_cyrillic_stage1(profile_biography, author_profile_snapshot.title)
            or check_cyrillic_stage2(aggregated_text)
        )

        if not has_cyrillic:
            logger.info(
                "Account %s (account_id: %d) REJECTED: No Cyrillic characters found in %d fetched posts.",
                platform_id,
                account_id,
                len(recent_items),
            )
            async with self.session_maker() as session:
                stmt = (
                    update(Account)
                    .where(Account.id == account_id, Account.status != "verified")
                    .values(status="rejected", updated_at=datetime.now(timezone.utc))
                )
                await session.execute(stmt)
                await session.commit()
            return 0

        logger.debug(
            "Stage2 Cyrillic validation PASSED for account_id: %d. Proceeding with content parsing.",
            account_id,
        )

        items_data: list[dict[str, Any]] = []
        items_needing_transcripts: list[tuple[str, str]] = []

        max_post_age_days = self.settings.max_post_age_days
        now_utc = datetime.now(timezone.utc)

        for item in recent_items:
            raw_id = item.get("id") or (
                f"{item['pk']}_{item.get('user', {}).get('pk', '')}"
                if item.get("pk") and isinstance(item.get("user"), dict) and item.get("user", {}).get("pk")
                else item.get("pk")
            )
            if not raw_id:
                continue
            item_id = str(raw_id)

            content_text = extract_instagram_content_text(item)
            description = content_text or ""

            hashtags = normalize_instagram_hashtags(item, description)

            combined_text = description + " " + " ".join(hashtags)

            likes, comments, views = extract_instagram_metrics(item)

            raw_duration = item.get("video_duration") or item.get("duration")
            if raw_duration is not None:
                try:
                    duration: float | None = float(raw_duration)
                except (ValueError, TypeError):
                    duration = None
            else:
                duration = None

            post_type, has_media = detect_content_media(
                platform="INSTAGRAM", payload=item, duration=duration
            )

            if has_media is True:
                video_url = extract_instagram_video_url(item)
            else:
                video_url = None

            shortcode = item.get("code") or item.get("shortcode")
            post_url = f"https://instagram.com/p/{shortcode}" if shortcode else item.get("url")

            try:
                published_dt = _to_utc_aware(extract_instagram_published_at(item))
            except Exception:
                published_dt = now_utc

            post_age_days = (now_utc - published_dt).days
            is_stale = post_age_days > max_post_age_days

            duration_eligible = duration is None or (3.0 <= duration <= 120.0)

            needs_transcript = (
                has_media
                and duration_eligible
                and not is_stale
            )

            if has_media:
                transcription_status = "pending"
                if not needs_transcript:
                    if is_stale:
                        transcription_status = "skipped_stale"
                    else:
                        transcription_status = "skipped"
            else:
                transcription_status = "skipped"

            if needs_transcript and post_url:
                items_needing_transcripts.append((item_id, post_url))

            item_data: dict[str, Any] = {
                "item": item,
                "item_id": item_id,
                "content_text": content_text,
                "likes": likes,
                "comments": comments,
                "views": views,
                "video_url": video_url,
                "post_type": post_type,
                "has_media": has_media,
                "post_url": post_url,
                "transcript": None,
                "hashtags": hashtags,
                "combined_text": combined_text,
                "transcription_status": transcription_status,
                "published_at": published_dt,
            }

            items_data.append(item_data)

        candidate_item_ids = [d["item_id"] for d in items_data]

        terminal_tx_statuses = ("completed", "rejected", "skipped", "skipped_stale", "skipped_no_speech")

        already_transcribed: dict[str, str] = {}
        skip_transcript_ids: set[str] = set()
        if candidate_item_ids:
            async with self.session_maker() as session:
                stmt = (
                    select(
                        Content.platform_content_id,
                        Content.transcription,
                        Content.raw_metadata["transcription_status"].astext,
                    )
                    .where(
                        Content.platform_content_id.in_(candidate_item_ids),
                        or_(
                            Content.transcription.isnot(None),
                            Content.raw_metadata["transcription_status"].astext.in_(terminal_tx_statuses),
                        ),
                    )
                )
                result = await session.execute(stmt)
                rows = result.all()
                already_transcribed = {row[0]: row[1] for row in rows if row[1] is not None}
                skip_transcript_ids = {row[0] for row in rows}

        items_needing_transcripts = [
            (item_id, post_url)
            for item_id, post_url in items_needing_transcripts
            if item_id not in skip_transcript_ids
        ]

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

            tx_status = "completed" if item_id in already_transcribed else item_data.get("transcription_status", "pending")

            content_metadata = InstagramContentMetadata.create_with_timestamp(
                video_url=video_url,
                post_type=post_type,
                platform_metrics=platform_metrics,
                author_profile_snapshot=author_profile_snapshot,
                raw_item_payload=prune_instagram_payload(item),
                hashtags=hashtags,
                post_url=item_data["post_url"],
                transcription_status=tx_status,
            )

            raw_meta_dict = content_metadata.model_dump(mode="json", exclude_none=False)

            final_content_values.append(
                {
                    "account_id": account_id,
                    "platform_content_id": item_id,
                    "content": item_data["content_text"],
                    "published_at": item_data["published_at"],
                    "transcription": already_transcribed.get(item_id),
                    "views": views,
                    "reactions_count": likes,
                    "comments_count": comments,
                    "shares_count": None,
                    "has_media": item_data["has_media"],
                    "is_embedded": False,
                    "graph_status": 0,
                    "raw_metadata": raw_meta_dict,
                    "updated_at": datetime.now(timezone.utc),
                }
            )

        contacts_dict, spider_count = await process_and_queue_discovered_contacts(
            session_maker=self.session_maker,
            parent_username=profile.get("username", ""),
            profile_biography=profile_biography,
            profile_external_url=profile_external_url,
            items_data=items_data,
            enable_contact_extraction=self.settings.enable_contact_extraction,
        )

        account_metadata = self._build_account_metadata(
            profile,
            profile_biography,
            contacts_dict=contacts_dict,
            context_text=aggregated_text,
        )

        if final_content_values:
            async with self.session_maker() as session:
                await bulk_upsert_content(
                    session=session,
                    content_values=final_content_values,
                )

                await update_account_profile_metadata(
                    session=session,
                    account_id=account_id,
                    platform="INSTAGRAM",
                    biography=profile_biography or "",
                    external_url=profile_external_url,
                    raw_profile_payload=profile,
                    posts_count=profile.get("media_count") or profile.get("posts_count"),
                    account_metadata=account_metadata,
                )

                await session.commit()
                logger.debug(
                    "Bulk upserted %d Instagram content items for account_id: %d",
                    len(final_content_values),
                    account_id,
                )

        for t_item_id, t_post_url in items_needing_transcripts:
            task: asyncio.Task[None] = asyncio.create_task(
                self._transcribe_and_update_content(
                    item_id=t_item_id, post_url=t_post_url,
                )
            )
            self.client.background_tasks.add(task)
            task.add_done_callback(self.client.background_tasks.discard)

        return spider_count

