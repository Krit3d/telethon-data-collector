"""
Instagram platform parser using Scrape Creators API.

Implements Instagram-specific profile parsing and content ingestion into PostgreSQL.
Extracts author profile metadata and stores it inside Content.raw_metadata.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
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
        """Fetch Instagram profile, apply filters, upsert account."""
        logger.info("Starting Instagram profile parse for handle: %s", handle)

        profile = await self._get_cached_or_fetch_profile(handle)
        if not profile:
            return None

        # Subscriber threshold filter
        subscribers = extract_instagram_subscribers(profile)
        if not (MIN_SUBSCRIBERS <= subscribers <= MAX_SUBSCRIBERS):
            logger.info(
                "Instagram handle %s has %d subscribers, outside range [%d, %d]. Rejecting.",
                handle, subscribers, MIN_SUBSCRIBERS, MAX_SUBSCRIBERS,
            )
            await self._upsert_account(profile, "rejected")
            return None

        # Content quality filters
        username = profile.get("username", "")
        biography = profile.get("biography", "")
        if not is_russian_text(biography) or is_slop_or_theme_page(username, biography):
            logger.info("Instagram handle %s failed content filters. Rejecting.", handle)
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
            handle, account_id, subscribers,
        )

        # Post-parse actions
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

        # Queue discovered accounts from contacts
        contacts_dict = parse_profile_contacts(biography, profile.get("external_url"))
        async with self.session_maker() as session:
            await queue_discovered_accounts(
                session=session,
                contacts_dict=contacts_dict,
                parent_handle=handle,
            )

        return account_id

    async def _get_cached_or_fetch_profile(self, handle: str) -> dict[str, Any] | None:
        """Get profile from cache or fetch from API."""
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

        except Exception as e:
            logger.error("API request failed for Instagram profile %s: %s", handle, e, exc_info=True)
            return None

    async def _upsert_account(
        self, profile: dict[str, Any], status: str
    ) -> int:
        """Select-then-upsert account to avoid database conflicts."""
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
        """Parse timeline content, fetch transcripts, and bulk upsert."""
        logger.info(
            "Starting Instagram content parse for account_id: %d, platform_id: %s",
            account_id, platform_id,
        )

        profile = await self._get_cached_or_fetch_profile(platform_id)
        if not profile:
            return

        # Extract and deduplicate media nodes
        edges = []
        for key in ["edge_owner_to_timeline_media", "edge_felix_video_timeline"]:
            edges.extend(profile.get(key, {}).get("edges", []))

        seen_nodes: Dict[str, dict] = {}
        for edge in edges:
            node = edge.get("node", {})
            if node_id := node.get("id"):
                seen_nodes[node_id] = node

        # Sort by date (newest first) and slice
        sorted_nodes = sorted(
            seen_nodes.values(),
            key=lambda x: extract_instagram_published_at(x) or datetime.min,
            reverse=True,
        )
        target_nodes = sorted_nodes[:max_items]

        if not target_nodes:
            logger.info("No Instagram content found for account_id: %d", account_id)
            return

        # Get existing transcripts to skip already-processed content
        platform_ids = [n["id"] for n in target_nodes if n.get("id")]
        existing_transcripts = await self._get_existing_transcripts(account_id, platform_ids)

        # Build author metadata once
        author_metadata = build_instagram_author_metadata(profile)

        # Fetch transcripts concurrently for eligible videos
        semaphore = asyncio.Semaphore(5)
        transcript_tasks = []
        eligible_nodes = []
        for node in target_nodes:
            node_id = node.get("id")
            if not node_id or node_id in existing_transcripts:
                continue
            if not node.get("is_video") or (node.get("video_duration") or 0.0) > 120.0:
                continue
            eligible_nodes.append(node)
            transcript_tasks.append(self._fetch_transcript(semaphore, node_id))

        transcripts = {}
        if transcript_tasks:
            results = await asyncio.gather(*transcript_tasks, return_exceptions=True)
            for node, result in zip(eligible_nodes, results):
                if isinstance(result, Exception):
                    logger.error("Transcript fetch failed for %s: %s", node.get("id"), result)
                else:
                    transcripts[node["id"]] = result

        # Build and upsert content items
        content_values = []
        for node in target_nodes:
            node_id = node.get("id")
            if not node_id:
                continue

            likes, comments = extract_instagram_metrics(node)
            content_values.append({
                "account_id": account_id,
                "platform_content_id": node_id,
                "content": extract_instagram_content_text(node),
                "published_at": extract_instagram_published_at(node),
                "transcription": transcripts.get(node_id),
                "views": node.get("video_view_count") or node.get("play_count"),
                "reactions_count": likes,
                "comments_count": comments,
                "shares_count": None,
                "has_media": bool(node.get("is_video")),
                "is_embedded": False,
                "is_graph_extracted": False,
                "raw_metadata": {
                    "platform_metrics": {
                        "likes": likes,
                        "comments": comments,
                        "video_url": extract_instagram_video_url(node) if node.get("is_video") else None,
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
                        len(content_values), account_id,
                    )

    async def _fetch_transcript(self, semaphore: asyncio.Semaphore, media_id: str) -> str | None:
        """Fetch media transcript with rate limiting."""
        async with semaphore:
            try:
                response = await self.client.get(
                    endpoint="/v2/instagram/media/transcript",
                    params={"media_id": media_id},
                )
                # Parse the plural "transcripts" list from the API response
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
                logger.error("Transcript fetch failed for %s: %s", media_id, e)
                return None

    async def _get_existing_transcripts(
        self, account_id: int, platform_ids: List[str]
    ) -> set[str]:
        """Get set of platform_content_ids that already have transcripts."""
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
