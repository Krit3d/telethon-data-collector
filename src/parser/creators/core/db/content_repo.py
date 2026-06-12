import logging
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Account, Content
from src.parser.creators.core.contacts import URL_PATTERN
from src.parser.creators.core.db.discovery_repo import queue_single_account
from src.parser.creators.core.db.helpers import (
    clean_content_raw_metadata,
    extract_platform_info,
)

logger = logging.getLogger(__name__)


IGNORE_TAGS: frozenset[str] = frozenset({
    "(music)", "[music]", "(laughter)", "[laughter]",
    "[applause]", "(applause)", "[no speech]", "(silence)",
    "music", "[sound]", "(sound)",
})

SYSTEM_ERROR_PATTERNS: tuple[str, ...] = (
    "hi guys, i'm sorry",
    "there is no recording",
    "please provide",
    "no speech detected",
    "transcription failed",
)

WHISPER_NOISE_RE = re.compile(r"^[\(\[][^\]\)]+[\)\]]$")


def clean_and_validate_transcription(text: str | None) -> str | None:
    if not text:
        return None

    text = text.strip()
    if not text:
        return None

    if WHISPER_NOISE_RE.match(text):
        return None

    lower_text = text.lower()
    if lower_text in IGNORE_TAGS:
        return None

    for pattern in SYSTEM_ERROR_PATTERNS:
        if pattern in lower_text:
            return None

    words = lower_text.split()
    if len(words) > 8:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.30:
            return None

    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) > 4:
        duplicate_ratio = 1 - len(set(lines)) / len(lines)
        if duplicate_ratio > 0.35:
            return None

    all_words = re.findall(r'\b[a-zA-Zа-яА-ЯёЁ]{3,}\b', text)
    if len(all_words) > 15:
        unique_words = set(w.lower() for w in all_words)
        if len(unique_words) / len(all_words) < 0.48:
            return None

    return text


async def bulk_upsert_content(
    session: AsyncSession,
    content_values: list[dict[str, Any]],
) -> None:
    if not content_values:
        return

    now = datetime.now(timezone.utc)

    insert_values = []
    for values in content_values:
        if "account_id" not in values or "platform_content_id" not in values:
            logger.warning("Skipping content value missing required fields: %s", values)
            continue

        raw_metadata = values.get("raw_metadata")
        if raw_metadata is not None:
            raw_metadata = clean_content_raw_metadata(raw_metadata)

        transcription = clean_and_validate_transcription(values.get("transcription"))

        prepared = {
            "account_id": values["account_id"],
            "platform_content_id": values["platform_content_id"],
            "content": values.get("content"),
            "transcription": transcription,
            "published_at": values.get("published_at", now),
            "views": values.get("views"),
            "reactions_count": values.get("reactions_count"),
            "comments_count": values.get("comments_count"),
            "shares_count": values.get("shares_count"),
            "raw_metadata": raw_metadata,
            "updated_at": now,
            "created_at": now,
            "is_embedded": values.get("is_embedded", False),
            "is_graph_extracted": values.get("is_graph_extracted", False),
            "has_media": values.get("has_media", False),
        }
        insert_values.append(prepared)

    if not insert_values:
        return

    stmt = pg_insert(Content).values(insert_values)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_content_account_platform_id",
        set_=dict(
            content=stmt.excluded.content,
            transcription=func.coalesce(stmt.excluded.transcription, Content.transcription),
            views=stmt.excluded.views,
            reactions_count=stmt.excluded.reactions_count,
            comments_count=stmt.excluded.comments_count,
            raw_metadata=stmt.excluded.raw_metadata,
            is_embedded=case(
                (Content.transcription.is_(None) & stmt.excluded.transcription.isnot(None), False),
                else_=Content.is_embedded,
            ),
            is_graph_extracted=case(
                (Content.transcription.is_(None) & stmt.excluded.transcription.isnot(None), False),
                else_=Content.is_graph_extracted,
            ),
            updated_at=stmt.excluded.updated_at,
        ),
    )
    await session.execute(stmt)
    await session.flush()

    logger.debug("Bulk upserted %d content items", len(insert_values))

    try:
        await process_content_external_links(session, content_values)
    except Exception as e:
        logger.error("Failed to process external links from content batch: %s", e, exc_info=True)


async def process_content_external_links(
    session: AsyncSession,
    content_values: list[dict[str, Any]],
    status: str = "pending",
) -> None:
    if not content_values:
        return

    account_ids = {v["account_id"] for v in content_values if "account_id" in v}
    if not account_ids:
        return

    stmt = select(Account).where(Account.id.in_(account_ids))
    result = await session.execute(stmt)
    parent_handle_map: dict[int, str] = {}
    parent_category_map: dict[int, str | None] = {}

    for account in result.scalars():
        account_id = account.id
        handle = account.username or account.platform_id or str(account_id)
        parent_handle_map[account_id] = handle
        category = None
        if account.raw_metadata and isinstance(account.raw_metadata, dict):
            category = account.raw_metadata.get("category")
        parent_category_map[account_id] = category

    for content_dict in content_values:
        account_id = content_dict.get("account_id")
        if not account_id:
            continue

        parent_handle = parent_handle_map.get(account_id, str(account_id))
        parent_category = parent_category_map.get(account_id)

        text_parts = []
        content = content_dict.get("content")
        transcription = content_dict.get("transcription")

        if content:
            text_parts.append(content)
        if transcription:
            text_parts.append(transcription)

        if not text_parts:
            continue

        combined_text = " ".join(text_parts)

        urls = URL_PATTERN.findall(combined_text)

        for url in urls:
            platform, platform_id = extract_platform_info(url)
            if not platform or not platform_id:
                continue

            if platform in ("WEBSITE", "LINK_IN_BIO"):
                continue

            await queue_single_account(
                session, platform, platform_id, parent_handle, status, parent_category
            )
