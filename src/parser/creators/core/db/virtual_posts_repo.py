import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Content
from src.parser.creators.core.db.helpers import clean_account_raw_metadata
from src.parser.creators.core.schemas import AccountMetadata

logger = logging.getLogger(__name__)


async def upsert_virtual_bio_post(
    session: AsyncSession,
    account_id: int,
    platform: str,
    platform_id: str,
    username: str | None,
    full_name: str | None,
    biography: str | None,
    subscribers_count: int,
    raw_metadata: AccountMetadata | dict[str, Any] | None = None,
) -> None:
    virtual_content_id = f"profile_bio_{platform_id}"
    compiled_text = (
        f"[PROFILE METADATA]\n"
        f"Platform: {platform}\n"
        f"Username: @{username or 'unknown'}\n"
        f"Title: {full_name or 'Unknown'}\n"
        f"Subscribers: {subscribers_count}\n"
        f"Bio: {biography or 'N/A'}"
    )

    now = datetime.now(timezone.utc)

    processed_metadata = clean_account_raw_metadata(raw_metadata)

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
        raw_metadata=processed_metadata,
        updated_at=now,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_content_account_platform_id",
        set_=dict(
            content=stmt.excluded.content,
            raw_metadata=stmt.excluded.raw_metadata,
            updated_at=stmt.excluded.updated_at,
        ),
    )
    await session.execute(stmt)
    logger.debug(
        "Upserted virtual profile post for account_id: %d (platform: %s, platform_content_id: %s)",
        account_id,
        platform,
        virtual_content_id,
    )
