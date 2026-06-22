import asyncio
import logging
import os
import sys

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db.database import Database
from src.db.models import Account, Content

logger = logging.getLogger("repair_metrics")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

BATCH_SIZE = 1000


def _safe_int(value: str | int | float | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_from_dict(data: dict | None, *keys: str) -> int | None:
    if data is None:
        return None
    for key in keys:
        val = data.get(key)
        if val is not None:
            return _safe_int(val)
    return None


def _extract_nested_count(data: dict | None, *paths: tuple[str, str]) -> int | None:
    if data is None:
        return None
    for outer_key, inner_key in paths:
        nested = data.get(outer_key)
        if isinstance(nested, dict):
            val = nested.get(inner_key)
            if val is not None:
                return _safe_int(val)
    return None


def extract_metrics(raw_metadata: dict | None) -> dict[str, int | None] | None:
    if not raw_metadata:
        return None

    raw_item_payload = raw_metadata.get("raw_item_payload")
    if not raw_item_payload or not isinstance(raw_item_payload, dict):
        return None

    likes = (
        _extract_from_dict(raw_item_payload, "like_count", "likes")
        or _extract_nested_count(raw_item_payload, ("edge_media_preview_like", "count"))
    )

    comments = (
        _extract_from_dict(raw_item_payload, "comment_count", "comments")
        or _extract_nested_count(raw_item_payload, ("edge_media_to_parent_comment", "count"))
    )

    plays = _extract_from_dict(raw_item_payload, "play_count", "plays")

    views = (
        _extract_from_dict(raw_item_payload, "view_count", "video_view_count")
        or plays
    )

    shares = _extract_from_dict(raw_item_payload, "share_count", "shares")

    platform_metrics: dict[str, int | None] = {
        "likes": likes,
        "comments_count": comments,
        "views": views,
        "shares": shares,
        "plays": plays,
    }

    return platform_metrics


async def process_batch(session: AsyncSession, last_id: int) -> tuple[int, int]:
    stmt = (
        select(Content)
        .join(Account, Content.account_id == Account.id)
        .where(
            and_(
                Content.id > last_id,
                Account.platform == "INSTAGRAM",
                or_(Content.views.is_(None), Content.comments_count.is_(None)),
            )
        )
        .order_by(Content.id.asc())
        .limit(BATCH_SIZE)
    )

    result = await session.execute(stmt)
    rows = list(result.scalars().all())

    if not rows:
        return last_id, 0

    updated = 0
    for row in rows:
        raw_metadata = row.raw_metadata
        if not raw_metadata or not isinstance(raw_metadata, dict):
            last_id = row.id
            continue

        metrics = extract_metrics(raw_metadata)
        if metrics is None:
            last_id = row.id
            continue

        raw_metadata["platform_metrics"] = metrics
        row.raw_metadata = raw_metadata

        if metrics["likes"] is not None:
            row.reactions_count = metrics["likes"]
        if metrics["comments_count"] is not None:
            row.comments_count = metrics["comments_count"]
        if metrics["views"] is not None:
            row.views = metrics["views"]

        last_id = row.id
        updated += 1

    await session.commit()

    return last_id, updated


async def main() -> None:
    db_url = os.environ.get("DB_URL", "").strip()

    if not db_url:
        try:
            from src.config.config import load_settings
            db_url = load_settings().db_url
            logger.info("Loaded DB_URL from config settings")
        except Exception:
            raise ValueError(
                "DB_URL environment variable is not set and could not load from src.config.config"
            )

    db = Database(db_url)
    total_processed = 0
    total_updated = 0
    last_id = 0

    logger.info("Starting repair_metrics backfill for INSTAGRAM posts")

    async with db.async_session() as session:
        while True:
            try:
                new_last_id, updated = await process_batch(session, last_id)
            except Exception as exc:
                logger.error("Database error at last_id=%d: %s", last_id, exc)
                break

            batch_count = new_last_id - last_id if new_last_id > last_id else 0
            total_processed += batch_count
            total_updated += updated

            if batch_count == 0:
                logger.info(
                    "No more rows to process. Final totals: scanned=%d, updated=%d",
                    total_processed,
                    total_updated,
                )
                break

            last_id = new_last_id
            logger.info(
                "Updated batch up to ID %d. Total processed in this session: %d, updated: %d",
                last_id,
                total_processed,
                total_updated,
            )

    await db.engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
