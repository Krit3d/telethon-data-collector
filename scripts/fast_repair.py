import asyncio
import logging
import sys
import time

from sqlalchemy import text, select, or_
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from src.config.config import load_settings
from src.db.database import Database
from src.db.models import Content, Account

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("repair")

CONCURRENCY: int = 2
BATCH_SIZE: int = 1000

REGISTER_FUNCTION_SQL: str = r"""
CREATE OR REPLACE FUNCTION safe_to_int(val text) RETURNS integer AS $$
    SELECT CASE
        WHEN val IS NULL OR trim(val) = '' THEN NULL
        WHEN val ~ '^\s*[-+]?[0-9]*\.?[0-9]+\s*$' THEN (val::numeric)::integer
        ELSE NULL
    END;
$$ LANGUAGE sql IMMUTABLE;
"""

UPDATE_BATCH_SQL: str = """
UPDATE content
SET
    reactions_count = COALESCE(
        safe_to_int(raw_metadata->'raw_item_payload'->>'like_count'),
        safe_to_int(raw_metadata->'raw_item_payload'->'edge_media_preview_like'->>'count'),
        reactions_count,
        0
    ),
    comments_count = COALESCE(
        safe_to_int(raw_metadata->'raw_item_payload'->>'comment_count'),
        safe_to_int(raw_metadata->'raw_item_payload'->'edge_media_to_parent_comment'->>'count'),
        comments_count,
        0
    ),
    views = COALESCE(
        safe_to_int(raw_metadata->'raw_item_payload'->>'view_count'),
        safe_to_int(raw_metadata->'raw_item_payload'->>'video_view_count'),
        safe_to_int(raw_metadata->'raw_item_payload'->>'play_count'),
        views
    ),
    raw_metadata = jsonb_set(
        COALESCE(raw_metadata, '{}'::jsonb),
        '{platform_metrics}',
        jsonb_build_object(
            'likes', COALESCE(
                safe_to_int(raw_metadata->'raw_item_payload'->>'like_count'),
                safe_to_int(raw_metadata->'raw_item_payload'->'edge_media_preview_like'->>'count'),
                0
            ),
            'comments_count', COALESCE(
                safe_to_int(raw_metadata->'raw_item_payload'->>'comment_count'),
                safe_to_int(raw_metadata->'raw_item_payload'->'edge_media_to_parent_comment'->>'count'),
                0
            ),
            'views', COALESCE(
                safe_to_int(raw_metadata->'raw_item_payload'->>'view_count'),
                safe_to_int(raw_metadata->'raw_item_payload'->>'video_view_count'),
                safe_to_int(raw_metadata->'raw_item_payload'->>'play_count')
            ),
            'shares', NULL,
            'plays', safe_to_int(raw_metadata->'raw_item_payload'->>'play_count')
        )
    )
WHERE id = ANY(:ids);
"""


async def process_batch(
    session_factory: async_sessionmaker[AsyncSession],
    chunk_ids: list[int],
    batch_idx: int,
    total_batches: int,
    semaphore: asyncio.Semaphore,
    progress_lock: asyncio.Lock,
    counters: dict[str, int],
    global_start: float,
) -> None:
    async with semaphore:
        batch_start = time.perf_counter()
        async with session_factory() as session:
            await session.execute(
                text(UPDATE_BATCH_SQL),
                {"ids": chunk_ids},
            )
            await session.commit()

        batch_elapsed = time.perf_counter() - batch_start
        global_elapsed = time.perf_counter() - global_start
        rows_processed = len(chunk_ids)
        rps = rows_processed / batch_elapsed if batch_elapsed > 0 else 0

        async with progress_lock:
            counters["updated"] += rows_processed
            counters["batches_done"] += 1
            done = counters["batches_done"]
            percent = (done / total_batches) * 100
            logger.info(
                f"[{done}/{total_batches} ({percent:.1f}%)] Batch {batch_idx} completed in "
                f"{batch_elapsed:.3f}s - {rows_processed} rows "
                f"({rps:,.0f} rows/s) | Total elapsed: {global_elapsed:.2f}s"
            )


async def main() -> None:
    script_start = time.perf_counter()
    settings = load_settings()
    db = Database(settings.db_url)

    async with db.async_session() as session:
        await session.execute(text(REGISTER_FUNCTION_SQL))
        await session.commit()
        logger.info("Registered SQL function safe_to_int")

    async with db.async_session() as session:
        await session.execute(text("DROP INDEX IF EXISTS idx_content_raw_metadata_gin;"))
        await session.commit()
        logger.info("Dropped GIN index idx_content_raw_metadata_gin")

    async with db.async_session() as session:
        fetch_start = time.perf_counter()
        id_query = await session.execute(
            select(Content.id)
            .join(Account, Content.account_id == Account.id)
            .where(
                Account.platform == "INSTAGRAM",
                Account.status == "verified",
                or_(
                    Content.reactions_count.is_(None),
                    Content.comments_count.is_(None),
                ),
            )
        )
        target_ids: list[int] = [row[0] for row in id_query.all()]
        fetch_elapsed = time.perf_counter() - fetch_start

    logger.info(f"Found {len(target_ids)} rows requiring repair in {fetch_elapsed:.2f}s")
    logger.info(f"Concurrency level: {CONCURRENCY} | Batch size: {BATCH_SIZE}")

    if not target_ids:
        logger.warning("Nothing to repair, exiting")
        await db.close()
        return

    chunks: list[list[int]] = [
        target_ids[i : i + BATCH_SIZE]
        for i in range(0, len(target_ids), BATCH_SIZE)
    ]
    total_batches = len(chunks)

    logger.info(f"Processing {total_batches} batches...")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    progress_lock = asyncio.Lock()
    counters: dict[str, int] = {"updated": 0, "batches_done": 0}
    global_start = time.perf_counter()

    tasks = [
        process_batch(
            db.async_session,
            chunk_ids,
            idx,
            total_batches,
            semaphore,
            progress_lock,
            counters,
            global_start,
        )
        for idx, chunk_ids in enumerate(chunks, start=1)
    ]

    await asyncio.gather(*tasks)

    async with db.async_session() as session:
        await session.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_content_raw_metadata_gin "
                "ON content USING gin (raw_metadata jsonb_path_ops);"
            )
        )
        await session.commit()
        logger.info("Rebuilt GIN index idx_content_raw_metadata_gin")

    total_elapsed = time.perf_counter() - script_start
    total_updated = counters["updated"]
    overall_rps = total_updated / total_elapsed if total_elapsed > 0 else 0

    logger.info("=" * 60)
    logger.info(f"Done. Total rows updated: {total_updated:,}")
    logger.info(f"Overall speed: {overall_rps:,.0f} rows/s")
    logger.info(f"Total elapsed: {total_elapsed:.2f}s")
    logger.info("=" * 60)

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
