import asyncio
import logging
import time
from typing import cast as type_cast

from sqlalchemy import select, text
from sqlalchemy.engine import CursorResult

from src.config.config import load_settings
from src.db.database import Database
from src.db.models import Account

logger = logging.getLogger(__name__)

BATCH_SIZE: int = 2000

UPDATE_AVG_ER_SQL: str = """
WITH target_accounts AS (
    SELECT id
    FROM accounts
    WHERE id = ANY(:ids)
      AND status = 'verified'
),
agg AS (
    SELECT c.account_id AS account_id,
           AVG(
               LEAST(
                   30.0,
                   ((COALESCE(c.comments_count, 0) + COALESCE(c.shares_count, 0) + c.reactions_count)::double precision / c.views) * 100.0
               )
           ) AS avg_er
    FROM content AS c
    WHERE c.account_id = ANY(:ids)
      AND c.reactions_count IS NOT NULL
      AND c.views IS NOT NULL
      AND c.views >= 50
    GROUP BY c.account_id
)
UPDATE accounts AS a
SET static_avg_er = COALESCE(agg.avg_er, 0.0)
FROM target_accounts
LEFT JOIN agg ON agg.account_id = target_accounts.id
WHERE a.id = target_accounts.id
"""


async def process_batch(db: Database, account_ids: list[int]) -> int:
    async with db.async_session() as session:
        async with session.begin():
            result = type_cast(
                CursorResult,
                await session.execute(
                    text(UPDATE_AVG_ER_SQL),
                    {"ids": account_ids},
                ),
            )
            return result.rowcount or 0


async def main(batch_size: int | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings = load_settings()
    db = Database(settings.db_url)
    await db.init_db()

    try:
        async with db.async_session() as session:
            result = await session.execute(
                select(Account.id)
                .where(Account.status == "verified")
                .order_by(Account.id)
            )
            account_ids: list[int] = list(result.scalars().all())

        total_accounts = len(account_ids)
        logger.info("Found %d verified accounts to recalculate", total_accounts)

        if total_accounts == 0:
            logger.warning("No verified accounts found, exiting")
            return

        chunk_size: int = batch_size or BATCH_SIZE
        chunks: list[list[int]] = [
            account_ids[i : i + chunk_size]
            for i in range(0, total_accounts, chunk_size)
        ]
        total_batches = len(chunks)
        total_updated = 0
        script_start = time.perf_counter()

        for batch_idx, chunk in enumerate(chunks, start=1):
            batch_start = time.perf_counter()
            updated = await process_batch(db, chunk)
            batch_elapsed = time.perf_counter() - batch_start
            total_updated += updated
            logger.info(
                "Batch %d/%d | ids %d..%d | updated %d accounts | %.3f s",
                batch_idx,
                total_batches,
                chunk[0],
                chunk[-1],
                updated,
                batch_elapsed,
            )

        total_elapsed = time.perf_counter() - script_start
        logger.info(
            "Done. Updated %d accounts in %d batches, total %.3f s",
            total_updated,
            total_batches,
            total_elapsed,
        )
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())