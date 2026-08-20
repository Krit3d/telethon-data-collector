from __future__ import annotations

import asyncio
import logging
import time

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from sqlalchemy import select

from src.config.config import load_settings
from src.db.database import Database
from src.db.models import Account

logger = logging.getLogger(__name__)

ACCOUNT_BATCH_SIZE = 500


async def create_payload_index(client: AsyncQdrantClient, collection_name: str) -> None:
    try:
        await client.create_payload_index(
            collection_name=collection_name,
            field_name="account_id",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        logger.info("Created payload index for account_id")
    except Exception as e:
        logger.warning("Payload index account_id may already exist: %s", e)

    try:
        await client.create_payload_index(
            collection_name=collection_name,
            field_name="is_author_blog",
            field_schema=models.PayloadSchemaType.BOOL,
        )
        logger.info("Created payload index for is_author_blog")
    except Exception as e:
        logger.warning("Payload index is_author_blog may already exist: %s", e)


async def fetch_account_ids(db: Database) -> tuple[list[int], list[int]]:
    stmt = (
        select(Account.id, Account.is_author_blog)
        .where(
            Account.status == "verified",
            Account.is_author_blog.in_([True, False]),
        )
    )

    author_account_ids: list[int] = []
    business_account_ids: list[int] = []

    async with db.async_session() as session:
        result = await session.execute(stmt)
        for row in result:
            if row.is_author_blog is True:
                author_account_ids.append(row.id)
            elif row.is_author_blog is False:
                business_account_ids.append(row.id)

    return author_account_ids, business_account_ids


async def main() -> None:
    settings = load_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    collection_name = settings.qdrant_collection_name or "social_posts"

    if not settings.qdrant_url:
        logger.error("QDRANT_URL is not configured")
        return

    db = Database(settings.db_url)
    qdrant_client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout,
    )

    try:
        await create_payload_index(qdrant_client, collection_name)

        start = time.monotonic()
        author_account_ids, business_account_ids = await fetch_account_ids(db)
        elapsed = time.monotonic() - start
        logger.info(
            "Fetched %d author + %d business accounts from PostgreSQL in %.2f s",
            len(author_account_ids),
            len(business_account_ids),
            elapsed,
        )

        start = time.monotonic()
        total_processed = 0

        for is_author, batch_ids in [(True, author_account_ids), (False, business_account_ids)]:
            for i in range(0, len(batch_ids), ACCOUNT_BATCH_SIZE):
                chunk = batch_ids[i : i + ACCOUNT_BATCH_SIZE]
                await qdrant_client.set_payload(
                    collection_name=collection_name,
                    payload={"is_author_blog": is_author},
                    points=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="account_id",
                                match=models.MatchAny(any=chunk),
                            )
                        ]
                    ),
                )
                total_processed += len(chunk)
                logger.info(
                    "Processed %d/%d accounts (is_author_blog=%s)",
                    min(i + ACCOUNT_BATCH_SIZE, len(batch_ids)),
                    len(batch_ids),
                    is_author,
                )

        elapsed = time.monotonic() - start
        logger.info(
            "Migration complete: %d accounts updated in %.2f s",
            total_processed,
            elapsed,
        )
    finally:
        await qdrant_client.close()


if __name__ == "__main__":
    asyncio.run(main())