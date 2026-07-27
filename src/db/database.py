import asyncio
import functools
import logging
import random
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any, cast as type_cast

from sqlalchemy import and_, case, func, or_, select, update, text, cast
from sqlalchemy.dialects.postgresql import insert, JSONPATH
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import joinedload

from src.db.models import Base, Account, Content

logger = logging.getLogger(__name__)


def with_retry_on_deadlock(
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None

            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (OperationalError, DBAPIError) as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    if hasattr(e, "orig") and e.orig is not None:
                        error_msg += " " + str(e.orig).lower()

                    is_deadlock = "40p01" in error_msg or "deadlock detected" in error_msg
                    is_serialization = "40001" in error_msg or "serialization failure" in error_msg
                    is_lock_timeout = "55p03" in error_msg or "lock timeout" in error_msg or "locknotavailable" in error_msg

                    if not (is_deadlock or is_serialization or is_lock_timeout):
                        raise

                    if attempt == max_retries:
                        logger.error(
                            "Failed after %d retries on deadlock/serialization/lock timeout failure: %s",
                            max_retries,
                            e,
                        )
                        raise

                    delay = base_delay * (2 ** attempt)
                    jitter = random.uniform(0, 0.1)
                    total_delay = delay + jitter

                    logger.warning(
                        "Attempt %d/%d failed with retryable error: %s. Retrying in %.2f s...",
                        attempt,
                        max_retries,
                        e,
                        total_delay,
                    )
                    await asyncio.sleep(total_delay)

            if last_exception is not None:
                raise last_exception

        return wrapper

    return decorator


class Database:
    def __init__(self, db_url: str, echo: bool = False, pool_size: int = 20, max_overflow: int = 10) -> None:
        self.engine = create_async_engine(
            db_url,
            echo=echo,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=15.0,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={
                "command_timeout": 60,
                "server_settings": {
                    "lock_timeout": "10000",
                    "idle_in_transaction_session_timeout": "30000",
                },
            },
        )
        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        timeout: float = 120.0,
        graph_name: str = "social_graph",
    ) -> None:
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", graph_name):
            raise ValueError(
                f"Invalid graph name: '{graph_name}'. "
                "Must be a valid SQL identifier matching ^[a-zA-Z_][a-zA-Z0-9_]*$"
            )

        async with self.engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT EXISTS ("
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = 'content'"
                    ") AND EXISTS ("
                    "SELECT 1 FROM ag_catalog.ag_graph WHERE name = :graph_name"
                    ")"
                ),
                {"graph_name": graph_name},
            )
            already_initialized = result.scalar()
            if already_initialized:
                logger.info("Database already initialized, skipping setup")
                return

        last_exception: Exception | None = None
        start_time = asyncio.get_event_loop().time()

        vertex_labels = ["Actor", "Entity", "Event", "Place", "Account", "Content"]

        for attempt in range(1, max_retries + 1):
            try:
                async with self.engine.connect() as lock_conn:
                    await lock_conn.execute(text("SELECT pg_advisory_lock(8675309)"))
                    try:
                        async with self.engine.begin() as conn:
                            await conn.run_sync(Base.metadata.create_all)
                            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS age;"))
                            await conn.execute(text("LOAD 'age';"))
                            await conn.execute(
                                text('SET search_path = ag_catalog, "$user", public;')
                            )
                            await conn.execute(
                                text(
                                    f"SELECT create_graph('{graph_name}') WHERE NOT EXISTS "
                                    f"(SELECT 1 FROM ag_graph WHERE name = '{graph_name}');"
                                )
                            )

                            for label in vertex_labels:
                                try:
                                    await conn.execute(text(f"""
                                        DO $$
                                        BEGIN
                                            IF NOT EXISTS (
                                                SELECT 1 FROM information_schema.tables
                                                WHERE table_schema = '{graph_name}'
                                                AND table_name = '{label}'
                                            ) THEN
                                                PERFORM create_vlabel('{graph_name}', '{label}');
                                            END IF;
                                        END
                                        $$;
                                    """))
                                except Exception as e:
                                    logger.warning("Failed to create vertex label %s: %s", label, e)

                        async with self.engine.connect() as conn:
                            for label in vertex_labels:
                                try:
                                    await conn.execute(text(f"""
                                        CREATE INDEX IF NOT EXISTS idx_{label.lower()}_id
                                        ON {graph_name}."{label}"
                                        USING btree (agtype_access_operator(properties, '"id"'));
                                    """))
                                except Exception as e:
                                    logger.debug("Skipped index creation for %s: %s", label, e)
                            await conn.commit()

                        # async with self.engine.connect() as conn:
                        #     conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
                        #     try:
                        #         await conn.execute(text("VACUUM ANALYZE;"))
                        #         logger.info("VACUUM ANALYZE completed")
                        #     except Exception as e:
                        #         logger.warning("VACUUM ANALYZE failed (non-critical): %s", e)

                        logger.info("Database initialization successful")
                        return
                    finally:
                        await lock_conn.execute(text("SELECT pg_advisory_unlock(8675309)"))

            except Exception as e:
                last_exception = e
                if attempt == max_retries:
                    logger.error(
                        "Database initialization failed after %d attempts: %s",
                        max_retries,
                        e,
                    )
                    raise RuntimeError(
                        f"Database initialization failed after {max_retries} attempts"
                    ) from e

                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    logger.error(
                        "Database initialization exceeded timeout of %.1f s after %d attempts",
                        timeout,
                        attempt,
                    )
                    raise RuntimeError(
                        f"Database initialization timed out after {elapsed:.1f} seconds"
                    ) from last_exception

                delay = min(base_delay * (2 ** (attempt - 1)), 60)
                jitter = random.uniform(0, delay * 0.1)
                total_delay = delay + jitter

                logger.warning(
                    "Database init attempt %d/%d failed: %s. Retrying in %.2f s...",
                    attempt,
                    max_retries,
                    e,
                    total_delay,
                )
                await asyncio.sleep(total_delay)

    async def reset_orphaned_processing_accounts(self) -> None:
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    update(Account)
                    .where(Account.status == "processing")
                    .where(Account.platform == "TELEGRAM")
                    .values(status="pending")
                )
                result = type_cast(CursorResult, await session.execute(stmt))

                if result.rowcount > 0:
                    logger.info(
                        "Reset %d orphaned processing accounts back to pending",
                        result.rowcount,
                    )
                else:
                    logger.debug("No orphaned processing accounts found")

    async def close(self) -> None:
        await self.engine.dispose()
        logger.info("Database connections closed")

    @with_retry_on_deadlock()
    async def count_pending_creator_accounts(self, platform: str) -> int:
        async with self.async_session() as session:
            stmt = (
                select(func.count(Account.id))
                .where(Account.platform == platform)
                .where(Account.status == "pending")
            )
            result = await session.execute(stmt)
            count = result.scalar()
            return count if count is not None else 0

    @with_retry_on_deadlock()
    async def claim_creator_accounts(
        self, platforms: list[str], batch_size: int, status_threshold_hours: int
    ) -> list[Account]:
        threshold_time = datetime.now(timezone.utc) - timedelta(hours=status_threshold_hours)

        async with self.async_session() as session:
            async with session.begin():
                subq = (
                    select(Account.id)
                    .where(Account.platform.in_(platforms))
                    .where(
                        (Account.status == "pending")
                        | (
                            (Account.status == "failed")
                            & (Account.updated_at < threshold_time)
                        )
                    )
                    .order_by(Account.updated_at.asc())
                    .limit(batch_size)
                    .with_for_update(skip_locked=True)
                    .cte("eligible_accounts")
                )

                stmt = (
                    update(Account)
                    .where(Account.id.in_(select(subq.c.id)))
                    .values(status="processing", updated_at=datetime.now(timezone.utc))
                    .returning(Account)
                )

                result = await session.execute(stmt)
                claimed_accounts = list(result.scalars().all())

                if claimed_accounts:
                    logger.info(
                        "Claimed %d creator accounts for processing",
                        len(claimed_accounts),
                    )
                else:
                    logger.debug("No creator accounts available to claim")

                return claimed_accounts

    async def _update_account_status(
        self,
        account_id: int,
        status: str,
        platform: str | list[str] | None = None,
    ) -> None:
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    update(Account)
                    .where(Account.id == account_id)
                    .values(status=status, updated_at=datetime.now(timezone.utc))
                )
                if platform is not None:
                    if isinstance(platform, list):
                        stmt = stmt.where(Account.platform.in_(platform))
                    else:
                        stmt = stmt.where(Account.platform == platform)

                result = type_cast(CursorResult, await session.execute(stmt))

                if result.rowcount > 0:
                    logger.debug(
                        "Updated account id=%d status to '%s'", account_id, status
                    )
                else:
                    logger.warning(
                        "Account id=%d not found when updating status to '%s'",
                        account_id,
                        status,
                    )

    @with_retry_on_deadlock()
    async def update_creator_account_status(
        self, account_id: int, status: str
    ) -> None:
        await self._update_account_status(account_id, status)

    @with_retry_on_deadlock()
    async def reset_orphaned_creator_accounts(
        self, platforms: list[str]
    ) -> int:
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    update(Account)
                    .where(Account.platform.in_(platforms))
                    .where(Account.status == "processing")
                    .values(status="pending", updated_at=datetime.now(timezone.utc))
                )
                result = type_cast(CursorResult, await session.execute(stmt))
                reset_count = result.rowcount

                if reset_count > 0:
                    logger.info(
                        "Reset %d orphaned creator accounts to pending",
                        reset_count,
                    )
                else:
                    logger.debug("No orphaned creator accounts found")

                return reset_count

    @with_retry_on_deadlock()
    async def upsert_account(self, account_data: dict[str, Any]) -> Account:
        stmt = insert(Account).values(**account_data)
        update_columns = {
            "username": stmt.excluded.username,
            "title": stmt.excluded.title,
            "description": stmt.excluded.description,
            "subscribers_count": stmt.excluded.subscribers_count,
            "access_hash": stmt.excluded.access_hash,
            "is_author_blog": stmt.excluded.is_author_blog,
            "updated_at": stmt.excluded.updated_at,
            "status": case(
                (
                    Account.status.in_(["parsed", "ready_for_parsing"]),
                    Account.status,
                ),
                else_=stmt.excluded.status,
            ),
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_=update_columns,
        ).returning(Account)

        async with self.async_session() as session:
            async with session.begin():
                result = await session.execute(stmt)
                account = result.scalar_one()

                logger.debug("Upserted account: %s", account)
                return account

    @with_retry_on_deadlock()
    async def upsert_content(self, content_data: dict[str, Any]) -> Content:
        stmt = insert(Content).values(**content_data)
        update_columns = {
            "views": stmt.excluded.views,
            "comments_count": stmt.excluded.comments_count,
            "shares_count": stmt.excluded.shares_count,
            "reactions_count": stmt.excluded.reactions_count,
            "fwd_from_channel_id": stmt.excluded.fwd_from_channel_id,
            "grouped_id": stmt.excluded.grouped_id,
            "has_media": stmt.excluded.has_media,
            "raw_metadata": stmt.excluded.raw_metadata,
            "updated_at": stmt.excluded.updated_at,
        }
        stmt = stmt.on_conflict_do_update(
            constraint="uq_content_account_platform_id",
            set_=update_columns,
        ).returning(Content)

        async with self.async_session() as session:
            async with session.begin():
                result = await session.execute(stmt)
                content = result.scalar_one()

                logger.debug("Upserted content: %s", content)
                return content

    @with_retry_on_deadlock()
    async def get_accounts_batch(
        self, account_ids: Sequence[int]
    ) -> dict[int, Account]:
        if not account_ids:
            return {}

        async with self.async_session() as session:
            stmt = select(Account).where(Account.id.in_(account_ids))
            result = await session.execute(stmt)
            accounts = result.scalars().all()
            return {acc.id: acc for acc in accounts}

    async def get_content_by_ids(
        self, content_ids: list[int]
    ) -> dict[int, Any]:
        if not content_ids:
            return {}

        async with self.async_session() as session:
            stmt = (
                select(Content)
                .options(joinedload(Content.account))
                .where(Content.id.in_(content_ids))
            )
            result = await session.execute(stmt)
            content_items = result.scalars().all()
            return {item.id: item for item in content_items}

    async def get_recent_content(self, limit: int = 100) -> list[Content]:
        async with self.async_session() as session:
            stmt = select(Content).order_by(Content.id.desc()).limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @with_retry_on_deadlock()
    async def get_random_pending_account(
        self, require_hash: bool = False
    ) -> Account | None:
        async with self.async_session() as session:
            async with session.begin():
                stmt = select(Account).where(
                    Account.status == "pending",
                    Account.platform == "TELEGRAM",
                )
                if require_hash:
                    stmt = stmt.where(Account.access_hash.is_not(None))

                stmt = (
                    stmt.order_by(func.random())
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                result = await session.execute(stmt)
                account = result.scalar_one_or_none()

                if account is not None:
                    account.status = "processing"
                    logger.debug(
                        "Claimed account id=%d username=%s for processing",
                        account.id,
                        account.username,
                    )
                else:
                    logger.debug("No pending accounts available")

                return account

    @with_retry_on_deadlock()
    async def mark_account_processed(self, account_id: int) -> None:
        await self._update_account_status(account_id, "ready_for_parsing", "TELEGRAM")

    @with_retry_on_deadlock()
    async def mark_account_rejected(self, account_id: int) -> None:
        await self._update_account_status(account_id, "rejected", "TELEGRAM")

    @with_retry_on_deadlock()
    async def mark_account_parsed(self, account_id: int) -> None:
        await self._update_account_status(account_id, "parsed", "TELEGRAM")

    @with_retry_on_deadlock()
    async def mark_account_pending(self, account_id: int) -> None:
        await self._update_account_status(account_id, "pending", "TELEGRAM")

    @with_retry_on_deadlock()
    async def get_account_for_parsing(
        self,
        session_index: int | None = None,
        total_sessions: int | None = None,
    ) -> Account | None:
        async with self.async_session() as session:
            async with session.begin():
                base_conditions = [
                    Account.status == "ready_for_parsing",
                    Account.platform == "TELEGRAM",
                    Account.is_author_blog == True,
                    Account.access_hash.is_not(None),
                ]

                use_shard = (
                    session_index is not None
                    and total_sessions is not None
                    and total_sessions > 0
                )

                if use_shard:
                    shard_conditions = base_conditions + [
                        func.mod(Account.id, total_sessions) == session_index
                    ]
                    shard_stmt = (
                        select(Account)
                        .where(*shard_conditions)
                        .order_by(func.random())
                        .limit(1)
                        .with_for_update(skip_locked=True)
                    )
                    result = await session.execute(shard_stmt)
                    account = result.scalar_one_or_none()

                    if account is not None:
                        account.status = "processing"
                        logger.debug(
                            "Claimed shard account id=%d username=%s (%d/%d)",
                            account.id,
                            account.username,
                            session_index,
                            total_sessions,
                        )
                        return account

                    logger.debug(
                        "No accounts in shard %d/%d, trying fallback",
                        session_index,
                        total_sessions,
                    )

                fallback_stmt = (
                    select(Account)
                    .where(*base_conditions)
                    .order_by(func.random())
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                result = await session.execute(fallback_stmt)
                account = result.scalar_one_or_none()

                if account is not None:
                    account.status = "processing"
                    logger.debug(
                        "Claimed account (fallback) id=%d username=%s",
                        account.id,
                        account.username,
                    )
                else:
                    logger.debug("No accounts available for parsing")

                return account

    @with_retry_on_deadlock()
    async def update_account_access_hash(
        self, account_id: int, access_hash: int
    ) -> None:
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    update(Account)
                    .where(Account.id == account_id)
                    .where(Account.platform == "TELEGRAM")
                    .values(access_hash=access_hash, updated_at=datetime.now(timezone.utc))
                )
                result = type_cast(CursorResult, await session.execute(stmt))

                if result.rowcount > 0:
                    logger.debug(
                        "Updated access_hash for account id=%d", account_id
                    )
                else:
                    logger.debug(
                        "Account id=%d not found when updating access_hash",
                        account_id,
                    )

    async def get_search_candidates(
        self,
        content_ids: list[int],
        location: str | None = None,
        min_followers: int | None = None,
        is_author_blog: bool | None = None,
    ) -> list[dict[str, Any]]:
        if not content_ids:
            return []

        async with self.async_session() as session:
            stmt = (
                select(
                    Content.id,
                    Content.account_id,
                    Content.content,
                    Content.transcription,
                    Content.created_at,
                    Content.message_id,
                    Content.platform_content_id,
                    Account.platform,
                    Account.username,
                    Account.title,
                    Account.description,
                    Account.subscribers_count,
                    Account.raw_metadata,
                    Account.static_avg_er,
                    Account.explanation,
                    Account.category_id,
                    Account.category_path,
                    Account.category_extension,
                    Account.is_author_blog,
                )
                .join(Account, Content.account_id == Account.id)
                .where(Content.id.in_(content_ids))
                .where(Account.status == "verified")
                .where(Content.is_enriched == True)
            )

            if location:
                location_escaped = re.escape(location)
                location_escaped = location_escaped.replace('\\', '\\\\').replace('"', '\\"')
                location_pattern = f".*{location_escaped}.*"

                stmt = stmt.where(
                    or_(
                        func.jsonb_path_exists(
                            Account.raw_metadata,
                            cast(f'$.geo_data.country ? (@ like_regex "{location_pattern}" flag "i")', JSONPATH),
                        ),
                        func.jsonb_path_exists(
                            Account.raw_metadata,
                            cast(f'$.geo_data.city ? (@ like_regex "{location_pattern}" flag "i")', JSONPATH),
                        ),
                        func.jsonb_path_exists(
                            Account.raw_metadata,
                            cast(f'$.location.country ? (@ like_regex "{location_pattern}" flag "i")', JSONPATH),
                        ),
                        func.jsonb_path_exists(
                            Account.raw_metadata,
                            cast(f'$.location.city ? (@ like_regex "{location_pattern}" flag "i")', JSONPATH),
                        ),
                    )
                )

            if min_followers is not None:
                stmt = stmt.where(Account.subscribers_count >= min_followers)

            if is_author_blog is not None:
                stmt = stmt.where(Account.is_author_blog == is_author_blog)

            result = await session.execute(stmt)

            return [
                {
                    "id": row.id,
                    "account_id": row.account_id,
                    "content": row.content,
                    "transcription": row.transcription,
                    "created_at": row.created_at,
                    "message_id": row.message_id,
                    "platform_content_id": row.platform_content_id,
                    "platform": row.platform,
                    "username": row.username,
                    "account_title": row.title,
                    "description": row.description,
                    "subscribers_count": row.subscribers_count,
                    "raw_metadata": row.raw_metadata,
                    "static_avg_er": row.static_avg_er,
                    "explanation": row.explanation,
                    "category_id": row.category_id,
                    "category_path": row.category_path,
                    "category_extension": row.category_extension,
                    "is_author_blog": row.is_author_blog,
                }
                for row in result
            ]

    async def get_latest_message_id(self, account_id: int) -> int | None:
        async with self.async_session() as session:
            stmt = (
                select(func.max(Content.message_id))
                .where(
                    Content.account_id == account_id,
                    Content.message_id.isnot(None),
                )
            )
            result = await session.execute(stmt)
            return result.scalar()
