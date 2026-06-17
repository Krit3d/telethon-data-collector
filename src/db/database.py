import asyncio
import functools
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Sequence, TypeVar, cast

from sqlalchemy import case, func, select, update, text
from sqlalchemy.engine import CursorResult
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.sql.expression import Select
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import joinedload

from src.db.models import Base, Account, Content

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_retry_on_deadlock(
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: OperationalError | DBAPIError | Exception | None = None

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

                    if not (is_deadlock or is_serialization):
                        raise

                    if attempt == max_retries:
                        logger.error(
                            "Failed after %d retries on deadlock/serialization failure: %s",
                            max_retries,
                            e,
                        )
                        raise

                    delay = base_delay * (2 ** attempt)
                    jitter = random.uniform(0, 0.1)
                    total_delay = delay + jitter

                    logger.warning(
                        "Attempt %d/%d failed with retryable error: %s. "
                        "Retrying in %.2f seconds...",
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
    def __init__(self, db_url: str, echo: bool = False) -> None:
        self.engine = create_async_engine(
            db_url,
            echo=echo,
            pool_size=20,
            max_overflow=10,
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
    ) -> None:
        last_exception: Exception | None = None
        start_time = asyncio.get_event_loop().time()

        for attempt in range(1, max_retries + 1):
            try:
                async with self.engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)

                    try:
                        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS age;"))
                        await conn.execute(text("LOAD 'age';"))
                        await conn.execute(
                            text('SET search_path = ag_catalog, "$user", public;')
                        )
                        await conn.execute(
                            text(
                                "SELECT create_graph('telegram_graph') WHERE NOT EXISTS "
                                "(SELECT 1 FROM ag_graph WHERE name = 'telegram_graph');"
                            )
                        )

                        vertex_labels = [
                            "Actor", "Entity", "Event", "Place", "Account", "Content",
                        ]
                        for label in vertex_labels:
                            try:
                                await conn.execute(text(f"""
                                    DO $$
                                    BEGIN
                                        IF NOT EXISTS (
                                            SELECT 1 FROM information_schema.tables
                                            WHERE table_schema = 'telegram_graph'
                                            AND table_name = '{label}'
                                        ) THEN
                                            PERFORM create_vlabel('telegram_graph', '{label}');
                                        END IF;
                                    END
                                    $$;
                                """))
                                logger.debug("Ensured vertex label exists: %s", label)
                            except Exception as e:
                                logger.warning("Failed to create vertex label %s: %s", label, e)

                        logger.info("Apache AGE extension and graph initialized")
                    except Exception as e:
                        logger.error("Failed to initialize Apache AGE: %s", e)
                        raise

                async with self.engine.connect() as conn:
                    labels = ["Actor", "Entity", "Event", "Place", "Account", "Content"]
                    for label in labels:
                        try:
                            await conn.execute(text(f"""
                                CREATE INDEX IF NOT EXISTS idx_{label.lower()}_id
                                ON telegram_graph."{label}"
                                USING btree (agtype_access_operator(properties, '"id"'));
                            """))
                        except Exception as e:
                            logger.debug("Skipped index creation for %s: %s", label, e)

                    await conn.commit()

                async with self.engine.connect() as conn:
                    conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
                    try:
                        await conn.execute(text("VACUUM ANALYZE;"))
                        logger.info("VACUUM ANALYZE completed successfully")
                    except Exception as e:
                        logger.warning("VACUUM ANALYZE failed (non-critical): %s", e)

                logger.info("Database initialization successful")
                return

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
                        "Database initialization exceeded timeout of %.1f seconds after %d attempts",
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
                    "Database initialization attempt %d/%d failed: %s. Retrying in %.2f seconds...",
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
                result = cast(CursorResult, await session.execute(stmt))
                count = result.rowcount

                if count > 0:
                    logger.info(
                        "Reset orphaned processing accounts back to pending: %d accounts",
                        count,
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

    @with_retry_on_deadlock()
    async def update_creator_account_status(
        self, account_id: int, status: str
    ) -> None:
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    update(Account)
                    .where(Account.id == account_id)
                    .values(status=status, updated_at=datetime.now(timezone.utc))
                )
                result = cast(CursorResult, await session.execute(stmt))

                if result.rowcount > 0:
                    logger.debug(
                        "Updated account id=%s status to '%s'",
                        account_id,
                        status,
                    )
                else:
                    logger.warning(
                        "Account id=%s not found when updating status",
                        account_id,
                    )

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
                result = cast(CursorResult, await session.execute(stmt))
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
        )

        async with self.async_session() as session:
            async with session.begin():
                await session.execute(stmt)
                account = await session.get(Account, account_data["id"])

                if account is None:
                    account = Account(**account_data)
                    session.add(account)
                    await session.flush()

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
        )

        async with self.async_session() as session:
            async with session.begin():
                await session.execute(stmt)

                content = await self._get_content_by_unique(
                    session,
                    content_data["account_id"],
                    content_data["platform_content_id"],
                )

                if content is None:
                    content = Content(**content_data)
                    session.add(content)
                    await session.flush()

                logger.debug("Upserted content: %s", content)
                return content

    async def _get_content_by_unique(
        self, session: AsyncSession, account_id: int, platform_content_id: str
    ) -> Content | None:
        stmt = select(Content).where(
            Content.account_id == account_id,
            Content.platform_content_id == platform_content_id,
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

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
    ) -> dict[int, Content]:
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

    async def get_unembedded_content(
        self, limit: int, priority_mode: bool
    ) -> list[Content]:
        async with self.async_session() as session:
            stmt = (
                select(Content)
                .options(joinedload(Content.account))
                .where(
                    Content.is_embedded == False,
                    Content.content.isnot(None),
                    Content.content != "",
                )
            )

            if priority_mode:
                stmt = stmt.order_by(Content.published_at.desc())
            else:
                stmt = stmt.order_by(Content.published_at.asc())

            stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @with_retry_on_deadlock()
    async def mark_content_embedded(self, content_ids: list[int]) -> None:
        if not content_ids:
            return

        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    update(Content)
                    .where(Content.id.in_(content_ids))
                    .values(
                        is_embedded=True,
                        updated_at=datetime.now(timezone.utc),
                    )
                )
                result = cast(CursorResult, await session.execute(stmt))

                if result.rowcount > 0:
                    logger.debug(
                        "Marked %d content items as embedded",
                        result.rowcount,
                    )
                else:
                    logger.warning(
                        "No content items found to mark as embedded for IDs: %s",
                        content_ids[:10],
                    )

    async def _build_ungraphed_query(
        self,
        require_content: bool,
        priority_mode: bool,
    ) -> Select[tuple[Content]]:
        conditions = [Content.is_graph_extracted == False]

        if require_content:
            conditions.append(Content.content.isnot(None))
            conditions.append(Content.content != "")

        stmt = select(Content).where(*conditions)

        if priority_mode:
            stmt = stmt.order_by(Content.published_at.desc())
        elif require_content:
            stmt = stmt.order_by(Content.published_at.asc())
        else:
            stmt = stmt.order_by(Content.id.asc())

        return stmt

    async def get_unextracted_content(
        self, limit: int = 50, priority_mode: bool = False
    ) -> list[Content]:
        async with self.async_session() as session:
            stmt = await self._build_ungraphed_query(
                require_content=False,
                priority_mode=priority_mode,
            )
            stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_ungraphed_content(
        self, limit: int, priority_mode: bool
    ) -> list[Content]:
        async with self.async_session() as session:
            stmt = await self._build_ungraphed_query(
                require_content=True,
                priority_mode=priority_mode,
            )
            stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @with_retry_on_deadlock()
    async def _mark_content_graph_extracted(
        self, content_id: int, update_updated_at: bool
    ) -> None:
        values: dict[str, Any] = {"is_graph_extracted": True}
        if update_updated_at:
            values["updated_at"] = datetime.now(timezone.utc)

        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    update(Content)
                    .where(Content.id == content_id)
                    .values(**values)
                )
                result = cast(CursorResult, await session.execute(stmt))

                if result.rowcount > 0:
                    logger.debug(
                        "Marked content id=%s as graph extracted", content_id
                    )
                else:
                    logger.warning(
                        "Content id=%s not found when marking as graph extracted",
                        content_id,
                    )

    @with_retry_on_deadlock()
    async def mark_content_extracted(self, content_id: int) -> None:
        await self._mark_content_graph_extracted(content_id, update_updated_at=False)

    @with_retry_on_deadlock()
    async def mark_content_graphed(self, content_id: int) -> None:
        await self._mark_content_graph_extracted(content_id, update_updated_at=True)

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
                        "Claimed account id=%s username=%s for processing",
                        account.id,
                        account.username,
                    )
                else:
                    logger.debug("No pending accounts available")

                return account

    async def _set_telegram_account_status(
        self, account_id: int, status: str
    ) -> None:
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    update(Account)
                    .where(Account.id == account_id)
                    .where(Account.platform == "TELEGRAM")
                    .values(status=status, updated_at=datetime.now(timezone.utc))
                )
                result = cast(CursorResult, await session.execute(stmt))

                if result.rowcount > 0:
                    logger.debug(
                        "Marked account id=%s as %s", account_id, status
                    )
                else:
                    logger.debug(
                        "Account id=%s not found when setting status to %s",
                        account_id,
                        status,
                    )

    @with_retry_on_deadlock()
    async def mark_account_processed(self, account_id: int) -> None:
        await self._set_telegram_account_status(account_id, "ready_for_parsing")

    @with_retry_on_deadlock()
    async def mark_account_rejected(self, account_id: int) -> None:
        await self._set_telegram_account_status(account_id, "rejected")

    @with_retry_on_deadlock()
    async def mark_account_parsed(self, account_id: int) -> None:
        await self._set_telegram_account_status(account_id, "parsed")

    @with_retry_on_deadlock()
    async def mark_account_pending(self, account_id: int) -> None:
        await self._set_telegram_account_status(account_id, "pending")

    @with_retry_on_deadlock()
    async def get_account_for_parsing(
        self,
        session_index: int | None = None,
        total_sessions: int | None = None,
    ) -> Account | None:
        async with self.async_session() as session:
            async with session.begin():
                account = None

                if (
                    session_index is not None
                    and total_sessions is not None
                    and total_sessions > 0
                ):
                    shard_stmt = (
                        select(Account)
                        .where(Account.status == "ready_for_parsing")
                        .where(Account.platform == "TELEGRAM")
                        .where(Account.is_author_blog == True)
                        .where(Account.access_hash.is_not(None))
                        .where(
                            func.mod(Account.id, total_sessions)
                            == session_index
                        )
                        .order_by(func.random())
                        .limit(1)
                        .with_for_update(skip_locked=True)
                    )
                    result = await session.execute(shard_stmt)
                    account = result.scalar_one_or_none()

                    if account:
                        account.status = "processing"
                        logger.debug(
                            "Parser claimed shard account id=%s username=%s (session_index=%s/%s)",
                            account.id,
                            account.username,
                            session_index,
                            total_sessions,
                        )
                        return account

                    logger.debug(
                        "No accounts available in shard %s/%s, trying fallback",
                        session_index,
                        total_sessions,
                    )

                fallback_stmt = (
                    select(Account)
                    .where(Account.status == "ready_for_parsing")
                    .where(Account.platform == "TELEGRAM")
                    .where(Account.is_author_blog == True)
                    .where(Account.access_hash.is_not(None))
                    .order_by(func.random())
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                result = await session.execute(fallback_stmt)
                account = result.scalar_one_or_none()

                if account:
                    account.status = "processing"
                    logger.debug(
                        "Parser claimed account (fallback) id=%s username=%s (has access_hash)",
                        account.id,
                        account.username,
                    )
                else:
                    logger.debug("No pending accounts available for parsing")

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
                result = cast(CursorResult, await session.execute(stmt))

                if result.rowcount > 0:
                    logger.debug(
                        "Updated access_hash for account id=%s", account_id
                    )
                else:
                    logger.debug(
                        "Account id=%s not found when updating access_hash",
                        account_id,
                    )

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
