"""
Asynchronous CRUD operations for accounts and content using SQLAlchemy 2.0.
"""

import asyncio
import logging
import random
from typing import Any, Sequence

from sqlalchemy import case, func, select, update, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import joinedload

from src.db.models import Base, Account, Content

logger = logging.getLogger(__name__)


class Database:
    """Asynchronous PostgreSQL connection manager."""

    def __init__(self, db_url: str, echo: bool = False) -> None:
        """
        Initialize the database manager.

        Args:
            db_url: Connection string in postgresql+asyncpg:// format.
            echo: Enable SQL query logging (for debugging).
        """

        # command_timeout is reduced to 60s for more responsive query termination.
        # pool_timeout prevents infinite blocking when the connection pool is exhausted.
        # server_settings configure PostgreSQL-level timeouts to avoid silent hangs
        # on locks or idle transactions over high-latency VPN tunnels (Tailscale).
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
        """
        Create all tables defined in the models (if they don't exist).

        Implements retry with exponential backoff and jitter for resilience
        against temporary network or database unavailability during startup.
        Also enforces an overall timeout to prevent indefinite blocking.

        Args:
            max_retries: Maximum number of retry attempts (default: 5).
            base_delay: Base delay in seconds for exponential backoff (default: 1.0).
            timeout: Overall timeout in seconds for the entire initialization process (default: 120.0).

        Raises:
            RuntimeError: If initialization fails after all retries or times out.
        """

        last_exception: Exception | None = None
        start_time = asyncio.get_event_loop().time()

        for attempt in range(1, max_retries + 1):
            try:
                # ===================================================================
                # STAGE 1: Core Setup (Transactional)
                # ===================================================================
                # Use engine.begin() for automatic transaction management
                async with self.engine.begin() as conn:
                    # Create SQLAlchemy-managed tables
                    await conn.run_sync(Base.metadata.create_all)

                    # Initialize Apache AGE extension and graph
                    try:
                        # Create extension if not exists
                        await conn.execute(
                            text("CREATE EXTENSION IF NOT EXISTS age;")
                        )
                        # Load AGE library
                        await conn.execute(text("LOAD 'age';"))
                        # Set search path for AGE
                        await conn.execute(
                            text(
                                'SET search_path = ag_catalog, "$user", public;'
                            )
                        )
                        # Create the base graph (ignore if already exists)
                        await conn.execute(
                            text(
                                "SELECT create_graph('telegram_graph') WHERE NOT EXISTS (SELECT 1 FROM ag_graph WHERE name = 'telegram_graph');"
                            )
                        )

                        # Initialize vertex labels for the graph
                        vertex_labels = [
                            "Actor",
                            "Entity",
                            "Event",
                            "Place",
                            "Account",
                            "Content",
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
                                logger.debug(
                                    "Ensured vertex label exists: %s", label
                                )
                            except Exception as e:
                                logger.warning(
                                    "Failed to create vertex label %s: %s",
                                    label,
                                    e,
                                )

                        logger.info(
                            "Apache AGE extension and graph initialized"
                        )
                    except Exception as e:
                        logger.error("Failed to initialize Apache AGE: %s", e)
                        raise
                # Transaction is automatically committed when the context exits

                # ===================================================================
                # STAGE 2: Graph Indexes (Non-transactional)
                # ===================================================================
                # Use connect() without begin() for manual transaction control
                async with self.engine.connect() as conn:
                    labels = [
                        "Actor",
                        "Entity",
                        "Event",
                        "Place",
                        "Account",
                        "Content",
                    ]
                    for label in labels:
                        try:
                            # B-Tree index using agtype_access_operator for @> operator (optimal for id lookups)
                            await conn.execute(text(f"""
                                CREATE INDEX IF NOT EXISTS idx_{label.lower()}_id
                                ON telegram_graph."{label}"
                                USING btree (agtype_access_operator(properties, '"id"'));
                            """))
                        except Exception as e:
                            logger.debug(
                                "Skipped index creation for %s: %s", label, e
                            )

                    # Explicitly commit the index creations (connection is not in autocommit mode)
                    await conn.commit()

                # ===================================================================
                # STAGE 3: Maintenance (Autocommit)
                # ===================================================================
                # Create a separate connection with AUTOCOMMIT isolation level
                # VACUUM cannot run inside a transaction block
                async with self.engine.connect() as conn:
                    # Set autocommit BEFORE executing any statements
                    conn = await conn.execution_options(
                        isolation_level="AUTOCOMMIT"
                    )
                    try:
                        await conn.execute(text("VACUUM ANALYZE;"))
                        logger.info("VACUUM ANALYZE completed successfully")
                    except Exception as e:
                        # VACUUM failure should not crash the entire startup
                        logger.warning(
                            "VACUUM ANALYZE failed (non-critical): %s", e
                        )

                logger.info("Database initialization successful")
                return  # Success, exit the retry loop

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

                # Check if we've exceeded the overall timeout
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

                # Exponential backoff with jitter to avoid thundering herd
                delay = min(
                    base_delay * (2 ** (attempt - 1)), 60
                )  # Cap at 60 seconds
                jitter = random.uniform(0, delay * 0.1)  # 10% jitter
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
        """Reset all accounts with status='processing' back to 'pending'.

        This recovers from crashes/restarts where accounts were left in processing state.
        """
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    update(Account)
                    .where(Account.status == "processing")
                    .values(status="pending")
                )
                result = await session.execute(stmt)
                count = result.rowcount  # type: ignore[attr-defined]

                if count > 0:
                    logger.info(
                        "Reset orphaned processing accounts back to pending: %d accounts",
                        count,
                    )
                else:
                    logger.debug("No orphaned processing accounts found")

    async def close(self) -> None:
        """Close all database connections."""
        await self.engine.dispose()
        logger.info("Database connections closed")

    async def upsert_account(self, account_data: dict[str, Any]) -> Account:
        """
        Insert or update an account record.

        Conflict is detected on the id field (Telegram channel_id).
        On conflict, all mutable fields except the primary key are updated.
        Status is preserved if it's already 'parsed' or 'ready_for_parsing'.

        Args:
            account_data: Dictionary with fields matching the Account model.

        Returns:
            The persisted Account object.
        """

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
                # Retrieve the current record (may have existed already)
                account = await session.get(Account, account_data["id"])

                if account is None:
                    # Fallback manual creation (should not happen under normal circumstances)
                    account = Account(**account_data)
                    session.add(account)
                    await session.flush()

                logger.debug("Upserted account: %s", account)

                return account

    async def upsert_content(self, content_data: dict[str, Any]) -> Content:
        """
        Insert or update a content record.

        Conflict is detected on the composite unique key (account_id, message_id).
        On conflict, only metrics (views, comments, shares, reactions) are updated.
        Content and published_at are preserved to avoid overwriting existing data.

        Args:
            content_data: Dictionary with fields matching the Content model.

        Returns:
            The persisted Content object.
        """

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
            constraint="uq_content_account_message",
            set_=update_columns,
        )

        async with self.async_session() as session:
            async with session.begin():
                await session.execute(stmt)

                # Retrieve the saved object
                content = await self._get_content_by_unique(
                    session, content_data["account_id"], content_data["message_id"]
                )

                if content is None:
                    # Fallback manual creation if upsert failed unexpectedly
                    content = Content(**content_data)
                    session.add(content)
                    await session.flush()

                logger.debug("Upserted content: %s", content)
                return content

    async def _get_content_by_unique(
        self, session: AsyncSession, account_id: int, message_id: int
    ) -> Content | None:
        """Helper method to fetch a content by its composite natural key."""
        stmt = select(Content).where(
            Content.account_id == account_id, Content.message_id == message_id
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_accounts_batch(
        self, account_ids: Sequence[int]
    ) -> dict[int, Account]:
        """
        Return a dictionary of existing accounts by a list of IDs.

        Useful for checking which accounts are already in the DB before parsing.

        Args:
            account_ids: List of Telegram account IDs.

        Returns:
            Dictionary mapping account_id to Account object.
        """

        if not account_ids:
            return {}

        async with self.async_session() as session:
            stmt = select(Account).where(Account.id.in_(account_ids))
            result = await session.execute(stmt)
            accounts = result.scalars().all()
            return {acc.id: acc for acc in accounts}

    async def get_content_by_ids(self, content_ids: list[int]) -> dict[int, Content]:
        """
        Fetch content by a list of content IDs with their associated accounts eagerly loaded.

        Args:
            content_ids: List of PostgreSQL content IDs (primary keys).

        Returns:
            Dictionary mapping content_id to Content object (with .account populated) for efficient lookup.
        """
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
        """Fetch recent content from the database for indexing."""
        async with self.async_session() as session:
            stmt = select(Content).order_by(Content.id.desc()).limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_unextracted_content(
        self, limit: int = 50, priority_mode: bool = False
    ) -> list[Content]:
        """
        Fetch content that has not yet been extracted to knowledge graph.

        Args:
            limit: Maximum number of content to return.
            priority_mode: If True, order by published_at DESC (most recent first).
                          If False, order by id ASC (oldest first).

        Returns:
            List of Content objects where is_graph_extracted is False.
        """

        async with self.async_session() as session:
            stmt = select(Content).where(Content.is_graph_extracted == False)  # noqa: E712

            if priority_mode:
                # Priority mode: process most recent content first (for search relevance)
                stmt = stmt.order_by(Content.published_at.desc())
            else:
                # Default: process oldest content first (FIFO)
                stmt = stmt.order_by(Content.id.asc())

            stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def mark_content_extracted(self, content_id: int) -> None:
        """Mark a content as extracted (is_graph_extracted = True).

        Uses a direct atomic UPDATE statement to avoid FOR UPDATE issues
        with outer joins caused by lazy="joined" relationships.

        Args:
            content_id: The database ID of the content to mark as extracted.
        """

        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    update(Content)
                    .where(Content.id == content_id)
                    .values(is_graph_extracted=True)
                )
                result = await session.execute(stmt)

                if result.rowcount > 0:  # type: ignore[attr-defined]
                    logger.debug("Marked content id=%s as extracted", content_id)
                else:
                    logger.warning(
                        "Content id=%s not found when marking as extracted",
                        content_id,
                    )

    async def get_random_pending_account(
        self, require_hash: bool = False
    ) -> Account | None:
        """
        Fetch a random account with status='pending' and mark it as 'processing'.

        Uses FOR UPDATE SKIP LOCKED to avoid race conditions between workers.
        After marking, the transaction is committed so the account is immediately
        visible to other workers as 'processing'.

        Args:
            require_hash: If True, only return accounts with non-null access_hash.
                This allows "weak" accounts to fetch only accounts that can be
                accessed directly without global search.

        Returns:
            The selected Account entity, or None if no pending accounts exist.
        """

        async with self.async_session() as session:
            # Start transaction with row-level lock
            async with session.begin():
                # Build query with optional hash requirement
                stmt = select(Account).where(Account.status == "pending")
                if require_hash:
                    stmt = stmt.where(Account.access_hash.is_not(None))

                # Get random pending account and lock it
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

    async def mark_account_processed(self, account_id: int) -> None:
        """Mark an account as successfully processed (status='parsed')."""
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Account)
                    .where(Account.id == account_id)
                    .with_for_update()
                )
                result = await session.execute(stmt)
                account = result.scalar_one_or_none()

                if account is not None:
                    account.status = "ready_for_parsing"
                    logger.debug("Marked account id=%s as processed", account_id)

    async def mark_account_rejected(self, account_id: int) -> None:
        """Mark an account as rejected (status='rejected')."""
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Account)
                    .where(Account.id == account_id)
                    .with_for_update()
                )
                result = await session.execute(stmt)
                account = result.scalar_one_or_none()

                if account is not None:
                    account.status = "rejected"
                    logger.debug("Marked account id=%s as rejected", account_id)

    async def get_account_for_parsing(
        self, session_index: int | None = None, total_sessions: int | None = None
    ) -> Account | None:
        """Fetch an account ready for CONTENT PARSING and mark as processing.

        Implements session-account sharding to reuse Telethon's local SQLite cache
        and avoid FloodWait on username resolution. Each session claims accounts
        from its shard first, then falls back to any available account.

        Args:
            session_index: Optional zero-based index of the current session (0 to total_sessions-1).
            total_sessions: Optional total number of sessions for sharding.

        Returns:
            The claimed Account entity, or None if no accounts are available.
        """
        async with self.async_session() as session:
            async with session.begin():
                account = None

                # Stage 1: Try to claim an account from this session's shard
                if session_index is not None and total_sessions is not None and total_sessions > 0:
                    shard_stmt = (
                        select(Account)
                        .where(Account.status == "ready_for_parsing")
                        .where(Account.is_author_blog == True)  # noqa: E712
                        .where(Account.access_hash.is_not(None))
                        # Shard condition: Account.id % total_sessions == session_index
                        .where(func.mod(Account.id, total_sessions) == session_index)
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

                # Stage 2: Fallback - claim ANY available account
                # This prevents idle workers when some sessions are in cooldown
                fallback_stmt = (
                    select(Account)
                    .where(Account.status == "ready_for_parsing")
                    .where(Account.is_author_blog == True)  # noqa: E712
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

    async def mark_account_parsed(self, account_id: int) -> None:
        """Mark an account as completely parsed (content are saved)."""
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Account)
                    .where(Account.id == account_id)
                    .with_for_update()
                )
                result = await session.execute(stmt)
                account = result.scalar_one_or_none()

                if account is not None:
                    account.status = "parsed"
                    logger.debug(
                        "Marked account id=%s as COMPLETELY PARSED", account_id
                    )

    async def mark_account_pending(self, account_id: int) -> None:
        """Return an account to pending status (e.g., if a worker failed due to a shadowban)."""
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Account)
                    .where(Account.id == account_id)
                    .with_for_update()
                )
                result = await session.execute(stmt)
                account = result.scalar_one_or_none()

                if account is not None:
                    account.status = "pending"
                    logger.debug(
                        "Returned account id=%s to pending state", account_id
                    )

    async def update_account_access_hash(
        self, account_id: int, access_hash: int
    ) -> None:
        """Update the access_hash for an account.

        This is used to store the session-local correct access_hash after
        successful resolution by username.

        Args:
            account_id: Telegram account ID.
            access_hash: The resolved access_hash for this session.
        """
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Account)
                    .where(Account.id == account_id)
                    .with_for_update()
                )
                result = await session.execute(stmt)
                account = result.scalar_one_or_none()

                if account is not None:
                    account.access_hash = access_hash
                    logger.debug(
                        "Updated access_hash for account id=%s", account_id
                    )

    async def get_latest_message_id(self, account_id: int) -> int | None:
        """Fetch the latest (highest) message ID for a given account.

        This is used to implement smart skip logic in the parser,
        allowing us to avoid re-fetching messages we already have.

        Args:
            account_id: Telegram account ID.

        Returns:
            The highest message_id for the account, or None if no content exist.
        """
        async with self.async_session() as session:
            stmt = select(func.max(Content.message_id)).where(
                Content.account_id == account_id
            )
            result = await session.execute(stmt)
            return result.scalar()  # Returns None if no rows exist
