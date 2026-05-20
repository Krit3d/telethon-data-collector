"""
Asynchronous CRUD operations for channels and posts using SQLAlchemy 2.0.
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


from src.db.models import Base, Channel, Post

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

        # command_timeout is set to 120s to accommodate graph queries over
        # high-latency VPN tunnels (Tailscale, 300ms+).  Individual graph
        # operations are additionally guarded by Python-level asyncio.wait_for
        # timeouts in GraphRepository (default 15s per label query).
        self.engine = create_async_engine(
            db_url,
            echo=echo,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={
                "command_timeout": 120,
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
                            "Channel",
                            "Post",
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
                                    "Failed to create vertex label %s: %s", label, e
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
                        "Channel",
                        "Post",
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

    async def reset_orphaned_processing_channels(self) -> None:
        """Reset all channels with status='processing' back to 'pending'.

        This recovers from crashes/restarts where channels were left in processing state.
        """
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    update(Channel)
                    .where(Channel.status == "processing")
                    .values(status="pending")
                )
                result = await session.execute(stmt)
                count = result.rowcount  # type: ignore[attr-defined]

                if count > 0:
                    logger.info(
                        "Reset orphaned processing channels back to pending: %d channels",
                        count,
                    )
                else:
                    logger.debug("No orphaned processing channels found")

    async def close(self) -> None:
        """Close all database connections."""
        await self.engine.dispose()
        logger.info("Database connections closed")

    async def upsert_channel(self, channel_data: dict[str, Any]) -> Channel:
        """
        Insert or update a channel record.

        Conflict is detected on the id field (Telegram channel_id).
        On conflict, all mutable fields except the primary key are updated.
        Status is preserved if it's already 'parsed' or 'ready_for_parsing'.

        Args:
            channel_data: Dictionary with fields matching the Channel model.

        Returns:
            The persisted Channel object.
        """

        stmt = insert(Channel).values(**channel_data)
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
                    Channel.status.in_(["parsed", "ready_for_parsing"]),
                    Channel.status,
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
                channel = await session.get(Channel, channel_data["id"])

                if channel is None:
                    # Fallback manual creation (should not happen under normal circumstances)
                    channel = Channel(**channel_data)
                    session.add(channel)
                    await session.flush()

                logger.debug("Upserted channel: %s", channel)

                return channel

    async def upsert_post(self, post_data: dict[str, Any]) -> Post:
        """
        Insert or update a post record.

        Conflict is detected on the composite unique key (channel_id, message_id).
        On conflict, only metrics (views, comments, shares, reactions) are updated.
        Content and published_at are preserved to avoid overwriting existing data.

        Args:
            post_data: Dictionary with fields matching the Post model.

        Returns:
            The persisted Post object.
        """

        stmt = insert(Post).values(**post_data)

        update_columns = {
            "views": stmt.excluded.views,
            "comments_count": stmt.excluded.comments_count,
            "shares_count": stmt.excluded.shares_count,
            "reactions_count": stmt.excluded.reactions_count,
            "author": stmt.excluded.author,
            "has_media": stmt.excluded.has_media,
            "geo_lat": stmt.excluded.geo_lat,
            "geo_long": stmt.excluded.geo_long,
            "language": stmt.excluded.language,
            "raw_metadata": stmt.excluded.raw_metadata,
            "updated_at": stmt.excluded.updated_at,
        }

        stmt = stmt.on_conflict_do_update(
            constraint="uq_post_channel_message",
            set_=update_columns,
        )

        async with self.async_session() as session:
            async with session.begin():
                await session.execute(stmt)

                # Retrieve the saved object
                post = await self._get_post_by_unique(
                    session, post_data["channel_id"], post_data["message_id"]
                )

                if post is None:
                    # Fallback manual creation if upsert failed unexpectedly
                    post = Post(**post_data)
                    session.add(post)
                    await session.flush()

                logger.debug("Upserted post: %s", post)
                return post

    async def _get_post_by_unique(
        self, session: AsyncSession, channel_id: int, message_id: int
    ) -> Post | None:
        """Helper method to fetch a post by its composite natural key."""
        stmt = select(Post).where(
            Post.channel_id == channel_id, Post.message_id == message_id
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_channels_batch(
        self, channel_ids: Sequence[int]
    ) -> dict[int, Channel]:
        """
        Return a dictionary of existing channels by a list of IDs.

        Useful for checking which channels are already in the DB before parsing.

        Args:
            channel_ids: List of Telegram channel IDs.

        Returns:
            Dictionary mapping channel_id to Channel object.
        """

        if not channel_ids:
            return {}

        async with self.async_session() as session:
            stmt = select(Channel).where(Channel.id.in_(channel_ids))
            result = await session.execute(stmt)
            channels = result.scalars().all()
            return {ch.id: ch for ch in channels}

    async def get_posts_by_ids(self, post_ids: list[int]) -> dict[int, Post]:
        """
        Fetch posts by a list of post IDs with their associated channels eagerly loaded.

        Args:
            post_ids: List of PostgreSQL post IDs (primary keys).

        Returns:
            Dictionary mapping post_id to Post object (with .channel populated) for efficient lookup.
        """
        if not post_ids:
            return {}

        async with self.async_session() as session:
            stmt = (
                select(Post)
                .options(joinedload(Post.channel))
                .where(Post.id.in_(post_ids))
            )
            result = await session.execute(stmt)
            posts = result.scalars().all()
            return {post.id: post for post in posts}

    async def get_recent_posts(self, limit: int = 100) -> list[Post]:
        """Fetch recent posts from the database for indexing."""
        async with self.async_session() as session:
            stmt = select(Post).order_by(Post.id.desc()).limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_unextracted_posts(
        self, limit: int = 50, priority_mode: bool = False
    ) -> list[Post]:
        """
        Fetch posts that have not yet been extracted to knowledge graph.

        Args:
            limit: Maximum number of posts to return.
            priority_mode: If True, order by published_at DESC (most recent first).
                          If False, order by id ASC (oldest first).

        Returns:
            List of Post objects where is_extracted is False.
        """

        async with self.async_session() as session:
            stmt = select(Post).where(Post.is_extracted == False)  # noqa: E712

            if priority_mode:
                # Priority mode: process most recent posts first (for search relevance)
                stmt = stmt.order_by(Post.published_at.desc())
            else:
                # Default: process oldest posts first (FIFO)
                stmt = stmt.order_by(Post.id.asc())

            stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def mark_post_extracted(self, post_id: int) -> None:
        """Mark a post as extracted (is_extracted = True).

        Uses a direct atomic UPDATE statement to avoid FOR UPDATE issues
        with outer joins caused by lazy="joined" relationships.

        Args:
            post_id: The database ID of the post to mark as extracted.
        """

        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    update(Post)
                    .where(Post.id == post_id)
                    .values(is_extracted=True)
                )
                result = await session.execute(stmt)

                if result.rowcount > 0:  # type: ignore[attr-defined]
                    logger.debug("Marked post id=%s as extracted", post_id)
                else:
                    logger.warning(
                        "Post id=%s not found when marking as extracted",
                        post_id,
                    )

    async def get_random_pending_channel(
        self, require_hash: bool = False
    ) -> Channel | None:
        """
        Fetch a random channel with status='pending' and mark it as 'processing'.

        Uses FOR UPDATE SKIP LOCKED to avoid race conditions between workers.
        After marking, the transaction is committed so the channel is immediately
        visible to other workers as 'processing'.

        Args:
            require_hash: If True, only return channels with non-null access_hash.
                This allows "weak" accounts to fetch only channels that can be
                accessed directly without global search.

        Returns:
            The selected Channel entity, or None if no pending channels exist.
        """

        async with self.async_session() as session:
            # Start transaction with row-level lock
            async with session.begin():
                # Build query with optional hash requirement
                stmt = select(Channel).where(Channel.status == "pending")
                if require_hash:
                    stmt = stmt.where(Channel.access_hash.is_not(None))

                # Get random pending channel and lock it
                stmt = (
                    stmt.order_by(func.random())
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                result = await session.execute(stmt)
                channel = result.scalar_one_or_none()

                if channel is not None:
                    channel.status = "processing"
                    logger.debug(
                        "Claimed channel id=%s username=%s for processing",
                        channel.id,
                        channel.username,
                    )
                else:
                    logger.debug("No pending channels available")

                return channel

    async def mark_channel_processed(self, channel_id: int) -> None:
        """Mark a channel as successfully processed (status='parsed')."""
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Channel)
                    .where(Channel.id == channel_id)
                    .with_for_update()
                )
                result = await session.execute(stmt)
                channel = result.scalar_one_or_none()

                if channel is not None:
                    channel.status = "ready_for_parsing"
                    logger.debug("Marked channel id=%s as parsed", channel_id)

    async def mark_channel_rejected(self, channel_id: int) -> None:
        """Mark a channel as rejected (status='rejected')."""
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Channel)
                    .where(Channel.id == channel_id)
                    .with_for_update()
                )
                result = await session.execute(stmt)
                channel = result.scalar_one_or_none()

                if channel is not None:
                    channel.status = "rejected"
                    logger.debug("Marked channel id=%s as rejected", channel_id)

    async def get_channel_for_parsing(self) -> Channel | None:
        """Fetch a channel ready for POST PARSING and mark as processing.

        Only returns channels that have an access_hash to avoid global search
        rate limits and potential account bans.
        """
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Channel)
                    .where(Channel.status == "ready_for_parsing")
                    .where(Channel.is_author_blog == True)  # noqa: E712
                    .where(Channel.access_hash.is_not(None))
                    .order_by(func.random())
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                result = await session.execute(stmt)
                channel = result.scalar_one_or_none()

                if channel:
                    channel.status = "processing"
                    logger.debug(
                        "Parser claimed channel id=%s username=%s (has access_hash)",
                        channel.id,
                        channel.username,
                    )

                return channel

    async def mark_channel_parsed(self, channel_id: int) -> None:
        """Mark a channel as completely parsed (posts are saved)."""
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Channel)
                    .where(Channel.id == channel_id)
                    .with_for_update()
                )
                result = await session.execute(stmt)
                channel = result.scalar_one_or_none()

                if channel is not None:
                    channel.status = "parsed"
                    logger.debug(
                        "Marked channel id=%s as COMPLETELY PARSED", channel_id
                    )

    async def mark_channel_pending(self, channel_id: int) -> None:
        """Return a channel to pending status (e.g., if a worker failed due to a shadowban)."""
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Channel)
                    .where(Channel.id == channel_id)
                    .with_for_update()
                )
                result = await session.execute(stmt)
                channel = result.scalar_one_or_none()

                if channel is not None:
                    channel.status = "pending"
                    logger.debug(
                        "Returned channel id=%s to pending state", channel_id
                    )

    async def update_channel_access_hash(
        self, channel_id: int, access_hash: int
    ) -> None:
        """Update the access_hash for a channel.

        This is used to store the session-local correct access_hash after
        successful resolution by username.

        Args:
            channel_id: Telegram channel ID.
            access_hash: The resolved access_hash for this session.
        """
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Channel)
                    .where(Channel.id == channel_id)
                    .with_for_update()
                )
                result = await session.execute(stmt)
                channel = result.scalar_one_or_none()

                if channel is not None:
                    channel.access_hash = access_hash
                    logger.debug(
                        "Updated access_hash for channel id=%s", channel_id
                    )

    async def get_latest_message_id(self, channel_id: int) -> int | None:
        """Fetch the latest (highest) message ID for a given channel.

        This is used to implement smart skip logic in the parser,
        allowing us to avoid re-fetching messages we already have.

        Args:
            channel_id: Telegram channel ID.

        Returns:
            The highest message_id for the channel, or None if no posts exist.
        """
        async with self.async_session() as session:
            stmt = (
                select(func.max(Post.message_id))
                .where(Post.channel_id == channel_id)
            )
            result = await session.execute(stmt)
            return result.scalar()  # Returns None if no rows exist
