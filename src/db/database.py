"""
Asynchronous CRUD operations for channels and posts using SQLAlchemy 2.0.
"""

import logging
from typing import Any, Sequence

from sqlalchemy import case, func, select, update, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import event

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

        self.engine = create_async_engine(
            db_url,
            echo=echo,
            pool_size=10,
            max_overflow=20,
        )

        # Configure AGE settings for every new connection
        @event.listens_for(self.engine.sync_engine, "connect")
        def set_age_search_path(dbapi_connection, connection_record):
            """Set search_path for Apache AGE on each new connection."""
            try:
                cursor = dbapi_connection.cursor()
                cursor.execute("LOAD 'age';")
                cursor.execute("SET search_path = ag_catalog, \"$user\", public;")
                cursor.close()
            except Exception as e:
                logger.warning("Failed to set AGE search_path: %s", e)

        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self) -> None:
        """Create all tables defined in the models (if they don't exist)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

            # Initialize Apache AGE extension and graph
            try:
                # Create extension if not exists
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS age;"))
                # Load AGE library
                await conn.execute(text("LOAD 'age';"))
                # Set search path for AGE
                await conn.execute(
                    text("SET search_path = ag_catalog, \"$user\", public;")
                )
                # Create the base graph (ignore if already exists)
                await conn.execute(
                    text("SELECT create_graph('telegram_graph') WHERE NOT EXISTS (SELECT 1 FROM ag_graph WHERE name = 'telegram_graph');")
                )
                logger.info("Apache AGE extension and graph initialized")
            except Exception as e:
                logger.error("Failed to initialize Apache AGE: %s", e)
                raise

        logger.info("Database tables created (if not exist)")

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
            "avatar_url": stmt.excluded.avatar_url,
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
        Fetch posts by a list of post IDs.

        Args:
            post_ids: List of PostgreSQL post IDs (primary keys).

        Returns:
            Dictionary mapping post_id to Post object for efficient lookup.
        """
        if not post_ids:
            return {}

        async with self.async_session() as session:
            stmt = select(Post).where(Post.id.in_(post_ids))
            result = await session.execute(stmt)
            posts = result.scalars().all()
            return {post.id: post for post in posts}

    async def get_recent_posts(self, limit: int = 100) -> list[Post]:
        """Fetch recent posts from the database for indexing."""
        async with self.async_session() as session:
            stmt = select(Post).order_by(Post.id.desc()).limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

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
                    .where(Channel.is_author_blog == True)
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

    async def execute_cypher(self, query: str) -> Any:
        """
        Execute a raw Cypher query against the Apache AGE graph.

        This is a placeholder method for future graph operations.
        The query will be executed in the context of the telegram_graph.

        Args:
            query: Cypher query string (e.g., "SELECT * FROM cypher('telegram_graph', $$ MATCH (n) RETURN n $$) AS (n agtype);")

        Returns:
            Raw query result (will be refined when graph queries are implemented).
        """
        async with self.async_session() as session:
            async with session.begin():
                result = await session.execute(text(query))
                return result.scalars().all()
