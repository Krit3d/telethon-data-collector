"""
Asynchronous CRUD operations for channels and posts using SQLAlchemy 2.0.
"""

import logging
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

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
        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self) -> None:
        """Create all tables defined in the models (if they don't exist)."""

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database tables created (if not exist)")

    async def close(self) -> None:
        """Close all database connections."""

        await self.engine.dispose()

        logger.info("Database connections closed")

    async def upsert_channel(self, channel_data: dict[str, Any]) -> Channel:
        """
        Insert or update a channel record.

        Conflict is detected on the id field (Telegram channel_id).
        On conflict, all mutable fields except the primary key are updated.

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
            "updated_at": stmt.excluded.updated_at,
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
        On conflict, content, metrics, and publication date are updated.

        Args:
            post_data: Dictionary with fields matching the Post model.

        Returns:
            The persisted Post object.
        """

        stmt = insert(Post).values(**post_data)

        update_columns = {
            "content": stmt.excluded.content,
            "published_at": stmt.excluded.published_at,
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
