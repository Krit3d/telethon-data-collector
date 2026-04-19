"""
SQLAlchemy models for channels and posts tables.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    ForeignKey,
    String,
    Text,
    Integer,
    DateTime,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

if TYPE_CHECKING:
    from datetime import datetime as Datetime


class Base(DeclarativeBase):
    """Base class for declarative models."""

    pass


class Channel(Base):
    """Table storing Telegram channel information."""

    __tablename__ = "channels"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        comment="Telegram channel ID (can be negative for supergroups)",
    )
    username: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Channel username without @",
    )
    title: Mapped[str] = mapped_column(
        String(255), nullable=False, comment="Channel title"
    )
    description: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Channel description"
    )
    subscribers_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Number of subscribers (may be hidden)"
    )
    avatar_url: Mapped[str | None] = mapped_column(
        String(512), nullable=True, comment="URL to channel avatar"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Timestamp when the record was first inserted",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Timestamp of the last record update",
    )

    def __repr__(self) -> str:
        return f"<Channel(id={self.id}, username={self.username})>"


class Post(Base):
    """Table storing posts from Telegram channels."""

    __tablename__ = "posts"
    __table_args__ = (
        UniqueConstraint(
            "channel_id", "message_id", name="uq_post_channel_message"
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Surrogate primary key",
    )
    channel_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("channels.id", ondelete="CASCADE"),
        nullable=False,
        comment="Foreign key referencing the channel",
    )
    message_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="Telegram message ID (unique within a channel)",
    )

    content: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Text content of the post"
    )
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Publication date of the post in Telegram",
    )

    views: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Number of views"
    )
    comments_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Number of comments"
    )
    shares_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Number of shares/reposts"
    )
    reactions_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Number of reactions (likes)"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Timestamp when the post was first saved",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Timestamp of the last update (metrics/content)",
    )

    def __repr__(self) -> str:
        return f"<Post(channel_id={self.channel_id}, message_id={self.message_id})>"
