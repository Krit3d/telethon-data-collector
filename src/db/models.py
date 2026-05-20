"""
SQLAlchemy models for channels and posts tables.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    String,
    Text,
    Integer,
    DateTime,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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
    access_hash: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        comment="Telegram access_hash for direct entity resolving",
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        server_default="pending",
        nullable=False,
        index=True,  # index for faster search of pending channels
        comment="Lifecycle status: pending, processing, parsed, rejected",
    )
    is_author_blog: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        comment="True if channel has video notes or author keywords",
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

    # Relationship back to posts
    posts: Mapped[list["Post"]] = relationship(
        back_populates="channel", cascade="all, delete-orphan"
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
    is_extracted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="false",
        index=True,
        comment="True if knowledge graph and embeddings are extracted",
    )
    channel_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("channels.id", ondelete="CASCADE"),
        nullable=False,
        comment="Foreign key referencing the channel",
    )
    # Relationship to Channel for eager loading and access to channel data
    channel: Mapped["Channel"] = relationship(
        back_populates="posts",
        lazy="joined",  # Eager load by default when querying posts
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

    # OpenSPG Knowledge Graph metadata fields
    author: Mapped[str | None] = mapped_column(
        String(255), nullable=True, comment="Author of the post"
    )
    fwd_from_channel_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="Forwarded from channel ID"
    )
    grouped_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="Grouped ID for grouped messages"
    )
    has_media: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", comment="Whether the post contains media"
    )
    geo_lat: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Geolocation latitude"
    )
    geo_long: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Geolocation longitude"
    )
    language: Mapped[str | None] = mapped_column(
        String(10), nullable=True, comment="Detected language code"
    )

    # JSONB raw_metadata column for OpenSPG raw metadata extraction
    # Stores arbitrary nested attributes for different domains (it-sector, finance, blogging, etc.)
    raw_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Arbitrary raw metadata for OpenSPG domain processing",
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
