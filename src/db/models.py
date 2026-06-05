"""
SQLAlchemy models for accounts, content, and comments tables.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
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


def get_platform_id_default(context):
    """Dynamic default for platform_id: extracts 'id' from insert parameters and casts to string."""
    params = context.get_current_parameters()
    return str(params.get('id')) if params.get('id') is not None else None


class Account(Base):
    """Table storing platform account information (e.g., Telegram channels)."""

    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        comment="Platform account ID (e.g., Telegram channel ID, can be negative for supergroups)",
    )
    platform: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        default='TELEGRAM',
        comment="Platform name (e.g., 'TELEGRAM', 'YOUTUBE')",
    )
    platform_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        default=get_platform_id_default,
        comment="Account ID on the platform (as string)",
    )
    username: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Account username without @",
    )
    title: Mapped[str] = mapped_column(
        String(255), nullable=False, comment="Account title/name"
    )
    description: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Account description"
    )
    subscribers_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Number of subscribers (may be hidden)"
    )
    access_hash: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        comment="Platform-specific access hash for direct entity resolving",
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        server_default="pending",
        nullable=False,
        index=True,
        comment="Lifecycle status: pending, processing, parsed, rejected",
    )
    is_author_blog: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        comment="True if account has video notes or author keywords",
    )

    raw_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Arbitrary raw metadata of the author account for OpenSPG domain processing",
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

    # Relationship back to content
    content: Mapped[list["Content"]] = relationship(
        back_populates="account", cascade="all, delete-orphan"
    )
    comments: Mapped[list["Comment"]] = relationship(
        back_populates="account"
    )

    def __repr__(self) -> str:
        return f"<Account(id={self.id}, platform={self.platform}, platform_id={self.platform_id})>"


class Content(Base):
    """Table storing content from platform accounts (e.g., Telegram posts)."""

    __tablename__ = "content"
    __table_args__ = (
        UniqueConstraint(
            "account_id", "platform_content_id", name="uq_content_account_platform_id"
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Surrogate primary key",
    )
    account_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("accounts.id", ondelete="CASCADE"),
        nullable=False,
        comment="Foreign key referencing the account",
    )
    # Relationship to Account for eager loading and access to account data
    account: Mapped["Account"] = relationship(
        back_populates="content",
        lazy="joined",
    )
    message_id: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        comment="Platform message ID (unique within an account, nullable for non-Telegram platforms)",
    )
    platform_content_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Content ID on the platform (as string)",
    )

    content: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Text content of the post"
    )
    transcription: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Transcribed text from media"
    )
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Publication date of the content on the platform",
    )

    is_embedded: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="false",
        comment="True if vector embeddings are generated",
    )
    is_graph_extracted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="false",
        comment="True if knowledge graph is extracted",
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
    fwd_from_channel_id: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        index=True,
        comment="Forwarded from channel ID",
    )
    grouped_id: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        index=True,
        comment="Grouped ID for grouped messages",
    )
    has_media: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="false",
        index=True,
        comment="Whether the content contains media",
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
        comment="Timestamp when the content was first saved",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Timestamp of the last update (metrics/content)",
    )

    # Relationship to comments
    comments: Mapped[list["Comment"]] = relationship(
        back_populates="content", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Content(account_id={self.account_id}, message_id={self.message_id})>"


class Comment(Base):
    """Table storing comments on content."""

    __tablename__ = "comments"
    __table_args__ = (
        UniqueConstraint(
            "content_id", "platform_comment_id", name="uq_comment_content_platform"
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="Surrogate primary key",
    )
    content_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("content.id", ondelete="CASCADE"),
        nullable=False,
        comment="Foreign key referencing the content",
    )
    # Relationship to Content
    content: Mapped["Content"] = relationship(
        back_populates="comments",
        lazy="joined",
    )
    account_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("accounts.id", ondelete="SET NULL"),
        nullable=True,
        comment="Foreign key referencing the comment author account",
    )
    # Relationship to Account (comment author)
    account: Mapped["Account | None"] = relationship(
        back_populates="comments",
    )
    platform_comment_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Comment ID on the platform (as string)",
    )
    text: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Text content of the comment"
    )
    author_id: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        comment="Platform ID of the comment author",
    )
    author_username: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Username of the comment author",
    )
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Publication date of the comment on the platform",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Timestamp when the comment was first saved",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="Timestamp of the last update",
    )

    def __repr__(self) -> str:
        return f"<Comment(content_id={self.content_id}, platform_comment_id={self.platform_comment_id})>"


# Backward compatibility alias for legacy Channel class references
Channel = Account
