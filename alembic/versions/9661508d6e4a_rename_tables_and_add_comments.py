"""Rename tables channels->accounts and posts->content, add new columns, and create comments table.

This migration performs a data-safe schema evolution:
1. Renames tables while preserving all data
2. Renames all constraints and indexes to maintain schema cleanliness
3. Adds platform-related columns to accounts
4. Adds content-related columns and removes deprecated is_extracted
5. Creates the new comments table

Revision ID: 9661508d6e4a
Revises:
Create Date: 2026-05-25 13:40:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# Revision identifiers, used by Alembic.
revision: str = "9661508d6e4a"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema with table renames, column additions, and new comments table."""

    # ==========================================================================
    # Step 1: Rename table `channels` to `accounts`
    # ==========================================================================
    op.rename_table("channels", "accounts")

    # Rename primary key constraint for accounts
    op.execute("ALTER TABLE accounts RENAME CONSTRAINT channels_pkey TO accounts_pkey")

    # Rename indexes for accounts
    op.execute("ALTER INDEX ix_channels_username RENAME TO ix_accounts_username")
    op.execute("ALTER INDEX ix_channels_status RENAME TO ix_accounts_status")

    # ==========================================================================
    # Step 2: Add columns to `accounts` table
    # ==========================================================================
    op.add_column("accounts", sa.Column("platform", sa.String(50), nullable=True))
    op.add_column("accounts", sa.Column("platform_id", sa.String(255), nullable=True))

    # ==========================================================================
    # Step 3: Data Migration for accounts - populate new columns
    # ==========================================================================
    op.execute(
        "UPDATE accounts SET platform = 'TELEGRAM', platform_id = CAST(id AS TEXT)"
    )

    # Alter columns to NOT NULL after data population
    op.alter_column("accounts", "platform", nullable=False)
    op.alter_column("accounts", "platform_id", nullable=False)

    # Create indexes on new columns in accounts
    op.create_index("ix_accounts_platform", "accounts", ["platform"], unique=False)
    op.create_index("ix_accounts_platform_id", "accounts", ["platform_id"], unique=False)

    # ==========================================================================
    # Step 4: Rename table `posts` to `content`
    # ==========================================================================
    op.rename_table("posts", "content")

    # Rename primary key constraint for content
    op.execute("ALTER TABLE content RENAME CONSTRAINT posts_pkey TO content_pkey")

    # Rename indexes for content
    op.execute("ALTER INDEX ix_posts_is_extracted RENAME TO ix_content_is_extracted")
    op.execute("ALTER INDEX ix_posts_published_at RENAME TO ix_content_published_at")

    # ==========================================================================
    # Step 5: Rename column `channel_id` to `account_id` in `content` table
    # ==========================================================================
    op.alter_column("content", "channel_id", new_column_name="account_id")

    # ==========================================================================
    # Step 6: Update Foreign Key constraint for content
    # ==========================================================================
    # Drop the old foreign key constraint pointing to channels
    op.drop_constraint("posts_channel_id_fkey", "content", type_="foreignkey")

    # Create new foreign key constraint pointing to accounts
    op.create_foreign_key(
        "fk_content_account_id",
        "content",
        "accounts",
        ["account_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # ==========================================================================
    # Step 7: Rename unique constraint for content
    # ==========================================================================
    op.execute(
        "ALTER TABLE content RENAME CONSTRAINT uq_post_channel_message TO uq_content_account_message"
    )

    # ==========================================================================
    # Step 8: Add columns to `content` table
    # ==========================================================================
    op.add_column(
        "content",
        sa.Column("platform_content_id", sa.String(255), nullable=True),
    )
    op.add_column(
        "content",
        sa.Column(
            "is_embedded",
            sa.Boolean(),
            server_default="false",
            nullable=False,
        ),
    )
    op.add_column(
        "content",
        sa.Column(
            "is_graph_extracted",
            sa.Boolean(),
            server_default="false",
            nullable=False,
        ),
    )
    op.add_column(
        "content",
        sa.Column("transcription", sa.Text(), nullable=True),
    )

    # ==========================================================================
    # Step 9: Data Migration for content - populate new columns
    # ==========================================================================
    op.execute(
        "UPDATE content SET "
        "platform_content_id = CAST(message_id AS TEXT), "
        "is_embedded = is_extracted, "
        "is_graph_extracted = is_extracted"
    )

    # Alter platform_content_id to NOT NULL after data population
    op.alter_column("content", "platform_content_id", nullable=False)

    # ==========================================================================
    # Step 10: Drop the old column `is_extracted` from `content` table
    # ==========================================================================
    op.drop_column("content", "is_extracted")

    # ==========================================================================
    # Step 11: Create new `comments` table
    # ==========================================================================
    op.create_table(
        "comments",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "content_id",
            sa.BigInteger(),
            sa.ForeignKey("content.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "account_id",
            sa.BigInteger(),
            sa.ForeignKey("accounts.id", ondelete="SET NULL"),
            nullable=True,
            comment="Foreign key referencing the comment author account",
        ),
        sa.Column("platform_comment_id", sa.String(255), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("author_id", sa.BigInteger(), nullable=True),
        sa.Column("author_username", sa.String(255), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "content_id", "platform_comment_id", name="uq_comment_content_platform"
        ),
    )

    # Create indexes for comments table
    op.create_index("ix_comments_content_id", "comments", ["content_id"], unique=False)
    op.create_index(
        "ix_comments_platform_comment_id",
        "comments",
        ["platform_comment_id"],
        unique=False,
    )
    op.create_index("ix_comments_account_id", "comments", ["account_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema to revert all changes."""

    # ==========================================================================
    # Step 1: Drop `comments` table
    # ==========================================================================
    op.drop_table("comments")

    # ==========================================================================
    # Step 2: Reverse content table modifications
    # ==========================================================================
    # Add back old column `is_extracted`
    op.add_column(
        "content",
        sa.Column(
            "is_extracted",
            sa.Boolean(),
            default=False,
            server_default="false",
            nullable=False,
        ),
    )

    # Restore is_extracted values from is_embedded
    op.execute("UPDATE content SET is_extracted = is_embedded")

    # Drop new columns from content table
    op.drop_column("content", "platform_content_id")
    op.drop_column("content", "is_embedded")
    op.drop_column("content", "is_graph_extracted")
    op.drop_column("content", "transcription")

    # Rename unique constraint back to old name
    op.execute(
        "ALTER TABLE content RENAME CONSTRAINT uq_content_account_message TO uq_post_channel_message"
    )

    # Drop new foreign key constraint
    op.drop_constraint("fk_content_account_id", "content", type_="foreignkey")

    # Rename account_id back to channel_id
    op.alter_column("content", "account_id", new_column_name="channel_id")

    # Recreate old foreign key constraint pointing to accounts (will be channels after rename)
    op.create_foreign_key(
        "posts_channel_id_fkey",
        "content",
        "accounts",
        ["channel_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # Rename indexes back to old names
    op.execute("ALTER INDEX ix_content_published_at RENAME TO ix_posts_published_at")
    # Note: is_extracted column is added back, so we need to recreate its index
    op.create_index("ix_posts_is_extracted", "content", ["is_extracted"], unique=False)

    # Rename primary key constraint back
    op.execute("ALTER TABLE content RENAME CONSTRAINT content_pkey TO posts_pkey")

    # Rename table content back to posts
    op.rename_table("content", "posts")

    # ==========================================================================
    # Step 3: Reverse accounts table modifications
    # ==========================================================================
    # Drop indexes on platform and platform_id
    op.drop_index("ix_accounts_platform_id", table_name="accounts")
    op.drop_index("ix_accounts_platform", table_name="accounts")

    # Drop new columns from accounts table
    op.drop_column("accounts", "platform_id")
    op.drop_column("accounts", "platform")

    # Rename indexes back to old names
    op.execute("ALTER INDEX ix_accounts_status RENAME TO ix_channels_status")
    op.execute("ALTER INDEX ix_accounts_username RENAME TO ix_channels_username")

    # Rename primary key constraint back
    op.execute("ALTER TABLE accounts RENAME CONSTRAINT accounts_pkey TO channels_pkey")

    # Rename table accounts back to channels
    op.rename_table("accounts", "channels")
