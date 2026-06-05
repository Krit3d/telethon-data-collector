"""add_raw_metadata_to_accounts

Revision ID: c40da6c6c069
Revises: 089b5a83d5f
Create Date: 2026-06-05 08:29:56.932959

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'c40da6c6c069'
down_revision: str | Sequence[str] | None = '089b5a83d5f'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add raw_metadata column to accounts table for OpenSPG domain processing
    op.add_column('accounts', sa.Column('raw_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True, comment='Arbitrary raw metadata of the author account for OpenSPG domain processing'))
    
    # Update column comments for accounts table
    op.alter_column('accounts', 'id',
               existing_type=sa.BIGINT(),
               comment='Platform account ID (e.g., Telegram channel ID, can be negative for supergroups)',
               existing_comment='Telegram channel ID (can be negative for supergroups)',
               existing_nullable=False)
    op.alter_column('accounts', 'platform',
               existing_type=sa.VARCHAR(length=50),
               comment="Platform name (e.g., 'TELEGRAM', 'YOUTUBE')",
               existing_nullable=False)
    op.alter_column('accounts', 'platform_id',
               existing_type=sa.VARCHAR(length=255),
               comment='Account ID on the platform (as string)',
               existing_nullable=False)
    op.alter_column('accounts', 'username',
               existing_type=sa.VARCHAR(length=255),
               comment='Account username without @',
               existing_comment='Channel username without @',
               existing_nullable=True)
    op.alter_column('accounts', 'title',
               existing_type=sa.VARCHAR(length=255),
               comment='Account title/name',
               existing_comment='Channel title',
               existing_nullable=False)
    op.alter_column('accounts', 'description',
               existing_type=sa.TEXT(),
               comment='Account description',
               existing_comment='Channel description',
               existing_nullable=True)
    op.alter_column('accounts', 'access_hash',
               existing_type=sa.BIGINT(),
               comment='Platform-specific access hash for direct entity resolving',
               existing_nullable=True)
    op.alter_column('accounts', 'is_author_blog',
               existing_type=sa.BOOLEAN(),
               comment='True if account has video notes or author keywords',
               existing_comment='True if channel has video notes or author keywords',
               existing_nullable=True)
    
    # Update column comments for comments table
    op.alter_column('comments', 'id',
               existing_type=sa.BIGINT(),
               comment='Surrogate primary key',
               existing_nullable=False)
    op.alter_column('comments', 'content_id',
               existing_type=sa.BIGINT(),
               comment='Foreign key referencing the content',
               existing_nullable=False)
    op.alter_column('comments', 'platform_comment_id',
               existing_type=sa.VARCHAR(length=255),
               comment='Comment ID on the platform (as string)',
               existing_nullable=False)
    op.alter_column('comments', 'text',
               existing_type=sa.TEXT(),
               comment='Text content of the comment',
               existing_nullable=True)
    op.alter_column('comments', 'author_id',
               existing_type=sa.BIGINT(),
               comment='Platform ID of the comment author',
               existing_nullable=True)
    op.alter_column('comments', 'author_username',
               existing_type=sa.VARCHAR(length=255),
               comment='Username of the comment author',
               existing_nullable=True)
    op.alter_column('comments', 'published_at',
               existing_type=postgresql.TIMESTAMP(timezone=True),
               comment='Publication date of the comment on the platform',
               existing_nullable=False)
    op.alter_column('comments', 'created_at',
               existing_type=postgresql.TIMESTAMP(timezone=True),
               comment='Timestamp when the comment was first saved',
               existing_nullable=False)
    op.alter_column('comments', 'updated_at',
               existing_type=postgresql.TIMESTAMP(timezone=True),
               comment='Timestamp of the last update',
               existing_nullable=False)
    
    # Update column comments for content table
    op.alter_column('content', 'account_id',
               existing_type=sa.BIGINT(),
               comment='Foreign key referencing the account',
               existing_comment='Foreign key referencing the channel',
               existing_nullable=False)
    op.alter_column('content', 'message_id',
               existing_type=sa.BIGINT(),
               comment='Platform message ID (unique within an account, nullable for non-Telegram platforms)',
               existing_comment='Telegram message ID (unique within a channel)',
               existing_nullable=True)
    op.alter_column('content', 'platform_content_id',
               existing_type=sa.VARCHAR(length=255),
               comment='Content ID on the platform (as string)',
               existing_nullable=False)
    op.alter_column('content', 'transcription',
               existing_type=sa.TEXT(),
               comment='Transcribed text from media',
               existing_nullable=True)
    op.alter_column('content', 'published_at',
               existing_type=postgresql.TIMESTAMP(timezone=True),
               comment='Publication date of the content on the platform',
               existing_comment='Publication date of the post in Telegram',
               existing_nullable=False)
    op.alter_column('content', 'is_embedded',
               existing_type=sa.BOOLEAN(),
               comment='True if vector embeddings are generated',
               existing_nullable=False,
               existing_server_default=sa.text('false'))
    op.alter_column('content', 'is_graph_extracted',
               existing_type=sa.BOOLEAN(),
               comment='True if knowledge graph is extracted',
               existing_nullable=False,
               existing_server_default=sa.text('false'))
    op.alter_column('content', 'fwd_from_channel_id',
               existing_type=sa.BIGINT(),
               comment='Forwarded from channel ID',
               existing_nullable=True)
    op.alter_column('content', 'grouped_id',
               existing_type=sa.BIGINT(),
               comment='Grouped ID for grouped messages',
               existing_nullable=True)
    
    # Make has_media column production-safe: update NULL values before setting NOT NULL
    op.execute("UPDATE content SET has_media = FALSE WHERE has_media IS NULL")
    op.alter_column('content', 'has_media',
               existing_type=sa.BOOLEAN(),
               nullable=False,
               comment='Whether the content contains media',
               existing_server_default=sa.text('false'))
    
    op.alter_column('content', 'raw_metadata',
               existing_type=postgresql.JSONB(astext_type=sa.Text()),
               comment='Arbitrary raw metadata for OpenSPG domain processing',
               existing_nullable=True)
    op.alter_column('content', 'created_at',
               existing_type=postgresql.TIMESTAMP(timezone=True),
               comment='Timestamp when the content was first saved',
               existing_comment='Timestamp when the post was first saved',
               existing_nullable=False)
    
    # Create new indexes for content table with safety checks (IF NOT EXISTS)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_content_fwd_from_channel_id 
        ON content (fwd_from_channel_id)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_content_grouped_id 
        ON content (grouped_id)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_content_has_media 
        ON content (has_media)
    """)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop new indexes from content table with safety checks
    op.execute("DROP INDEX IF EXISTS ix_content_has_media")
    op.execute("DROP INDEX IF EXISTS ix_content_grouped_id")
    op.execute("DROP INDEX IF EXISTS ix_content_fwd_from_channel_id")
    
    # Reverse column comments for content table
    op.alter_column('content', 'created_at',
               existing_type=postgresql.TIMESTAMP(timezone=True),
               comment='Timestamp when the post was first saved',
               existing_comment='Timestamp when the content was first saved',
               existing_nullable=False)
    op.alter_column('content', 'raw_metadata',
               existing_type=postgresql.JSONB(astext_type=sa.Text()),
               comment=None,
               existing_comment='Arbitrary raw metadata for OpenSPG domain processing',
               existing_nullable=True)
    
    # Reverse has_media column change safely
    op.alter_column('content', 'has_media',
               existing_type=sa.BOOLEAN(),
               nullable=True,
               comment=None,
               existing_comment='Whether the content contains media',
               existing_server_default=sa.text('false'))
    
    op.alter_column('content', 'grouped_id',
               existing_type=sa.BIGINT(),
               comment=None,
               existing_comment='Grouped ID for grouped messages',
               existing_nullable=True)
    op.alter_column('content', 'fwd_from_channel_id',
               existing_type=sa.BIGINT(),
               comment=None,
               existing_comment='Forwarded from channel ID',
               existing_nullable=True)
    op.alter_column('content', 'is_graph_extracted',
               existing_type=sa.BOOLEAN(),
               comment=None,
               existing_comment='True if knowledge graph is extracted',
               existing_nullable=False,
               existing_server_default=sa.text('false'))
    op.alter_column('content', 'is_embedded',
               existing_type=sa.BOOLEAN(),
               comment=None,
               existing_comment='True if vector embeddings are generated',
               existing_nullable=False,
               existing_server_default=sa.text('false'))
    op.alter_column('content', 'published_at',
               existing_type=postgresql.TIMESTAMP(timezone=True),
               comment='Publication date of the post in Telegram',
               existing_comment='Publication date of the content on the platform',
               existing_nullable=False)
    op.alter_column('content', 'transcription',
               existing_type=sa.TEXT(),
               comment=None,
               existing_comment='Transcribed text from media',
               existing_nullable=True)
    op.alter_column('content', 'platform_content_id',
               existing_type=sa.VARCHAR(length=255),
               comment=None,
               existing_comment='Content ID on the platform (as string)',
               existing_nullable=False)
    op.alter_column('content', 'message_id',
               existing_type=sa.BIGINT(),
               comment='Telegram message ID (unique within a channel)',
               existing_comment='Platform message ID (unique within an account, nullable for non-Telegram platforms)',
               existing_nullable=True)
    op.alter_column('content', 'account_id',
               existing_type=sa.BIGINT(),
               comment='Foreign key referencing the channel',
               existing_comment='Foreign key referencing the account',
               existing_nullable=False)
    
    # Reverse column comments for comments table
    op.alter_column('comments', 'updated_at',
               existing_type=postgresql.TIMESTAMP(timezone=True),
               comment=None,
               existing_comment='Timestamp of the last update',
               existing_nullable=False)
    op.alter_column('comments', 'created_at',
               existing_type=postgresql.TIMESTAMP(timezone=True),
               comment=None,
               existing_comment='Timestamp when the comment was first saved',
               existing_nullable=False)
    op.alter_column('comments', 'published_at',
               existing_type=postgresql.TIMESTAMP(timezone=True),
               comment=None,
               existing_comment='Publication date of the comment on the platform',
               existing_nullable=False)
    op.alter_column('comments', 'author_username',
               existing_type=sa.VARCHAR(length=255),
               comment=None,
               existing_comment='Username of the comment author',
               existing_nullable=True)
    op.alter_column('comments', 'author_id',
               existing_type=sa.BIGINT(),
               comment=None,
               existing_comment='Platform ID of the comment author',
               existing_nullable=True)
    op.alter_column('comments', 'text',
               existing_type=sa.TEXT(),
               comment=None,
               existing_comment='Text content of the comment',
               existing_nullable=True)
    op.alter_column('comments', 'platform_comment_id',
               existing_type=sa.VARCHAR(length=255),
               comment=None,
               existing_comment='Comment ID on the platform (as string)',
               existing_nullable=False)
    op.alter_column('comments', 'content_id',
               existing_type=sa.BIGINT(),
               comment=None,
               existing_comment='Foreign key referencing the content',
               existing_nullable=False)
    op.alter_column('comments', 'id',
               existing_type=sa.BIGINT(),
               comment=None,
               existing_comment='Surrogate primary key',
               existing_nullable=False)
    
    # Reverse column comments for accounts table
    op.alter_column('accounts', 'is_author_blog',
               existing_type=sa.BOOLEAN(),
               comment='True if channel has video notes or author keywords',
               existing_comment='True if account has video notes or author keywords',
               existing_nullable=True)
    op.alter_column('accounts', 'access_hash',
               existing_type=sa.BIGINT(),
               comment=None,
               existing_comment='Platform-specific access hash for direct entity resolving',
               existing_nullable=True)
    op.alter_column('accounts', 'description',
               existing_type=sa.TEXT(),
               comment='Channel description',
               existing_comment='Account description',
               existing_nullable=True)
    op.alter_column('accounts', 'title',
               existing_type=sa.VARCHAR(length=255),
               comment='Channel title',
               existing_comment='Account title/name',
               existing_nullable=False)
    op.alter_column('accounts', 'username',
               existing_type=sa.VARCHAR(length=255),
               comment='Channel username without @',
               existing_comment='Account username without @',
               existing_nullable=True)
    op.alter_column('accounts', 'platform_id',
               existing_type=sa.VARCHAR(length=255),
               comment=None,
               existing_comment='Account ID on the platform (as string)',
               existing_nullable=False)
    op.alter_column('accounts', 'platform',
               existing_type=sa.VARCHAR(length=50),
               comment=None,
               existing_comment="Platform name (e.g., 'TELEGRAM', 'YOUTUBE')",
               existing_nullable=False)
    op.alter_column('accounts', 'id',
               existing_type=sa.BIGINT(),
               comment='Telegram channel ID (can be negative for supergroups)',
               existing_comment='Platform account ID (e.g., Telegram channel ID, can be negative for supergroups)',
               existing_nullable=False)
    
    # Drop raw_metadata column from accounts table
    op.drop_column('accounts', 'raw_metadata')
