"""add_gin_indexes_raw_metadata

Revision ID: a1b2c3d4e5f6
Revises: c40da6c6c069
Create Date: 2026-06-09 05:04:00.000000

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: str | Sequence[str] | None = 'c40da6c6c069'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: create GIN indexes on raw_metadata JSONB columns."""
    # Create GIN index on content.raw_metadata using jsonb_path_ops
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_content_raw_metadata_gin "
        "ON content USING gin (raw_metadata jsonb_path_ops)"
    )

    # Create GIN index on accounts.raw_metadata using jsonb_path_ops
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_accounts_raw_metadata_gin "
        "ON accounts USING gin (raw_metadata jsonb_path_ops)"
    )


def downgrade() -> None:
    """Downgrade schema: drop GIN indexes."""
    op.execute("DROP INDEX IF EXISTS idx_content_raw_metadata_gin")
    op.execute("DROP INDEX IF EXISTS idx_accounts_raw_metadata_gin")
