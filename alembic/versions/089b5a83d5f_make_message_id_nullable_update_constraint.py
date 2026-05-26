"""Make message_id nullable and update unique constraint for non-numeric content IDs.

This migration supports Instagram and TikTok integration by:
1. Making message_id nullable to accommodate platforms without numeric IDs
2. Replacing the unique constraint from (account_id, message_id) to
   (account_id, platform_content_id) to use the string-based platform_content_id
   which works across all platforms (Telegram, TikTok, Instagram, YouTube).

Revision ID: 089b5a83d5f
Revises: 9661508d6e4a
Create Date: 2026-05-26 08:27:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# Revision identifiers, used by Alembic.
revision: str = "089b5a83d5f"
down_revision: Union[str, Sequence[str], None] = "9661508d6e4a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to support non-numeric content IDs from Instagram and TikTok."""

    # ==========================================================================
    # Step 1: Drop the old unique constraint on (account_id, message_id)
    # ==========================================================================
    op.drop_constraint(
        "uq_content_account_message", "content", type_="unique"
    )

    # ==========================================================================
    # Step 2: Make message_id column nullable
    # ==========================================================================
    op.alter_column("content", "message_id", nullable=True)

    # ==========================================================================
    # Step 3: Create new unique constraint on (account_id, platform_content_id)
    # ==========================================================================
    op.create_unique_constraint(
        "uq_content_account_platform_id",
        "content",
        ["account_id", "platform_content_id"],
    )


def downgrade() -> None:
    """Downgrade schema to revert to numeric message_id constraint."""

    # ==========================================================================
    # Step 1: Drop the new unique constraint on (account_id, platform_content_id)
    # ==========================================================================
    op.drop_constraint(
        "uq_content_account_platform_id", "content", type_="unique"
    )

    # ==========================================================================
    # Step 2: Make message_id column NOT NULL again
    # ==========================================================================
    # Note: This may fail if there are NULL values in the column.
    # In production, ensure all NULL message_id rows are handled before downgrading.
    op.alter_column("content", "message_id", nullable=False)

    # ==========================================================================
    # Step 3: Recreate the old unique constraint on (account_id, message_id)
    # ==========================================================================
    op.create_unique_constraint(
        "uq_content_account_message", "content", ["account_id", "message_id"]
    )
