from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = '7e4b9d2f6a3c'
down_revision: str | None = 'f3a9c2e8b1d4'
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_index(
        "uq_accounts_platform_username_lower",
        "accounts",
        ["platform", sa.text("lower(username)")],
        unique=True,
        postgresql_where=sa.text("username IS NOT NULL AND username != ''"),
    )
    op.create_index(
        "uq_accounts_platform_id_lower",
        "accounts",
        ["platform", sa.text("lower(platform_id)")],
        unique=True,
        postgresql_where=sa.text("platform_id IS NOT NULL AND platform_id != ''"),
    )


def downgrade() -> None:
    op.drop_index("uq_accounts_platform_id_lower", table_name="accounts")
    op.drop_index("uq_accounts_platform_username_lower", table_name="accounts")
