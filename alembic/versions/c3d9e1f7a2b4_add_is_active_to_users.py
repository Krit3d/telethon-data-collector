from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'c3d9e1f7a2b4'
down_revision: str | Sequence[str] | None = '7e4b9d2f6a3c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'users',
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        schema='public',
    )


def downgrade() -> None:
    op.drop_column('users', 'is_active', schema='public')
