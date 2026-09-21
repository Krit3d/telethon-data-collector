from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = 'f3a9c2e8b1d4'
down_revision: str | Sequence[str] | None = '9b2c3d4e5f6a'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'user_shortlists',
        sa.Column('source', sa.String(length=20), server_default='search', nullable=False),
    )


def downgrade() -> None:
    op.drop_column('user_shortlists', 'source')