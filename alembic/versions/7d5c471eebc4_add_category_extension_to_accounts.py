from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '7d5c471eebc4'
down_revision: str | Sequence[str] | None = 'f1e2d3c4b5a6'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('accounts', sa.Column('category_extension', sa.String(length=100), nullable=True))


def downgrade() -> None:
    op.drop_column('accounts', 'category_extension')