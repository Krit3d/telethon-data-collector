from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'ad396ecf90d6'
down_revision: str | Sequence[str] | None = '80b4c5d6e7f8'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('accounts', sa.Column('country', sa.String(length=2), nullable=True))
    op.add_column('accounts', sa.Column('city', sa.String(length=100), nullable=True))
    op.add_column('accounts', sa.Column('gender', sa.String(length=20), nullable=True))
    op.create_index('ix_accounts_country', 'accounts', ['country'], unique=False)
    op.create_index('ix_accounts_city', 'accounts', ['city'], unique=False)
    op.create_index('ix_accounts_gender', 'accounts', ['gender'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_accounts_gender', table_name='accounts')
    op.drop_index('ix_accounts_city', table_name='accounts')
    op.drop_index('ix_accounts_country', table_name='accounts')
    op.drop_column('accounts', 'gender')
    op.drop_column('accounts', 'city')
    op.drop_column('accounts', 'country')