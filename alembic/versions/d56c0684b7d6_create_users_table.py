from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'd56c0684b7d6'
down_revision: str | Sequence[str] | None = 'ad396ecf90d6'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        schema='public',
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True, schema='public')


def downgrade() -> None:
    op.drop_index('ix_users_email', table_name='users', schema='public', if_exists=True)
    op.execute('DROP TABLE IF EXISTS public.users CASCADE')
    op.execute('DROP TABLE IF EXISTS ag_catalog.users CASCADE')