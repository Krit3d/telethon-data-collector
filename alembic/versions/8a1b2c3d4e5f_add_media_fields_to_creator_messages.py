from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = '8a1b2c3d4e5f'
down_revision: str | Sequence[str] | None = '7f8a9b0c1d2e'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column('creator_messages', 'text', existing_type=sa.Text(), nullable=True)
    op.add_column('creator_messages', sa.Column('media_url', sa.String(length=1024), nullable=True))
    op.add_column('creator_messages', sa.Column('media_name', sa.String(length=255), nullable=True))
    op.add_column('creator_messages', sa.Column('media_type', sa.String(length=50), nullable=True))


def downgrade() -> None:
    op.drop_column('creator_messages', 'media_type')
    op.drop_column('creator_messages', 'media_name')
    op.drop_column('creator_messages', 'media_url')
    op.alter_column('creator_messages', 'text', existing_type=sa.Text(), nullable=False)