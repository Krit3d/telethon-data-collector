from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = '7f8a9b0c1d2e'
down_revision: str | Sequence[str] | None = '5c9e2a7f3b1d'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('creator_messages', sa.Column('channel_type', sa.String(length=30), nullable=True))
    op.add_column('creator_messages', sa.Column('channel_target', sa.String(length=255), nullable=True))
    op.add_column('creator_messages', sa.Column('external_message_id', sa.String(length=255), nullable=True))
    op.create_index('ix_creator_messages_account_channel', 'creator_messages', ['account_id', 'channel_type'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_creator_messages_account_channel', table_name='creator_messages', if_exists=True)
    op.drop_column('creator_messages', 'external_message_id')
    op.drop_column('creator_messages', 'channel_target')
    op.drop_column('creator_messages', 'channel_type')