from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = '5c9e2a7f3b1d'
down_revision: str | Sequence[str] | None = '65e1cd3d7a8d'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("TRUNCATE TABLE deal_messages")

    op.rename_table('deal_messages', 'creator_messages')

    op.drop_index('ix_deal_messages_deal_id', table_name='creator_messages')

    op.execute("ALTER TABLE creator_messages DROP CONSTRAINT IF EXISTS deal_messages_deal_id_fkey")

    op.alter_column('creator_messages', 'deal_id', existing_type=sa.BigInteger(), nullable=True)

    op.add_column('creator_messages', sa.Column('user_id', sa.Integer(), nullable=False))
    op.add_column('creator_messages', sa.Column('account_id', sa.BigInteger(), nullable=False))

    op.create_foreign_key(
        'fk_creator_messages_deal_id_deals',
        'creator_messages',
        'deals',
        ['deal_id'],
        ['id'],
        ondelete='SET NULL',
    )
    op.create_foreign_key(
        'fk_creator_messages_user_id_users',
        'creator_messages',
        'users',
        ['user_id'],
        ['id'],
        ondelete='CASCADE',
    )
    op.create_foreign_key(
        'fk_creator_messages_account_id_accounts',
        'creator_messages',
        'accounts',
        ['account_id'],
        ['id'],
        ondelete='CASCADE',
    )
    op.create_index(
        'ix_creator_messages_user_account_created',
        'creator_messages',
        ['user_id', 'account_id', 'created_at'],
        unique=False,
    )
    op.create_index('ix_creator_messages_account_id', 'creator_messages', ['account_id'], unique=False)
    op.create_index('ix_creator_messages_deal_id', 'creator_messages', ['deal_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_creator_messages_deal_id', table_name='creator_messages', if_exists=True)
    op.drop_index('ix_creator_messages_account_id', table_name='creator_messages', if_exists=True)
    op.drop_index('ix_creator_messages_user_account_created', table_name='creator_messages', if_exists=True)

    op.drop_constraint('fk_creator_messages_account_id_accounts', 'creator_messages', type_='foreignkey')
    op.drop_constraint('fk_creator_messages_user_id_users', 'creator_messages', type_='foreignkey')
    op.drop_constraint('fk_creator_messages_deal_id_deals', 'creator_messages', type_='foreignkey')

    op.execute("DELETE FROM creator_messages WHERE deal_id IS NULL")

    op.drop_column('creator_messages', 'account_id')
    op.drop_column('creator_messages', 'user_id')

    op.alter_column('creator_messages', 'deal_id', existing_type=sa.BigInteger(), nullable=False)

    op.create_foreign_key(
        'deal_messages_deal_id_fkey',
        'creator_messages',
        'deals',
        ['deal_id'],
        ['id'],
        ondelete='CASCADE',
    )
    op.create_index('ix_deal_messages_deal_id', 'creator_messages', ['deal_id'], unique=False)

    op.rename_table('creator_messages', 'deal_messages')