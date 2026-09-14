from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = '65e1cd3d7a8d'
down_revision: str | Sequence[str] | None = 'd56c0684b7d6'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'deals',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('account_id', sa.BigInteger(), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('stage', sa.SmallInteger(), server_default=sa.text('1'), nullable=False),
        sa.Column('budget', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('type', sa.String(length=50), server_default=sa.text("'Stories'"), nullable=False),
        sa.Column('brand_name', sa.String(length=255), nullable=True),
        sa.Column('pub_date', sa.String(length=50), nullable=True),
        sa.Column('terms', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['account_id'], ['accounts.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_deals_user_id', 'deals', ['user_id'], unique=False)
    op.create_index('ix_deals_account_id', 'deals', ['account_id'], unique=False)
    op.create_index('ix_deals_stage', 'deals', ['stage'], unique=False)
    op.create_index('ix_deals_user_stage', 'deals', ['user_id', 'stage'], unique=False)
    op.create_index('ix_deals_user_account', 'deals', ['user_id', 'account_id'], unique=False)

    op.create_table(
        'deal_messages',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('deal_id', sa.BigInteger(), nullable=False),
        sa.Column('sender_type', sa.String(length=20), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('is_read', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['deal_id'], ['deals.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_deal_messages_deal_id', 'deal_messages', ['deal_id'], unique=False)
    op.create_index('ix_deal_messages_created_at', 'deal_messages', ['created_at'], unique=False)
    op.create_index('ix_deal_messages_deal_id_created_at', 'deal_messages', ['deal_id', 'created_at'], unique=False)
    op.create_index('ix_deal_messages_deal_unread', 'deal_messages', ['deal_id', 'is_read'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_deal_messages_deal_unread', table_name='deal_messages', if_exists=True)
    op.drop_index('ix_deal_messages_deal_id_created_at', table_name='deal_messages', if_exists=True)
    op.drop_index('ix_deal_messages_created_at', table_name='deal_messages', if_exists=True)
    op.drop_index('ix_deal_messages_deal_id', table_name='deal_messages', if_exists=True)
    op.drop_table('deal_messages', if_exists=True)

    op.drop_index('ix_deals_user_stage', table_name='deals', if_exists=True)
    op.drop_index('ix_deals_user_account', table_name='deals', if_exists=True)
    op.drop_index('ix_deals_stage', table_name='deals', if_exists=True)
    op.drop_index('ix_deals_account_id', table_name='deals', if_exists=True)
    op.drop_index('ix_deals_user_id', table_name='deals', if_exists=True)
    op.drop_table('deals', if_exists=True)
