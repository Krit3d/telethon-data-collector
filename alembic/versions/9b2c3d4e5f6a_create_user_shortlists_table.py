from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = '9b2c3d4e5f6a'
down_revision: str | Sequence[str] | None = '8a1b2c3d4e5f'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'user_shortlists',
        sa.Column('user_id', sa.Integer(), nullable=False, comment='Foreign key referencing the user'),
        sa.Column('account_id', sa.BigInteger(), nullable=False, comment='Foreign key referencing the account'),
        sa.Column('status', sa.String(length=30), server_default='Свободен', nullable=False, comment="Custom manager status (e.g., 'Свободен')"),
        sa.Column('notes', sa.Text(), nullable=True, comment='Manager custom notes for this shortlist entry'),
        sa.Column('custom_cpm', sa.Integer(), nullable=True, comment='Custom CPM override set by the manager'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Timestamp when the record was first inserted'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Timestamp of the last record update'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['account_id'], ['accounts.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('user_id', 'account_id'),
        comment='User shortlist mapping table with manager notes and custom status',
    )
    op.create_index('ix_user_shortlists_user_created', 'user_shortlists', ['user_id', sa.text('created_at DESC')], unique=False)
    op.create_index('ix_user_shortlists_user_status', 'user_shortlists', ['user_id', 'status'], unique=False)
    op.create_index('ix_user_shortlists_account_id', 'user_shortlists', ['account_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_user_shortlists_user_status', table_name='user_shortlists', if_exists=True)
    op.drop_index('ix_user_shortlists_user_created', table_name='user_shortlists', if_exists=True)
    op.drop_index('ix_user_shortlists_account_id', table_name='user_shortlists', if_exists=True)
    op.drop_table('user_shortlists', if_exists=True)