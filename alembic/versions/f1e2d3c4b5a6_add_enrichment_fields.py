from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'f1e2d3c4b5a6'
down_revision: str | Sequence[str] | None = '9e8d7c6b5a4f'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('accounts', sa.Column('category_id', sa.String(50), nullable=True, comment="Hard unique category ID from IAB 3.1 taxonomy (e.g., '653' for dentists). B-Tree indexed for fast filtering"))
    op.add_column('accounts', sa.Column('category_path', sa.String(512), nullable=True, comment="Full category breadcrumb (e.g., 'Medical Health > Dental Health'). Used for UI display and parent category search"))
    op.add_column('accounts', sa.Column('explanation', sa.Text(), nullable=True, comment="Pre-generated author expertise description in Russian for UI display per specification requirements"))
    op.add_column('accounts', sa.Column('static_avg_er', sa.Float(), nullable=True, comment="Pre-calculated average engagement rate from the author's last posts. Avoids heavy runtime aggregations during search"))
    op.add_column('content', sa.Column('is_enriched', sa.Boolean(), nullable=False, server_default=sa.text('false'), comment="Marker for content readiness. Worker processes only posts with is_enriched=False, then marks as True after handling. Enables efficient incremental pipeline without reprocessing"))
    op.create_index('ix_accounts_category_id', 'accounts', ['category_id'], unique=False)
    op.create_index('ix_content_is_enriched', 'content', ['is_enriched'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_content_is_enriched', table_name='content')
    op.drop_column('content', 'is_enriched')
    op.drop_index('ix_accounts_category_id', table_name='accounts')
    op.drop_column('accounts', 'static_avg_er')
    op.drop_column('accounts', 'explanation')
    op.drop_column('accounts', 'category_path')
    op.drop_column('accounts', 'category_id')