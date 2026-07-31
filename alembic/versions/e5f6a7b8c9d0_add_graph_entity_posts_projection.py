from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'e5f6a7b8c9d0'
down_revision: str | Sequence[str] | None = 'd4e5f6a7b8c9'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
    op.create_table(
        'graph_entity_posts',
        sa.Column('entity_name_lower', sa.Text(), nullable=False),
        sa.Column('post_id', sa.BigInteger(), nullable=False),
        sa.Column('entity_type', sa.String(50), nullable=True),
        sa.Column('is_author_blog', sa.Boolean(), nullable=True),
        sa.Column('distance', sa.Integer(), nullable=True),
        sa.Column('weight', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('entity_name_lower', 'post_id'),
        schema='public',
    )
    op.execute('CREATE INDEX IF NOT EXISTS idx_gep_entity_author ON public.graph_entity_posts (entity_name_lower, is_author_blog)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_gep_post_id ON public.graph_entity_posts (post_id)')
    op.execute('CREATE INDEX IF NOT EXISTS idx_gep_trgm ON public.graph_entity_posts USING gin (entity_name_lower gin_trgm_ops)')


def downgrade() -> None:
    op.execute('DROP INDEX IF EXISTS public.idx_gep_trgm')
    op.execute('DROP INDEX IF EXISTS public.idx_gep_post_id')
    op.execute('DROP INDEX IF EXISTS public.idx_gep_entity_author')
    op.drop_table('graph_entity_posts', schema='public')