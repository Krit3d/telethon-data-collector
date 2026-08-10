from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '80b4c5d6e7f8'
down_revision: str | Sequence[str] | None = '2f3a4b5c6d7e'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('content', sa.Column('graph_status', sa.SmallInteger(), nullable=False, server_default=sa.text('0')))

    op.drop_column('content', 'is_graph_extracted')

    op.execute('CREATE INDEX IF NOT EXISTS idx_content_pending ON content (account_id, published_at DESC) WHERE graph_status = 0')
    op.execute('CREATE INDEX IF NOT EXISTS idx_content_is_enriched ON content (is_enriched) WHERE is_enriched = false')
    op.execute('CREATE INDEX IF NOT EXISTS idx_content_is_embedded ON content (is_embedded) WHERE is_embedded = false')
    op.execute('CREATE INDEX IF NOT EXISTS idx_content_account_id ON content (account_id)')


def downgrade() -> None:
    op.execute('DROP INDEX IF EXISTS idx_content_pending')
    op.execute('DROP INDEX IF EXISTS idx_content_is_enriched')
    op.execute('DROP INDEX IF EXISTS idx_content_is_embedded')
    op.execute('DROP INDEX IF EXISTS idx_content_account_id')

    op.add_column('content', sa.Column('is_graph_extracted', sa.Boolean(), nullable=False, server_default=sa.text('false')))

    op.drop_column('content', 'graph_status')