from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'b7d8e9f0a1b2'
down_revision: str | Sequence[str] | None = 'a1b2c3d4e5f6'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index('idx_content_stale_graph_processing', 'content', [sa.text("(raw_metadata->>'claimed_at')")], postgresql_where=sa.text("is_graph_extracted = false AND raw_metadata->>'graph_status' = 'processing'"))


def downgrade() -> None:
    op.drop_index('idx_content_stale_graph_processing', table_name='content')
