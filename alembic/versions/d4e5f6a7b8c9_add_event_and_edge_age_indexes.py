from collections.abc import Sequence

from alembic import op

revision: str = 'd4e5f6a7b8c9'
down_revision: str | Sequence[str] | None = '7d5c471eebc4'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute('SET search_path TO social_graph, ag_catalog, public')
    op.execute('CREATE INDEX IF NOT EXISTS idx_event_db_post_id ON social_graph."Event" USING btree (ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, \'"db_post_id"\'::ag_catalog.agtype]))')
    op.execute('CREATE INDEX IF NOT EXISTS idx_event_engagement_rate ON social_graph."Event" USING btree (ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, \'"engagement_rate"\'::ag_catalog.agtype]))')
    op.execute('CREATE INDEX IF NOT EXISTS idx_ag_label_edge_start_end ON social_graph._ag_label_edge USING btree (start_id, end_id)')


def downgrade() -> None:
    op.execute('DROP INDEX IF EXISTS social_graph.idx_event_db_post_id')
    op.execute('DROP INDEX IF EXISTS social_graph.idx_event_engagement_rate')
    op.execute('DROP INDEX IF EXISTS social_graph.idx_ag_label_edge_start_end')
