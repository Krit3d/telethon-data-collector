from collections.abc import Sequence

from alembic import op

from src.config.config import Settings

revision: str = '4fd80d3eff64'
down_revision: str | Sequence[str] | None = 'b7d8e9f0a1b2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    settings = Settings()  # type: ignore[call-arg]
    graph_name = settings.graph_name
    op.execute(f'CREATE INDEX IF NOT EXISTS entity_properties_id_idx ON "{graph_name}"."Entity" USING btree (ag_catalog.agtype_access_operator(properties, \'"id"\'::ag_catalog.agtype))')
    op.execute(f'CREATE INDEX IF NOT EXISTS event_properties_db_post_id_idx ON "{graph_name}"."Event" USING btree (ag_catalog.agtype_access_operator(properties, \'"db_post_id"\'::ag_catalog.agtype))')


def downgrade() -> None:
    settings = Settings()  # type: ignore[call-arg]
    graph_name = settings.graph_name
    op.execute(f'DROP INDEX IF EXISTS "{graph_name}".entity_properties_id_idx')
    op.execute(f'DROP INDEX IF EXISTS "{graph_name}".event_properties_db_post_id_idx')
