from collections.abc import Sequence

from alembic import op

from src.config.config import Settings

revision: str = '9e8d7c6b5a4f'
down_revision: str | Sequence[str] | None = '4fd80d3eff64'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    settings = Settings()  # type: ignore[call-arg]
    graph_name = settings.graph_name
    op.execute(f"""
        UPDATE "{graph_name}"."Entity"
        SET properties = (
            (properties::text::jsonb || jsonb_build_object('name_lower', lower(coalesce(properties::text::jsonb->>'name', ''))))
        )::text::ag_catalog.agtype
    """)
    op.execute(f'CREATE INDEX IF NOT EXISTS entity_properties_name_lower_idx ON "{graph_name}"."Entity" USING btree (ag_catalog.agtype_access_operator(properties, \'"name_lower"\'::ag_catalog.agtype))')


def downgrade() -> None:
    settings = Settings()  # type: ignore[call-arg]
    graph_name = settings.graph_name
    op.execute(f'DROP INDEX IF EXISTS "{graph_name}".entity_properties_name_lower_idx')
    op.execute(f"""
        UPDATE "{graph_name}"."Entity"
        SET properties = (
            (properties::text::jsonb - 'name_lower')
        )::text::ag_catalog.agtype
    """)
