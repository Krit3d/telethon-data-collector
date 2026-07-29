from alembic import op

revision: str = 'e7f8a9b0c1d2'
down_revision: str | None = 'd4e5f6a7b8c9'
branch_labels: str | list[str] | None = None
depends_on: str | list[str] | None = None


def upgrade() -> None:
    label_tables = ["Entity", "Actor", "Topic", "Event", "Place", "Hashtag"]
    for table in label_tables:
        op.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table.lower()}_fts
            ON social_graph."{table}"
            USING gin (to_tsvector('simple', properties::text));
        """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_ag_edge_start_id
        ON social_graph._ag_label_edge (start_id);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_ag_edge_end_id
        ON social_graph._ag_label_edge (end_id);
    """)


def downgrade() -> None:
    label_tables = ["Entity", "Actor", "Topic", "Event", "Place", "Hashtag"]
    for table in label_tables:
        op.execute(f'DROP INDEX IF EXISTS social_graph.idx_{table.lower()}_fts;')
    op.execute("DROP INDEX IF EXISTS social_graph.idx_ag_edge_start_id;")
    op.execute("DROP INDEX IF EXISTS social_graph.idx_ag_edge_end_id;")