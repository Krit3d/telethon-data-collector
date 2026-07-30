import logging

from sqlalchemy import text
from sqlalchemy.engine import Result
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class GraphSearchRepository:

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def search_posts_by_entities(self, entities: list[str], author_type: str, limit: int = 600) -> dict[int, float]:
        entities_clean = [e.strip().lower() for e in entities if e.strip()]
        entity_patterns = [f"%{e}%" for e in entities_clean if len(e) >= 2]
        if not entity_patterns:
            return {}

        apply_author_filter = author_type in ("expert", "business")
        is_author_blog = (author_type == "expert")

        query = text("""
            WITH matched_vertices AS (
                SELECT id
                FROM social_graph._ag_label_vertex
                WHERE lower((properties::text::jsonb)->>'name') LIKE ANY(:entity_patterns)
            ),
            matched_edges AS (
                SELECT
                    e.start_id AS connected_vertex_id,
                    (e.properties::text::jsonb)->>'db_post_id' AS edge_post_id
                FROM matched_vertices v
                JOIN social_graph._ag_label_edge e ON e.end_id = v.id
                UNION ALL
                SELECT
                    e.end_id AS connected_vertex_id,
                    (e.properties::text::jsonb)->>'db_post_id' AS edge_post_id
                FROM matched_vertices v
                JOIN social_graph._ag_label_edge e ON e.start_id = v.id
            ),
            candidate_posts AS (
                SELECT
                    coalesce(
                        (other_v.properties::text::jsonb)->>'db_post_id',
                        me.edge_post_id,
                        (other_v.properties::text::jsonb)->>'id'
                    ) AS raw_post_ref,
                    COUNT(*)::float AS raw_graph_score
                FROM matched_edges me
                JOIN social_graph._ag_label_vertex other_v ON other_v.id = me.connected_vertex_id
                GROUP BY 1
            ),
            valid_candidate_posts AS (
                SELECT
                    regexp_replace(raw_post_ref, '[^0-9]', '', 'g') AS clean_id_str,
                    raw_graph_score
                FROM candidate_posts
                WHERE raw_post_ref IS NOT NULL
                  AND length(regexp_replace(raw_post_ref, '[^0-9]', '', 'g')) BETWEEN 1 AND 18
            )
            SELECT
                c.id AS post_id,
                vcp.raw_graph_score
            FROM valid_candidate_posts vcp
            JOIN public.content c ON c.id = vcp.clean_id_str::bigint
            JOIN public.accounts a ON a.id = c.account_id
            WHERE (:apply_author_filter = False OR a.is_author_blog = :is_author_blog)
              AND c.is_enriched = True
            ORDER BY vcp.raw_graph_score DESC
            LIMIT :limit;
        """)

        try:
            result: Result = await self.session.execute(
                query,
                {
                    "entity_patterns": entity_patterns,
                    "apply_author_filter": apply_author_filter,
                    "is_author_blog": is_author_blog,
                    "limit": limit,
                }
            )

            rows = result.fetchall()
            logger.info("Apache AGE returned %d candidate posts from graph using pattern matching", len(rows))
            if not rows:
                count_result = await self.session.execute(
                    text("SELECT COUNT(*) FROM social_graph._ag_label_vertex WHERE lower((properties::text::jsonb)->>'name') LIKE ANY(:entity_patterns)"),
                    {"entity_patterns": entity_patterns}
                )
                vertex_count = count_result.scalar()
                logger.info("No graph matches found for entities: %s (matched_vertices=%s)", entity_patterns, vertex_count)
                return {}

            max_score = max(row.raw_graph_score for row in rows)
            if max_score == 0:
                return {}

            return {
                int(row.post_id): (row.raw_graph_score / max_score)
                for row in rows
            }
        except Exception as e:
            await self.session.rollback()
            logger.exception("Direct AGE SQL search query failed: %s", e)
            return {}
