import json
import logging

from sqlalchemy import text
from sqlalchemy.engine import Result
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class GraphSearchRepository:

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def search_posts_by_entities(self, entities: list[str], author_type: str, limit: int = 600) -> dict[int, float]:
        if not entities:
            return {}

        entities_clean = [e.strip().lower() for e in entities if e.strip()]
        if not entities_clean:
            return {}

        apply_author_filter = (author_type in ("expert", "business"))
        is_author_blog = (author_type == "expert")

        query = text("""
            SELECT
                (g.post_id::text)::bigint AS post_id,
                (g.raw_graph_score::text)::float AS raw_graph_score
            FROM cypher('social_graph', $$
                MATCH (e:Entity)-[r:MENTIONED_IN]->(p:Post)
                WHERE e.name_lower IN $entities
                RETURN p.db_id AS post_id, p.account_id AS account_id, SUM(r.weight) AS raw_graph_score
            $$, CAST(:entities_json AS agtype)) AS g(post_id agtype, account_id agtype, raw_graph_score agtype)
            JOIN public.accounts a ON a.id = (g.account_id::text)::bigint
            JOIN public.content c ON c.id = (g.post_id::text)::bigint
            WHERE (:apply_author_filter = False OR a.is_author_blog = :is_author_blog)
              AND c.is_enriched = True
            ORDER BY (g.raw_graph_score::text)::float DESC
            LIMIT :limit;
        """)

        try:
            await self.session.execute(text("SET search_path = ag_catalog, public;"))

            result: Result = await self.session.execute(
                query,
                {
                    "entities_json": json.dumps({"entities": entities_clean}),
                    "apply_author_filter": apply_author_filter,
                    "is_author_blog": is_author_blog,
                    "limit": limit,
                }
            )

            rows = result.fetchall()
            if not rows:
                return {}

            max_score = max(row.raw_graph_score for row in rows)

            return {
                row.post_id: (row.raw_graph_score / max_score) if max_score > 0 else 0.0
                for row in rows
            }
        except Exception:
            logger.exception("AGE graph search query failed, rolling back transaction")
            await self.session.rollback()
            return {}
