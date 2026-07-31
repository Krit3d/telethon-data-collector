import logging

from sqlalchemy import text
from sqlalchemy.engine import Result
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class GraphSearchRepository:

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def search_posts_by_entities(self, entities: list[str], author_type: str, limit: int = 600) -> dict[int, float]:
        clean_entities = [e.strip().lower() for e in entities if e.strip()]
        if not clean_entities:
            return {}

        if author_type == "expert":
            apply_author_filter = True
            target_is_author_blog = True
        elif author_type == "business":
            apply_author_filter = True
            target_is_author_blog = False
        else:
            apply_author_filter = False
            target_is_author_blog = True

        sql = """
            SELECT
                post_id,
                SUM(weight)::float AS accumulative_raw_score
            FROM public.graph_entity_posts
            WHERE entity_name_lower = ANY(:clean_entities)
              AND (:apply_author_filter = False OR is_author_blog = :target_is_author_blog)
            GROUP BY post_id
            ORDER BY accumulative_raw_score DESC
            LIMIT :limit;
        """

        try:
            result: Result = await self.session.execute(
                text(sql),
                {
                    "clean_entities": clean_entities,
                    "apply_author_filter": apply_author_filter,
                    "target_is_author_blog": target_is_author_blog,
                    "limit": limit,
                },
            )

            rows = result.fetchall()
            logger.info(
                "Graph projection returned %d candidate posts from materialized index (author_type=%s)",
                len(rows), author_type,
            )
            if not rows:
                return {}

            raw_scores = [float(row.accumulative_raw_score) for row in rows]
            max_score = max(raw_scores)
            if max_score == 0.0:
                return {}

            return {
                int(row.post_id): (float(row.accumulative_raw_score) / max_score)
                for row in rows
            }
        except Exception:
            await self.session.rollback()
            logger.exception("Graph projection index query failed")
            return {}
