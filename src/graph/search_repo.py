from __future__ import annotations

from src.graph.client import Neo4jClient


class Neo4jSearchRepository:

    def __init__(self, client: Neo4jClient) -> None:
        self._client = client

    async def find_candidate_posts_with_weights(
        self,
        canonical_entity_ids: list[str],
        target_topics: list[str],
        limit: int = 500,
    ) -> dict[int, float]:
        clean_entities = [e for e in canonical_entity_ids if e and e.strip()]
        clean_topics = [t for t in target_topics if t and t.strip()]

        if not clean_entities and not clean_topics:
            return {}

        query = """
            CALL {
                UNWIND $entity_ids AS eid
                MATCH (e {id: eid})
                MATCH (p:Post)-[r:MENTIONS|TAGGED_AT|TAGGED_WITH|ABOUT]->(e)
                WITH p, r,
                     CASE type(r)
                         WHEN 'ABOUT' THEN 1.5
                         WHEN 'TAGGED_AT' THEN 1.2
                         WHEN 'TAGGED_WITH' THEN 1.1
                         ELSE 1.0
                     END AS w
                RETURN p.content_id AS content_id, w AS weight
                UNION ALL
                UNWIND $topics AS top
                MATCH (c:Concept)
                WHERE c.code = top OR LOWER(c.name) = top OR LOWER(c.tier_1) = top
                MATCH (p:Post)-[:ABOUT]->(target)
                WHERE target = c OR (target:MicroConcept AND (target)-[:BELONGS_TO]->(c))
                RETURN p.content_id AS content_id, 1.5 AS weight
            }
            WITH content_id, SUM(weight) AS total_weight
            WHERE content_id IS NOT NULL
            RETURN content_id, total_weight
            ORDER BY total_weight DESC
            LIMIT $limit
        """

        results = await self._client.execute_read(query, {
            "entity_ids": clean_entities,
            "topics": clean_topics,
            "limit": limit,
        })

        return {int(row["content_id"]): float(row["total_weight"]) for row in results}

    async def find_authors_by_concepts(
        self,
        category_ids: list[str],
        limit: int = 200,
    ) -> list[int]:
        if not category_ids:
            return []

        query = """
            MATCH (c:Concept)
            WHERE c.code IN $category_ids OR c.id IN $category_ids
            MATCH (a:Actor)-[:COVERS_TOPIC]->(c)
            RETURN DISTINCT a.account_id AS account_id
            LIMIT $limit
        """

        results = await self._client.execute_read(query, {
            "category_ids": category_ids,
            "limit": limit,
        })

        return [int(row["account_id"]) for row in results]
