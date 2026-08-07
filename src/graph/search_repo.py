from __future__ import annotations

from src.graph.client import Neo4jClient


class Neo4jSearchRepository:

    def __init__(self, client: Neo4jClient) -> None:
        self._client = client

    async def find_candidate_post_ids(
        self,
        entities: list[str],
        target_topics: list[str],
        limit: int = 300,
    ) -> list[int]:
        clean_entities = [e.strip().lower() for e in entities if e and e.strip()]
        clean_topics = [t.strip().lower() for t in target_topics if t and t.strip()]

        if not clean_entities and not clean_topics:
            return []

        query = """
            MATCH (p:Post)
            WHERE EXISTS {
                MATCH (p)-[:MENTIONS]->(e:Entity)
                WHERE e.name IN $entities OR e.label IN $entities
            }
            OR EXISTS {
                MATCH (p)-[:BELONGS_TO]->(c:Concept)
                WHERE c.name IN $topics OR c.tier_1 IN $topics OR c.tier_2 IN $topics
            }
            RETURN DISTINCT p.id AS id
            LIMIT $limit
        """

        results = await self._client.execute_read(query, {
            "entities": clean_entities,
            "topics": clean_topics,
            "limit": limit,
        })

        return [int(row["id"]) for row in results]

    async def find_authors_by_concepts(
        self,
        category_ids: list[str],
        min_followers: int | None = None,
        limit: int = 100,
    ) -> list[int]:
        if not category_ids:
            return []

        query = """
            MATCH (a:Account)
            WHERE a.category_id IN $category_ids
            AND ($min_followers IS NULL OR a.subscribers_count >= $min_followers)
            RETURN DISTINCT a.id AS id
            LIMIT $limit
        """

        results = await self._client.execute_read(query, {
            "category_ids": category_ids,
            "min_followers": min_followers,
            "limit": limit,
        })

        return [int(row["id"]) for row in results]