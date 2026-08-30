from src.api.schemas import GraphAuthorEvidence
from src.graph.client import Neo4jClient


class Neo4jSearchRepository:

    def __init__(self, client: Neo4jClient) -> None:
        self._client = client

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

    async def get_authors_graph_evidence(
        self,
        account_ids: list[int],
        search_tokens: list[str],
        target_languages: list[str] | None = None,
    ) -> dict[int, GraphAuthorEvidence]:
        if not account_ids:
            return {}

        query = """
            MATCH (a:Actor)
            WHERE a.account_id IN $account_ids
            WITH a,
                size([(a)-[:COVERS_TOPIC]->() | 1]) AS total_topics_count,
                size([(a)-[:PUBLISHED]->(:Post) | 1]) AS graphed_posts_count,
                [(a)-[r:COVERS_TOPIC]->(c:Concept)
                    WHERE c.name_lower IN $search_tokens
                    | {name: c.name_lower, posts_count: coalesce(r.posts_count, 0)}] AS concept_topics,
                [(a)-[r:COVERS_TOPIC]->(mc:MicroConcept)
                    WHERE mc.name_lower IN $search_tokens
                    | {name: mc.name_lower, posts_count: coalesce(r.posts_count, 0)}] AS micro_topics,
                [(a)-[:PUBLISHED]->(:Post)-[:MENTIONS]->(e)
                    WHERE (e:Entity OR e:Organization OR e:Product OR e:Event) AND e.name_lower IN $search_tokens
                    | e.name_lower] AS mentioned_entities,
                [(a)-[:PUBLISHED]->(:Post)-[:ABOUT]->(mc:MicroConcept)
                    WHERE mc.name_lower IN $search_tokens
                    | mc.name_lower] AS about_micros,
                [(a)-[:USES_TECH]->(tech)
                    WHERE tech.name_lower IN $search_tokens
                    | tech.name_lower] AS tech_matches,
                [(a)-[:WORKS_AT]->(org:Organization)
                    WHERE org.name_lower IN $search_tokens
                    | org.name_lower] AS org_matches,
                [(a)-[:PARTICIPATED_IN]->(ev:Event)
                    WHERE ev.name_lower IN $search_tokens
                    | ev.name_lower] AS ev_matches,
                [(a)-[r:PRODUCES]->(prod:Product)
                    WHERE prod.name_lower IN $search_tokens
                    | coalesce(r.relation_subtype, 'creator')] AS prod_subtypes,
                EXISTS { MATCH (a)-[:PUBLISHED]->(p:Post) WHERE p.is_spam_or_gambling = true } AS is_spam
            WITH a, is_spam,
                total_topics_count, graphed_posts_count,
                concept_topics, micro_topics,
                REDUCE(acc = [], x IN [t IN concept_topics | t.name] | CASE WHEN x IN acc THEN acc ELSE acc + x END) AS uniq_concept_names,
                REDUCE(acc = [], x IN [t IN micro_topics | t.name] | CASE WHEN x IN acc THEN acc ELSE acc + x END) AS uniq_micro_names,
                size(REDUCE(acc = [], x IN mentioned_entities | CASE WHEN x IN acc THEN acc ELSE acc + x END)) AS uniq_entities_count,
                size(REDUCE(acc = [], x IN about_micros | CASE WHEN x IN acc THEN acc ELSE acc + x END)) AS uniq_about_count,
                size(tech_matches) > 0 AS has_tech_relation,
                (size(org_matches) > 0 OR size(ev_matches) > 0) AS has_role_relation,
                'creator' IN prod_subtypes AS is_creator,
                'promoter' IN prod_subtypes AS is_promoter
            RETURN
                a.account_id AS account_id,
                a.location_name AS location_name,
                a.primary_language AS primary_language,
                CASE WHEN $target_languages IS NOT NULL AND size(coalesce($target_languages, [])) > 0 THEN coalesce(a.primary_language, '') IN $target_languages ELSE true END AS matched_language,
                total_topics_count,
                graphed_posts_count,
                size(uniq_concept_names) + size(uniq_micro_names) AS matched_topics_count,
                REDUCE(s = 0, x IN concept_topics | s + x.posts_count) + REDUCE(s = 0, x IN micro_topics | s + x.posts_count) AS total_posts_count,
                uniq_concept_names AS matched_concept_names,
                uniq_micro_names AS matched_micro_names,
                uniq_entities_count AS entity_matches_count,
                uniq_entities_count + uniq_about_count AS direct_mentions_count,
                has_role_relation,
                has_tech_relation,
                is_creator,
                is_promoter,
                is_spam AS is_spam_or_gambling
        """

        params: dict[str, object] = {
            "account_ids": account_ids,
            "search_tokens": search_tokens,
            "target_languages": [lang.lower().strip() for lang in target_languages] if target_languages else None,
        }

        rows = await self._client.execute_read(query, params)

        result: dict[int, GraphAuthorEvidence] = {}
        for row in rows:
            account_id = int(row["account_id"])
            total_posts_count = float(row["total_posts_count"])
            matched_entities_count = int(row["entity_matches_count"])
            direct_mentions_count = int(row["direct_mentions_count"])
            has_role_relation = bool(row["has_role_relation"])
            has_tech_relation = bool(row["has_tech_relation"])
            is_creator = bool(row["is_creator"])
            is_promoter = bool(row["is_promoter"])
            is_spam_or_gambling = bool(row["is_spam_or_gambling"])
            location_name = str(row["location_name"]) if row.get("location_name") else None
            primary_language = str(row["primary_language"]) if row.get("primary_language") else None

            topic_coverage_weight = min(1.0, total_posts_count / 12.0)

            matched_concepts = [str(n) for n in row["matched_concept_names"] if n]
            matched_microconcepts = [str(n) for n in row["matched_micro_names"] if n]

            result[account_id] = GraphAuthorEvidence(
                account_id=account_id,
                topic_coverage_weight=topic_coverage_weight,
                matched_concepts=matched_concepts,
                matched_microconcepts=matched_microconcepts,
                total_topics_count=int(row["total_topics_count"]),
                matched_topics_count=int(row["matched_topics_count"]),
                matched_entities_count=matched_entities_count,
                direct_mentions_count=direct_mentions_count,
                has_role_relation=has_role_relation,
                has_tech_relation=has_tech_relation,
                is_creator=is_creator,
                is_promoter=is_promoter,
                is_spam_or_gambling=is_spam_or_gambling,
                location_name=location_name,
                primary_language=primary_language,
                raw_graph_score=0.0,
            )

        return result
