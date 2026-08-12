from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import update

from src.config.config import Settings
from src.db.database import Database
from src.db.models import Content
from src.graph.client import Neo4jClient
from src.graph.builder.reader import PostBatchContext
from src.graph.ontology import (
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    OpenSPGExtractionResult,
    RelationType,
)
from src.graph.utils import clean_name_lower


class GraphWriter:

    def __init__(self, settings: Settings, neo4j_client: Neo4jClient, db: Database) -> None:
        self._neo4j_client = neo4j_client
        self._db = db

    async def write_extraction_result(
        self,
        extraction_result: OpenSPGExtractionResult,
        context: PostBatchContext,
    ) -> None:
        actor_node = {
            "id": context.author_node_id,
            "account_id": context.account_id,
            "name": context.author_title,
            "name_lower": clean_name_lower(context.author_title),
            "handle": context.author_username,
            "platform": context.platform,
        }

        post_node = {
            "id": context.pub_node_id,
            "account_id": context.account_id,
            "content_id": context.content_id,
            "published_at": int(context.published_at.timestamp()),
            "platform": context.platform,
            "post_type": context.post_type,
            "language": extraction_result.psychographics.language or "ru",
            "sentiment": extraction_result.psychographics.sentiment or "neutral",
            "tone": extraction_result.psychographics.primary_tone.value,
            "secondary_tone": extraction_result.psychographics.secondary_tones[0].value if extraction_result.psychographics.secondary_tones else None,
            "primary_hormone": extraction_result.psychographics.primary_hormone.value,
            "secondary_hormone": extraction_result.psychographics.secondary_hormone.value if extraction_result.psychographics.secondary_hormone else None,
            "score_dopamine": extraction_result.psychographics.scores.get("score_dopamine", 0.0),
            "score_oxytocin": extraction_result.psychographics.scores.get("score_oxytocin", 0.0),
            "score_serotonin": extraction_result.psychographics.scores.get("score_serotonin", 0.0),
            "score_cortisol": extraction_result.psychographics.scores.get("score_cortisol", 0.0),
            "score_adrenaline": extraction_result.psychographics.scores.get("score_adrenaline", 0.0),
            "score_endorphin": extraction_result.psychographics.scores.get("score_endorphin", 0.0),
            "is_video": context.is_video,
            "is_spam_or_gambling": extraction_result.is_spam_or_gambling,
        }

        nodes_by_label: dict[str, list[dict[str, Any]]] = {}
        nodes_by_label["Actor"] = [actor_node]
        nodes_by_label["Post"] = [post_node]

        for entity in extraction_result.entities:
            if not entity.id:
                continue
            label_str = entity.label.value
            if label_str not in nodes_by_label:
                nodes_by_label[label_str] = []
            node_dict: dict[str, Any] = {
                "id": entity.id,
                "name": entity.name,
                "name_lower": entity.name_lower,
            }
            node_dict.update(entity.properties)
            nodes_by_label[label_str].append(node_dict)

        for label_str, node_list in nodes_by_label.items():
            await self._neo4j_client.batch_merge_nodes(label_str, node_list)

        rel_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for rel in extraction_result.relations:
            key = (rel.source_label.value, rel.target_label.value, rel.relation_type.value)
            if key not in rel_groups:
                rel_groups[key] = []
            rel_groups[key].append({
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "properties": rel.properties,
            })

        for (src_label, tgt_label, rel_type), rels in rel_groups.items():
            await self._neo4j_client.batch_merge_relations(src_label, tgt_label, rel_type, rels)

        async with self._db.async_session() as session:
            async with session.begin():
                await session.execute(
                    update(Content)
                    .where(Content.id == context.content_id)
                    .values(
                        graph_status=2,
                        updated_at=datetime.now(timezone.utc),
                    )
                )

        await self._run_post_aggregations(context)

    async def _run_post_aggregations(self, context: PostBatchContext) -> None:
        covers_query = (
            "MATCH (a:Actor {id: $author_id})-[:PUBLISHED]->(p:Post) "
            "OPTIONAL MATCH (p)-[:ABOUT]->(mc:MicroConcept)-[:BELONGS_TO]->(c:Concept) "
            "OPTIONAL MATCH (p)-[:ABOUT]->(c2:Concept) "
            "WITH a, p, COLLECT(DISTINCT c) + COLLECT(DISTINCT c2) AS concepts "
            "UNWIND concepts AS concept "
            "WITH a, concept, COUNT(DISTINCT p) AS post_count "
            "WHERE concept IS NOT NULL AND post_count >= 2 "
            "MERGE (a)-[r:COVERS_TOPIC]->(concept)"
        )
        try:
            await self._neo4j_client.execute_write(covers_query, {"author_id": context.author_node_id})
        except Exception:
            pass

        profile_query = (
            "MATCH (a:Actor {id: $author_id})-[:PUBLISHED]->(p:Post) "
            "WITH p ORDER BY p.published_at DESC LIMIT 30 "
            "WITH COLLECT(p.tone) AS tones, "
            "     COLLECT(p.primary_hormone) AS primary_hormones, "
            "     COLLECT(p.secondary_hormone) AS secondary_hormones "
            "WITH "
            "  [t IN tones WHERE t IS NOT NULL] AS clean_tones, "
            "  [h IN primary_hormones WHERE h IS NOT NULL] AS clean_primary, "
            "  [h IN secondary_hormones WHERE h IS NOT NULL] AS clean_secondary "
            "WITH "
            "  CASE WHEN SIZE(clean_tones) > 0 "
            "    THEN REDUCE(m = HEAD(clean_tones), t IN TAIL(clean_tones) | "
            "      CASE WHEN SIZE([x IN clean_tones WHERE x = t]) > SIZE([x IN clean_tones WHERE x = m]) THEN t ELSE m END) "
            "    ELSE NULL END AS dominant_tone, "
            "  CASE WHEN SIZE(clean_primary) > 0 "
            "    THEN REDUCE(m = HEAD(clean_primary), h IN TAIL(clean_primary) | "
            "      CASE WHEN SIZE([x IN clean_primary WHERE x = h]) > SIZE([x IN clean_primary WHERE x = m]) THEN h ELSE m END) "
            "    ELSE NULL END AS dominant_primary_hormone, "
            "  CASE WHEN SIZE(clean_secondary) > 0 "
            "    THEN REDUCE(m = HEAD(clean_secondary), h IN TAIL(clean_secondary) | "
            "      CASE WHEN SIZE([x IN clean_secondary WHERE x = h]) > SIZE([x IN clean_secondary WHERE x = m]) THEN h ELSE m END) "
            "    ELSE NULL END AS dominant_secondary_hormone "
            "MATCH (a:Actor {id: $author_id}) "
            "SET a.primary_tone = COALESCE(dominant_tone, a.primary_tone), "
            "    a.primary_hormone = COALESCE(dominant_primary_hormone, a.primary_hormone), "
            "    a.secondary_hormone = COALESCE(dominant_secondary_hormone, a.secondary_hormone)"
        )
        try:
            await self._neo4j_client.execute_write(profile_query, {"author_id": context.author_node_id})
        except Exception:
            pass