from __future__ import annotations

import asyncio
import logging
import time
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
    OpenSPGExtractionResult,
)
from src.graph.utils import clean_name_lower, sanitize_properties

logger = logging.getLogger(__name__)


class GraphWriter:

    def __init__(self, settings: Settings, neo4j_client: Neo4jClient, db: Database) -> None:
        self._neo4j_client = neo4j_client
        self._db = db
        self._background_tasks: set[asyncio.Task] = set()

    async def write_extraction_result(
        self,
        extraction_result: OpenSPGExtractionResult,
        context: PostBatchContext,
    ) -> None:
        await self.write_extraction_results_batch([(extraction_result, context)])

    async def write_extraction_results_batch(
        self,
        items: list[tuple[OpenSPGExtractionResult, PostBatchContext]],
    ) -> None:
        if not items:
            return

        t0 = time.perf_counter()

        nodes_by_label: dict[str, list[dict[str, Any]]] = {}
        rel_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        claimed_ids: list[int] = []
        unique_author_ids: set[str] = set()

        for extraction_result, context in items:
            claimed_ids.append(context.content_id)
            unique_author_ids.add(context.author_node_id)

            actor_node = sanitize_properties({
                "id": context.author_node_id,
                "account_id": context.account_id,
                "name": context.author_title,
                "name_lower": clean_name_lower(context.author_title),
                "handle": context.author_username,
                "platform": context.platform,
                "platform_id": context.platform_id,
                "location_name": context.location_name,
            })

            post_node = sanitize_properties({
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
            })

            nodes_by_label.setdefault("Actor", []).append(actor_node)
            nodes_by_label.setdefault("Post", []).append(post_node)

            for entity in extraction_result.entities:
                if not entity.id:
                    continue
                label_str = entity.label.value
                props = sanitize_properties(dict(entity.properties))
                if label_str == EntityType.Entity.value:
                    props.pop("mentions_count", None)
                else:
                    props.setdefault("mentions_count", 1)
                node_dict = {"id": entity.id, "name": entity.name, "name_lower": entity.name_lower}
                node_dict.update(props)
                nodes_by_label.setdefault(label_str, []).append(node_dict)

            for rel in extraction_result.relations:
                key = (rel.source_label.value, rel.target_label.value, rel.relation_type.value)
                rel_groups.setdefault(key, []).append({
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "properties": sanitize_properties(rel.properties),
                })

        total_nodes = sum(len(v) for v in nodes_by_label.values())
        t1 = time.perf_counter()
        for label_str, node_list in nodes_by_label.items():
            await self._neo4j_client.batch_merge_nodes(label_str, node_list)
        node_elapsed = (time.perf_counter() - t1) * 1000

        total_rels = sum(len(v) for v in rel_groups.values())
        t2 = time.perf_counter()
        for (src_label, tgt_label, rel_type), rels in rel_groups.items():
            await self._neo4j_client.batch_merge_relations(src_label, tgt_label, rel_type, rels)
        rel_elapsed = (time.perf_counter() - t2) * 1000

        t3 = time.perf_counter()
        async with self._db.async_session() as session:
            async with session.begin():
                await session.execute(
                    update(Content)
                    .where(Content.id.in_(claimed_ids))
                    .values(
                        graph_status=2,
                        updated_at=datetime.now(timezone.utc),
                    )
                )
        pg_elapsed = (time.perf_counter() - t3) * 1000

        total_elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Batch writer done in %.1fms | items: %d | nodes: %d (%d labels, %.1fms) | rels: %d (%d types, %.1fms) | PG status=2: %.1fms",
            total_elapsed, len(items), total_nodes, len(nodes_by_label), node_elapsed,
            total_rels, len(rel_groups), rel_elapsed, pg_elapsed,
        )

        for author_id in unique_author_ids:
            task = asyncio.create_task(self._run_post_aggregations(author_id))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _run_post_aggregations(self, author_id: str) -> None:
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
            await self._neo4j_client.execute_write(covers_query, {"author_id": author_id})
        except Exception as exc:
            logger.warning("COVERS_TOPIC aggregation failed for author %s: %s", author_id, exc)

        profile_query = (
            "OPTIONAL MATCH (a:Actor {id: $author_id})-[:PUBLISHED]->(p:Post) "
            "WITH a, p ORDER BY p.published_at DESC LIMIT 30 "
            "WITH a, [x IN COLLECT(p) WHERE x IS NOT NULL] AS posts "
            "WITH a, posts, "
            "  [p IN posts WHERE p.language IS NOT NULL | p.language] AS languages, "
            "  [p IN posts WHERE p.sentiment IS NOT NULL | p.sentiment] AS sentiments, "
            "  [p IN posts WHERE p.tone IS NOT NULL | p.tone] AS tones, "
            "  [p IN posts WHERE p.primary_hormone IS NOT NULL | p.primary_hormone] AS primary_hormones, "
            "  [p IN posts WHERE p.secondary_hormone IS NOT NULL | p.secondary_hormone] AS secondary_hormones "
            "WITH a, languages, sentiments, tones, primary_hormones, secondary_hormones, "
            "  CASE WHEN SIZE(languages) > 0 "
            "    THEN REDUCE(m = HEAD(languages), v IN TAIL(languages) | "
            "      CASE WHEN SIZE([x IN languages WHERE x = v]) > SIZE([x IN languages WHERE x = m]) THEN v ELSE m END) "
            "    ELSE NULL END AS dominant_language, "
            "  CASE WHEN SIZE(sentiments) > 0 "
            "    THEN REDUCE(m = HEAD(sentiments), v IN TAIL(sentiments) | "
            "      CASE WHEN SIZE([x IN sentiments WHERE x = v]) > SIZE([x IN sentiments WHERE x = m]) THEN v ELSE m END) "
            "    ELSE NULL END AS dominant_sentiment, "
            "  CASE WHEN SIZE(tones) > 0 "
            "    THEN REDUCE(m = HEAD(tones), v IN TAIL(tones) | "
            "      CASE WHEN SIZE([x IN tones WHERE x = v]) > SIZE([x IN tones WHERE x = m]) THEN v ELSE m END) "
            "    ELSE NULL END AS dominant_tone, "
            "  CASE WHEN SIZE(primary_hormones) > 0 "
            "    THEN REDUCE(m = HEAD(primary_hormones), v IN TAIL(primary_hormones) | "
            "      CASE WHEN SIZE([x IN primary_hormones WHERE x = v]) > SIZE([x IN primary_hormones WHERE x = m]) THEN v ELSE m END) "
            "    ELSE NULL END AS dominant_primary_hormone, "
            "  CASE WHEN SIZE(secondary_hormones) > 0 "
            "    THEN REDUCE(m = HEAD(secondary_hormones), v IN TAIL(secondary_hormones) | "
            "      CASE WHEN SIZE([x IN secondary_hormones WHERE x = v]) > SIZE([x IN secondary_hormones WHERE x = m]) THEN v ELSE m END) "
            "    ELSE NULL END AS dominant_secondary_hormone "
            "SET a.primary_language = COALESCE(dominant_language, a.primary_language, 'ru'), "
            "    a.primary_sentiment = COALESCE(dominant_sentiment, a.primary_sentiment, 'neutral'), "
            "    a.primary_tone = COALESCE(dominant_tone, a.primary_tone, 'casual'), "
            "    a.primary_hormone = COALESCE(dominant_primary_hormone, a.primary_hormone, 'dopamine'), "
            "    a.secondary_hormone = COALESCE(dominant_secondary_hormone, a.secondary_hormone)"
        )
        try:
            await self._neo4j_client.execute_write(profile_query, {"author_id": author_id})
        except Exception as exc:
            logger.warning("Actor profile aggregation failed for author %s: %s", author_id, exc)