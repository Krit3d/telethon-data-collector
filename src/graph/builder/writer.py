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
            })

            post_node = sanitize_properties({
                "id": context.pub_node_id,
                "account_id": context.account_id,
                "content_id": context.content_id,
                "published_at": int(context.published_at.timestamp()),
                "platform": context.platform,
                "post_type": context.post_type,
                "language": extraction_result.psychographics.language,
                "sentiment": extraction_result.psychographics.sentiment.value if extraction_result.psychographics.sentiment else None,
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
            "WITH a, COUNT(p) AS total_posts "
            "MATCH (a)-[:PUBLISHED]->(p:Post) "
            "OPTIONAL MATCH (p)-[:ABOUT]->(:MicroConcept)-[:BELONGS_TO]->(concept:Concept) "
            "WITH a, total_posts, concept, COUNT(DISTINCT p) AS post_count "
            "WHERE concept IS NOT NULL AND post_count >= 2 "
            "MERGE (a)-[r:COVERS_TOPIC]->(concept) "
            "SET r.posts_count = post_count, "
            "    r.weight = toFloat(post_count) / total_posts, "
            "    r.last_updated = timestamp()"
        )
        try:
            await self._neo4j_client.execute_write(covers_query, {"author_id": author_id})
        except Exception as exc:
            logger.warning("COVERS_TOPIC aggregation failed for author %s: %s", author_id, exc)

        profile_query = (
            "MATCH (a:Actor {id: $author_id})-[:PUBLISHED]->(p:Post) "
            "WITH a, COLLECT(p) AS posts "
            "WITH a, posts, "
            "  [x IN posts WHERE x.language IS NOT NULL | x.language] AS langs, "
            "  [x IN posts WHERE x.sentiment IS NOT NULL | x.sentiment] AS sents "
            "WITH a, posts, langs, sents "
            "UNWIND (CASE WHEN SIZE(langs) > 0 THEN langs ELSE [null] END) AS lang "
            "WITH a, posts, sents, lang, COUNT(*) AS cnt "
            "ORDER BY cnt DESC "
            "WITH a, posts, sents, "
            "  COLLECT(lang) AS lang_order, "
            "  COLLECT(cnt) AS lang_counts, "
            "  REDUCE(s = 0, c IN COLLECT(cnt) | s + c) AS total_lang_count "
            "WITH a, posts, sents, "
            "  CASE WHEN SIZE(lang_order) > 0 AND lang_counts[0] >= total_lang_count * 0.15 "
            "    THEN lang_order[0] "
            "    ELSE NULL END AS primary_language, "
            "  CASE WHEN SIZE(sents) > 0 "
            "    THEN REDUCE(m = HEAD(sents), v IN TAIL(sents) | "
            "      CASE WHEN SIZE([x IN sents WHERE x = v]) > SIZE([x IN sents WHERE x = m]) THEN v ELSE m END) "
            "    ELSE NULL END AS primary_sentiment, "
            "  REDUCE(arr = [], p IN posts | "
            "    arr + CASE WHEN p.tone IS NOT NULL THEN [{v: p.tone, w: 2}] ELSE [] END "
            "       + CASE WHEN p.secondary_tone IS NOT NULL THEN [{v: p.secondary_tone, w: 1}] ELSE [] END "
            "  ) AS tone_entries, "
            "  REDUCE(arr = [], p IN posts | "
            "    arr + CASE WHEN p.primary_hormone IS NOT NULL THEN [{v: p.primary_hormone, w: 2}] ELSE [] END "
            "       + CASE WHEN p.secondary_hormone IS NOT NULL THEN [{v: p.secondary_hormone, w: 1}] ELSE [] END "
            "  ) AS hormone_entries "
            "UNWIND (CASE WHEN size(tone_entries) > 0 THEN tone_entries ELSE [{v: null, w: 0}] END) AS te "
            "WITH a, primary_language, primary_sentiment, hormone_entries, te.v AS tone_val, SUM(te.w) AS tone_score "
            "ORDER BY tone_score DESC "
            "WITH a, primary_language, primary_sentiment, hormone_entries, "
            "  COLLECT(tone_val) AS tone_order, COLLECT(tone_score) AS tone_scores "
            "WITH a, primary_language, primary_sentiment, hormone_entries, tone_order, tone_scores, "
            "  tone_order[0] AS primary_tone, "
            "  tone_order[1] AS secondary_tone_candidate, "
            "  tone_scores[0] AS primary_tone_score, "
            "  tone_scores[1] AS secondary_tone_score, "
            "  REDUCE(s = 0, sc IN tone_scores | s + sc) AS total_tone_score "
            "WITH a, primary_language, primary_sentiment, hormone_entries, "
            "  primary_tone, "
            "  CASE WHEN secondary_tone_candidate IS NOT NULL AND total_tone_score > 0 "
            "       AND toFloat(secondary_tone_score) / total_tone_score >= 0.15 "
            "    THEN secondary_tone_candidate "
            "    ELSE NULL END AS secondary_tone "
            "UNWIND (CASE WHEN size(hormone_entries) > 0 THEN hormone_entries ELSE [{v: null, w: 0}] END) AS he "
            "WITH a, primary_language, primary_sentiment, primary_tone, secondary_tone, "
            "  he.v AS hormone_val, SUM(he.w) AS hormone_score "
            "ORDER BY hormone_score DESC "
            "WITH a, primary_language, primary_sentiment, primary_tone, secondary_tone, "
            "  COLLECT(hormone_val) AS hormone_order, COLLECT(hormone_score) AS hormone_scores "
            "WITH a, primary_language, primary_sentiment, primary_tone, secondary_tone, "
            "  hormone_order, hormone_scores, "
            "  hormone_order[0] AS primary_hormone, "
            "  hormone_order[1] AS secondary_hormone_candidate, "
            "  hormone_scores[0] AS primary_hormone_score, "
            "  hormone_scores[1] AS secondary_hormone_score, "
            "  REDUCE(s = 0, sc IN hormone_scores | s + sc) AS total_hormone_score "
            "WITH a, primary_language, primary_sentiment, primary_tone, secondary_tone, "
            "  primary_hormone, "
            "  CASE WHEN secondary_hormone_candidate IS NOT NULL AND total_hormone_score > 0 "
            "       AND toFloat(secondary_hormone_score) / total_hormone_score >= 0.15 "
            "    THEN secondary_hormone_candidate "
            "    ELSE NULL END AS secondary_hormone "
            "SET a.primary_language = COALESCE(primary_language, a.primary_language, 'ru'), "
            "    a.primary_sentiment = COALESCE(primary_sentiment, a.primary_sentiment), "
            "    a.primary_tone = COALESCE(primary_tone, a.primary_tone), "
            "    a.secondary_tone = secondary_tone, "
            "    a.primary_hormone = COALESCE(primary_hormone, a.primary_hormone), "
            "    a.secondary_hormone = secondary_hormone"
        )
        try:
            await self._neo4j_client.execute_write(profile_query, {"author_id": author_id})
        except Exception as exc:
            logger.warning("Actor profile aggregation failed for author %s: %s", author_id, exc)