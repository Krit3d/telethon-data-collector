from __future__ import annotations

import logging
import math
import re
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update

from src.config.config import Settings
from src.db.database import Database
from src.db.models import Account, Content
from src.graph.client import Neo4jClient
from src.graph.builder.reader import PostBatchContext
from src.graph.ontology import (
    EntityType,
    OpenSPGExtractionResult,
    RelationType,
)
from src.graph.utils import sanitize_properties

logger = logging.getLogger(__name__)


def _to_val(val: Any, default: Any = None) -> Any:
    if val is None:
        return default
    return val.value if hasattr(val, "value") else val


class GraphWriter:

    def __init__(self, settings: Settings, neo4j_client: Neo4jClient, db: Database) -> None:
        self._settings = settings
        self._neo4j_client = neo4j_client
        self._db = db

    async def write_extraction_result(
        self,
        extraction_result: OpenSPGExtractionResult,
        context: PostBatchContext,
        is_first_chunk: bool = True,
    ) -> None:
        await self.write_extraction_results_batch([(extraction_result, context)], is_first_chunk=is_first_chunk)

    async def write_extraction_chunk(
        self,
        extraction_result: OpenSPGExtractionResult,
        context: PostBatchContext,
        is_first_chunk: bool,
    ) -> None:
        if is_first_chunk:
            await self.write_extraction_results_batch([(extraction_result, context)], is_first_chunk=True)
            return

        t0 = time.perf_counter()

        nodes_by_label: dict[str, dict[str, dict[str, Any]]] = {}
        rel_groups: dict[tuple[str, str, str], dict[tuple[str, str], dict[str, Any]]] = {}

        psychographics = extraction_result.psychographics

        post_node = sanitize_properties({
            "id": context.pub_node_id,
            "account_id": context.account_id,
            "content_id": context.content_id,
            "published_at": int(context.published_at.timestamp()),
            "platform": context.platform,
            "post_type": context.post_type,
            "language": psychographics.language,
            "tone": _to_val(psychographics.primary_tone, None),
            "secondary_tone": _to_val(psychographics.secondary_tone, None),
            "tone_confidence": psychographics.tone_confidence,
            "primary_hormone": _to_val(psychographics.primary_hormone, None),
            "secondary_hormone": _to_val(psychographics.secondary_hormone, None),
            "score_dopamine": psychographics.score_dopamine,
            "score_oxytocin": psychographics.score_oxytocin,
            "score_serotonin": psychographics.score_serotonin,
            "score_cortisol": psychographics.score_cortisol,
            "score_adrenaline": psychographics.score_adrenaline,
            "score_endorphin": psychographics.score_endorphin,
            "is_video": context.is_video,
            "is_spam_or_gambling": extraction_result.is_spam_or_gambling,
        })

        nodes_by_label.setdefault("Post", {})[post_node["id"]] = post_node

        published_key = (EntityType.Actor.value, EntityType.Post.value, RelationType.PUBLISHED.value)
        published_rel_dict = rel_groups.setdefault(published_key, {})
        published_pair_key = (context.author_node_id, context.pub_node_id)
        published_rel_dict[published_pair_key] = {
            "source_id": context.author_node_id,
            "target_id": context.pub_node_id,
            "properties": sanitize_properties({"published_at": int(context.published_at.timestamp())}),
        }

        for entity in extraction_result.entities:
            if not entity.id:
                continue
            if entity.label == EntityType.Actor:
                continue
            label_str = entity.label.value
            props = sanitize_properties(dict(entity.properties))
            node_dict = {"id": entity.id, "name": entity.name, "name_lower": entity.name_lower, "mentions_count": 1}
            node_dict.update(props)
            label_dict = nodes_by_label.setdefault(label_str, {})
            existing = label_dict.get(node_dict["id"])
            if existing:
                existing.update(node_dict)
            else:
                label_dict[node_dict["id"]] = node_dict

        for rel in extraction_result.relations:
            key = (rel.source_label.value, rel.target_label.value, rel.relation_type.value)
            rel_dict = rel_groups.setdefault(key, {})
            pair_key = (rel.source_id, rel.target_id)
            rel_props = {
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "properties": sanitize_properties(rel.properties),
            }
            if rel.relation_type == RelationType.BELONGS_TO:
                sim = rel.properties.get("similarity")
                if sim is not None:
                    rel_props["properties"]["similarity"] = float(sim)
            existing = rel_dict.get(pair_key)
            if existing:
                existing["properties"].update(rel_props["properties"])
            else:
                rel_dict[pair_key] = rel_props

        total_nodes = sum(len(v) for v in nodes_by_label.values())
        t1 = time.perf_counter()
        for label_str in sorted(nodes_by_label.keys()):
            node_dict = nodes_by_label[label_str]
            await self._neo4j_client.batch_merge_nodes(label_str, sorted(node_dict.values(), key=lambda x: str(x["id"])))
        node_elapsed = (time.perf_counter() - t1) * 1000

        total_rels = sum(len(v) for v in rel_groups.values())
        t2 = time.perf_counter()
        for rel_key in sorted(rel_groups.keys()):
            src_label, tgt_label, rel_type = rel_key
            rel_dict = rel_groups[rel_key]
            await self._neo4j_client.batch_merge_relations(src_label, tgt_label, rel_type, sorted(rel_dict.values(), key=lambda x: (str(x["source_id"]), str(x["target_id"]))))
        rel_elapsed = (time.perf_counter() - t2) * 1000

        total_elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Chunk writer done in %.1fms | nodes: %d (%d labels, %.1fms) | rels: %d (%d types, %.1fms)",
            total_elapsed, total_nodes, len(nodes_by_label), node_elapsed,
            total_rels, len(rel_groups), rel_elapsed,
        )

    async def write_extraction_results_batch(
        self,
        items: list[tuple[OpenSPGExtractionResult, PostBatchContext]],
        is_first_chunk: bool = True,
    ) -> None:
        if not items:
            return

        t0 = time.perf_counter()

        nodes_by_label: dict[str, dict[str, dict[str, Any]]] = {}
        rel_groups: dict[tuple[str, str, str], dict[tuple[str, str], dict[str, Any]]] = {}
        claimed_ids: list[int] = []
        unique_author_ids: set[str] = set()

        for extraction_result, context in items:
            claimed_ids.append(context.content_id)
            unique_author_ids.add(context.author_node_id)

            psychographics = extraction_result.psychographics

            post_node = sanitize_properties({
                "id": context.pub_node_id,
                "account_id": context.account_id,
                "content_id": context.content_id,
                "published_at": int(context.published_at.timestamp()),
                "platform": context.platform,
                "post_type": context.post_type,
                "language": psychographics.language,
                "tone": _to_val(psychographics.primary_tone, None),
                "secondary_tone": _to_val(psychographics.secondary_tone, None),
                "tone_confidence": psychographics.tone_confidence,
                "primary_hormone": _to_val(psychographics.primary_hormone, None),
                "secondary_hormone": _to_val(psychographics.secondary_hormone, None),
                "score_dopamine": psychographics.score_dopamine,
                "score_oxytocin": psychographics.score_oxytocin,
                "score_serotonin": psychographics.score_serotonin,
                "score_cortisol": psychographics.score_cortisol,
                "score_adrenaline": psychographics.score_adrenaline,
                "score_endorphin": psychographics.score_endorphin,
                "is_video": context.is_video,
                "is_spam_or_gambling": extraction_result.is_spam_or_gambling,
            })

            post_dict = nodes_by_label.setdefault("Post", {})
            existing_post = post_dict.get(post_node["id"])
            if existing_post:
                existing_post.update(post_node)
            else:
                post_dict[post_node["id"]] = post_node

            published_key = (EntityType.Actor.value, EntityType.Post.value, RelationType.PUBLISHED.value)
            published_rel_dict = rel_groups.setdefault(published_key, {})
            published_pair_key = (context.author_node_id, context.pub_node_id)
            published_rel_dict[published_pair_key] = {
                "source_id": context.author_node_id,
                "target_id": context.pub_node_id,
                "properties": sanitize_properties({"published_at": int(context.published_at.timestamp())}),
            }

            for entity in extraction_result.entities:
                if not entity.id:
                    continue
                if entity.label == EntityType.Actor:
                    continue
                label_str = entity.label.value
                props = sanitize_properties(dict(entity.properties))
                node_dict = {"id": entity.id, "name": entity.name, "name_lower": entity.name_lower, "mentions_count": 1}
                node_dict.update(props)
                label_dict = nodes_by_label.setdefault(label_str, {})
                existing = label_dict.get(node_dict["id"])
                if existing:
                    existing.update(node_dict)
                else:
                    label_dict[node_dict["id"]] = node_dict

            for rel in extraction_result.relations:
                key = (rel.source_label.value, rel.target_label.value, rel.relation_type.value)
                rel_dict = rel_groups.setdefault(key, {})
                pair_key = (rel.source_id, rel.target_id)
                rel_props = {
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "properties": sanitize_properties(rel.properties),
                }
                if rel.relation_type == RelationType.BELONGS_TO:
                    sim = rel.properties.get("similarity")
                    if sim is not None:
                        rel_props["properties"]["similarity"] = float(sim)
                existing = rel_dict.get(pair_key)
                if existing:
                    existing["properties"].update(rel_props["properties"])
                else:
                    rel_dict[pair_key] = rel_props

        total_nodes = sum(len(v) for v in nodes_by_label.values())
        t1 = time.perf_counter()
        for label_str in sorted(nodes_by_label.keys()):
            node_dict = nodes_by_label[label_str]
            await self._neo4j_client.batch_merge_nodes(label_str, sorted(node_dict.values(), key=lambda x: str(x["id"])))
        node_elapsed = (time.perf_counter() - t1) * 1000

        total_rels = sum(len(v) for v in rel_groups.values())
        t2 = time.perf_counter()
        for rel_key in sorted(rel_groups.keys()):
            src_label, tgt_label, rel_type = rel_key
            rel_dict = rel_groups[rel_key]
            await self._neo4j_client.batch_merge_relations(src_label, tgt_label, rel_type, sorted(rel_dict.values(), key=lambda x: (str(x["source_id"]), str(x["target_id"]))))
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
            await self._run_post_aggregations(author_id)

    async def _run_post_aggregations(self, author_id: str) -> None:
        covers_query = (
            "MATCH (a:Actor {id: $author_id})-[:PUBLISHED]->(p:Post) "
            "WITH a, COUNT(p) AS total_posts "
            "MATCH (a)-[:PUBLISHED]->(p:Post) "
            "OPTIONAL MATCH (p)-[:ABOUT]->(:MicroConcept)-[:BELONGS_TO]->(concept:Concept) "
            "WITH a, total_posts, concept, COUNT(DISTINCT p) AS post_count "
            "WHERE concept IS NOT NULL AND post_count >= 2 "
            "WITH a, total_posts, concept, post_count "
            "ORDER BY concept.id "
            "MERGE (a)-[r:COVERS_TOPIC]->(concept) "
            "SET r.posts_count = post_count, "
            "    r.weight = toFloat(post_count) / total_posts, "
            "    r.last_updated = timestamp()"
        )
        covers_micro_query = (
            "MATCH (a:Actor {id: $author_id})-[:PUBLISHED]->(p:Post) "
            "WITH a, COUNT(p) AS total_posts "
            "MATCH (a)-[:PUBLISHED]->(p:Post)-[:ABOUT]->(mc:MicroConcept) "
            "WITH a, total_posts, mc, COUNT(DISTINCT p) AS post_count "
            "WHERE post_count >= 2 "
            "WITH a, total_posts, mc, post_count "
            "ORDER BY mc.id "
            "MERGE (a)-[r:COVERS_TOPIC]->(mc) "
            "SET r.posts_count = post_count, "
            "    r.weight = toFloat(post_count) / total_posts, "
            "    r.last_updated = timestamp()"
        )
        try:
            await self._neo4j_client.execute_write(covers_query, {"author_id": author_id})
        except Exception as exc:
            logger.warning("COVERS_TOPIC aggregation failed for author %s: %s", author_id, exc)

        try:
            await self._neo4j_client.execute_write(covers_micro_query, {"author_id": author_id})
        except Exception as exc:
            logger.warning("COVERS_TOPIC MicroConcept aggregation failed for author %s: %s", author_id, exc)

        try:
            await self._aggregate_actor_profile(author_id)
        except Exception as exc:
            logger.warning("Actor profile Python aggregation failed for author %s: %s", author_id, exc)

    async def _aggregate_actor_profile(self, author_id: str) -> None:
        posts_query = (
            "MATCH (a:Actor {id: $author_id})-[:PUBLISHED]->(p:Post) "
            "RETURN a.account_id AS account_id, "
            "       p.published_at AS published_at, "
            "       p.score_dopamine AS score_dopamine, "
            "       p.score_oxytocin AS score_oxytocin, "
            "       p.score_serotonin AS score_serotonin, "
            "       p.score_cortisol AS score_cortisol, "
            "       p.score_adrenaline AS score_adrenaline, "
            "       p.score_endorphin AS score_endorphin, "
            "       p.tone AS tone, "
            "       p.secondary_tone AS secondary_tone, "
            "       p.tone_confidence AS tone_confidence, "
            "       p.language AS language"
        )
        rows = await self._neo4j_client.execute_read(posts_query, {"author_id": author_id})
        if not rows:
            return

        account_id = rows[0].get("account_id")

        now_ts = time.time()
        use_time_decay = self._settings.use_time_decay
        half_life_days = self._settings.time_decay_half_life_days

        sum_dopamine = 0.0
        sum_oxytocin = 0.0
        sum_serotonin = 0.0
        sum_cortisol = 0.0
        sum_adrenaline = 0.0
        sum_endorphin = 0.0

        tone_counter: dict[str, float] = {}
        secondary_tone_counter: dict[str, float] = {}
        language_counter: dict[str, float] = {}

        for row in rows:
            published_at = row.get("published_at")
            w = 1.0
            if use_time_decay:
                if published_at is None:
                    delta_days = 0.0
                else:
                    delta_days = max(0.0, (now_ts - published_at) / 86400.0)
                w = math.exp(-math.log(2) * delta_days / half_life_days)

            sum_dopamine += (row.get("score_dopamine") or 0.0) * w
            sum_oxytocin += (row.get("score_oxytocin") or 0.0) * w
            sum_serotonin += (row.get("score_serotonin") or 0.0) * w
            sum_cortisol += (row.get("score_cortisol") or 0.0) * w
            sum_adrenaline += (row.get("score_adrenaline") or 0.0) * w
            sum_endorphin += (row.get("score_endorphin") or 0.0) * w

            tone_conf = row.get("tone_confidence")
            tone_weight_factor = float(tone_conf) if tone_conf is not None else 1.0
            effective_tone_weight = w * tone_weight_factor

            tone = row.get("tone")
            if tone is not None:
                tone_counter[tone] = tone_counter.get(tone, 0.0) + effective_tone_weight

            secondary_tone = row.get("secondary_tone")
            if secondary_tone is not None:
                secondary_tone_counter[secondary_tone] = secondary_tone_counter.get(secondary_tone, 0.0) + 0.5 * effective_tone_weight

            language = row.get("language")
            if language is not None:
                language_counter[language] = language_counter.get(language, 0.0) + w

        N = len(rows)
        total_hormone_score = (
            sum_dopamine + sum_oxytocin + sum_serotonin
            + sum_cortisol + sum_adrenaline + sum_endorphin
        )
        mean_intensity = total_hormone_score / N

        primary_hormone: str | None = None
        secondary_hormone: str | None = None

        if total_hormone_score > 0 and mean_intensity >= 0.05:
            sum_dopamine /= total_hormone_score
            sum_oxytocin /= total_hormone_score
            sum_serotonin /= total_hormone_score
            sum_cortisol /= total_hormone_score
            sum_adrenaline /= total_hormone_score
            sum_endorphin /= total_hormone_score

            hormone_scores: list[tuple[str, float]] = [
                ("dopamine", sum_dopamine),
                ("oxytocin", sum_oxytocin),
                ("serotonin", sum_serotonin),
                ("cortisol", sum_cortisol),
                ("adrenaline", sum_adrenaline),
                ("endorphin", sum_endorphin),
            ]
            hormone_scores.sort(key=lambda item: item[1], reverse=True)

            primary_hormone = hormone_scores[0][0]
            if len(hormone_scores) > 1 and hormone_scores[1][1] >= 0.15:
                secondary_hormone = hormone_scores[1][0]

        primary_tone = max(tone_counter, key=lambda k: tone_counter[k]) if tone_counter else None

        combined_secondary: dict[str, float] = {}
        for t, count in tone_counter.items():
            if t != primary_tone:
                combined_secondary[t] = combined_secondary.get(t, 0.0) + count * 0.5
        for st, count in secondary_tone_counter.items():
            if st != primary_tone:
                combined_secondary[st] = combined_secondary.get(st, 0.0) + count

        total_tone_mass = sum(tone_counter.values()) + sum(secondary_tone_counter.values())
        secondary_tone: str | None = None
        if combined_secondary and total_tone_mass > 0:
            candidate = max(combined_secondary, key=lambda k: combined_secondary[k])
            candidate_score = combined_secondary[candidate]
            if candidate_score / total_tone_mass >= 0.15:
                secondary_tone = candidate

        primary_language = max(language_counter, key=lambda k: language_counter[k]) if language_counter else None

        if primary_language is None and account_id is not None:
            async with self._db.async_session() as session:
                result = await session.execute(
                    select(Account.description, Account.title).where(Account.id == account_id)
                )
                row = result.one_or_none()
            if row is not None:
                description, title = row
                if description is not None and len(description.split()) > 3:
                    if re.search(r'[а-яА-ЯеЁ]', description):
                        primary_language = 'ru'
                    elif re.search(r'[a-zA-Z]', description):
                        primary_language = 'en'
                if primary_language is None and title is not None:
                    if re.search(r'[а-яА-ЯеЁ]', title):
                        primary_language = 'ru'
                    elif re.search(r'[a-zA-Z]', title):
                        primary_language = 'en'

        update_query = (
            "MATCH (a:Actor {id: $author_id}) "
            "SET a.primary_language = $primary_language, "
            "    a.primary_tone = $primary_tone, "
            "    a.secondary_tone = $secondary_tone, "
            "    a.primary_hormone = $primary_hormone, "
            "    a.secondary_hormone = $secondary_hormone, "
            "    a.updated_at = timestamp()"
        )
        await self._neo4j_client.execute_write(
            update_query,
            {
                "author_id": author_id,
                "primary_language": _to_val(primary_language, None),
                "primary_tone": _to_val(primary_tone, None),
                "secondary_tone": _to_val(secondary_tone, None),
                "primary_hormone": _to_val(primary_hormone, None),
                "secondary_hormone": _to_val(secondary_hormone, None),
            },
        )