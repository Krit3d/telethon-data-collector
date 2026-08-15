from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict

from openai import AsyncOpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from src.config.config import Settings
from src.graph.builder.reader import PostBatchContext
from src.graph.client import Neo4jClient
from src.graph.ontology import (
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    HashtagItem,
    OpenSPGExtractionResult,
    RelationType,
    VECTORIZABLE_ENTITY_LABELS,
    extract_entity_subtype,
)
from src.graph.utils import build_node_id, clean_name_lower, format_bge_representation

logger = logging.getLogger(__name__)

_L1_CACHE_MAX = 50_000
_EMBEDDING_RETRIES = 3
_ENTITY_SCORE_THRESHOLD = 0.88
_CATEGORY_SCORE_GAP = 0.03
_CATEGORY_MIN_SCORE = 0.55
_HASHTAG_ENTITY_SCORE_THRESHOLD = 0.85

_CANONICAL_ENTITY_LABELS: frozenset[str] = frozenset({
    EntityType.Product.value,
    EntityType.Organization.value,
    EntityType.Entity.value,
    EntityType.Event.value,
})


class Aligner:

    def __init__(self, settings: Settings, neo4j_client: Neo4jClient) -> None:
        self._openai_client = AsyncOpenAI(
            base_url=settings.cloud_ru_base_url,
            api_key=settings.cloud_ru_api_key,
        )
        self._embedding_model = settings.cloud_ru_embedding_model
        self._qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self._neo4j_client = neo4j_client
        self._l1_cache: OrderedDict[str, str] = OrderedDict()

    def _cache_set(self, key: str, value: str) -> None:
        self._l1_cache[key] = value
        self._l1_cache.move_to_end(key)
        while len(self._l1_cache) > _L1_CACHE_MAX:
            self._l1_cache.popitem(last=False)

    async def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        t0 = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(_EMBEDDING_RETRIES):
            try:
                response = await self._openai_client.embeddings.create(
                    model=self._embedding_model,
                    input=texts,
                )
                ordered = sorted(response.data, key=lambda item: item.index)
                elapsed = (time.perf_counter() - t0) * 1000
                logger.debug("BGE-M3 embedding batch of %d texts done in %.1fms", len(texts), elapsed)
                return [item.embedding for item in ordered]
            except (APIError, APITimeoutError, APIConnectionError, RateLimitError) as exc:
                last_error = exc
                if attempt < _EMBEDDING_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
        raise last_error or RuntimeError("Embedding generation failed after all retries")

    async def _disambiguate_entities(
        self,
        missed: list[ExtractedEntity],
        id_map: dict[str, str],
    ) -> None:
        if not missed:
            return
        bge_texts: list[str] = []
        for e in missed:
            _, subtype = extract_entity_subtype(e.label, e.properties)
            bge_texts.append(format_bge_representation(e.label.value, e.name, subtype))
        try:
            embeddings = await self._get_embeddings_batch(bge_texts)
        except Exception:
            logger.warning("Embedding generation failed in _disambiguate_entities, skipping")
            return
        if len(embeddings) != len(missed):
            return
        for entity, emb in zip(missed, embeddings):
            entity.embedding = emb
        t0 = time.perf_counter()
        try:
            batch_response = await self._qdrant_client.query_batch_points(
                collection_name="social_entities",
                requests=[
                    models.QueryRequest(
                        query=emb,
                        using="text",
                        limit=1,
                        score_threshold=_ENTITY_SCORE_THRESHOLD,
                        with_payload=True,
                    )
                    for emb in embeddings
                ],
            )
        except Exception:
            logger.warning("Qdrant query_batch_points failed in _disambiguate_entities, skipping")
            return
        qdrant_elapsed = (time.perf_counter() - t0) * 1000
        resolved = 0
        for entity, response in zip(missed, batch_response):
            if not entity.id or entity.id in id_map:
                continue
            if not response.points:
                continue
            hit = response.points[0]
            if hit.score < _ENTITY_SCORE_THRESHOLD:
                continue
            payload = hit.payload or {}
            canonical_label = payload.get("label") or payload.get("Label")
            canonical_id = (
                payload.get("id")
                or payload.get("original_id")
                or str(hit.id)
            )
            if not canonical_id:
                continue
            key = f"{entity.label.value}:{clean_name_lower(entity.name)}"
            self._cache_set(key, canonical_id)
            id_map[entity.id] = canonical_id
            resolved += 1
        logger.debug(
            "Qdrant social_entities search for %d entities done in %.1fms, resolved %d",
            len(missed), qdrant_elapsed, resolved,
        )

    async def _link_microconcepts(
        self,
        microconcepts: list[ExtractedEntity],
        context: PostBatchContext,
        relations: list[ExtractedRelation],
        entities: list[ExtractedEntity],
    ) -> None:
        if not microconcepts:
            return
        candidates = [
            mc for mc in microconcepts
            if mc.id and mc.properties.get("is_classified") is not True
        ]
        if not candidates:
            return
        seen_ids: set[str] = set()
        unique: list[ExtractedEntity] = []
        for mc in candidates:
            mc_id = mc.id
            if not mc_id or mc_id in seen_ids:
                continue
            seen_ids.add(mc_id)
            unique.append(mc)
        names = [mc.name for mc in unique]
        try:
            embeddings = await self._get_embeddings_batch(names)
        except Exception:
            logger.warning("Embedding generation failed in _link_microconcepts, skipping")
            return
        if len(embeddings) != len(unique):
            return
        t0 = time.perf_counter()
        try:
            batch_response = await self._qdrant_client.query_batch_points(
                collection_name="categories",
                requests=[
                    models.QueryRequest(
                        query=emb,
                        using="text",
                        limit=3,
                        with_payload=True,
                    )
                    for emb in embeddings
                ],
            )
        except Exception:
            logger.warning("Qdrant query_batch_points failed in _link_microconcepts, skipping")
            return
        qdrant_elapsed = (time.perf_counter() - t0) * 1000
        linked = 0
        for mc, response in zip(unique, batch_response):
            mc_id = mc.id
            if not mc_id or not response.points:
                continue
            hit_candidates = [
                {"score": hit.score, "payload": hit.payload or {}}
                for hit in response.points
            ]
            chosen = hit_candidates[0]
            if float(chosen["score"]) < _CATEGORY_MIN_SCORE:
                continue
            if (
                len(hit_candidates) >= 2
                and (hit_candidates[0]["score"] - hit_candidates[1]["score"]) < _CATEGORY_SCORE_GAP
                and context.account_category_id
            ):
                matched = next(
                    (c for c in hit_candidates if str(c["payload"].get("code")).strip().lower() == str(context.account_category_id).strip().lower()),
                    None,
                )
                if matched is not None:
                    chosen = matched
            code = chosen["payload"].get("code")
            if not code:
                continue
            concept_node_id = build_node_id(EntityType.Concept, code)
            concept_name = str(chosen["payload"].get("name") or code)
            tier_1 = chosen["payload"].get("tier_1")
            tier_2 = chosen["payload"].get("tier_2")
            tier_3 = chosen["payload"].get("tier_3")
            tier_4 = chosen["payload"].get("tier_4")
            extension = chosen["payload"].get("extension")
            entities.append(
                ExtractedEntity(
                    id=concept_node_id,
                    name=concept_name,
                    label=EntityType.Concept,
                    properties={
                        "code": code,
                        "tier_1": tier_1,
                        "tier_2": tier_2,
                        "tier_3": tier_3,
                        "tier_4": tier_4,
                        "extension": extension,
                    },
                )
            )
            mc.properties["is_classified"] = True
            key = f"{EntityType.MicroConcept.value}:{clean_name_lower(mc.name)}"
            self._cache_set(key, mc_id)
            relations.append(
                ExtractedRelation(
                    source_id=mc_id,
                    source_label=EntityType.MicroConcept,
                    target_id=concept_node_id,
                    target_label=EntityType.Concept,
                    relation_type=RelationType.BELONGS_TO,
                    properties={"similarity": round(float(chosen["score"]), 4)},
                )
            )
            linked += 1
        logger.debug(
            "Qdrant iab_categories search for %d microconcepts done in %.1fms, linked %d",
            len(unique), qdrant_elapsed, linked,
        )

    async def _link_author_category(
        self,
        extraction_result: OpenSPGExtractionResult,
        context: PostBatchContext,
    ) -> None:
        if not context.account_category_id:
            return
        if context.account_category_path:
            parts = [p.strip() for p in context.account_category_path.split(" > ")]
        else:
            parts = [context.account_category_id]
        name = parts[-1]
        concept_node_id = build_node_id(EntityType.Concept, context.account_category_id)
        properties: dict[str, str] = {"code": context.account_category_id}
        if len(parts) > 0:
            properties["tier_1"] = parts[0]
        if len(parts) > 1:
            properties["tier_2"] = parts[1]
        if len(parts) > 2:
            properties["tier_3"] = parts[2]
        if len(parts) > 3:
            properties["tier_4"] = parts[3]
        extraction_result.entities.append(
            ExtractedEntity(
                id=concept_node_id,
                name=name,
                label=EntityType.Concept,
                properties=properties,
            )
        )
        extraction_result.relations.append(
            ExtractedRelation(
                source_id=context.author_node_id,
                source_label=EntityType.Actor,
                target_id=concept_node_id,
                target_label=EntityType.Concept,
                relation_type=RelationType.BELONGS_TO,
                properties={},
            )
        )

    async def _link_hashtags(
        self,
        extraction_result: OpenSPGExtractionResult,
        context: PostBatchContext,
    ) -> None:
        if not extraction_result.hashtags:
            return

        seen_raw: set[str] = set()
        unique: list[HashtagItem] = []
        for ht in extraction_result.hashtags:
            if ht.raw in seen_raw:
                continue
            seen_raw.add(ht.raw)
            unique.append(ht)

        for ht in unique:
            ht_node_id = build_node_id(EntityType.Hashtag, ht.raw)
            extraction_result.entities.append(
                ExtractedEntity(
                    id=ht_node_id,
                    name=f"#{ht.raw}",
                    label=EntityType.Hashtag,
                    properties={"raw": ht.raw, "normalized": ht.normalized},
                )
            )
            if context.pub_node_id:
                extraction_result.relations.append(
                    ExtractedRelation(
                        source_id=context.pub_node_id,
                        source_label=EntityType.Post,
                        target_id=ht_node_id,
                        target_label=EntityType.Hashtag,
                        relation_type=RelationType.TAGGED_WITH,
                    )
                )

        normalized_texts = [ht.normalized for ht in unique]
        try:
            embeddings = await self._get_embeddings_batch(normalized_texts)
        except Exception:
            logger.warning("Embedding generation failed in _link_hashtags, skipping")
            return

        if len(embeddings) != len(unique):
            return

        t0 = time.perf_counter()

        async def _query_social_entities() -> list[models.QueryResponse] | None:
            try:
                return await self._qdrant_client.query_batch_points(
                    collection_name="social_entities",
                    requests=[
                        models.QueryRequest(
                            query=emb,
                            using="text",
                            limit=1,
                            score_threshold=_HASHTAG_ENTITY_SCORE_THRESHOLD,
                            with_payload=True,
                        )
                        for emb in embeddings
                    ],
                )
            except Exception:
                logger.warning("Qdrant social_entities query failed in _link_hashtags, skipping")
                return None

        async def _query_categories() -> list[models.QueryResponse] | None:
            try:
                return await self._qdrant_client.query_batch_points(
                    collection_name="categories",
                    requests=[
                        models.QueryRequest(
                            query=emb,
                            using="text",
                            limit=1,
                            score_threshold=_CATEGORY_MIN_SCORE,
                            with_payload=True,
                        )
                        for emb in embeddings
                    ],
                )
            except Exception:
                logger.warning("Qdrant categories query failed in _link_hashtags, skipping")
                return None

        social_resp, cat_resp = await asyncio.gather(
            _query_social_entities(),
            _query_categories(),
        )

        qdrant_elapsed = (time.perf_counter() - t0) * 1000

        existing_entity_ids: set[str] = {e.id for e in extraction_result.entities if e.id}
        existing_concept_ids: set[str] = {
            e.id for e in extraction_result.entities
            if e.id and e.label == EntityType.Concept
        }

        maps_to_count = 0
        belongs_to_count = 0

        if social_resp is not None:
            for ht, response in zip(unique, social_resp):
                if not response.points:
                    continue
                hit = response.points[0]
                if hit.score < _HASHTAG_ENTITY_SCORE_THRESHOLD:
                    continue
                payload = hit.payload or {}
                canonical_label = payload.get("label") or payload.get("Label")
                if not canonical_label or canonical_label not in _CANONICAL_ENTITY_LABELS:
                    continue
                canonical_id = (
                    payload.get("id")
                    or payload.get("original_id")
                    or str(hit.id)
                )
                if not canonical_id:
                    continue
                try:
                    target_label = EntityType(canonical_label)
                except ValueError:
                    continue
                canonical_name = str(payload.get("name") or payload.get("title") or canonical_id)
                ht_node_id = build_node_id(EntityType.Hashtag, ht.raw)
                if canonical_id not in existing_entity_ids:
                    extraction_result.entities.append(
                        ExtractedEntity(
                            id=canonical_id,
                            name=canonical_name,
                            label=target_label,
                        )
                    )
                    existing_entity_ids.add(canonical_id)
                extraction_result.relations.append(
                    ExtractedRelation(
                        source_id=ht_node_id,
                        source_label=EntityType.Hashtag,
                        target_id=canonical_id,
                        target_label=target_label,
                        relation_type=RelationType.MAPS_TO,
                        properties={},
                    )
                )
                maps_to_count += 1

        if cat_resp is not None:
            for ht, response in zip(unique, cat_resp):
                if not response.points:
                    continue
                hit = response.points[0]
                if hit.score < _CATEGORY_MIN_SCORE:
                    continue
                payload = hit.payload or {}
                code = payload.get("code")
                if not code:
                    continue
                concept_node_id = build_node_id(EntityType.Concept, code)
                ht_node_id = build_node_id(EntityType.Hashtag, ht.raw)
                if concept_node_id not in existing_concept_ids:
                    concept_name = str(payload.get("name") or code)
                    tier_1 = payload.get("tier_1")
                    tier_2 = payload.get("tier_2")
                    tier_3 = payload.get("tier_3")
                    tier_4 = payload.get("tier_4")
                    extension = payload.get("extension")
                    extraction_result.entities.append(
                        ExtractedEntity(
                            id=concept_node_id,
                            name=concept_name,
                            label=EntityType.Concept,
                            properties={
                                "code": code,
                                "tier_1": tier_1,
                                "tier_2": tier_2,
                                "tier_3": tier_3,
                                "tier_4": tier_4,
                                "extension": extension,
                            },
                        )
                    )
                    existing_concept_ids.add(concept_node_id)
                extraction_result.relations.append(
                    ExtractedRelation(
                        source_id=ht_node_id,
                        source_label=EntityType.Hashtag,
                        target_id=concept_node_id,
                        target_label=EntityType.Concept,
                        relation_type=RelationType.BELONGS_TO,
                        properties={"similarity": round(hit.score, 4)},
                    )
                )
                belongs_to_count += 1

        logger.debug(
            "Qdrant hashtag search for %d hashtags done in %.1fms | MAPS_TO: %d, BELONGS_TO: %d",
            len(unique), qdrant_elapsed, maps_to_count, belongs_to_count,
        )

    async def align(
        self,
        extraction_result: OpenSPGExtractionResult,
        context: PostBatchContext,
    ) -> OpenSPGExtractionResult:
        t0 = time.perf_counter()
        id_map: dict[str, str] = {}

        l1_hits = 0
        missed: list[ExtractedEntity] = []
        for entity in extraction_result.entities:
            if not entity.id:
                continue
            key = f"{entity.label.value}:{clean_name_lower(entity.name)}"
            cached = self._l1_cache.get(key)
            if cached is not None:
                id_map[entity.id] = cached
                self._l1_cache.move_to_end(key)
                if entity.label == EntityType.MicroConcept:
                    entity.properties["is_classified"] = True
                l1_hits += 1
            else:
                missed.append(entity)

        neo4j_elapsed = 0.0
        if missed:
            missed_ids = [e.id for e in missed if e.id]
            t1 = time.perf_counter()
            existing_ids = await self._neo4j_client.lookup_existing_ids(missed_ids)
            neo4j_elapsed = (time.perf_counter() - t1) * 1000
            for entity in missed:
                if entity.id and entity.id in existing_ids:
                    key = f"{entity.label.value}:{clean_name_lower(entity.name)}"
                    self._cache_set(key, entity.id)
                    id_map[entity.id] = entity.id
                    if entity.label == EntityType.MicroConcept:
                        entity.properties["is_classified"] = True

        still_missed = [
            e for e in missed
            if e.id and e.id not in id_map and e.label in VECTORIZABLE_ENTITY_LABELS
        ]
        t_qdrant_entities = time.perf_counter()
        await self._disambiguate_entities(still_missed, id_map)
        qdrant_entities_elapsed = (time.perf_counter() - t_qdrant_entities) * 1000

        microconcepts = [
            e for e in extraction_result.entities
            if e.label == EntityType.MicroConcept
        ]
        unique_microconcepts: list[ExtractedEntity] = []
        seen_microconcept_ids: set[str] = set()
        for mc in microconcepts:
            if not mc.id:
                continue
            canonical_id = id_map.get(mc.id, mc.id)
            if canonical_id in seen_microconcept_ids:
                continue
            seen_microconcept_ids.add(canonical_id)
            if mc.id in id_map:
                mc.id = canonical_id
            unique_microconcepts.append(mc)

        t_qdrant_categories = time.perf_counter()
        await self._link_microconcepts(
            unique_microconcepts,
            context,
            extraction_result.relations,
            extraction_result.entities,
        )
        qdrant_categories_elapsed = (time.perf_counter() - t_qdrant_categories) * 1000

        await self._link_author_category(extraction_result, context)

        await self._link_hashtags(extraction_result, context)

        if id_map:
            for entity in extraction_result.entities:
                if entity.id and entity.id in id_map:
                    entity.id = id_map[entity.id]

        deduped_entities: list[ExtractedEntity] = []
        seen_entity_ids: set[str] = set()
        for entity in extraction_result.entities:
            if not entity.id:
                continue
            if entity.id in seen_entity_ids:
                continue
            seen_entity_ids.add(entity.id)
            deduped_entities.append(entity)
        extraction_result.entities = deduped_entities

        if id_map:
            for relation in extraction_result.relations:
                if relation.source_id in id_map:
                    relation.source_id = id_map[relation.source_id]
                if relation.target_id in id_map:
                    relation.target_id = id_map[relation.target_id]

        deduped_relations: list[ExtractedRelation] = []
        seen_relation_keys: set[tuple[str, str, str]] = set()
        for relation in extraction_result.relations:
            if relation.source_id == relation.target_id:
                continue
            key = (relation.source_id, relation.target_id, relation.relation_type.value)
            if key in seen_relation_keys:
                continue
            seen_relation_keys.add(key)
            deduped_relations.append(relation)
        extraction_result.relations = deduped_relations

        total_elapsed = (time.perf_counter() - t0) * 1000
        l1_miss = len(missed)
        logger.debug(
            "Aligner done in %.1fms | L1 cache: %d hit, %d miss | Neo4j lookup: %.1fms | Qdrant social_entities: %.1fms | Qdrant categories: %.1fms | entities: %d, relations: %d",
            total_elapsed, l1_hits, l1_miss, neo4j_elapsed,
            qdrant_entities_elapsed, qdrant_categories_elapsed,
            len(extraction_result.entities), len(extraction_result.relations),
        )
        return extraction_result
