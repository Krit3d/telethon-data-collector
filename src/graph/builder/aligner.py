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
    OpenSPGExtractionResult,
    RelationType,
)
from src.graph.utils import clean_name_lower, format_bge_representation

logger = logging.getLogger(__name__)

_L1_CACHE_MAX = 50_000
_EMBEDDING_RETRIES = 3
_ENTITY_SCORE_THRESHOLD = 0.88
_CATEGORY_SCORE_GAP = 0.03


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
        bge_texts = [
            format_bge_representation(e.label.value, e.name, e.properties)
            for e in missed
        ]
        try:
            embeddings = await self._get_embeddings_batch(bge_texts)
        except Exception:
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
        names = [mc.name for mc in microconcepts]
        try:
            embeddings = await self._get_embeddings_batch(names)
        except Exception:
            return
        if len(embeddings) != len(microconcepts):
            return
        t0 = time.perf_counter()
        try:
            batch_response = await self._qdrant_client.query_batch_points(
                collection_name="iab_categories",
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
            return
        qdrant_elapsed = (time.perf_counter() - t0) * 1000
        linked = 0
        for mc, response in zip(microconcepts, batch_response):
            if not mc.id:
                continue
            if not response.points:
                continue
            candidates = [
                {"score": hit.score, "payload": hit.payload or {}}
                for hit in response.points
            ]
            chosen = candidates[0]
            if (
                len(candidates) >= 2
                and (candidates[0]["score"] - candidates[1]["score"]) < _CATEGORY_SCORE_GAP
                and context.account_category_id
            ):
                matched = next(
                    (c for c in candidates if c["payload"].get("code") == context.account_category_id),
                    None,
                )
                if matched is not None:
                    chosen = matched
            code = chosen["payload"].get("code")
            if not code:
                continue
            concept_node_id = f"concept_{code}"
            concept_name = str(chosen["payload"].get("name") or code)
            tier_1 = chosen["payload"].get("tier_1")
            entities.append(
                ExtractedEntity(
                    id=concept_node_id,
                    name=concept_name,
                    name_lower=clean_name_lower(concept_name),
                    label=EntityType.Concept,
                    properties={
                        "code": code,
                        "tier_1": tier_1,
                    },
                )
            )
            mc.properties["is_classified"] = True
            relations.append(
                ExtractedRelation(
                    source_id=mc.id,
                    source_label=EntityType.MicroConcept,
                    target_id=concept_node_id,
                    target_label=EntityType.Concept,
                    relation_type=RelationType.BELONGS_TO,
                )
            )
            linked += 1
        logger.debug(
            "Qdrant iab_categories search for %d microconcepts done in %.1fms, linked %d",
            len(microconcepts), qdrant_elapsed, linked,
        )

    async def align(
        self,
        extraction_result: OpenSPGExtractionResult,
        context: PostBatchContext,
    ) -> OpenSPGExtractionResult:
        t0 = time.perf_counter()
        id_map: dict[str, str] = {}

        non_micro = [
            e for e in extraction_result.entities
            if e.label != EntityType.MicroConcept
        ]

        l1_hits = 0
        missed: list[ExtractedEntity] = []
        for entity in non_micro:
            if not entity.id:
                continue
            key = f"{entity.label.value}:{clean_name_lower(entity.name)}"
            cached = self._l1_cache.get(key)
            if cached is not None:
                id_map[entity.id] = cached
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

        still_missed = [
            e for e in missed
            if e.id and e.id not in id_map
        ]
        await self._disambiguate_entities(still_missed, id_map)

        microconcepts = [
            e for e in extraction_result.entities
            if e.label == EntityType.MicroConcept
        ]
        await self._link_microconcepts(microconcepts, context, extraction_result.relations, extraction_result.entities)

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
            "Aligner done in %.1fms | L1 cache: %d hit, %d miss | Neo4j lookup: %.1fms | entities: %d, relations: %d",
            total_elapsed, l1_hits, l1_miss, neo4j_elapsed,
            len(extraction_result.entities), len(extraction_result.relations),
        )
        return extraction_result