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
from src.embeddings.client import CATEGORIES_COLLECTION, ENTITIES_COLLECTION
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
from src.graph.utils import build_node_id, clean_identifier, clean_name_lower, format_bge_representation, format_display_name

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
    EntityType.Hashtag.value,
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
        self._classified_mc_ids: set[str] = set()

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
        except Exception as exc:
            logger.warning("Embedding generation failed in _disambiguate_entities: %s", exc)
            return
        if len(embeddings) != len(missed):
            return
        for entity, emb in zip(missed, embeddings):
            entity.embedding = emb
        t0 = time.perf_counter()
        try:
            batch_response = await self._qdrant_client.query_batch_points(
                collection_name=ENTITIES_COLLECTION,
                requests=[
                    models.QueryRequest(
                        query=emb,
                        using="text",
                        filter=models.Filter(must=[models.FieldCondition(key="label", match=models.MatchValue(value=entity.label.value))]),
                        limit=1,
                        score_threshold=_ENTITY_SCORE_THRESHOLD,
                        with_payload=True,
                    )
                    for entity, emb in zip(missed, embeddings)
                ],
            )
        except Exception as exc:
            logger.warning("Qdrant query_batch_points failed in _disambiguate_entities: %s", exc)
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
            canonical_name = payload.get("name") or payload.get("title")
            if not canonical_name:
                continue
            canonical_name = format_display_name(canonical_name, entity.label)
            canonical_id = build_node_id(entity.label, canonical_name)
            entity.name = canonical_name
            entity.name_lower = clean_name_lower(canonical_name)
            key = f"{entity.label.value}:{clean_name_lower(canonical_name)}"
            self._cache_set(key, canonical_id)
            id_map[entity.id] = canonical_id
            resolved += 1
        logger.debug(
            "Qdrant social_entities search for %d entities done in %.1fms, resolved %d",
            len(missed), qdrant_elapsed, resolved,
        )

    async def _resolve_actor_mentions(
        self,
        extraction_result: OpenSPGExtractionResult,
        context: PostBatchContext,
        id_map: dict[str, str],
    ) -> None:
        candidates: list[ExtractedEntity] = []
        for entity in extraction_result.entities:
            if not entity.id or entity.id in id_map:
                continue
            if entity.label == EntityType.Entity and entity.properties.get("entity_type") == "person":
                candidates.append(entity)
            elif "@" in entity.name:
                candidates.append(entity)
        if not candidates:
            return

        resolved_ids: set[str] = set()
        resolved_count = 0
        for entity in candidates:
            entity_id = entity.id
            if not entity_id:
                continue
            actor_id: str | None = None
            name_lower = clean_name_lower(entity.name)
            cache_key = f"Actor:{name_lower}"
            cached = self._l1_cache.get(cache_key)
            if cached is not None:
                actor_id = cached
                self._l1_cache.move_to_end(cache_key)
            else:
                handle: str | None = None
                if "@" in entity.name:
                    raw = entity.name.strip()
                    if raw.startswith("@"):
                        handle = raw[1:].strip()
                    else:
                        at_idx = raw.find("@")
                        if at_idx >= 0:
                            handle = raw[at_idx + 1:].strip().split()[0]
                try:
                    if handle:
                        rows = await self._neo4j_client.execute_read(
                            "MATCH (a:Actor) WHERE a.handle = $handle RETURN a.id AS id LIMIT 1",
                            {"handle": handle},
                        )
                        if rows:
                            raw_id = rows[0].get("id")
                            if isinstance(raw_id, str):
                                actor_id = raw_id
                                self._cache_set(cache_key, actor_id)
                    if not actor_id:
                        rows = await self._neo4j_client.execute_read(
                            "MATCH (a:Actor) WHERE a.name_lower = $name_lower RETURN a.id AS id LIMIT 1",
                            {"name_lower": name_lower},
                        )
                        if rows:
                            raw_id = rows[0].get("id")
                            if isinstance(raw_id, str):
                                actor_id = raw_id
                                self._cache_set(cache_key, actor_id)
                except Exception as exc:
                    logger.warning("Neo4j Actor lookup failed for %s: %s", entity.name, exc)
                    continue
            if not actor_id:
                continue
            resolved_id: str = actor_id
            id_map[entity_id] = resolved_id
            entity.id = resolved_id
            entity.label = EntityType.Actor
            resolved_ids.add(entity_id)
            resolved_count += 1

        if resolved_ids:
            extraction_result.entities = [
                e for e in extraction_result.entities
                if e.id not in resolved_ids
            ]

        if resolved_count:
            logger.debug("Actor mention resolution: resolved %d entities to Actor nodes", resolved_count)

    async def _link_microconcepts(
        self,
        microconcepts: list[ExtractedEntity],
        context: PostBatchContext,
        relations: list[ExtractedRelation],
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
        except Exception as exc:
            logger.warning("Embedding generation failed in _link_microconcepts: %s", exc)
            return
        if len(embeddings) != len(unique):
            return
        t0 = time.perf_counter()
        try:
            batch_response = await self._qdrant_client.query_batch_points(
                collection_name=CATEGORIES_COLLECTION,
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
        except Exception as exc:
            logger.warning("Qdrant query_batch_points failed in _link_microconcepts: %s", exc)
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
            mc.properties["is_classified"] = True
            self._classified_mc_ids.add(mc_id)
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

        existing_entity_ids: set[str] = {e.id for e in extraction_result.entities if e.id}
        existing_relation_keys: set[tuple[str, str, str]] = {
            (r.source_id, r.target_id, r.relation_type.value)
            for r in extraction_result.relations
            if r.source_id and r.target_id
        }

        for ht in unique:
            ht_node_id = build_node_id(EntityType.Hashtag, ht.raw)
            if ht_node_id not in existing_entity_ids:
                extraction_result.entities.append(
                    ExtractedEntity(
                        id=ht_node_id,
                        name=f"#{ht.raw}",
                        name_lower=clean_name_lower(ht.raw),
                        label=EntityType.Hashtag,
                        properties={"raw": ht.raw, "normalized": ht.normalized},
                    )
                )
                existing_entity_ids.add(ht_node_id)
            if context.pub_node_id:
                tagged_key = (context.pub_node_id, ht_node_id, RelationType.TAGGED_WITH.value)
                if tagged_key not in existing_relation_keys:
                    extraction_result.relations.append(
                        ExtractedRelation(
                            source_id=context.pub_node_id,
                            source_label=EntityType.Post,
                            target_id=ht_node_id,
                            target_label=EntityType.Hashtag,
                            relation_type=RelationType.TAGGED_WITH,
                        )
                    )
                    existing_relation_keys.add(tagged_key)

        normalized_texts = [ht.normalized for ht in unique]
        try:
            embeddings = await self._get_embeddings_batch(normalized_texts)
        except Exception as exc:
            logger.warning("Embedding generation failed in _link_hashtags: %s", exc)
            return

        if len(embeddings) != len(unique):
            return

        t0 = time.perf_counter()

        async def _query_social_entities() -> list[models.QueryResponse] | None:
            try:
                return await self._qdrant_client.query_batch_points(
                    collection_name=ENTITIES_COLLECTION,
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
            except Exception as exc:
                logger.warning("Qdrant social_entities query failed in _link_hashtags: %s", exc)
                return None

        async def _query_categories() -> list[models.QueryResponse] | None:
            try:
                return await self._qdrant_client.query_batch_points(
                    collection_name=CATEGORIES_COLLECTION,
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
            except Exception as exc:
                logger.warning("Qdrant categories query failed in _link_hashtags: %s", exc)
                return None

        social_resp, cat_resp = await asyncio.gather(
            _query_social_entities(),
            _query_categories(),
        )

        qdrant_elapsed = (time.perf_counter() - t0) * 1000

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
                try:
                    target_label = EntityType(canonical_label)
                except ValueError:
                    continue
                canonical_name = format_display_name(str(payload.get("name") or payload.get("title") or canonical_label), target_label)
                canonical_id = build_node_id(target_label, canonical_name)
                ht_node_id = build_node_id(EntityType.Hashtag, ht.raw)
                if canonical_id not in existing_entity_ids:
                    match target_label:
                        case EntityType.Product:
                            type_val = payload.get("product_type")
                            if not type_val:
                                logger.warning("Hashtag '%s' -> Product: missing 'product_type' in payload, skipping MAPS_TO for canonical '%s'", ht.raw, canonical_name)
                                continue
                            props: dict[str, object] = {"product_type": type_val}
                        case EntityType.Event:
                            type_val = payload.get("event_type")
                            if not type_val:
                                logger.warning("Hashtag '%s' -> Event: missing 'event_type' in payload, skipping MAPS_TO for canonical '%s'", ht.raw, canonical_name)
                                continue
                            props = {"event_type": type_val}
                        case EntityType.Organization:
                            type_val = payload.get("org_type")
                            if not type_val:
                                logger.warning("Hashtag '%s' -> Organization: missing 'org_type' in payload, skipping MAPS_TO for canonical '%s'", ht.raw, canonical_name)
                                continue
                            props = {"org_type": type_val}
                        case EntityType.Entity:
                            type_val = payload.get("entity_type")
                            if not type_val:
                                logger.warning("Hashtag '%s' -> Entity: missing 'entity_type' in payload, skipping MAPS_TO for canonical '%s'", ht.raw, canonical_name)
                                continue
                            props = {"entity_type": type_val}
                        case _:
                            props = {}
                    extraction_result.entities.append(
                        ExtractedEntity(
                            id=canonical_id,
                            name=canonical_name,
                            name_lower=clean_name_lower(canonical_name),
                            label=target_label,
                            properties=props,
                        )
                    )
                    existing_entity_ids.add(canonical_id)
                maps_to_key = (ht_node_id, canonical_id, RelationType.MAPS_TO.value)
                if maps_to_key not in existing_relation_keys:
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
                    existing_relation_keys.add(maps_to_key)
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
                belongs_to_key = (ht_node_id, concept_node_id, RelationType.BELONGS_TO.value)
                if belongs_to_key not in existing_relation_keys:
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
                    existing_relation_keys.add(belongs_to_key)
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

        t_actor_resolve = time.perf_counter()
        await self._resolve_actor_mentions(extraction_result, context, id_map)
        actor_resolve_elapsed = (time.perf_counter() - t_actor_resolve) * 1000

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
            if canonical_id in self._classified_mc_ids:
                mc.properties["is_classified"] = True
            unique_microconcepts.append(mc)

        t_qdrant_categories = time.perf_counter()
        await self._link_microconcepts(
            unique_microconcepts,
            context,
            extraction_result.relations,
        )
        qdrant_categories_elapsed = (time.perf_counter() - t_qdrant_categories) * 1000

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

        id_to_label: dict[str, EntityType] = {}
        for entity in extraction_result.entities:
            if entity.id:
                id_to_label[entity.id] = entity.label
        if context.author_node_id:
            id_to_label[context.author_node_id] = EntityType.Actor
        if context.pub_node_id:
            id_to_label[context.pub_node_id] = EntityType.Post
        for relation in extraction_result.relations:
            resolved_source = id_to_label.get(relation.source_id)
            if resolved_source is not None:
                relation.source_label = resolved_source
            resolved_target = id_to_label.get(relation.target_id)
            if resolved_target is not None:
                relation.target_label = resolved_target

        deduped_relations: list[ExtractedRelation] = []
        seen_relation_keys: dict[tuple[str, str, str], ExtractedRelation] = {}
        for relation in extraction_result.relations:
            if relation.source_id == relation.target_id:
                continue
            key = (relation.source_id, relation.target_id, relation.relation_type.value)
            existing = seen_relation_keys.get(key)
            if existing is not None:
                for k, v in relation.properties.items():
                    if v not in (None, "", {}, []) and k not in existing.properties:
                        existing.properties[k] = v
                continue
            seen_relation_keys[key] = relation
            deduped_relations.append(relation)
        extraction_result.relations = deduped_relations

        concept_ids: set[str] = set()
        for relation in extraction_result.relations:
            if relation.relation_type == RelationType.BELONGS_TO and relation.target_label == EntityType.Concept and relation.target_id:
                concept_ids.add(relation.target_id)
        for entity in extraction_result.entities:
            if entity.label == EntityType.Concept and entity.id:
                concept_ids.add(entity.id)

        coauthor_ids = {
            f"actor_{context.platform.lower()}_{clean_identifier(ca)}"
            for ca in context.post_coauthors
            if clean_identifier(ca)
        }
        author_handle = (getattr(context, "author_handle", "") or getattr(context, "author_username", "") or "").strip()
        forbidden_names = {clean_name_lower(context.author_title)}
        if author_handle:
            forbidden_names.add(clean_name_lower(author_handle))
        extraction_result.sanitize_and_validate(
            allowed_ids={context.pub_node_id, context.author_node_id} | coauthor_ids | concept_ids,
            forbidden_names=forbidden_names,
            author_title=context.author_title,
            author_handle=author_handle,
        )

        total_elapsed = (time.perf_counter() - t0) * 1000
        l1_miss = len(missed)
        logger.debug(
            "Aligner done in %.1fms | L1 cache: %d hit, %d miss | Neo4j lookup: %.1fms | Qdrant social_entities: %.1fms | Actor resolve: %.1fms | Qdrant categories: %.1fms | entities: %d, relations: %d",
            total_elapsed, l1_hits, l1_miss, neo4j_elapsed,
            qdrant_entities_elapsed, actor_resolve_elapsed, qdrant_categories_elapsed,
            len(extraction_result.entities), len(extraction_result.relations),
        )
        return extraction_result
