from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any

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
from src.graph.utils import build_node_id, clean_compact_name, clean_identifier, clean_name_lower, format_bge_representation, format_display_name

logger = logging.getLogger(__name__)

_L1_CACHE_MAX = 50_000
_EMBEDDING_RETRIES = 3
_ENTITY_SCORE_THRESHOLD = 0.88
_CATEGORY_MIN_SCORE = 0.68
_CATEGORY_DEPTH_DELTA = 0.05
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
            prefer_grpc=settings.qdrant_prefer_grpc,
            grpc_port=settings.qdrant_grpc_port,
            timeout=settings.qdrant_timeout,
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
            canonical_id = payload.get("id") or build_node_id(entity.label, canonical_name)
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

    async def _resolve_coauthors(
        self,
        extraction_result: OpenSPGExtractionResult,
        context: PostBatchContext,
        id_map: dict[str, str],
    ) -> None:
        if not context.post_coauthors:
            return

        unresolved_temp_ids: set[str] = set()
        for ca in context.post_coauthors:
            clean_ca = clean_identifier(ca)
            if not clean_ca:
                continue
            temp_id = build_node_id(EntityType.Actor, clean_ca, platform=context.platform)
            cache_key = f"Actor:{clean_name_lower(clean_ca)}"
            cached = self._l1_cache.get(cache_key)
            if cached is not None:
                id_map[temp_id] = cached
                self._l1_cache.move_to_end(cache_key)
                continue
            try:
                rows = await self._neo4j_client.execute_read(
                    "MATCH (a:Actor) WHERE (a.handle = $handle OR a.name_lower = $name_lower) AND toLower(a.platform) = $platform RETURN a.id AS id LIMIT 1",
                    {"handle": clean_ca, "name_lower": clean_name_lower(clean_ca), "platform": context.platform.lower()},
                )
                if rows:
                    raw_id = rows[0].get("id")
                    if isinstance(raw_id, str):
                        self._cache_set(cache_key, raw_id)
                        id_map[temp_id] = raw_id
                        continue
            except Exception as exc:
                logger.warning("Neo4j coauthor lookup failed for %s: %s", ca, exc)
                continue
            unresolved_temp_ids.add(temp_id)

        if unresolved_temp_ids:
            extraction_result.relations = [
                r for r in extraction_result.relations
                if not (r.relation_type == RelationType.COAUTHOR and r.target_id in unresolved_temp_ids)
            ]

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
                        limit=10,
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
            valid: list[dict[str, Any]] = [
                cand for cand in hit_candidates
                if float(cand["score"]) >= _CATEGORY_MIN_SCORE
            ]
            if not valid:
                continue
            for cand in valid:
                payload = cand["payload"]
                if payload.get("tier_4"):
                    cand["_depth"] = 4
                elif payload.get("tier_3"):
                    cand["_depth"] = 3
                elif payload.get("tier_2"):
                    cand["_depth"] = 2
                elif payload.get("tier_1"):
                    cand["_depth"] = 1
                else:
                    cand["_depth"] = 0
            max_score = max(float(cand["score"]) for cand in valid)
            delta_window = [
                cand for cand in valid
                if float(cand["score"]) >= max_score - _CATEGORY_DEPTH_DELTA
            ]
            chosen: dict[str, Any] | None = None
            if context.account_category_id:
                matched = next(
                    (c for c in delta_window if str(c["payload"].get("code")).strip().lower() == str(context.account_category_id).strip().lower()),
                    None,
                )
                if matched is not None:
                    chosen = matched
            if chosen is None:
                max_depth = max(int(cand["_depth"]) for cand in delta_window)
                deepest = [cand for cand in delta_window if int(cand["_depth"]) == max_depth]
                chosen = max(deepest, key=lambda c: float(c["score"]))
            payload = chosen["payload"]
            code = payload.get("code")
            if not code:
                continue
            canonical_name = payload.get("name") or payload.get("title") or code
            canonical_id = payload.get("id") or build_node_id(EntityType.MicroConcept, canonical_name)
            mc.properties["is_classified"] = True
            self._classified_mc_ids.add(mc_id)
            key = f"{EntityType.MicroConcept.value}:{clean_name_lower(mc.name)}"
            self._cache_set(key, mc_id)
            relations.append(
                ExtractedRelation(
                    source_id=mc_id,
                    source_label=EntityType.MicroConcept,
                    target_id=canonical_id,
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

        unresolved: list[HashtagItem] = []
        maps_to_count = 0

        for ht in unique:
            ht_node_id = build_node_id(EntityType.Hashtag, ht.raw)
            raw_lower = clean_name_lower(ht.raw)
            norm_lower = clean_name_lower(ht.normalized)
            raw_compact = clean_compact_name(ht.raw)
            norm_compact = clean_compact_name(ht.normalized)

            resolved = False
            for entity in extraction_result.entities:
                if entity.label not in (EntityType.MicroConcept, EntityType.Entity, EntityType.Product, EntityType.Organization, EntityType.Event):
                    continue
                if not entity.id:
                    continue
                el_lower = clean_name_lower(entity.name)
                el_compact = clean_compact_name(entity.name)

                matched = False
                if el_lower == raw_lower or el_lower == norm_lower:
                    matched = True
                elif el_compact and (el_compact == raw_compact or el_compact == norm_compact):
                    matched = True
                elif len(el_compact) >= 4 and (raw_compact and el_compact in raw_compact or norm_compact and el_compact in norm_compact):
                    matched = True

                if matched:
                    maps_to_key = (ht_node_id, entity.id, RelationType.MAPS_TO.value)
                    if maps_to_key not in existing_relation_keys:
                        extraction_result.relations.append(
                            ExtractedRelation(
                                source_id=ht_node_id,
                                source_label=EntityType.Hashtag,
                                target_id=entity.id,
                                target_label=entity.label,
                                relation_type=RelationType.MAPS_TO,
                                properties={},
                            )
                        )
                        existing_relation_keys.add(maps_to_key)
                        maps_to_count += 1
                    resolved = True
                    break
            if not resolved:
                unresolved.append(ht)

        if unresolved:
            l2_names: list[str] = []
            l2_node_ids: list[str] = []
            for ht in unresolved:
                raw_lower = clean_name_lower(ht.raw)
                norm_lower = clean_name_lower(ht.normalized)
                l2_names.append(raw_lower)
                if norm_lower != raw_lower:
                    l2_names.append(norm_lower)
                mc_id_raw = build_node_id(EntityType.MicroConcept, ht.raw)
                mc_id_norm = build_node_id(EntityType.MicroConcept, ht.normalized)
                l2_node_ids.append(mc_id_raw)
                if mc_id_norm != mc_id_raw:
                    l2_node_ids.append(mc_id_norm)

            l2_rows: list[dict[str, Any]] = []
            try:
                l2_rows = await self._neo4j_client.execute_read(
                    "MATCH (n) WHERE (n:MicroConcept OR n:Entity OR n:Product OR n:Organization OR n:Event) "
                    "AND (n.name_lower IN $names OR n.id IN $node_ids) "
                    "RETURN n.id AS id, labels(n) AS labels, n.name AS name, n.name_lower AS name_lower, properties(n) AS props",
                    {"names": l2_names, "node_ids": l2_node_ids},
                )
            except Exception as exc:
                logger.warning("Neo4j L2 hashtag lookup failed: %s", exc)

            if l2_rows:
                l2_found_ids: set[str] = set()
                for row in l2_rows:
                    node_id = row.get("id")
                    if not node_id:
                        continue
                    node_labels: list[str] = row.get("labels", [])
                    node_name = row.get("name") or ""
                    node_name_lower = row.get("name_lower") or clean_name_lower(node_name)
                    canonical_label: EntityType | None = None
                    for lbl in (EntityType.MicroConcept, EntityType.Entity, EntityType.Product, EntityType.Organization, EntityType.Event):
                        if lbl.value in node_labels:
                            canonical_label = lbl
                            break
                    if canonical_label is None:
                        continue
                    if node_id not in existing_entity_ids:
                        node_props: dict[str, Any] = row.get("props") or {}
                        match canonical_label:
                            case EntityType.Product:
                                props: dict[str, object] = {"product_type": str(node_props.get("product_type") or "service")}
                            case EntityType.Organization:
                                props = {"org_type": str(node_props.get("org_type") or "company")}
                            case EntityType.Entity:
                                props = {"entity_type": str(node_props.get("entity_type") or "general")}
                            case EntityType.Event:
                                props = {"event_type": str(node_props.get("event_type") or "incident")}
                            case EntityType.MicroConcept:
                                props = {"is_classified": bool(node_props.get("is_classified", False))}
                            case _:
                                props = {}
                        extraction_result.entities.append(
                            ExtractedEntity(
                                id=node_id,
                                name=node_name,
                                name_lower=node_name_lower,
                                label=canonical_label,
                                properties=props,
                            )
                        )
                        existing_entity_ids.add(node_id)
                    l2_found_ids.add(node_id)

                for ht in unresolved:
                    ht_node_id = build_node_id(EntityType.Hashtag, ht.raw)
                    if any(
                        (ht_node_id, eid, RelationType.MAPS_TO.value) in existing_relation_keys
                        for eid in l2_found_ids
                    ):
                        continue
                    raw_lower = clean_name_lower(ht.raw)
                    norm_lower = clean_name_lower(ht.normalized)
                    mc_id_raw = build_node_id(EntityType.MicroConcept, ht.raw)
                    mc_id_norm = build_node_id(EntityType.MicroConcept, ht.normalized)
                    matched_id: str | None = None
                    matched_label: EntityType | None = None
                    for row in l2_rows:
                        node_id = row.get("id")
                        if not node_id:
                            continue
                        row_name_lower = row.get("name_lower") or ""
                        if row_name_lower in (raw_lower, norm_lower) or node_id in (mc_id_raw, mc_id_norm):
                            matched_id = node_id
                            node_labels: list[str] = row.get("labels", [])
                            for lbl in (EntityType.MicroConcept, EntityType.Entity, EntityType.Product, EntityType.Organization, EntityType.Event):
                                if lbl.value in node_labels:
                                    matched_label = lbl
                                    break
                            break
                    if matched_id and matched_label:
                        maps_to_key = (ht_node_id, matched_id, RelationType.MAPS_TO.value)
                        if maps_to_key not in existing_relation_keys:
                            extraction_result.relations.append(
                                ExtractedRelation(
                                    source_id=ht_node_id,
                                    source_label=EntityType.Hashtag,
                                    target_id=matched_id,
                                    target_label=matched_label,
                                    relation_type=RelationType.MAPS_TO,
                                    properties={},
                                )
                            )
                            existing_relation_keys.add(maps_to_key)
                            maps_to_count += 1

            still_unresolved = [
                ht for ht in unresolved
                if not any(
                    (build_node_id(EntityType.Hashtag, ht.raw), eid, RelationType.MAPS_TO.value) in existing_relation_keys
                    for eid in (l2_found_ids if l2_rows else set())
                )
            ]

            if still_unresolved:
                normalized_texts = [ht.normalized for ht in still_unresolved]
                try:
                    embeddings = await self._get_embeddings_batch(normalized_texts)
                except Exception as exc:
                    logger.warning("Embedding generation failed in _link_hashtags L3: %s", exc)
                    embeddings = []

                if embeddings and len(embeddings) == len(still_unresolved):
                    t0 = time.perf_counter()
                    try:
                        batch_response = await self._qdrant_client.query_batch_points(
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
                        logger.warning("Qdrant entities query failed in _link_hashtags L3: %s", exc)
                        batch_response = []

                    qdrant_elapsed = (time.perf_counter() - t0) * 1000

                    for ht, response in zip(still_unresolved, batch_response):
                        ht_node_id = build_node_id(EntityType.Hashtag, ht.raw)
                        if not response or not response.points:
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
                        canonical_name = format_display_name(
                            str(payload.get("name") or payload.get("title") or canonical_label),
                            target_label,
                        )
                        canonical_id = build_node_id(target_label, canonical_name)
                        if canonical_id not in existing_entity_ids:
                            match target_label:
                                case EntityType.Product:
                                    props: dict[str, object] = {"product_type": str(payload.get("product_type") or "service")}
                                case EntityType.Event:
                                    props = {"event_type": str(payload.get("event_type") or "incident")}
                                case EntityType.Organization:
                                    props = {"org_type": str(payload.get("org_type") or "company")}
                                case EntityType.Entity:
                                    props = {"entity_type": str(payload.get("entity_type") or "general")}
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

                    logger.debug(
                        "Qdrant hashtag L3 search for %d hashtags done in %.1fms | MAPS_TO: %d",
                        len(still_unresolved), qdrant_elapsed, maps_to_count,
                    )

        microconcepts_in_post = [
            e for e in extraction_result.entities
            if e.label == EntityType.MicroConcept and e.id
        ]
        if microconcepts_in_post:
            for ht in unresolved:
                ht_node_id = build_node_id(EntityType.Hashtag, ht.raw)
                already_mapped = any(
                    src == ht_node_id and rel == RelationType.MAPS_TO.value
                    for src, _, rel in existing_relation_keys
                )
                if already_mapped:
                    continue
                target_mc = microconcepts_in_post[0]
                mc_id = target_mc.id
                if not mc_id:
                    continue
                maps_to_key = (ht_node_id, mc_id, RelationType.MAPS_TO.value)
                if maps_to_key not in existing_relation_keys:
                    extraction_result.relations.append(
                        ExtractedRelation(
                            source_id=ht_node_id,
                            source_label=EntityType.Hashtag,
                            target_id=mc_id,
                            target_label=EntityType.MicroConcept,
                            relation_type=RelationType.MAPS_TO,
                            properties={},
                        )
                    )
                    existing_relation_keys.add(maps_to_key)
                    maps_to_count += 1

        logger.debug(
            "Hashtag linking complete: %d unique hashtags, %d MAPS_TO relations created",
            len(unique), maps_to_count,
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

        t_coauthor_resolve = time.perf_counter()
        await self._resolve_coauthors(extraction_result, context, id_map)
        coauthor_resolve_elapsed = (time.perf_counter() - t_coauthor_resolve) * 1000

        t_actor_resolve = time.perf_counter()
        await self._resolve_actor_mentions(extraction_result, context, id_map)
        actor_resolve_elapsed = (time.perf_counter() - t_actor_resolve) * 1000

        resolved_actor_ids = {v for v in id_map.values() if v.startswith("actor_")}

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

        coauthor_ids = {
            build_node_id(EntityType.Actor, clean_identifier(ca), platform=context.platform)
            for ca in context.post_coauthors
            if clean_identifier(ca)
        }

        concept_ids: set[str] = set()
        for relation in extraction_result.relations:
            if relation.relation_type == RelationType.BELONGS_TO and relation.target_label == EntityType.Concept and relation.target_id:
                concept_ids.add(relation.target_id)
        for entity in extraction_result.entities:
            if entity.label == EntityType.Concept and entity.id:
                concept_ids.add(entity.id)

        id_to_label: dict[str, EntityType] = {}
        for entity in extraction_result.entities:
            if entity.id:
                id_to_label[entity.id] = entity.label
        if context.author_node_id:
            id_to_label[context.author_node_id] = EntityType.Actor
        if context.pub_node_id:
            id_to_label[context.pub_node_id] = EntityType.Post
        for eid in resolved_actor_ids:
            id_to_label[eid] = EntityType.Actor
        for eid in coauthor_ids:
            id_to_label[eid] = EntityType.Actor
        for eid in concept_ids:
            id_to_label[eid] = EntityType.Concept
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

        author_handle = (getattr(context, "author_handle", "") or getattr(context, "author_username", "") or "").strip()
        forbidden_names = {clean_name_lower(context.author_title)}
        if author_handle:
            forbidden_names.add(clean_name_lower(author_handle))
        allowed_ids = {context.pub_node_id, context.author_node_id} | resolved_actor_ids | coauthor_ids | concept_ids
        allowed_ids.discard(None)
        extraction_result.sanitize_and_validate(
            allowed_ids=allowed_ids,
            forbidden_names=forbidden_names,
            author_title=context.author_title,
            author_handle=author_handle,
        )

        total_elapsed = (time.perf_counter() - t0) * 1000
        l1_miss = len(missed)
        logger.debug(
            "Aligner done in %.1fms | L1 cache: %d hit, %d miss | Neo4j lookup: %.1fms | Coauthor resolve: %.1fms | Qdrant social_entities: %.1fms | Actor resolve: %.1fms | Qdrant categories: %.1fms | entities: %d, relations: %d",
            total_elapsed, l1_hits, l1_miss, neo4j_elapsed,
            coauthor_resolve_elapsed, qdrant_entities_elapsed, actor_resolve_elapsed, qdrant_categories_elapsed,
            len(extraction_result.entities), len(extraction_result.relations),
        )
        return extraction_result
