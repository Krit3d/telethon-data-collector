from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid

from openai import AsyncOpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from src.config.config import Settings
from src.graph.ontology import EntityType, OpenSPGExtractionResult, VECTORIZABLE_ENTITY_LABELS, extract_entity_subtype
from src.graph.utils import format_bge_representation

logger = logging.getLogger(__name__)


_EMBEDDING_RETRIES = 3


class EntityVectorizer:

    def __init__(self, settings: Settings) -> None:
        self._openai_client = AsyncOpenAI(
            base_url=settings.cloud_ru_base_url,
            api_key=settings.cloud_ru_api_key,
        )
        self._qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self._model = settings.cloud_ru_embedding_model

    async def vectorize_and_upsert_entities(self, extraction_result: OpenSPGExtractionResult) -> None:
        entities = [e for e in extraction_result.entities if e.label in VECTORIZABLE_ENTITY_LABELS]
        if not entities:
            return

        ready: list[PointStruct] = []
        missing_entities: list[tuple[int, str]] = []

        for idx, entity in enumerate(entities):
            if not entity.id:
                continue
            point_id = uuid.uuid5(uuid.NAMESPACE_URL, entity.id)
            subtype_key, subtype_val = extract_entity_subtype(entity.label, entity.properties)
            if entity.embedding is not None:
                payload: dict[str, str | None] = {
                    "id": entity.id,
                    "name": entity.name,
                    "name_lower": entity.name_lower,
                    "label": entity.label.value,
                }
                if subtype_key and subtype_val:
                    payload[subtype_key] = subtype_val
                ready.append(PointStruct(
                    id=str(point_id),
                    vector={"text": entity.embedding},
                    payload=payload,
                ))
            else:
                missing_entities.append((
                    idx,
                    format_bge_representation(entity.label.value, entity.name, subtype_val),
                ))

        if missing_entities:
            missing_texts = [text for _, text in missing_entities]
            t0 = time.perf_counter()
            last_error: Exception | None = None
            for attempt in range(_EMBEDDING_RETRIES):
                try:
                    response = await self._openai_client.embeddings.create(
                        model=self._model,
                        input=missing_texts,
                    )
                    ordered = sorted(response.data, key=lambda item: item.index)
                    missing_embeddings = [item.embedding for item in ordered]
                    break
                except RateLimitError as exc:
                    last_error = exc
                    if attempt < _EMBEDDING_RETRIES - 1:
                        delay = (2 ** attempt) + random.uniform(0.0, 1.0)
                        await asyncio.sleep(delay)
                except (APIError, APITimeoutError, APIConnectionError) as exc:
                    last_error = exc
                    if attempt < _EMBEDDING_RETRIES - 1:
                        await asyncio.sleep(2 ** attempt)
            else:
                raise last_error or RuntimeError("Embedding generation failed after all retries")

            embed_elapsed = (time.perf_counter() - t0) * 1000

            if len(missing_embeddings) != len(missing_entities):
                raise RuntimeError(f"Embedding count mismatch: {len(missing_embeddings)} vs {len(missing_entities)} entities")

            for (orig_idx, _), embedding in zip(missing_entities, missing_embeddings):
                entity = entities[orig_idx]
                if not entity.id:
                    continue
                entity.embedding = embedding
                point_id = uuid.uuid5(uuid.NAMESPACE_URL, entity.id)
                payload: dict[str, str | None] = {
                    "id": entity.id,
                    "name": entity.name,
                    "name_lower": entity.name_lower,
                    "label": entity.label.value,
                }
                subtype_key, subtype_val = extract_entity_subtype(entity.label, entity.properties)
                if subtype_key and subtype_val:
                    payload[subtype_key] = subtype_val
                ready.append(PointStruct(
                    id=str(point_id),
                    vector={"text": embedding},
                    payload=payload,
                ))

        t1 = time.perf_counter()
        if ready:
            await self._qdrant_client.upsert(
                collection_name="social_entities",
                points=ready,
            )
        upsert_elapsed = (time.perf_counter() - t1) * 1000

        logger.debug(
            "Vectorizer: %d entities (%d reused, %d new) upserted %d points in %.1fms",
            len(entities), len(entities) - len(missing_entities), len(missing_entities), len(ready), upsert_elapsed,
        )