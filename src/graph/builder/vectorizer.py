from __future__ import annotations

import asyncio
import uuid

from openai import AsyncOpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from src.config.config import Settings
from src.graph.ontology import EntityType, OpenSPGExtractionResult
from src.graph.utils import format_bge_representation


_EMBEDDING_RETRIES = 3
_EXCLUDED_LABELS: frozenset[EntityType] = frozenset({
    EntityType.Concept,
    EntityType.Tone,
    EntityType.Language,
    EntityType.Hashtag,
})


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
        entities = [e for e in extraction_result.entities if e.label not in _EXCLUDED_LABELS]
        if not entities:
            return

        bge_texts = [
            format_bge_representation(e.label.value, e.name, e.properties)
            for e in entities
        ]

        last_error: Exception | None = None
        for attempt in range(_EMBEDDING_RETRIES):
            try:
                response = await self._openai_client.embeddings.create(
                    model=self._model,
                    input=bge_texts,
                )
                ordered = sorted(response.data, key=lambda item: item.index)
                embeddings = [item.embedding for item in ordered]
                break
            except (APIError, APITimeoutError, APIConnectionError, RateLimitError) as exc:
                last_error = exc
                if attempt < _EMBEDDING_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
        else:
            raise last_error or RuntimeError("Embedding generation failed after all retries")

        if len(embeddings) != len(entities):
            raise RuntimeError(f"Embedding count mismatch: {len(embeddings)} vs {len(entities)} entities")

        points: list[PointStruct] = []
        for entity, embedding in zip(entities, embeddings):
            if not entity.id:
                continue
            point_id = uuid.uuid5(uuid.NAMESPACE_URL, entity.id)
            points.append(PointStruct(
                id=point_id,
                vector={"text": embedding},
                payload={
                    "id": entity.id,
                    "name": entity.name,
                    "name_lower": entity.name_lower,
                    "label": entity.label.value,
                    "properties": entity.properties,
                },
            ))

        if points:
            await self._qdrant_client.upsert(
                collection_name="social_entities",
                points=points,
            )