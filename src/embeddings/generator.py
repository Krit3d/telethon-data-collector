from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import random
import struct
from typing import Final

import httpx
from openai import AsyncOpenAI, RateLimitError
from qdrant_client.http import models

from src.config.config import Settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM: Final[int] = 1024


class EmbeddingGenerator:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
            timeout=httpx.Timeout(60.0, connect=10.0, read=60.0, write=10.0),
        )
        self.openai_client = AsyncOpenAI(
            api_key=settings.cloud_ru_api_key,
            base_url=settings.cloud_ru_base_url,
            http_client=http_client,
        )

    @staticmethod
    def _make_fallback_sparse(text: str) -> models.SparseVector:
        tokens = text.lower().split()
        index_map: dict[int, float] = {}
        for token in tokens:
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = token_hash % 30_000
            weight = index_map.get(idx, 0.0) + 1.0
            index_map[idx] = weight
        sorted_items = sorted(index_map.items())
        if not sorted_items:
            return models.SparseVector(indices=[0], values=[0.0])
        return models.SparseVector(
            indices=[k for k, _ in sorted_items],
            values=[v for _, v in sorted_items],
        )

    async def generate_batch(
        self,
        texts: list[str],
    ) -> tuple[list[list[float]], list[models.SparseVector]]:
        if not texts:
            return [], []

        valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        if not valid_indices:
            return [[] for _ in texts], [self._make_fallback_sparse("") for _ in texts]

        valid_texts = [texts[i] for i in valid_indices]

        max_retries = 6
        base_delay = 1.0
        max_delay = 10.0
        for attempt in range(1, max_retries + 1):
            try:
                raw_response = await self.openai_client.embeddings.with_raw_response.create(
                    model=self.settings.cloud_ru_embedding_model,
                    input=valid_texts,
                )
                payload = raw_response.http_response.json()
                break
            except RateLimitError:
                if attempt == max_retries:
                    logger.error(
                        "Cloud.ru embedding API rate limits exceeded after %d attempts",
                        attempt,
                    )
                    raise RuntimeError("Cloud.ru embedding API rate limits were exceeded")
                sleep_duration = min(base_delay * (2 ** (attempt - 1)), max_delay)
                sleep_duration *= random.uniform(0.5, 1.5)
                logger.warning(
                    "Rate limited on attempt %d/%d, sleeping for %.2fs",
                    attempt,
                    max_retries,
                    sleep_duration,
                )
                await asyncio.sleep(sleep_duration)

        dense_map: dict[int, list[float]] = {}
        sparse_map: dict[int, models.SparseVector] = {}

        for local_idx, item in enumerate(payload.get("data", [])):
            global_idx = valid_indices[local_idx]

            dense_emb = item.get("embedding", [])
            if isinstance(dense_emb, str):
                try:
                    decoded = base64.b64decode(dense_emb)
                    dense_emb = list(struct.unpack(f"{len(decoded) // 4}f", decoded))
                except Exception:
                    dense_emb = [0.0] * EMBEDDING_DIM
            dense_map[global_idx] = dense_emb

            sparse_vector: models.SparseVector | None = None

            for key in ("sparse", "sparse_embedding"):
                raw_sparse = item.get(key)
                if raw_sparse is not None:
                    if isinstance(raw_sparse, dict):
                        indices = raw_sparse.get("indices")
                        values = raw_sparse.get("values")
                        if (
                            isinstance(indices, list)
                            and isinstance(values, list)
                            and indices
                            and values
                        ):
                            sparse_vector = models.SparseVector(
                                indices=[int(i) for i in indices],
                                values=[float(v) for v in values],
                            )
                            break
                        else:
                            sorted_items = sorted(
                                (int(k), float(v)) for k, v in raw_sparse.items()
                            )
                            if sorted_items:
                                sparse_vector = models.SparseVector(
                                    indices=[k for k, _ in sorted_items],
                                    values=[v for _, v in sorted_items],
                                )
                                break

            if sparse_vector is None:
                sparse_vector = self._make_fallback_sparse(valid_texts[local_idx])

            sparse_map[global_idx] = sparse_vector

        result_dense: list[list[float]] = []
        result_sparse: list[models.SparseVector] = []
        for i in range(len(texts)):
            result_dense.append(dense_map.get(i, [0.0] * EMBEDDING_DIM))
            result_sparse.append(
                sparse_map.get(i, self._make_fallback_sparse(texts[i]))
            )

        return result_dense, result_sparse

    async def close(self) -> None:
        try:
            await self.openai_client.close()
            logger.debug("OpenAI client closed")
        except Exception as e:
            logger.warning("Error closing OpenAI client", exc_info=e)