from __future__ import annotations

import asyncio
import base64
import logging
import math
import struct
import sys

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from src.config.config import load_settings

logger = logging.getLogger("verify_embeddings")

EMBEDDING_DIM = 1024
TEXT_TRUNCATE_LEN = 200

TEST_QUERIES: list[str] = [
    "театральная постановка и актерская игра",
    "свадебное оформление и флористика",
]


async def generate_dense_embedding(
    openai_client: AsyncOpenAI,
    model: str,
    text: str,
) -> list[list[float]]:
    raw_response = await openai_client.embeddings.with_raw_response.create(
        model=model,
        input=[text],
    )
    payload = raw_response.http_response.json()
    embeddings: list[list[float]] = []
    for item in payload.get("data", []):
        embedding = item.get("embedding", [])
        if isinstance(embedding, str):
            binary_data = base64.b64decode(embedding)
            num_floats = len(binary_data) // 4
            binary_data = binary_data[:num_floats * 4]
            decoded_vector = list(struct.unpack(f"<{num_floats}f", binary_data))
            embeddings.append(decoded_vector)
        else:
            embeddings.append(embedding)
    return embeddings


async def verify_collection_distance(
    client: AsyncQdrantClient,
    collection_name: str,
) -> None:
    info = await client.get_collection(collection_name)
    vectors_config = info.config.params.vectors
    if vectors_config is None:
        logger.error("Collection '%s' has no vectors config", collection_name)
        return
    if isinstance(vectors_config, dict):
        for vector_name, vector_params in vectors_config.items():
            if vector_name == "text":
                actual_distance = vector_params.distance
                expected_distance = models.Distance.COSINE
                if actual_distance == expected_distance:
                    logger.info(
                        "Collection '%s' vector '%s' distance metric is %s (correct)",
                        collection_name,
                        vector_name,
                        actual_distance,
                    )
                else:
                    logger.error(
                        "Collection '%s' vector '%s' distance metric is %s, expected %s",
                        collection_name,
                        vector_name,
                        actual_distance,
                        expected_distance,
                    )
    else:
        actual_distance = vectors_config.distance
        expected_distance = models.Distance.COSINE
        if actual_distance == expected_distance:
            logger.info(
                "Collection '%s' distance metric is %s (correct)",
                collection_name,
                actual_distance,
            )
        else:
            logger.error(
                "Collection '%s' distance metric is %s, expected %s",
                collection_name,
                actual_distance,
                expected_distance,
            )


async def run_search(
    qdrant_client: AsyncQdrantClient,
    openai_client: AsyncOpenAI,
    collection_name: str,
    embedding_model: str,
    query: str,
    limit: int = 3,
) -> None:
    logger.info("Running semantic search for query: '%s'", query)

    dense_list = await generate_dense_embedding(openai_client, embedding_model, query)
    dense_emb = dense_list[0]

    logger.debug("dense_emb type: %s", type(dense_emb))
    logger.debug("dense_emb length (dimension): %d", len(dense_emb))
    logger.debug("dense_emb first 5 values: %s", dense_emb[:5])
    logger.debug("dense_emb last 5 values: %s", dense_emb[-5:])

    invalid_indices: list[int] = []
    for idx, val in enumerate(dense_emb):
        if val is None or not isinstance(val, (float, int)) or math.isnan(val) or math.isinf(val):
            invalid_indices.append(idx)
    if invalid_indices:
        logger.error(
            "dense_emb contains %d invalid element(s) at indices: %s, values: %s",
            len(invalid_indices),
            invalid_indices,
            [(i, dense_emb[i]) for i in invalid_indices],
        )
    else:
        logger.debug("dense_emb validation passed: all %d elements are valid floats", len(dense_emb))

    response = await qdrant_client.query_points(
        collection_name=collection_name,
        query=dense_emb,
        using="text",
        limit=limit,
        with_payload=True,
    )

    if not response.points:
        logger.warning("No results found for query: '%s'", query)
        return

    logger.info(
        "Query: '%s' returned %d result(s)",
        query,
        len(response.points),
    )
    for rank, hit in enumerate(response.points, start=1):
        point_id = hit.id
        score = hit.score
        payload = hit.payload or {}
        platform = payload.get("platform", "N/A")
        text_raw = payload.get("text", "") or ""
        text_preview = text_raw[:TEXT_TRUNCATE_LEN]
        if len(text_raw) > TEXT_TRUNCATE_LEN:
            text_preview += "..."

        print(
            f"  [{rank}] Point ID: {point_id}\n"
            f"      Score:   {score:.6f}\n"
            f"      Platform: {platform}\n"
            f"      Text:    {text_preview}\n"
        )


async def main() -> None:
    settings = load_settings()

    if not settings.qdrant_url:
        logger.error("QDRANT_URL is not configured in environment/.env")
        sys.exit(1)

    collection_name = settings.qdrant_collection_name
    if not collection_name:
        logger.error("QDRANT_COLLECTION_NAME is not configured in environment/.env")
        sys.exit(1)

    embedding_model = settings.cloud_ru_embedding_model

    logger.info("Connecting to Qdrant at %s ...", settings.qdrant_url)
    qdrant_client = AsyncQdrantClient(
        url=settings.qdrant_url,
        timeout=settings.qdrant_timeout,
        api_key=settings.qdrant_api_key,
    )

    openai_client = AsyncOpenAI(
        api_key=settings.cloud_ru_api_key,
        base_url=settings.cloud_ru_base_url,
    )

    try:
        logger.info("Verifying collection '%s' ...", collection_name)
        await verify_collection_distance(qdrant_client, collection_name)

        collection_info = await qdrant_client.get_collection(collection_name)
        total_points = collection_info.points_count
        logger.info(
            "Collection '%s' contains %s point(s)",
            collection_name,
            total_points,
        )

        for query in TEST_QUERIES:
            print(f"\n{'=' * 70}")
            print(f"Query: {query}")
            print(f"{'=' * 70}")
            await run_search(
                qdrant_client,
                openai_client,
                collection_name,
                embedding_model,
                query,
                limit=3,
            )

        logger.info("Verification completed successfully")

    except Exception as e:
        logger.error("Verification failed with error: %s", e, exc_info=True)
        sys.exit(1)

    finally:
        await qdrant_client.close()
        await openai_client.close()
        logger.info("Clients closed")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(main())
