import asyncio
import logging

from qdrant_client import AsyncQdrantClient, models

from src.config.config import load_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    settings = load_settings()

    qdrant_client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    logger.info("Connected to Qdrant at %s", settings.qdrant_url)

    existing = await qdrant_client.get_collections()
    existing_names = [c.name for c in existing.collections]

    for collection_name in ("social_posts", "social_entities"):
        if collection_name in existing_names:
            logger.info("Deleting existing collection: %s", collection_name)
            await qdrant_client.delete_collection(collection_name=collection_name)

    logger.info("Creating collection: social_posts")
    await qdrant_client.create_collection(
        collection_name="social_posts",
        vectors_config={
            "text": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            "video_clip": models.VectorParams(size=512, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "text_sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=True),
            ),
        },
    )
    logger.info("Collection 'social_posts' created successfully")

    logger.info("Creating collection: social_entities")
    await qdrant_client.create_collection(
        collection_name="social_entities",
        vectors_config={
            "text": models.VectorParams(size=1024, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "text_sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=True),
            ),
        },
    )
    logger.info("Collection 'social_entities' created successfully")

    await qdrant_client.close()

    logger.info("Migration completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
