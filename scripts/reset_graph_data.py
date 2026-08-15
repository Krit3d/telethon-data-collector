from __future__ import annotations

import asyncio
import logging
import sys

from neo4j import AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.config.config import load_settings

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("reset_graph_data")

NEO4J_LABELS: list[str] = [
    "Actor",
    "Post",
    "Entity",
    "Place",
    "Organization",
    "Product",
    "Event",
    "MicroConcept",
    "Concept",
    "Hashtag",
]

DELETE_ALL_QUERY: str = "MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"


async def reset_neo4j(neo4j_url: str, neo4j_user: str, neo4j_password: str, neo4j_database: str) -> None:
    driver = AsyncGraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
    try:
        await driver.verify_connectivity()
        logger.info("Neo4j connection established")

        async with driver.session(database=neo4j_database) as session:
            logger.info("Deleting all nodes and relationships...")
            await session.run(DELETE_ALL_QUERY)  # type: ignore[arg-type]
            logger.info("All nodes and relationships deleted")

        for label in NEO4J_LABELS:
            try:
                async with driver.session(database=neo4j_database) as session:
                    cypher = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                    await session.run(cypher)  # type: ignore[arg-type]
                logger.info("Constraint ensured for label: %s", label)
            except Exception as exc:
                logger.warning("Failed to create constraint for label %s: %s", label, exc)

        logger.info("Neo4j reset completed")
    finally:
        await driver.close()


async def reset_qdrant(qdrant_url: str | None, qdrant_api_key: str | None) -> None:
    if qdrant_url is None:
        logger.warning("QDRANT_URL not configured, skipping Qdrant reset")
        return

    client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    try:
        existing = await client.get_collections()
        existing_names = [c.name for c in existing.collections]

        if "social_entities" in existing_names:
            logger.info("Deleting existing collection: social_entities")
            await client.delete_collection(collection_name="social_entities")

        logger.info("Creating collection: social_entities")
        await client.create_collection(
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

        await client.create_payload_index(
            collection_name="social_entities",
            field_name="label",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        await client.create_payload_index(
            collection_name="social_entities",
            field_name="name_lower",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        logger.info("Collection 'social_entities' created successfully with payload indexes")
    finally:
        await client.close()


async def reset_postgres_graph_status(db_url: str) -> None:
    engine = create_async_engine(db_url, pool_size=2, max_overflow=2)
    try:
        async with engine.begin() as conn:
            result = await conn.execute(
                text(
                    "UPDATE content "
                    "SET graph_status = 0 "
                    "FROM accounts "
                    "WHERE content.account_id = accounts.id "
                    "AND accounts.status = 'verified'"
                )
            )
            updated_count = result.rowcount
            logger.info("Reset graph_status to 0 for %d posts from verified accounts", updated_count)
    finally:
        await engine.dispose()


async def main() -> None:
    settings = load_settings()

    await reset_neo4j(
        neo4j_url=settings.neo4j_url,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        neo4j_database=settings.neo4j_database,
    )

    await reset_qdrant(
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
    )

    await reset_postgres_graph_status(db_url=settings.db_url)

    logger.info("Graph data reset completed successfully")


if __name__ == "__main__":
    asyncio.run(main())