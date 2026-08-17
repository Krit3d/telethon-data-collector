from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, cast

from neo4j import AsyncGraphDatabase
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.config.config import Settings, load_settings
from src.embeddings.client import ENTITIES_COLLECTION, QdrantClientManager

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

DELETE_ALL_QUERY: str = "MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"


async def reset_neo4j(neo4j_url: str, neo4j_user: str, neo4j_password: str, neo4j_database: str) -> None:
    driver = AsyncGraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
    try:
        await driver.verify_connectivity()
        logger.info("Neo4j connection established")

        async with driver.session(database=neo4j_database) as session:
            logger.info("Deleting all nodes and relationships...")
            await session.run(cast(Any, DELETE_ALL_QUERY))
            logger.info("All nodes and relationships deleted")

        for label in NEO4J_LABELS:
            try:
                async with driver.session(database=neo4j_database) as session:
                    cypher = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                    await session.run(cast(Any, cypher))
                logger.info("Constraint ensured for label: %s", label)
            except Exception as exc:
                logger.warning("Failed to create constraint for label %s: %s", label, exc)

        logger.info("Neo4j reset completed")
    finally:
        await driver.close()


async def reset_qdrant(settings: Settings) -> None:
    if settings.qdrant_url is None:
        logger.warning("QDRANT_URL not configured, skipping Qdrant reset")
        return

    client_manager = QdrantClientManager(settings)
    try:
        if await client_manager.client.collection_exists(ENTITIES_COLLECTION):
            logger.info("Deleting existing collection: %s", ENTITIES_COLLECTION)
            await client_manager.client.delete_collection(ENTITIES_COLLECTION)

        await client_manager.initialize()
        logger.info("Collection '%s' recreated with project settings", ENTITIES_COLLECTION)
    finally:
        await client_manager.close()


async def reset_postgres_graph_status(db_url: str) -> None:
    db_url = db_url.replace("@localhost:", "@db:").replace("@127.0.0.1:", "@db:")
    engine = create_async_engine(db_url, pool_size=2, max_overflow=2)
    try:
        async with engine.begin() as conn:
            result = await conn.execute(
                text(
                    "UPDATE content "
                    "SET graph_status = 0 "
                    "WHERE graph_status != 0 "
                    "AND account_id IN ("
                    "SELECT id "
                    "FROM accounts "
                    "WHERE LOWER(status) = 'verified'"
                    ")"
                )
            )
            updated_count = result.rowcount
            logger.info("Reset graph_status to 0 for %d posts from verified accounts", updated_count)
    finally:
        await engine.dispose()


async def main() -> None:
    settings = load_settings()

    try:
        await reset_neo4j(
            neo4j_url=settings.neo4j_url,
            neo4j_user=settings.neo4j_user,
            neo4j_password=settings.neo4j_password,
            neo4j_database=settings.neo4j_database,
        )
    except Exception as exc:
        logger.error("Ошибка на этапе Neo4j: %s", exc, exc_info=True)

    try:
        await reset_qdrant(settings=settings)
    except Exception as exc:
        logger.error("Ошибка на этапе Qdrant: %s", exc, exc_info=True)

    try:
        await reset_postgres_graph_status(db_url=settings.db_url)
    except Exception as exc:
        logger.error("Ошибка на этапе PostgreSQL: %s", exc, exc_info=True)

    logger.info("=== Завершение процесса сброса ===")


if __name__ == "__main__":
    asyncio.run(main())
