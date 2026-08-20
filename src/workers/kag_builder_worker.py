import asyncio
import logging
import signal

from src.config.config import load_settings
from src.db.database import Database
from src.graph.bootstrap import bootstrap_graph
from src.graph.client import Neo4jClient
from src.graph.builder.orchestrator import KagBuilderOrchestrator
from src.utils.logger import setup_logging


async def main() -> None:
    settings = load_settings()
    setup_logging(settings.log_level)

    for logger_name in ("httpx", "httpcore", "openai", "qdrant_client", "urllib3", "neo4j"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    pool_size = max(settings.graph_write_concurrency + 2, 5)
    max_overflow = 2

    db = Database(settings.db_url, pool_size=pool_size, max_overflow=max_overflow)
    neo4j_client = Neo4jClient(settings)

    await neo4j_client.connect()
    await bootstrap_graph(settings, db, neo4j_client)

    orchestrator = KagBuilderOrchestrator(settings, db, neo4j_client)

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, orchestrator.stop)
    loop.add_signal_handler(signal.SIGTERM, orchestrator.stop)

    try:
        await orchestrator.run_pipeline()
    finally:
        await neo4j_client.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())