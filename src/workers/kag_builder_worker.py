import asyncio
import signal

from src.config.config import load_settings
from src.db.database import Database
from src.graph.client import Neo4jClient
from src.graph.builder.orchestrator import KagBuilderOrchestrator
from src.utils.logger import setup_logging


async def main() -> None:
    settings = load_settings()
    setup_logging(settings.log_level)

    db = Database(settings.db_url)
    neo4j_client = Neo4jClient(settings)

    await neo4j_client.connect()

    orchestrator = KagBuilderOrchestrator(settings, db, neo4j_client)

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, orchestrator.stop)
    loop.add_signal_handler(signal.SIGTERM, orchestrator.stop)

    try:
        await orchestrator.run_pipeline()
    finally:
        await neo4j_client.close()


if __name__ == "__main__":
    asyncio.run(main())