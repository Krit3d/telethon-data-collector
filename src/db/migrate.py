"""
Database migration script.

This script is responsible for initializing the database schema and running
necessary migrations. It should be executed once the database container is healthy,
before starting worker services.
"""

import asyncio
import logging

from src.config.config import load_settings
from src.db.database import Database

logger = logging.getLogger(__name__)


async def main() -> None:
    """Run database migration: initialize schema and exit cleanly."""
    logger.info("Starting database migration process...")
    db = None
    
    try:
        settings = load_settings()
        # Mask credentials in logs by only showing the host/database part
        db_url_safe = settings.db_url.split("@")[-1] if "@" in settings.db_url else settings.db_url
        logger.info("Configuration loaded. Target database: %s", db_url_safe)

        db = Database(settings.db_url)
        logger.info("Initializing database schema...")
        await db.init_db()
        logger.info("Database migration completed successfully.")
    except Exception as e:
        logger.error("Database migration failed: %s", e, exc_info=True)
        raise
    finally:
        if db is not None:
            await db.close()
            logger.info("Database connection closed.")


if __name__ == "__main__":
    asyncio.run(main())
