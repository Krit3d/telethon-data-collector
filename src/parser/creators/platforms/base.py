"""
Abstract base class for platform-specific parsers.

Defines the interface that all platform parsers must implement for
profile parsing and content ingestion into the PostgreSQL database.
"""

import abc
import logging
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from src.config.config import Settings
from src.parser.creators.sc_client import ScrapeCreatorsClient

logger = logging.getLogger(__name__)


class BasePlatformParser(abc.ABC):
    """Abstract base class for platform-specific content parsers.

    Provides the common interface for parsing profiles and content from
    various social media platforms (Instagram, TikTok, etc.) and upserting
    the data into the PostgreSQL database using PostgreSQL-specific
    ON CONFLICT DO UPDATE for high-throughput concurrency.

    Attributes:
        session_maker: SQLAlchemy async session maker for database operations.
        client: ScrapeCreatorsClient instance for API requests.
        settings: Application settings containing configuration values.
    """

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        client: ScrapeCreatorsClient,
        settings: Settings,
    ) -> None:
        """Initialize the platform parser with required dependencies.

        Args:
            session_maker: SQLAlchemy async session maker for creating database sessions.
            client: ScrapeCreatorsClient instance for making API requests.
            settings: Application settings containing configuration values.
        """
        self.session_maker = session_maker
        self.client = client
        self.settings = settings

    @abc.abstractmethod
    async def parse_profile(self, handle: str) -> int | None:
        """Fetch profile data, upsert to accounts table, return database ID.

        Retrieves profile information for the given handle from the platform API,
        upserts the data into the accounts table, and returns the database ID
        of the upserted record.

        Args:
            handle: Platform-specific username/handle (without @ prefix).

        Returns:
            Database ID (int) of the upserted account record, or None if
            the profile could not be parsed or doesn't meet criteria.

        Raises:
            Exception: If a critical error occurs during parsing that should
                not be silently ignored.
        """
        raise NotImplementedError("Subclasses must implement parse_profile()")

    @abc.abstractmethod
    async def parse_content(
        self,
        account_id: int,
        platform_id: str,
        max_items: int = 50,
    ) -> None:
        """Fetch content/videos, parse, and bulk upsert to content table.

        Retrieves content items (videos, posts, etc.) for the given account
        from the platform API, parses the data, and performs a bulk upsert
        into the content table using PostgreSQL ON CONFLICT DO UPDATE.

        Args:
            account_id: Database ID of the parent account record.
            platform_id: Platform-specific account ID used in API calls.
            max_items: Maximum number of content items to fetch (default: 50).

        Raises:
            Exception: If a critical error occurs during content parsing that
                should not be silently ignored.
        """
        raise NotImplementedError("Subclasses must implement parse_content()")
