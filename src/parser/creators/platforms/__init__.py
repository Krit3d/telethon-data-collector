"""
Platform parser factory module.

Provides a factory function to dynamically resolve and instantiate platform-specific
parsers based on the platform string. Supports Instagram and TikTok platforms.
"""

import logging
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from src.config.config import Settings
from src.parser.creators.platforms.base import BasePlatformParser

if TYPE_CHECKING:
    from src.parser.creators.sc_client import ScrapeCreatorsClient

logger = logging.getLogger(__name__)

# Supported platforms mapping
SUPPORTED_PLATFORMS = {
    "INSTAGRAM": "InstagramParser",
    "TIKTOK": "TikTokParser",
}


def get_platform_parser(
    platform: str,
    session_maker: async_sessionmaker[AsyncSession],
    client: "ScrapeCreatorsClient",
    settings: Settings,
) -> BasePlatformParser:
    """
    Factory function to get the appropriate platform parser instance.

    Normalizes the platform string to uppercase and returns the corresponding
    platform parser instance initialized with the provided dependencies.

    Args:
        platform: Platform name string (e.g., "instagram", "TIKTOK").
        session_maker: SQLAlchemy async session maker for database operations.
        client: ScrapeCreatorsClient instance for API requests.
        settings: Application settings containing configuration values.

    Returns:
        An instance of the appropriate BasePlatformParser subclass.

    Raises:
        ValueError: If the platform is not supported.
    """
    normalized_platform = platform.upper()

    if normalized_platform == "INSTAGRAM":
        from src.parser.creators.platforms.instagram import InstagramParser

        logger.debug(f"Creating InstagramParser for platform: {platform}")
        return InstagramParser(
            session_maker=session_maker,
            client=client,
            settings=settings,
        )

    if normalized_platform == "TIKTOK":
        from src.parser.creators.platforms.tiktok import TikTokParser

        logger.debug(f"Creating TikTokParser for platform: {platform}")
        return TikTokParser(
            session_maker=session_maker,
            client=client,
            settings=settings,
        )

    logger.error(f"Unsupported platform requested: {platform}")
    raise ValueError(
        f"Unsupported platform: '{platform}'. "
        f"Supported platforms are: {list(SUPPORTED_PLATFORMS.keys())}"
    )
