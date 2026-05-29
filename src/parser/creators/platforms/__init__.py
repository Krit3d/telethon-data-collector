"""
Platform parser factory module.

Exposes `get_platform_parser` to dynamically instantiate platform-specific parsers
for supported social media platforms.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config.config import Settings
from src.parser.creators.sc_client import ScrapeCreatorsClient
from .base import BasePlatformParser
from .instagram import InstagramParser
from .threads import ThreadsParser
from .tiktok import TikTokParser
from .youtube import YouTubeParser


def get_platform_parser(
    platform: str,
    session_maker: async_sessionmaker[AsyncSession],
    client: ScrapeCreatorsClient,
    settings: Settings,
) -> BasePlatformParser:
    """Dynamically fetch and instantiate a platform parser for the given platform.

    Args:
        platform: Identifier for the target platform (e.g., "INSTAGRAM", "THREADS", "TIKTOK", "YOUTUBE").
            Case-insensitive matching is applied.
        session_maker: Async session maker for database transactions.
        client: Client for ScrapeCreators API interactions.
        settings: Application configuration settings.

    Returns:
        An initialized platform parser instance matching the requested platform.

    Raises:
        ValueError: If the provided platform string is not supported.
    """
    platform_registry = {
        "INSTAGRAM": InstagramParser,
        "THREADS": ThreadsParser,
        "TIKTOK": TikTokParser,
        "YOUTUBE": YouTubeParser,
    }

    normalized_platform = platform.upper()
    parser_cls = platform_registry.get(normalized_platform)

    if parser_cls is None:
        supported_platforms = ", ".join(platform_registry.keys())
        raise ValueError(
            f"Unsupported platform '{platform}'. Supported platforms: {supported_platforms}"
        )

    return parser_cls(
        session_maker=session_maker,
        client=client,
        settings=settings,
    )
