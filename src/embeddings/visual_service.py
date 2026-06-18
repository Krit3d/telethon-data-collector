from __future__ import annotations

import logging

from src.config.config import Settings
from src.embeddings.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class VisualEmbeddingService(VideoProcessor):

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def extract_visual_embedding(
        self, video_url: str | None
    ) -> list[float] | None:
        if not video_url:
            logger.debug("No video URL provided, skipping visual embedding extraction")
            return None

        if not self.settings.enable_visual_embeddings:
            logger.debug("Visual embeddings are disabled in settings")
            return None

        logger.debug("Extracting visual embedding for video: %s", video_url)

        return [0.0] * self.settings.visual_embedding_dim
