from abc import ABC, abstractmethod


class VideoProcessor(ABC):
    @abstractmethod
    async def extract_visual_embedding(self, video_url: str | None) -> list[float] | None:
        ...
