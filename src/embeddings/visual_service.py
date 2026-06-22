from __future__ import annotations

import logging

from src.config.config import Settings
from src.embeddings.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class VisualEmbeddingService(VideoProcessor):

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._disabled = not settings.enable_visual_embeddings
        self._session = None

        if self._disabled:
            logger.info(
                "Visual embedding service disabled via environment or settings"
            )
            return

        if _cuda_available():
            self._session = _build_cuda_session()
        else:
            logger.info(
                "CUDA provider not available; visual embedding service running on CPU"
            )
            self._session = _build_cpu_session()

    async def extract_visual_embedding(
        self, video_url: str | None
    ) -> list[float] | None:
        if not video_url:
            logger.debug("No video URL provided, skipping visual embedding extraction")
            return None

        if not self.settings.enable_visual_embeddings:
            logger.debug("Visual embeddings are disabled in settings")
            return None

        if self._session is None:
            logger.warning(
                "No ONNX session available; visual embedding cannot be computed"
            )
            return None

        logger.debug("Extracting visual embedding for video: %s", video_url)

        return [0.0] * self.settings.visual_embedding_dim


def _cuda_available() -> bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _build_cpu_session():
    try:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        return ort.InferenceSession(
            "visual_model.onnx",
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        logger.error("Failed to create CPU ONNX session: %s", exc)
        return None


def _build_cuda_session():
    try:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        return ort.InferenceSession(
            "visual_model.onnx",
            sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    except Exception as exc:
        logger.warning(
            "CUDA session creation failed (%s); falling back to CPU", exc
        )
        return _build_cpu_session()
