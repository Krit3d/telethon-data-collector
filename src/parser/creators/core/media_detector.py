from __future__ import annotations

from typing import Any

_INSTAGRAM_VIDEO_PRODUCT_TYPES = frozenset({"clips", "reels"})


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_transcription(transcription: str | None) -> bool:
    return bool(transcription and transcription.strip())


def _instagram_child_has_video(child: Any) -> bool:
    if not isinstance(child, dict):
        return False
    if child.get("media_type") == 2 or child.get("is_video") is True:
        return True
    return bool(child.get("video_versions"))


def _instagram_has_media(payload: dict[str, Any], transcription: str | None) -> bool:
    if _has_transcription(transcription):
        return True
    if payload.get("media_type") == 2 or payload.get("is_video") is True:
        return True
    if payload.get("video_versions"):
        return True
    if payload.get("product_type") in _INSTAGRAM_VIDEO_PRODUCT_TYPES:
        return True
    if payload.get("subtype_name_for_REST__") == "XDTClipsMedia":
        return True
    if "clips_metadata" in payload:
        return True
    if payload.get("media_type") == 8 or payload.get("carousel_media"):
        carousel = payload.get("carousel_media")
        if isinstance(carousel, (list, tuple)):
            for child in carousel:
                if _instagram_child_has_video(child):
                    return True
    return False


def _detect_instagram(
    payload: dict[str, Any],
    transcription: str | None,
    duration: float | None,
) -> tuple[str, bool]:
    is_carousel = (
        payload.get("media_type") == 8
        or bool(payload.get("carousel_media"))
        or payload.get("product_type") == "carousel_container"
    )
    if is_carousel:
        return "post", _instagram_has_media(payload, transcription)
    has_media = _instagram_has_media(payload, transcription)
    if not has_media:
        return "post", False
    if duration is None:
        duration = _to_float(payload.get("video_duration"))
        if duration is None:
            duration = _to_float(payload.get("duration"))
    if duration is not None and duration > 120.0:
        return "video", True
    return "reel", True


def _detect_youtube(
    payload: dict[str, Any],
    transcription: str | None,
    duration: float | None,
) -> tuple[str, bool]:
    has_media = bool(
        payload.get("video_id")
        or payload.get("video_stream")
        or payload.get("streaming_data")
        or payload.get("formats")
        or payload.get("is_video") is True
        or _has_transcription(transcription)
    )
    if not has_media:
        return "post", False
    if payload.get("is_short") is True:
        return "short", True
    if duration is not None and duration <= 60.0:
        return "short", True
    return "video", True


def _detect_tiktok(
    payload: dict[str, Any],
    transcription: str | None,
) -> tuple[str, bool]:
    has_media = bool(
        "video" in payload
        or "video_id" in payload
        or payload.get("video_versions")
        or payload.get("is_video") is True
        or _has_transcription(transcription)
    )
    if has_media:
        return "tiktok", True
    return "post", False


def _detect_telegram(
    payload: dict[str, Any],
    transcription: str | None,
) -> tuple[str, bool]:
    has_media = bool(
        "video" in payload
        or "video_note" in payload
        or _has_transcription(transcription)
    )
    if has_media:
        return "video", True
    return "post", False


def _detect_unknown(
    payload: dict[str, Any],
    transcription: str | None,
) -> tuple[str, bool]:
    if _has_transcription(transcription):
        return "video", True
    if (
        "video" in payload
        or "video_id" in payload
        or payload.get("video_versions")
        or payload.get("is_video") is True
    ):
        return "video", True
    return "post", False


def detect_content_media(
    platform: str,
    payload: dict[str, Any],
    transcription: str | None = None,
    duration: float | None = None,
) -> tuple[str, bool]:
    normalized = platform.strip().lower() if platform else ""
    if normalized == "instagram":
        return _detect_instagram(payload, transcription, duration)
    if normalized == "youtube":
        return _detect_youtube(payload, transcription, duration)
    if normalized == "tiktok":
        return _detect_tiktok(payload, transcription)
    if normalized == "telegram":
        return _detect_telegram(payload, transcription)
    return _detect_unknown(payload, transcription)
