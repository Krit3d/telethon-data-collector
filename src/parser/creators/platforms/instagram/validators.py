import logging
from typing import Any

from src.parser.creators.core.utils import is_russian_text

logger = logging.getLogger(__name__)

MIN_SUBSCRIBERS: int = 5000
MAX_SUBSCRIBERS: int = 1000000


def validate_follower_count(subscribers: int) -> bool:
    return MIN_SUBSCRIBERS <= subscribers <= MAX_SUBSCRIBERS


def check_cyrillic_stage1(biography: str | None, full_name: str | None) -> bool:
    has_cyrillic_bio = is_russian_text(biography)
    has_cyrillic_name = is_russian_text(full_name)
    return has_cyrillic_bio or has_cyrillic_name


def check_cyrillic_stage2(aggregated_text: str) -> bool:
    return is_russian_text(aggregated_text)


def has_commercial_music(item: dict[str, Any]) -> bool:
    if item.get("music_info") is not None or item.get("music_metadata") is not None:
        return True

    clips_metadata: dict[str, Any] = item.get("clips_metadata") or {}
    if clips_metadata.get("audio_type") == "music" or clips_metadata.get("music_info") is not None:
        return True

    original_sound_info: dict[str, Any] | None = clips_metadata.get("original_sound_info")
    if original_sound_info is not None and original_sound_info.get("music_canonical_id") is not None:
        return True

    clips_music_attribution_info: dict[str, Any] = item.get("clips_music_attribution_info") or {}
    if clips_music_attribution_info.get("uses_original_audio") is False:
        return True

    music_metadata: dict[str, Any] = item.get("music_metadata") or {}
    if music_metadata.get("uses_original_audio") is False:
        return True

    return False
