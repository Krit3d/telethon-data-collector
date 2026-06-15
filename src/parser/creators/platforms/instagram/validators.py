import logging

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


