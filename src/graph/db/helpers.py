import asyncio
import functools
import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar, cast

from sqlalchemy.exc import DBAPIError

try:
    import asyncpg.exceptions as _asyncpg_exc
except ImportError:
    _asyncpg_exc = None

logger = logging.getLogger(__name__)

ID_PREFIX_TO_LABEL: dict[str, str] = {
    "actor_": "Actor",
    "brand_": "Actor",
    "product_": "Entity",
    "topic_": "Entity",
    "hashtag_": "Entity",
    "entity_": "Entity",
    "event_": "Event",
    "publication_": "Event",
    "collaboration_": "Event",
    "place_": "Place",
    "region_": "Place",
    "content_": "Content",
}

F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])

_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 0.5


def _is_retryable_db_error(exc: BaseException) -> bool:
    if _asyncpg_exc is not None:
        if isinstance(exc, (_asyncpg_exc.SerializationError, _asyncpg_exc.DeadlockDetectedError)):
            return True
        if isinstance(exc, _asyncpg_exc.InternalServerError):
            if "Entity failed to be updated" in str(exc):
                return True
    if isinstance(exc, DBAPIError) and exc.orig is not None and _asyncpg_exc is not None:
        if isinstance(exc.orig, (_asyncpg_exc.SerializationError, _asyncpg_exc.DeadlockDetectedError)):
            return True
        if isinstance(exc.orig, _asyncpg_exc.InternalServerError):
            if "Entity failed to be updated" in str(exc):
                return True
    return False


def connection_retry(func: F) -> F:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        last_exc: BaseException | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                if not _is_retryable_db_error(exc) or attempt == _MAX_RETRIES:
                    raise
                last_exc = exc
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Retryable DB error on %s (attempt %d/%d): %s. "
                    "Retrying in %.2fs",
                    func.__qualname__,
                    attempt,
                    _MAX_RETRIES,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    return cast(F, wrapper)


def parse_agtype(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    text_val = value if isinstance(value, str) else str(value)
    text_val = text_val.strip()
    if len(text_val) >= 2 and (
        (text_val[0] == '"' and text_val[-1] == '"')
        or (text_val[0] == "'" and text_val[-1] == "'")
    ):
        text_val = text_val[1:-1]
    try:
        return json.loads(text_val)
    except (json.JSONDecodeError, TypeError):
        return text_val
