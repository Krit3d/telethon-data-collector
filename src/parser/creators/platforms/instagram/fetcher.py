import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _published_at_utc(item: dict[str, Any]) -> datetime:
    raw_ts = (
        item.get("taken_at")
        or item.get("published_at")
        or item.get("timestamp")
        or 0
    )
    try:
        ts = int(raw_ts)
    except (ValueError, TypeError):
        ts = 0
    return datetime.fromtimestamp(ts, tz=timezone.utc)


async def fetch_recent_instagram_posts(
    client: Any,
    handle: str,
    max_items: int,
) -> list[dict[str, Any]]:
    logger.info(
        "Fetching recent Instagram posts for handle: %s (max_items=%d)",
        handle,
        max_items,
    )

    try:
        response = await client.get(
            endpoint="/v2/instagram/user/posts",
            params={"handle": handle},
        )
    except Exception as e:
        logger.error(
            "Failed to fetch posts for %s: %s",
            handle,
            e,
        )
        raise

    items: list[dict[str, Any]] = response.get("items", [])

    if not items:
        logger.info("No items returned for handle: %s", handle)
        return []

    items.sort(key=_published_at_utc, reverse=True)
    items = items[:max_items]

    logger.info(
        "Collected %d recent posts for handle: %s",
        len(items),
        handle,
    )

    return items
