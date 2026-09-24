import logging
from typing import Any

from src.parser.creators.core.media_detector import detect_content_media

from .helpers import extract_instagram_published_at

logger = logging.getLogger(__name__)


async def fetch_recent_instagram_posts(
    client: Any,
    handle: str,
    target_posts: int = 4,
    target_reels: int = 8,
    max_pages: int = 4,
    max_total_items: int = 20,
) -> list[dict[str, Any]]:
    collected_items: dict[str, dict[str, Any]] = {}
    posts_count = 0
    reels_count = 0
    cursor: str | None = None
    pages_fetched = 0

    for page in range(max_pages):
        params: dict[str, Any] = {"handle": handle}
        if cursor:
            params["next_max_id"] = cursor

        response = await client.get(
            endpoint="/v2/instagram/user/posts",
            params=params,
        )

        pages_fetched = page + 1

        items = response.get("items", []) if isinstance(response, dict) else []
        if not items:
            break

        for item in items:
            if not isinstance(item, dict):
                continue

            item_id = str(item.get("id") or item.get("pk") or item.get("code") or "")
            if not item_id or item_id in collected_items:
                continue

            collected_items[item_id] = item

            post_type, _ = detect_content_media(platform="INSTAGRAM", payload=item)
            if post_type == "reel":
                reels_count += 1
            else:
                posts_count += 1

        if posts_count >= target_posts and reels_count >= target_reels:
            break

        more_available = response.get("more_available") if isinstance(response, dict) else False
        cursor = response.get("next_max_id") if isinstance(response, dict) else None

        if not more_available or not cursor:
            break

    if not collected_items:
        logger.info("No Instagram posts collected for handle: %s", handle)
        return []

    items_list = list(collected_items.values())
    items_list.sort(key=extract_instagram_published_at, reverse=True)
    result = items_list[:max_total_items]

    logger.info(
        "Collected %d Instagram items for handle: %s (posts=%d, reels=%d, pages=%d)",
        len(result),
        handle,
        posts_count,
        reels_count,
        pages_fetched,
    )

    return result
