import logging
from typing import Any

from src.parser.creators.core.media_detector import detect_content_media

from .helpers import extract_instagram_published_at

logger = logging.getLogger(__name__)


async def fetch_recent_instagram_posts(
    client: Any,
    handle: str,
    max_total_items: int = 12,
    target_posts: int | None = None,
    target_reels: int | None = None,
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    effective_reels = target_reels if target_reels is not None else (max_total_items * 2) // 3
    effective_posts = target_posts if target_posts is not None else max(0, max_total_items - effective_reels)
    effective_pages = max_pages if max_pages is not None else max(2, (max_total_items + 11) // 12)

    collected_reels: dict[str, dict[str, Any]] = {}
    collected_posts: dict[str, dict[str, Any]] = {}
    cursor: str | None = None
    pages_fetched = 0

    for page in range(effective_pages):
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
            if not item_id:
                continue

            if item_id in collected_reels or item_id in collected_posts:
                continue

            post_type, _ = detect_content_media(platform="INSTAGRAM", payload=item)
            if post_type == "reel":
                collected_reels[item_id] = item
            else:
                collected_posts[item_id] = item

        if len(collected_reels) >= effective_reels and len(collected_posts) >= effective_posts:
            break

        more_available = response.get("more_available") if isinstance(response, dict) else False
        cursor = response.get("next_max_id") if isinstance(response, dict) else None

        if not more_available or not cursor:
            break

    if not collected_reels and not collected_posts:
        logger.info("No Instagram posts collected for handle: %s", handle)
        return []

    sorted_reels = sorted(
        collected_reels.values(),
        key=extract_instagram_published_at,
        reverse=True,
    )
    sorted_posts = sorted(
        collected_posts.values(),
        key=extract_instagram_published_at,
        reverse=True,
    )

    selected_reels = sorted_reels[:effective_reels]
    selected_posts = sorted_posts[:effective_posts]

    remaining_slots = max_total_items - (len(selected_reels) + len(selected_posts))
    if remaining_slots > 0:
        selected_reels = selected_reels + sorted_reels[effective_reels : effective_reels + remaining_slots]

    remaining_slots = max_total_items - (len(selected_reels) + len(selected_posts))
    if remaining_slots > 0:
        selected_posts = selected_posts + sorted_posts[effective_posts : effective_posts + remaining_slots]

    result = selected_reels + selected_posts
    result.sort(key=extract_instagram_published_at, reverse=True)
    result = result[:max_total_items]

    logger.info(
        "Collected %d Instagram items for handle: %s (posts=%d, reels=%d, pages=%d)",
        len(result),
        handle,
        len(selected_posts),
        len(selected_reels),
        pages_fetched,
    )

    return result
