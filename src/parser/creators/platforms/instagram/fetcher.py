import logging
from typing import Any

logger = logging.getLogger(__name__)

_MIN_DURATION = 0.0
_MAX_DURATION = 120.0


def _is_valid_video_item(item: dict[str, Any]) -> bool:
    is_video = (
        item.get("media_type") == 2
        or item.get("is_video") is True
        or (
            isinstance(item.get("video_versions"), list)
            and bool(item.get("video_versions"))
        )
    )
    return is_video


def _is_duration_valid(item: dict[str, Any]) -> bool:
    duration = item.get("video_duration") or item.get("duration", 0.0)
    return _MIN_DURATION < duration <= _MAX_DURATION


def _filter_page_items(
    items: list[dict[str, Any]],
    seen_ids: set[str],
    remaining: int,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for item in items:
        if len(collected) >= remaining:
            break

        item_id = item.get("id") or item.get("media_id") or item.get("pk")
        if not item_id:
            continue

        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)

        if not _is_valid_video_item(item):
            continue

        if not _is_duration_valid(item):
            continue

        collected.append(item)

    return collected


async def fetch_valid_instagram_videos(
    client: Any,
    handle: str,
    max_items: int,
) -> list[dict[str, Any]]:
    logger.info(
        "Fetching valid Instagram videos for handle: %s (max_items=%d)",
        handle,
        max_items,
    )

    try:
        response_page1 = await client.get(
            endpoint="/v2/instagram/user/posts",
            params={"handle": handle},
        )
    except Exception as e:
        logger.error(
            "Failed to fetch page 1 posts for %s: %s",
            handle,
            e,
        )
        raise

    items_page1: list[dict[str, Any]] = response_page1.get("items", [])

    if not items_page1:
        logger.info("No items returned for handle: %s", handle)
        return []

    seen_ids: set[str] = set()
    valid_items: list[dict[str, Any]] = _filter_page_items(
        items_page1, seen_ids, max_items,
    )

    if len(valid_items) < max_items:
        more_available = response_page1.get("more_available", False)
        cursor = (
            response_page1.get("profile_grid_items_cursor")
            or response_page1.get("next_max_id")
        )

        if more_available and cursor:
            logger.info(
                "Page 1 yielded %d valid videos for %s, fetching page 2",
                len(valid_items),
                handle,
            )

            try:
                response_page2 = await client.get(
                    endpoint="/v2/instagram/user/posts",
                    params={"handle": handle, "cursor": cursor},
                )
            except Exception as e:
                logger.warning(
                    "Failed to fetch page 2 for %s: %s. Using page 1 results only.",
                    handle,
                    e,
                )
            else:
                items_page2: list[dict[str, Any]] = response_page2.get("items", [])
                remaining = max_items - len(valid_items)
                page2_valid = _filter_page_items(items_page2, seen_ids, remaining)
                valid_items.extend(page2_valid)

    logger.info(
        "Collected %d valid videos for handle: %s",
        len(valid_items),
        handle,
    )

    return valid_items
