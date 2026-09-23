import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_KEYS_TO_KEEP = {
    "user",
    "media_id",
    "pk",
    "code",
    "shortcode",
    "media_type",
    "video_duration",
    "duration",
    "is_video",
    "play_count",
    "video_view_count",
    "comment_count",
    "like_count",
    "video_url",
    "coauthor_producers",
    "edge_media_to_tagged_user",
    "usertags",
    "tagged_users",
    "carousel_media",
    "clips_metadata",
    "accessibility_caption",
    "hashtags",
}


def extract_instagram_subscribers(user_dict: dict[str, Any]) -> int:
    edge_followed_by = user_dict.get("edge_followed_by")
    if isinstance(edge_followed_by, dict):
        count = edge_followed_by.get("count")
        if count is not None:
            try:
                return int(count)
            except (ValueError, TypeError):
                pass

    followers = user_dict.get("followers") or user_dict.get("followers_count")
    if followers is not None:
        try:
            return int(followers)
        except (ValueError, TypeError):
            pass

    return 0


def extract_instagram_content_text(node_dict: dict[str, Any]) -> str | None:
    caption = node_dict.get("caption")
    if isinstance(caption, dict):
        text = caption.get("text")
        if text and isinstance(text, str):
            return text
    elif isinstance(caption, str) and caption:
        return caption

    caption_text = node_dict.get("caption_text")
    if caption_text and isinstance(caption_text, str):
        return caption_text

    text = node_dict.get("text")
    if text and isinstance(text, str):
        return text

    return None


def extract_instagram_published_at(node_dict: dict[str, Any]) -> datetime:
    raw_time: Any = (
        node_dict.get("taken_at")
        or node_dict.get("taken_at_timestamp")
        or node_dict.get("created_at")
        or node_dict.get("timestamp")
    )

    published_at: datetime = datetime.now(timezone.utc)

    if raw_time is not None:
        try:
            if isinstance(raw_time, (int, float)):
                ts_value: float = float(raw_time)

                if ts_value > 9999999999:
                    ts_value = ts_value / 1000.0

                published_at = datetime.fromtimestamp(ts_value, tz=timezone.utc)

            elif isinstance(raw_time, str):
                normalized: str = raw_time.replace("Z", "+00:00")
                published_at = datetime.fromisoformat(normalized)

                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
                else:
                    published_at = published_at.astimezone(timezone.utc)

        except Exception:
            published_at = datetime.now(timezone.utc)

    return published_at


def extract_instagram_video_url(node_dict: dict[str, Any]) -> str | None:
    video_url = node_dict.get("video_url")
    if isinstance(video_url, str) and video_url:
        return video_url

    video_versions = node_dict.get("video_versions")
    if isinstance(video_versions, list) and video_versions:
        first = video_versions[0]
        if isinstance(first, dict):
            url = first.get("url")
            if isinstance(url, str) and url:
                return url

    return None


def extract_instagram_metrics(
    node_dict: dict[str, Any],
) -> tuple[int | None, int | None]:
    likes_count: int | None = None
    raw_likes = node_dict.get("like_count") or node_dict.get("likes")
    if raw_likes is not None:
        try:
            likes_count = int(raw_likes)
        except (ValueError, TypeError):
            pass

    comments_count: int | None = None
    raw_comments = node_dict.get("comment_count") or node_dict.get("comments")
    if raw_comments is not None:
        try:
            comments_count = int(raw_comments)
        except (ValueError, TypeError):
            pass

    return (likes_count, comments_count)


def prune_instagram_payload(item: dict[str, Any]) -> dict[str, Any]:
    pruned: dict[str, Any] = {}
    for key in _KEYS_TO_KEEP:
        if key in item:
            pruned[key] = item[key]

    carousel_media = item.get("carousel_media")
    if isinstance(carousel_media, list):
        pruned_slides: list[dict[str, Any]] = []
        for slide in carousel_media:
            if not isinstance(slide, dict):
                continue
            pruned_slide: dict[str, Any] = {}
            for slide_key in ("pk", "media_type", "usertags"):
                if slide_key in slide:
                    pruned_slide[slide_key] = slide[slide_key]
            pruned_slides.append(pruned_slide)
        pruned["carousel_media"] = pruned_slides

    caption = item.get("caption")
    if caption is not None:
        if isinstance(caption, dict):
            pruned["caption"] = {"text": caption.get("text", "")}
        elif isinstance(caption, str):
            pruned["caption"] = caption

    return pruned
