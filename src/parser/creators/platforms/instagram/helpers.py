import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.parser.creators.core.contacts import parse_profile_contacts

logger = logging.getLogger(__name__)

_KEYS_TO_KEEP = {
    "id",
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
    "clips_metadata",
    "location",
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


def build_instagram_author_metadata(user_dict: dict[str, Any]) -> dict[str, Any]:
    biography: str | None = user_dict.get("biography")
    username: str | None = user_dict.get("username")

    external_url: str | None = user_dict.get("external_url")
    contacts_dict: dict[str, Any] = parse_profile_contacts(
        biography, external_url
    )

    profile_link: str | None = None
    if username:
        profile_link = f"https://instagram.com/{username}"

    contacts: list[str] = []
    for email in contacts_dict.get("emails", []):
        contacts.append(f"email:{email}")
    for handle in contacts_dict.get("telegram_handles", []):
        contacts.append(f"telegram:@{handle}")

    external_links: list[str] = contacts_dict.get("external_links", [])

    location: str | None = None
    business_address = user_dict.get("business_address_json")
    if business_address and isinstance(business_address, dict):
        location = business_address.get(
            "street_address"
        ) or business_address.get("city")
    elif user_dict.get("location"):
        location = user_dict.get("location")

    language: str | None = None
    geo_data: dict[str, float] | None = None

    author_metadata: dict[str, Any] = {
        "profile_link": profile_link,
        "bio_description": biography,
        "external_links": external_links if external_links else None,
        "contacts": contacts if contacts else None,
        "advertising_contacts": contacts if contacts else None,
        "language": language,
        "location": location,
        "geo_data": geo_data,
    }

    return {k: v for k, v in author_metadata.items() if v is not None}


def prune_instagram_payload(item: dict[str, Any]) -> dict[str, Any]:
    pruned: dict[str, Any] = {}
    for key in _KEYS_TO_KEEP:
        if key in item:
            pruned[key] = item[key]

    caption = item.get("caption")
    if caption is not None:
        if isinstance(caption, dict):
            pruned["caption"] = {"text": caption.get("text", "")}
        elif isinstance(caption, str):
            pruned["caption"] = caption

    return pruned


def extract_instagram_geo_data(
    profile: dict[str, Any],
    biography: str | None,
    full_name: str,
) -> tuple[str | None, dict[str, Any] | None]:
    location_str: str | None = None
    geo_data: dict[str, Any] | None = None

    try:
        business_address_raw = profile.get("business_address_json")
        if business_address_raw is not None:
            address_dict: dict[str, Any] | None = None
            if isinstance(business_address_raw, str):
                try:
                    parsed = json.loads(business_address_raw)
                    if isinstance(parsed, dict):
                        address_dict = parsed
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(business_address_raw, dict):
                address_dict = business_address_raw

            if address_dict is not None:
                city_name = address_dict.get("city_name")
                if city_name and isinstance(city_name, str):
                    location_str = city_name
                    parts = [p.strip() for p in city_name.split(",") if p.strip()]
                    city = parts[0] if parts else city_name
                    country = parts[1] if len(parts) > 1 else "Russia"
                    geo_data = {"city": city, "country": country}

        if geo_data is None:
            fallback_name = profile.get("city_name") or profile.get("location")
            if fallback_name and isinstance(fallback_name, str):
                location_str = fallback_name
                parts = [p.strip() for p in fallback_name.split(",") if p.strip()]
                city = parts[0] if parts else fallback_name
                country = parts[1] if len(parts) > 1 else "Russia"
                geo_data = {"city": city, "country": country}

        if geo_data is None:
            city_aliases: dict[str, tuple[str, str]] = {
                "москва": ("Moscow", "Russia"),
                "москве": ("Moscow", "Russia"),
                "мск": ("Moscow", "Russia"),
                "санкт-петербург": ("Saint Petersburg", "Russia"),
                "спб": ("Saint Petersburg", "Russia"),
                "питер": ("Saint Petersburg", "Russia"),
                "ташкент": ("Tashkent", "Uzbekistan"),
                "алматы": ("Almaty", "Kazakhstan"),
                "алмата": ("Almaty", "Kazakhstan"),
                "астана": ("Astana", "Kazakhstan"),
                "караганда": ("Karaganda", "Kazakhstan"),
                "минск": ("Minsk", "Belarus"),
                "киев": ("Kyiv", "Ukraine"),
                "новосибирск": ("Novosibirsk", "Russia"),
                "екатеринбург": ("Yekaterinburg", "Russia"),
                "казань": ("Kazan", "Russia"),
                "краснодар": ("Krasnodar", "Russia"),
                "ростов": ("Rostov-on-Don", "Russia"),
                "самара": ("Samara", "Russia"),
                "уфа": ("Ufa", "Russia"),
                "челябинск": ("Chelyabinsk", "Russia"),
            }
            bio_lower = (biography or "").lower()
            fn_lower = (full_name or "").lower()
            search_text = f"{fn_lower} {bio_lower}"
            for alias, (city, country) in city_aliases.items():
                if alias in search_text:
                    location_str = f"{city}, {country}"
                    geo_data = {"city": city, "country": country}
                    break

    except Exception:
        pass

    return (location_str, geo_data)
