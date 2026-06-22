from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, model_validator


def _safe_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    return None


HEAVY_PROFILE_KEYS: list[str] = [
    "chaining_results",
    "facebook_pages",
    "linked_facebook_page",
    "mutual_followers_data",
    "eligible_promotions",
    "ad_metadata",
    "hd_profile_pic_versions",
    "hd_profile_pic_url_info",
    "bio_links",
    "about_your_account_blurb",
    "edge_owner_to_timeline_media",
    "edge_felix_video_timeline",
    "edge_saved_media",
    "edge_media_collections",
    "edge_mutual_followed_by",
    "edge_related_profiles",
    "biography_with_entities",
    "fb_profile_biolink",
    "profile_pic_url",
    "profile_pic_url_hd",
]


class GeoData(BaseModel):
    city: str | None = None
    country: str | None = None
    coordinates: list[float] | None = None


class MetricsEntry(BaseModel):
    subscribers_count: int | None = None
    posts_count: int | None = None
    timestamp: str


class Contacts(BaseModel):
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    telegram_handles: list[str] = Field(default_factory=list)
    telegram_channels: list[str] = Field(default_factory=list)
    telegram_personal: list[str] = Field(default_factory=list)
    advertising_emails: list[str] = Field(default_factory=list)
    advertising_telegrams: list[str] = Field(default_factory=list)


class AccountMetadata(BaseModel):
    profile_url: str | None = None
    biography: str | None = None
    category: str | None = None
    language: str | None = None
    location: str | None = None
    contacts: Contacts = Field(default_factory=Contacts)
    external_platforms: dict[str, str | None] = Field(default_factory=dict)
    link_in_bio: str | None = None
    website: str | None = None
    geo_data: GeoData | None = None
    metrics_history: list[MetricsEntry] = Field(default_factory=list)
    external_links: list[str] = Field(default_factory=list)
    raw_profile_payload: dict[str, Any] | None = None
    extracted_at: str
    is_verified: bool = False
    is_business: bool = False

    @model_validator(mode="before")
    @classmethod
    def prune_raw_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        raw_payload = data.get("raw_profile_payload")
        if not isinstance(raw_payload, dict):
            return data

        if "is_verified" not in data:
            data["is_verified"] = raw_payload.get("is_verified", False)

        if "is_business" not in data:
            data["is_business"] = raw_payload.get(
                "is_business_account", raw_payload.get("is_professional_account", False)
            )

        for key in HEAVY_PROFILE_KEYS:
            raw_payload.pop(key, None)

        return data

    @classmethod
    def create_with_timestamp(cls, **kwargs) -> "AccountMetadata":
        if "extracted_at" not in kwargs:
            kwargs["extracted_at"] = datetime.now(timezone.utc).isoformat()
        return cls(**kwargs)


class PostGeoData(BaseModel):
    location_id: str | None = None
    name: str | None = None
    lat: float | None = None
    lng: float | None = None


class PlatformMetrics(BaseModel):
    likes: int | None = None
    comments_count: int | None = None
    views: int | None = None
    shares: int | None = None
    plays: int | None = None


class AuthorProfileSnapshot(BaseModel):
    username: str
    title: str | None = None


class ContentMetadata(BaseModel):
    video_url: str | None = None
    category: str | None = None
    language: str | None = None
    post_type: str
    post_url: str | None = None
    platform_metrics: PlatformMetrics | None = None
    geo_data: PostGeoData | None = None
    author_profile_snapshot: AuthorProfileSnapshot | None = None
    raw_item_payload: dict[str, Any] | None = None
    extracted_at: str
    local_clip_path: str | None = None
    video_processing_status: str = "pending"
    transcription_status: str = "pending"
    hashtags: list[str] = Field(default_factory=list)
    coauthors: list[str] = Field(default_factory=list)
    tagged_users: list[str] = Field(default_factory=list)
    music_title: str | None = None
    music_author: str | None = None
    accessibility_caption: str | None = None

    @model_validator(mode="before")
    @classmethod
    def prune_raw_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        raw_payload = data.get("raw_item_payload")
        if not isinstance(raw_payload, dict):
            return data

        if not data.get("accessibility_caption"):
            data["accessibility_caption"] = raw_payload.get("accessibility_caption")

        coauthor_producers = raw_payload.get("coauthor_producers")
        if isinstance(coauthor_producers, list) and not data.get("coauthors"):
            data["coauthors"] = [
                u.get("username", "")
                for u in coauthor_producers
                if isinstance(u, dict) and u.get("username")
            ]

        tagged_edges = raw_payload.get("edge_media_to_tagged_user")
        if isinstance(tagged_edges, dict) and not data.get("tagged_users"):
            edges = tagged_edges.get("edges")
            if isinstance(edges, list):
                data["tagged_users"] = [
                    e.get("node", {}).get("user", {}).get("username", "")
                    for e in edges
                    if isinstance(e, dict)
                    and isinstance(e.get("node"), dict)
                    and isinstance(e["node"].get("user"), dict)
                    and e["node"]["user"].get("username")
                ]

        clips_meta = raw_payload.get("clips_metadata")
        if isinstance(clips_meta, dict) and not data.get("music_title"):
            music_info = clips_meta.get("clips_music_attribution_info")
            if isinstance(music_info, dict):
                data["music_title"] = music_info.get("song_name")
            if isinstance(music_info, dict) and not data.get("music_author"):
                data["music_author"] = music_info.get("artist_name")

        location = raw_payload.get("location")
        if isinstance(location, dict) and not data.get("geo_data"):
            location_id = location.get("pk") or location.get("id")
            data["geo_data"] = {
                "location_id": str(location_id) if location_id is not None else None,
                "name": location.get("name"),
                "lat": location.get("lat") or location.get("latitude"),
                "lng": location.get("lng") or location.get("longitude"),
            }

        if "platform_metrics" not in data:
            likes = (
                _safe_int(raw_payload.get("like_count"))
                or _safe_int(raw_payload.get("likes"))
                or None
            )
            if likes is None:
                edge_like = raw_payload.get("edge_media_preview_like")
                if isinstance(edge_like, dict):
                    likes = _safe_int(edge_like.get("count"))

            comments_count = (
                _safe_int(raw_payload.get("comment_count"))
                or _safe_int(raw_payload.get("comments"))
                or None
            )
            if comments_count is None:
                edge_comment = raw_payload.get("edge_media_to_parent_comment")
                if isinstance(edge_comment, dict):
                    comments_count = _safe_int(edge_comment.get("count"))

            plays = _safe_int(raw_payload.get("play_count")) or _safe_int(raw_payload.get("plays")) or None

            views = (
                _safe_int(raw_payload.get("view_count"))
                or _safe_int(raw_payload.get("video_view_count"))
                or None
            )
            is_video = (
                raw_payload.get("media_type") == 2
                or raw_payload.get("product_type") in ("video", "clips", "reels")
                or data.get("post_type") in ("video", "reel")
            )
            if views is None and is_video:
                views = plays

            shares = _safe_int(raw_payload.get("share_count")) or _safe_int(raw_payload.get("shares")) or None

            data["platform_metrics"] = {
                "likes": likes,
                "comments_count": comments_count,
                "views": views,
                "shares": shares,
                "plays": plays,
            }

        heavy_fields = [
            "video_dash_manifest",
            "image_versions2",
            "user",
            "owner",
            "clips_metadata",
            "scrubber_spritesheet_info_candidates",
            "organic_tracking_token",
        ]

        for field in heavy_fields:
            raw_payload.pop(field, None)

        return data

    @classmethod
    def create_with_timestamp(cls, **kwargs) -> "ContentMetadata":
        if "extracted_at" not in kwargs:
            kwargs["extracted_at"] = datetime.now(timezone.utc).isoformat()
        return cls(**kwargs)


class InstagramContentMetadata(ContentMetadata):
    is_reel: bool | None = None


class TikTokContentMetadata(ContentMetadata):
    pass


class YouTubeContentMetadata(ContentMetadata):
    is_short: bool | None = None
    duration_seconds: float | None = None
