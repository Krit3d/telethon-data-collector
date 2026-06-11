from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, model_validator


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
    telegram_channels: list[str] = Field(default_factory=list)
    telegram_personal: list[str] = Field(default_factory=list)


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
    platform_metrics: PlatformMetrics | None = None
    geo_data: PostGeoData | None = None
    author_profile_snapshot: AuthorProfileSnapshot | None = None
    raw_item_payload: dict[str, Any] | None = None
    extracted_at: str

    @model_validator(mode="before")
    @classmethod
    def prune_raw_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        raw_payload = data.get("raw_item_payload")
        if not isinstance(raw_payload, dict):
            return data

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
