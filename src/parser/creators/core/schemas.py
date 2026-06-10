"""
Pydantic V2 schemas for OpenSPG-compliant social media parser.

This module defines data models for account metadata, post metadata,
and related structures used in social media parsing and contact extraction.
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, model_validator


class GeoData(BaseModel):
    """
    Geographical data for account location.

    Attributes:
        city: City name.
        country: Country name.
        coordinates: List of [longitude, latitude] coordinates.
    """

    city: str | None = None
    country: str | None = None
    coordinates: list[float] | None = None


class MetricsEntry(BaseModel):
    """
    Metrics snapshot at a specific point in time.

    Attributes:
        subscribers_count: Number of subscribers/followers.
        posts_count: Number of posts published.
        timestamp: ISO 8601 timestamp of the metrics snapshot.
    """

    subscribers_count: int | None = None
    posts_count: int | None = None
    timestamp: str


class Contacts(BaseModel):
    """
    Contact information extracted from profile.

    Attributes:
        emails: List of email addresses.
        phones: List of phone numbers.
        telegram_channels: List of Telegram channel handles.
        telegram_personal: List of Telegram personal profile handles.
    """

    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    telegram_channels: list[str] = Field(default_factory=list)
    telegram_personal: list[str] = Field(default_factory=list)


class ExternalPlatforms(BaseModel):
    """
    External platform links found in profile.

    Attributes:
        vk: VKontakte profile URL or handle.
        youtube: YouTube channel URL or handle.
        threads: Threads profile URL or handle.
        tiktok: TikTok profile URL or handle.
    """

    vk: str | None = None
    youtube: str | None = None
    threads: str | None = None
    tiktok: str | None = None


class AccountMetadata(BaseModel):
    """
    Complete metadata for a social media account.

    Attributes:
        profile_url: URL of the profile.
        biography: Profile biography text.
        category: Account category (e.g., "creator", "business").
        language: Detected or declared language.
        location: Location string from profile.
        contacts: Extracted contact information.
        external_platforms: Links to external platforms.
        link_in_bio: First matched link-in-bio URL (e.g., taplink, linktree).
        website: Primary website URL.
        geo_data: Geographical data for the account.
        metrics_history: Historical metrics entries.
        raw_profile_payload: Raw API response data.
        extracted_at: ISO 8601 timestamp when data was extracted.
    """

    profile_url: str | None = None
    biography: str | None = None
    category: str | None = None
    language: str | None = None
    location: str | None = None
    contacts: Contacts = Field(default_factory=Contacts)
    external_platforms: ExternalPlatforms = Field(default_factory=ExternalPlatforms)
    link_in_bio: str | None = None
    website: str | None = None
    geo_data: GeoData | None = None
    metrics_history: list[MetricsEntry] = Field(default_factory=list)
    raw_profile_payload: dict[str, Any] | None = None
    extracted_at: str

    @classmethod
    def create_with_timestamp(cls, **kwargs) -> "AccountMetadata":
        """
        Create an AccountMetadata instance with current UTC timestamp.

        Args:
            **kwargs: Fields to pass to the AccountMetadata constructor.

        Returns:
            AccountMetadata instance with extracted_at set to current UTC time.
        """
        if "extracted_at" not in kwargs:
            kwargs["extracted_at"] = datetime.now(timezone.utc).isoformat()
        return cls(**kwargs)


class PostGeoData(BaseModel):
    """
    Geographical data for a post location.

    Attributes:
        location_id: Platform-specific location identifier.
        name: Location name.
        lat: Latitude coordinate.
        lng: Longitude coordinate.
    """

    location_id: str | None = None
    name: str | None = None
    lat: float | None = None
    lng: float | None = None


class PlatformMetrics(BaseModel):
    """
    Platform-specific engagement metrics for a post.

    Attributes:
        likes: Number of likes/reactions.
        comments_count: Number of comments.
        views: Number of views.
        shares: Number of shares/reposts.
        plays: Number of video plays.
    """

    likes: int | None = None
    comments_count: int | None = None
    views: int | None = None
    shares: int | None = None
    plays: int | None = None


class AuthorProfileSnapshot(BaseModel):
    """
    Snapshot of author profile data at post extraction time.

    Attributes:
        username: Author's username/handle.
        title: Author's display name or title.
    """

    username: str
    title: str | None = None


class ContentMetadata(BaseModel):
    """
    Metadata for social media post content.

    Attributes:
        video_url: URL to video content if available.
        category: Content category.
        language: Detected or declared language.
        post_type: Type of post (e.g., "reel", "post", "carousel").
        platform_metrics: Engagement metrics for the post.
        geo_data: Geographical data for the post.
        author_profile_snapshot: Author information at extraction time.
        raw_item_payload: Raw API response data for the post.
        extracted_at: ISO 8601 timestamp when data was extracted.
    """

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
        """
        Prune heavy, redundant fields from raw_item_payload before validation.

        This validator removes memory-intensive fields from the raw API payload
        to prevent database bloat when storing in JSONB columns. The pruned
        fields are typically large nested objects that are not needed for
        analysis but consume significant storage space.

        Args:
            data: Input data dictionary or object before validation.

        Returns:
            Modified data with heavy fields removed from raw_item_payload,
            or unmodified data if not a dictionary.
        """
        # Only process if data is a dictionary
        if not isinstance(data, dict):
            return data

        # Check if raw_item_payload exists and is a dictionary
        raw_payload = data.get("raw_item_payload")
        if not isinstance(raw_payload, dict):
            return data

        # List of heavy fields to remove from raw_item_payload
        heavy_fields = [
            "video_dash_manifest",
            "image_versions2",
            "user",
            "owner",
            "clips_metadata",
            "scrubber_spritesheet_info_candidates",
            "organic_tracking_token",
        ]

        # Safely remove each heavy field if present
        for field in heavy_fields:
            raw_payload.pop(field, None)

        return data

    @classmethod
    def create_with_timestamp(cls, **kwargs) -> "ContentMetadata":
        """
        Create a ContentMetadata instance with current UTC timestamp.

        Args:
            **kwargs: Fields to pass to the ContentMetadata constructor.

        Returns:
            ContentMetadata instance with extracted_at set to current UTC time.
        """
        if "extracted_at" not in kwargs:
            kwargs["extracted_at"] = datetime.now(timezone.utc).isoformat()
        return cls(**kwargs)


class InstagramContentMetadata(ContentMetadata):
    """
    Instagram-specific content metadata.

    Inherits all fields from ContentMetadata and adds Instagram-specific
    fields for Reel detection and validation.

    Attributes:
        is_reel: Whether the post is an Instagram Reel.
    """

    is_reel: bool | None = None


class TikTokContentMetadata(ContentMetadata):
    """
    TikTok-specific content metadata.

    Inherits all fields from ContentMetadata for TikTok video content.
    TikTok videos are always short-form, so no additional type field is needed.
    """


class YouTubeContentMetadata(ContentMetadata):
    """
    YouTube-specific content metadata.

    Inherits all fields from ContentMetadata and adds YouTube-specific
    fields for Shorts detection and video duration tracking.

    Attributes:
        is_short: Whether the video is a YouTube Short.
        duration_seconds: Video duration in seconds.
    """

    is_short: bool | None = None
    duration_seconds: float | None = None
