"""
Instagram-specific JSON payload parsers for social media profile parsing.

This module provides:
    - Functions to extract subscriber counts, content text, timestamps, video URLs, and metrics
    - Helpers to build author profile metadata from Instagram API responses
"""

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def extract_instagram_subscribers(user_dict: dict[str, Any]) -> int:
    """Extract subscriber count from Instagram user data dictionary.

    Tries multiple possible fields to find the follower count:
    - edge_followed_by.count (GraphQL structure)
    - followers / followers_count

    Args:
        user_dict: User object dictionary from Instagram API response.

    Returns:
        Subscriber count as integer, or 0 if not found or invalid.
    """
    # Try edge_followed_by.count first (GraphQL structure)
    edge_followed_by = user_dict.get("edge_followed_by")
    if isinstance(edge_followed_by, dict):
        count = edge_followed_by.get("count")
        if count is not None:
            try:
                return int(count)
            except (ValueError, TypeError):
                pass

    # Try followers field
    followers = user_dict.get("followers") or user_dict.get("followers_count")
    if followers is not None:
        try:
            return int(followers)
        except (ValueError, TypeError):
            pass

    return 0


def extract_instagram_content_text(node_dict: dict[str, Any]) -> str | None:
    """Extract text content from Instagram content node dictionary.

    Tries multiple fields to find the content text:
    - edge_media_to_caption.edges[0].node.text
    - accessibility_caption (fallback)

    Args:
        node_dict: Content node dictionary from Instagram API response.

    Returns:
        Extracted text content, or None if not found.
    """
    # Try edge_media_to_caption.edges[0].node.text
    edge_media_to_caption = node_dict.get("edge_media_to_caption")
    if isinstance(edge_media_to_caption, dict):
        edges = edge_media_to_caption.get("edges", [])
        if isinstance(edges, list) and edges:
            first_edge = edges[0]
            if isinstance(first_edge, dict):
                text_node = first_edge.get("node", {})
                if isinstance(text_node, dict) and text_node.get("text"):
                    return str(text_node["text"])

    # Fallback to accessibility_caption
    accessibility_caption = node_dict.get("accessibility_caption")
    if accessibility_caption and isinstance(accessibility_caption, str):
        return accessibility_caption

    return None


def extract_instagram_published_at(node_dict: dict[str, Any]) -> datetime:
    """Extract and convert published timestamp to timezone-aware datetime (UTC).

    Robustly extracts timestamp from multiple possible fields:
    - taken_at, taken_at_timestamp, created_at, timestamp

    Handles:
    - Integer/float UNIX timestamps (seconds or milliseconds since epoch)
    - ISO 8601 strings (with 'Z' suffix or timezone offset)
    - Falls back to datetime.now(timezone.utc) if extraction fails

    This function NEVER returns None to prevent database NotNullViolationError.

    Args:
        node_dict: Content node dictionary from Instagram API response.

    Returns:
        Timezone-aware datetime in UTC (always valid, never None).
    """
    # Try multiple timestamp keys in order of likelihood
    raw_time: Any = (
        node_dict.get("taken_at")
        or node_dict.get("taken_at_timestamp")
        or node_dict.get("created_at")
        or node_dict.get("timestamp")
    )

    # Default fallback: current UTC time (prevents NotNullViolationError)
    published_at: datetime = datetime.now(timezone.utc)

    if raw_time is not None:
        try:
            if isinstance(raw_time, (int, float)):
                # Convert to float for consistent handling
                ts_value: float = float(raw_time)

                # Handle milliseconds timestamps (values > 9999999999)
                if ts_value > 9999999999:
                    ts_value = ts_value / 1000.0

                published_at = datetime.fromtimestamp(ts_value, tz=timezone.utc)

            elif isinstance(raw_time, str):
                # Handle ISO 8601 strings
                # Replace 'Z' suffix with '+00:00' for fromisoformat compatibility
                normalized: str = raw_time.replace("Z", "+00:00")
                published_at = datetime.fromisoformat(normalized)

                # Ensure timezone-aware: attach UTC if naive
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
                else:
                    # Convert to UTC for consistency
                    published_at = published_at.astimezone(timezone.utc)

        except Exception:
            # Any parsing failure: fall back to current UTC time
            published_at = datetime.now(timezone.utc)

    return published_at


def extract_instagram_video_url(node_dict: dict[str, Any]) -> str | None:
    """Extract high-quality direct MP4 URL from Instagram video node.

    Tries multiple fields to find the best quality video URL for
    downstream GPU worker processing:
    - video_url (highest quality)
    - display_url (if from cdninstagram)
    - video_resources (sorted by profile for best quality)

    Args:
        node_dict: Content node dictionary from Instagram API response.

    Returns:
        Direct MP4 URL string, or None if not found.
    """
    # Try video_url field first (highest quality)
    video_url: str | None = node_dict.get("video_url")
    if video_url and isinstance(video_url, str):
        return video_url

    # Try display_url as fallback (may be image, check)
    display_url: str | None = node_dict.get("display_url")
    if (
        display_url
        and isinstance(display_url, str)
        and "cdninstagram" in display_url
    ):
        return display_url

    # Try video_resources for highest quality
    video_resources = node_dict.get("video_resources")
    if isinstance(video_resources, list) and video_resources:
        # Sort by profile (higher is better quality) and get the best
        sorted_resources = sorted(
            video_resources,
            key=lambda r: r.get("profile", 0) if isinstance(r, dict) else 0,
            reverse=True,
        )
        best_resource = sorted_resources[0]
        if isinstance(best_resource, dict):
            return best_resource.get("src")

    return None


def extract_instagram_metrics(
    node_dict: dict[str, Any],
) -> tuple[int | None, int | None]:
    """Extract engagement metrics from Instagram content node.

    Extracts likes count and comments count from the node dictionary.

    Args:
        node_dict: Content node dictionary from Instagram API response.

    Returns:
        A tuple of (likes_count, comments_count).
        Each value is an integer if found, None otherwise.
    """
    # Extract likes count from edge_media_preview_like.count
    likes_count: int | None = None
    edge_media_preview_like = node_dict.get("edge_media_preview_like")
    if isinstance(edge_media_preview_like, dict):
        count = edge_media_preview_like.get("count")
        if count is not None:
            try:
                likes_count = int(count)
            except (ValueError, TypeError):
                pass

    # Extract comments count from edge_media_to_comment.count
    comments_count: int | None = None
    edge_media_to_comment = node_dict.get("edge_media_to_comment")
    if isinstance(edge_media_to_comment, dict):
        count = edge_media_to_comment.get("count")
        if count is not None:
            try:
                comments_count = int(count)
            except (ValueError, TypeError):
                pass

    return (likes_count, comments_count)


def build_instagram_author_metadata(user_dict: dict[str, Any]) -> dict[str, Any]:
    """Build author profile metadata dictionary from Instagram user data.

    Extracts external links, contact information, location, language,
    and other profile-level data. This metadata is typically stored inside
    Content.raw_metadata under the "author_profile_metadata" key.

    Uses parse_profile_contacts to extract emails, Telegram handles,
    and external links from the biography.

    Args:
        user_dict: User object dictionary from Instagram API response.

    Returns:
        Dictionary containing standardized author profile metadata:
        - profile_link: Instagram profile URL
        - bio_description: Biography text
        - external_links: List of external links
        - contacts: List of contact strings (emails, telegram handles)
        - advertising_contacts: Same as contacts
        - language: Language code (None if not available)
        - location: Location string if available
        - geo_data: Geo coordinates if available (None for Instagram)
    """
    from src.parser.creators.core.contacts import parse_profile_contacts

    biography: str | None = user_dict.get("biography")
    username: str | None = user_dict.get("username")
    full_name: str | None = user_dict.get("full_name")

    # Use shared utility to parse contacts from biography and external_url
    external_url: str | None = user_dict.get("external_url")
    contacts_dict: dict[str, Any] = parse_profile_contacts(
        biography, external_url
    )

    # Build profile link
    profile_link: str | None = None
    if username:
        profile_link = f"https://instagram.com/{username}"

    # Build contacts list in the format expected by OpenSPG
    contacts: list[str] = []
    for email in contacts_dict.get("emails", []):
        contacts.append(f"email:{email}")
    for handle in contacts_dict.get("telegram_handles", []):
        contacts.append(f"telegram:@{handle}")

    # Add external links as contact entries
    external_links: list[str] = contacts_dict.get("external_links", [])

    # Extract location from business address JSON if available
    location: str | None = None
    business_address = user_dict.get("business_address_json")
    if business_address and isinstance(business_address, dict):
        location = business_address.get(
            "street_address"
        ) or business_address.get("city")
    elif user_dict.get("location"):
        location = user_dict.get("location")

    # Language is not directly available from Instagram API
    language: str | None = None

    # Geo-data (latitude/longitude) is not typically available from basic API
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

    # Remove None values for cleaner JSON
    return {k: v for k, v in author_metadata.items() if v is not None}
