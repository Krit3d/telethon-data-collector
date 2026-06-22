import hashlib
import logging
from typing import Any

from pydantic import ValidationError

from src.graph.extractor.extraction_helpers import (
    clean_hashtag,
    find_entity,
    find_or_create_entity,
    find_or_create_relation,
    normalize_language,
    sanitize_id,
)
from src.graph.schema import ExtractedEntity, ExtractedRelation, PropertyType

logger = logging.getLogger(__name__)


def enrich_publication_node(
    entities: list[ExtractedEntity],
    relations: list[ExtractedRelation],
    source_node_id: str,
    post_id: int,
    post_metrics: dict[str, int | None],
    raw_metadata: dict[str, Any],
    platform: str | None,
    author_subscribers: int | None,
    process_language_data: bool,
) -> None:
    pub = find_entity(entities, source_node_id)
    if pub is None:
        logger.warning(
            "Publication entity %s not found in entities list",
            source_node_id,
        )
        return

    existing = {p.key for p in pub.properties}

    if "db_post_id" not in existing:
        try:
            pub.add_property("db_post_id", post_id, "numeric")
        except (ValidationError, ValueError) as exc:
            logger.warning(
                "Failed to enrich db_post_id for entity %s: %s",
                pub.id,
                exc,
            )

    for key in ("views", "reactions_count", "comments_count", "shares_count"):
        value = post_metrics.get(key)
        if value is not None and key not in existing:
            try:
                pub.add_property(key, value, "numeric")
            except (ValidationError, ValueError) as exc:
                logger.warning(
                    "Failed to enrich property %s for entity %s: %s",
                    key,
                    pub.id,
                    exc,
                )

    reactions_count = post_metrics.get("reactions_count")
    comments_count = post_metrics.get("comments_count")
    if (
        author_subscribers
        and author_subscribers > 0
        and isinstance(reactions_count, (int, float))
        and isinstance(comments_count, (int, float))
        and "engagement_rate" not in existing
    ):
        try:
            er = round((reactions_count + comments_count) / author_subscribers, 6)
            pub.add_property("engagement_rate", er, "numeric")
        except (ValidationError, ValueError) as exc:
            logger.warning(
                "Failed to enrich engagement_rate for entity %s: %s",
                pub.id,
                exc,
            )

    if "video_url" in raw_metadata and "video_url" not in existing:
        try:
            pub.add_property(
                "video_url", raw_metadata["video_url"], "text"
            )
        except (ValidationError, ValueError) as exc:
            logger.warning(
                "Failed to enrich property video_url for entity %s: %s",
                pub.id,
                exc,
            )

    if "published_at" in raw_metadata and "published_at" not in existing:
        try:
            pub.add_property(
                "published_at", str(raw_metadata["published_at"]), "text"
            )
        except (ValidationError, ValueError) as exc:
            logger.warning(
                "Failed to enrich property published_at for entity %s: %s",
                pub.id,
                exc,
            )

    transcription = raw_metadata.get("transcription")
    if transcription and "transcript" not in existing:
        try:
            pub.add_property("transcript", transcription, "text")
        except (ValidationError, ValueError) as exc:
            logger.warning(
                "Failed to enrich property transcript for entity %s: %s",
                pub.id,
                exc,
            )

    if (
        "accessibility_caption" in raw_metadata
        and "accessibility_caption" not in existing
    ):
        try:
            pub.add_property(
                "accessibility_caption",
                str(raw_metadata["accessibility_caption"]),
                "text",
            )
        except (ValidationError, ValueError) as exc:
            logger.warning(
                "Failed to enrich accessibility_caption for entity %s: %s",
                pub.id,
                exc,
            )

    hashtags = raw_metadata.get("hashtags")
    if isinstance(hashtags, list) and hashtags:
        for tag in hashtags:
            cleaned_tag = clean_hashtag(str(tag))
            if not cleaned_tag:
                continue
            tag_entity_id = f"hashtag_{cleaned_tag}"
            tag_entity = find_or_create_entity(
                entities,
                entity_id=tag_entity_id,
                label="Entity",
                name=f"#{cleaned_tag}",
            )
            existing_tag = {p.key for p in tag_entity.properties}
            if "type" not in existing_tag:
                tag_entity.add_property("type", "hashtag", "text")
            if "raw_tag" not in existing_tag:
                tag_entity.add_property(
                    "raw_tag", str(tag).strip(), "text"
                )
            find_or_create_relation(
                relations,
                source_id=source_node_id,
                relation_type="USES_HASHTAG",
                target_id=tag_entity_id,
            )

    geo_data = raw_metadata.get("geo_data")
    if isinstance(geo_data, dict):
        loc_name = geo_data.get("name")
        lat = geo_data.get("lat") or geo_data.get("latitude")
        lng = geo_data.get("lng") or geo_data.get("longitude")
        has_coords = (
            lat is not None
            and lng is not None
            and isinstance(lat, (int, float))
            and isinstance(lng, (int, float))
        )
        if isinstance(loc_name, str) and loc_name:
            loc_id = f"loc_{sanitize_id(loc_name)}"
            loc_display = loc_name
        elif has_coords:
            loc_id = f"loc_{lat}_{lng}"
            loc_display = f"{lat}, {lng}"
        else:
            loc_id = None
            loc_display = None

        if loc_id is not None and loc_display is not None:
            loc_entity = find_or_create_entity(
                entities,
                entity_id=loc_id,
                label="Place",
                name=loc_display,
            )
            existing_loc = {p.key for p in loc_entity.properties}
            if "type" not in existing_loc:
                loc_entity.add_property("type", "region", "text")
            if (
                isinstance(loc_name, str)
                and loc_name
                and "name" not in existing_loc
            ):
                loc_entity.add_property("name", loc_name, "text")
            if has_coords and "coordinates" not in existing_loc:
                assert isinstance(lat, (int, float))
                assert isinstance(lng, (int, float))
                loc_entity.add_property(
                    "coordinates", [float(lat), float(lng)], "geo"
                )
            find_or_create_relation(
                relations,
                source_id=source_node_id,
                relation_type="TAGGED_AT",
                target_id=loc_id,
            )

    audio_tracks = raw_metadata.get("audio_tracks")
    if isinstance(audio_tracks, list) and audio_tracks:
        for track in audio_tracks:
            if not isinstance(track, dict):
                continue
            track_title = str(track.get("title", "")).strip()
            track_author = str(track.get("author", "")).strip()
            if not track_title:
                continue
            track_hash = hashlib.md5(
                track_title.encode("utf-8")
            ).hexdigest()[:12]
            audio_entity_id = f"audio_{track_hash}"
            audio_entity = find_or_create_entity(
                entities,
                entity_id=audio_entity_id,
                label="Entity",
                name=track_title,
            )
            existing_audio = {p.key for p in audio_entity.properties}
            if "type" not in existing_audio:
                audio_entity.add_property("type", "audio", "text")
            if "title" not in existing_audio:
                audio_entity.add_property("title", track_title, "text")
            if track_author and "author" not in existing_audio:
                audio_entity.add_property("author", track_author, "text")
            find_or_create_relation(
                relations,
                source_id=source_node_id,
                relation_type="USES_AUDIO",
                target_id=audio_entity_id,
            )
    else:
        music_title = raw_metadata.get("music_title")
        if isinstance(music_title, str) and music_title.strip():
            music_author = str(
                raw_metadata.get("music_author", "")
            ).strip()
            track_hash = hashlib.md5(
                music_title.strip().encode("utf-8")
            ).hexdigest()[:12]
            audio_entity_id = f"audio_{track_hash}"
            audio_entity = find_or_create_entity(
                entities,
                entity_id=audio_entity_id,
                label="Entity",
                name=music_title.strip(),
            )
            existing_audio = {p.key for p in audio_entity.properties}
            if "type" not in existing_audio:
                audio_entity.add_property("type", "audio", "text")
            if "title" not in existing_audio:
                audio_entity.add_property(
                    "title", music_title.strip(), "text"
                )
            if music_author and "author" not in existing_audio:
                audio_entity.add_property("author", music_author, "text")
            find_or_create_relation(
                relations,
                source_id=source_node_id,
                relation_type="USES_AUDIO",
                target_id=audio_entity_id,
            )

    pub_lang = normalize_language(raw_metadata.get("language"))
    if process_language_data and pub_lang is not None and "language" not in existing:
        try:
            pub.add_property(
                "language", pub_lang, PropertyType.LANGUAGE
            )
        except (ValidationError, ValueError) as exc:
            logger.warning(
                "Failed to enrich language for entity %s: %s",
                pub.id,
                exc,
            )
