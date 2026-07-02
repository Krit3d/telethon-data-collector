import json
import logging
import urllib.parse
from typing import Any

from pydantic import ValidationError

from src.graph.extractor.extraction_helpers import (
    clean_telegram_link,
    find_entity,
    find_or_create_entity,
    find_or_create_relation,
    sanitize_id,
)
from src.graph.schema import ExtractedEntity, ExtractedRelation
from src.graph.utils import _convert_to_dict

logger = logging.getLogger(__name__)


def enrich_author_node(
    entities: list[ExtractedEntity],
    relations: list[ExtractedRelation],
    source_node_id: str,
    account_metadata: dict[str, Any],
    platform: str | None,
    process_language_data: bool,
) -> None:
    author = find_entity(entities, source_node_id)
    if author is None:
        logger.warning(
            "Author entity %s not found in entities list", source_node_id
        )
        return

    meta_copy = account_metadata.copy()
    meta_copy.pop("access_hash", None)
    existing = {p.key for p in author.properties}

    if "type" not in existing:
        try:
            author.add_property("type", "author", "text")
        except (ValidationError, ValueError) as exc:
            logger.warning(
                "Failed to enrich property type for entity %s: %s",
                author.id,
                exc,
            )

    base_field_map: dict[str, tuple[str, str]] = {
        "follower_count": ("numeric", "follower_count"),
        "subscribers_count": ("numeric", "follower_count"),
        "engagement_rate": ("numeric", "engagement_rate"),
        "handle": ("text", "handle"),
        "username": ("text", "handle"),
        "title": ("text", "display_name"),
    }

    for src_key, (prop_type, target_key) in base_field_map.items():
        value = meta_copy.get(src_key)
        if value is not None and target_key not in existing:
            if prop_type == "numeric":
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        logger.warning(
                            "Skipping non-numeric value '%s' for property %s on entity %s",
                            value,
                            target_key,
                            author.id,
                        )
                        continue
                elif not isinstance(value, (int, float)):
                    logger.warning(
                        "Skipping invalid numeric value %r for property %s on entity %s",
                        value,
                        target_key,
                        author.id,
                    )
                    continue
            try:
                author.add_property(target_key, value, prop_type)
            except (ValidationError, ValueError) as exc:
                logger.warning(
                    "Failed to enrich base property %s for entity %s: %s",
                    target_key,
                    author.id,
                    exc,
                )

    profile_field_map: dict[str, str] = {
        "biography": "biography",
        "website": "website",
        "link_in_bio": "link_in_bio",
        "profile_url": "profile_url",
    }

    telegram_link_keys = {"website", "link_in_bio"}
    for src_key, target_key in profile_field_map.items():
        value = meta_copy.get(src_key)
        if value is not None and target_key not in existing:
            try:
                cleaned = (
                    clean_telegram_link(str(value))
                    if src_key in telegram_link_keys
                    else str(value)
                )
                author.add_property(target_key, cleaned, "text")
            except (ValidationError, ValueError) as exc:
                logger.warning(
                    "Failed to enrich profile property %s for entity %s: %s",
                    target_key,
                    author.id,
                    exc,
                )

    contacts = _convert_to_dict(meta_copy.get("contacts"))
    if contacts:
        contact_list_keys = (
            "emails",
            "phones",
            "telegram_handles",
            "telegram_channels",
            "telegram_personal",
            "advertising_emails",
            "advertising_telegrams",
        )
        telegram_contact_keys = {
            "telegram_handles",
            "telegram_channels",
            "telegram_personal",
            "advertising_telegrams",
        }
        for ckey in contact_list_keys:
            cval = contacts.get(ckey)
            if not isinstance(cval, list) or not cval:
                continue
            for item in cval:
                item_str = str(item).strip()
                if not item_str:
                    continue
                if ckey in telegram_contact_keys:
                    item_str = clean_telegram_link(item_str)
                clean_val = sanitize_id(item_str)
                contact_entity_id = f"contact_{ckey}_{clean_val}"
                contact_entity = find_or_create_entity(
                    entities,
                    entity_id=contact_entity_id,
                    label="Entity",
                    name=item_str,
                )
                existing_contact = {p.key for p in contact_entity.properties}
                if "type" not in existing_contact:
                    contact_entity.add_property("type", "contact", "text")
                if "value" not in existing_contact:
                    contact_entity.add_property("value", item_str, "text")
                if "contact_type" not in existing_contact:
                    contact_entity.add_property("contact_type", ckey, "text")
                find_or_create_relation(
                    relations,
                    source_id=source_node_id,
                    relation_type="HAS_CONTACT",
                    target_id=contact_entity_id,
                )

    geo_data = meta_copy.get("geo_data")
    if isinstance(geo_data, dict):
        city = geo_data.get("city")
        country = geo_data.get("country")
        loc_parts = [p for p in (country, city) if isinstance(p, str) and p]
        if loc_parts:
            loc_id = "loc_" + "_".join(sanitize_id(p) for p in loc_parts)
            loc_name = ", ".join(loc_parts)
            loc_entity = find_or_create_entity(
                entities,
                entity_id=loc_id,
                label="Place",
                name=loc_name,
            )
            existing_loc = {p.key for p in loc_entity.properties}
            if "type" not in existing_loc:
                loc_entity.add_property("type", "region", "text")
            if isinstance(country, str) and country and "country" not in existing_loc:
                loc_entity.add_property("country", country, "text")
            if isinstance(city, str) and city and "city" not in existing_loc:
                loc_entity.add_property("city", city, "text")
            coords = geo_data.get("coordinates")
            if (
                isinstance(coords, list)
                and len(coords) == 2
                and all(isinstance(c, (int, float)) for c in coords)
                and "coordinates" not in existing_loc
            ):
                loc_entity.add_property(
                    "coordinates", [float(c) for c in coords], "geo"
                )
            find_or_create_relation(
                relations,
                source_id=source_node_id,
                relation_type="BASED_IN",
                target_id=loc_id,
            )

    region = meta_copy.get("region")
    if isinstance(region, str) and region.strip():
        region_id = f"loc_{sanitize_id(region)}"
        region_entity = find_or_create_entity(
            entities,
            entity_id=region_id,
            label="Place",
            name=region.strip(),
        )
        existing_region = {p.key for p in region_entity.properties}
        if "type" not in existing_region:
            region_entity.add_property("type", "region", "text")
        find_or_create_relation(
            relations,
            source_id=source_node_id,
            relation_type="BASED_IN",
            target_id=region_id,
        )

    ext_links = meta_copy.get("external_links")
    if isinstance(ext_links, list) and ext_links:
        seen_link_domains: set[str] = set()
        for link in ext_links:
            link_str = str(link).strip()
            if not link_str:
                continue
            try:
                parsed_url = urllib.parse.urlparse(link_str)
                domain = parsed_url.netloc or parsed_url.path
                domain = domain.lower().strip().rstrip("/")
                if not domain or domain in seen_link_domains:
                    continue
                seen_link_domains.add(domain)
            except (ValueError, TypeError):
                continue
            link_entity_id = f"link_{sanitize_id(domain)}"
            link_entity = find_or_create_entity(
                entities,
                entity_id=link_entity_id,
                label="Entity",
                name=domain,
            )
            existing_link = {p.key for p in link_entity.properties}
            if "type" not in existing_link:
                link_entity.add_property("type", "link", "text")
            if "url" not in existing_link:
                link_entity.add_property("url", link_str, "text")
            if "domain" not in existing_link:
                link_entity.add_property("domain", domain, "text")
            find_or_create_relation(
                relations,
                source_id=source_node_id,
                relation_type="HAS_LINK",
                target_id=link_entity_id,
            )

    ext_platforms = meta_copy.get("external_platforms")
    if (
        isinstance(ext_platforms, dict)
        and ext_platforms
        and "external_platforms" not in existing
    ):
        try:
            author.add_property(
                "external_platforms",
                json.dumps(ext_platforms, ensure_ascii=False),
                "text",
            )
        except (ValidationError, ValueError) as exc:
            logger.warning(
                "Failed to enrich external_platforms for entity %s: %s",
                author.id,
                exc,
            )

    if (
        platform is not None
        and platform.upper() == "TELEGRAM"
        and "is_author_blog" not in existing
    ):
        is_author_blog = meta_copy.get("is_author_blog")
        if is_author_blog is not None:
            try:
                author.add_property(
                    "is_author_blog",
                    "true" if is_author_blog else "false",
                    "text",
                )
            except (ValidationError, ValueError) as exc:
                logger.warning(
                    "Failed to enrich is_author_blog for entity %s: %s",
                    author.id,
                    exc,
                )

    audience_locations = meta_copy.get("audience_locations")
    if isinstance(audience_locations, list):
        for location_str in audience_locations:
            location_val = str(location_str).strip()
            if not location_val:
                continue
            loc_id = f"audience_loc_{sanitize_id(location_val)}"
            loc_entity = find_or_create_entity(
                entities,
                entity_id=loc_id,
                label="Place",
                name=location_val,
            )
            existing_loc = {p.key for p in loc_entity.properties}
            if "type" not in existing_loc:
                loc_entity.add_property("type", "region", "text")
            find_or_create_relation(
                relations,
                source_id=source_node_id,
                relation_type="HAS_AUDIENCE_IN",
                target_id=loc_id,
            )
