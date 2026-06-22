import asyncio
import hashlib
import json
import logging
import re
import time
import urllib.parse
from typing import Any

import aiohttp
from pydantic import ValidationError

from src.config.config import Settings
from src.graph.schema import (
    ExtractedEntity,
    ExtractedRelation,
    OpenSPGExtractionResult,
    PropertyType,
    get_open_spg_llm_prompt,
)
from src.graph.utils import _convert_to_dict, _repair_json

logger = logging.getLogger(__name__)

_DOMAIN_ENTITY_MAPPING = (
    "\n\nADDITIONAL DOMAIN ENTITY CLASSIFICATION RULES:\n"
    "You MUST classify each extracted entity using a 'type' marker property in its properties list.\n\n"
    "Classification mapping (domain concept -> OpenSPG label + type marker property):\n"
    "  - Topic (subject, theme, discussion point, trend, concept) -> label: Concept, "
    'add property: {"key": "type", "value": "topic", "type": "text"}\n'
    "  - Person (individual, public figure, influencer, expert) -> label: Actor, "
    'add property: {"key": "type", "value": "person", "type": "text"}\n'
    "  - Brand (company, product line, trademark, startup) -> label: Actor, "
    'add property: {"key": "type", "value": "brand", "type": "text"}\n'
    "  - Organization (institution, agency, team, group, department) -> label: Actor, "
    'add property: {"key": "type", "value": "organization", "type": "text"}\n'
    "  - Author (blogger, content creator, journalist) -> label: Actor, "
    'add property: {"key": "type", "value": "author", "type": "text"}\n'
    "  - Publication (article, post, news piece, announcement) -> label: Event, "
    'add property: {"key": "type", "value": "publication", "type": "text"}\n'
    "  - Region (geographic area, country, city, district) -> label: Place, "
    'add property: {"key": "type", "value": "region", "type": "text"}\n\n'
    "STRICT RELATIONSHIP RULES:\n"
    "  1. You MUST output ABOUT relations for every Publication entity that discusses a Topic.\n"
    "     Format: source_id=<publication_id>, relation_type=ABOUT, target_id=<topic_entity_id>\n"
    "  2. You MUST output MENTIONS relations when a Publication references a Brand, Person, or Organization.\n"
    "     Format: source_id=<publication_id>, relation_type=MENTIONS, target_id=<entity_id>\n"
    "  3. Every MENTIONS relation MUST include a sentiment property:\n"
    '     {{"key": "sentiment", "value": "<positive|negative|neutral>", "type": "text"}}\n'
    "     Determine sentiment based on the context and tone of the mention in the text.\n\n"
    "Every extracted entity MUST include the 'type' marker property. "
    "If an entity does not fit any of the above categories, use the most appropriate "
    "OpenSPG label (Actor/Entity/Event/Place) and set type to 'other'."
)

_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 2.0
_RATE_LIMIT_COOLDOWN = 60.0
_REQUEST_TIMEOUT = 120


def _clean_telegram_link(val: str) -> str:
    val = val.strip()
    if val.startswith("@"):
        return f"https://t.me/{val[1:]}"
    if val.startswith("+") and not val.startswith("https://"):
        after_plus = val[1:]
        if after_plus.isdigit():
            return val
        if any(c.isalpha() for c in after_plus):
            return f"https://t.me/{val}"
    return val


def _clean_hashtag(tag: str) -> str:
    cleaned = tag.strip().lstrip("#").strip()
    return re.sub(r"[^a-zA-Z0-9_]", "_", cleaned).strip("_").lower()


def _normalize_language(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip().lower()
    if len(stripped) == 2 and stripped.isalpha():
        return stripped
    return None


def _sanitize_id(value: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", value.lower().strip()).strip("_")


class KnowledgeExtractor:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT),
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.debug("KnowledgeExtractor: aiohttp session closed")

    def _build_prompt(
        self,
        text: str,
        author_id: int,
        platform: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        base = get_open_spg_llm_prompt(text, author_id, platform, metadata)
        return f"{base}{_DOMAIN_ENTITY_MAPPING}"

    async def _call_llm(
        self,
        text: str,
        author_id: int,
        post_id: int,
        metadata: dict[str, Any] | None = None,
        platform: str | None = None,
    ) -> OpenSPGExtractionResult:
        if not self.settings.llm_api_key:
            raise RuntimeError("LLM API key is not configured")

        session = await self._get_session()
        prompt = self._build_prompt(text, author_id, platform, metadata)
        schema = OpenSPGExtractionResult.model_json_schema()

        last_error: BaseException | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                async with session.post(
                    f"{self.settings.llm_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.llm_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.settings.llm_model_name,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a highly meticulous OpenSPG knowledge extraction engine. "
                                    "Extract entities and relations from the provided text following the "
                                    "schema and classification rules strictly. "
                                    "Every entity MUST include a 'type' marker property indicating its "
                                    "domain classification (topic, person, brand, organization, author, "
                                    "publication, region, or other)."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 4096,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "OpenSPGExtractionResult",
                                "strict": True,
                                "schema": schema,
                            },
                        },
                    },
                ) as response:
                    if response.status == 429:
                        delay = (
                            _RATE_LIMIT_COOLDOWN
                            if attempt < _MAX_RETRIES - 1
                            else _RETRY_BASE_DELAY * (2 ** attempt)
                        )
                        logger.warning(
                            "Rate limit (429) on attempt %d/%d for post_id=%d, "
                            "retrying in %.1fs",
                            attempt + 1,
                            _MAX_RETRIES,
                            post_id,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    if response.status != 200:
                        error_body = await response.text()
                        logger.error(
                            "LLM API error: status=%d, body=%s (post_id=%d)",
                            response.status,
                            error_body[:500],
                            post_id,
                        )
                        last_error = RuntimeError(
                            f"LLM API returned HTTP {response.status}"
                        )
                        if attempt < _MAX_RETRIES - 1:
                            await asyncio.sleep(
                                _RETRY_BASE_DELAY * (2 ** attempt)
                            )
                            continue
                        break

                    data = await response.json()
                    content = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )

                    if not content:
                        logger.warning(
                            "Empty LLM response on attempt %d/%d for post_id=%d",
                            attempt + 1,
                            _MAX_RETRIES,
                            post_id,
                        )
                        last_error = RuntimeError("LLM returned empty content")
                        if attempt < _MAX_RETRIES - 1:
                            continue
                        break

                    repaired = _repair_json(content)
                    if repaired != content:
                        logger.info(
                            "JSON repair applied for post_id=%d", post_id
                        )
                        content = repaired

                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError as json_err:
                        logger.error(
                            "JSON decode failed for post_id=%d: %s "
                            "(content[:300]=%s)",
                            post_id,
                            json_err,
                            content[:300],
                        )
                        last_error = RuntimeError(
                            f"JSON decode failed: {json_err}"
                        )
                        last_error.__cause__ = json_err
                        if attempt < _MAX_RETRIES - 1:
                            continue
                        break

                    try:
                        result = OpenSPGExtractionResult.model_validate(parsed)
                    except ValidationError as val_err:
                        logger.error(
                            "Pydantic validation failed for post_id=%d: %s",
                            post_id,
                            val_err,
                        )
                        last_error = RuntimeError(
                            f"Validation failed: {val_err}"
                        )
                        last_error.__cause__ = val_err
                        if attempt < _MAX_RETRIES - 1:
                            continue
                        break

                    logger.info(
                        "LLM extraction succeeded for post_id=%d: "
                        "%d entities, %d relations",
                        post_id,
                        len(result.entities),
                        len(result.relations),
                    )
                    return result

            except (aiohttp.ClientError, TimeoutError) as exc:
                last_error = exc
                logger.warning(
                    "Network error on attempt %d/%d for post_id=%d: %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    post_id,
                    exc,
                )
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(
                        _RETRY_BASE_DELAY * (2 ** attempt)
                    )
                    continue

        raise RuntimeError(
            f"LLM extraction failed after {_MAX_RETRIES} attempts "
            f"for post_id={post_id}"
        ) from last_error

    @staticmethod
    def _find_or_create_entity(
        entities: list[ExtractedEntity],
        entity_id: str,
        label: str,
        name: str,
    ) -> ExtractedEntity:
        for entity in entities:
            if entity.id == entity_id:
                return entity
        created = ExtractedEntity(
            id=entity_id,
            label=label,
            name=name,
            properties=[],
        )
        entities.append(created)
        return created

    @staticmethod
    def _find_entity(
        entities: list[ExtractedEntity],
        entity_id: str,
    ) -> ExtractedEntity | None:
        for entity in entities:
            if entity.id == entity_id:
                return entity
        return None

    @staticmethod
    def _find_or_create_relation(
        relations: list[ExtractedRelation],
        source_id: str,
        relation_type: str,
        target_id: str,
    ) -> ExtractedRelation:
        for rel in relations:
            if (
                rel.source_id == source_id
                and rel.relation_type == relation_type
                and rel.target_id == target_id
            ):
                return rel
        created = ExtractedRelation(
            source_id=source_id,
            relation_type=relation_type,
            target_id=target_id,
        )
        relations.append(created)
        return created

    @staticmethod
    def _enrich_author_node(
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
        source_node_id: str,
        account_metadata: dict[str, Any],
        platform: str | None = None,
        *,
        process_language_data: bool = False,
    ) -> None:
        author = KnowledgeExtractor._find_entity(entities, source_node_id)
        if author is None:
            logger.warning(
                "Author entity %s not found in entities list", source_node_id
            )
            return

        meta_copy = account_metadata.copy()
        meta_copy.pop("access_hash", None)
        existing = {p.key for p in author.properties}

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
                        _clean_telegram_link(str(value))
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
                        item_str = _clean_telegram_link(item_str)
                    clean_val = _sanitize_id(item_str)
                    contact_entity_id = f"contact_{ckey}_{clean_val}"
                    contact_entity = KnowledgeExtractor._find_or_create_entity(
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
                    KnowledgeExtractor._find_or_create_relation(
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
                loc_id = "loc_" + "_".join(_sanitize_id(p) for p in loc_parts)
                loc_name = ", ".join(loc_parts)
                loc_entity = KnowledgeExtractor._find_or_create_entity(
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
                KnowledgeExtractor._find_or_create_relation(
                    relations,
                    source_id=source_node_id,
                    relation_type="BASED_IN",
                    target_id=loc_id,
                )

        region = meta_copy.get("region")
        if isinstance(region, str) and region.strip():
            region_id = f"loc_{_sanitize_id(region)}"
            region_entity = KnowledgeExtractor._find_or_create_entity(
                entities,
                entity_id=region_id,
                label="Place",
                name=region.strip(),
            )
            existing_region = {p.key for p in region_entity.properties}
            if "type" not in existing_region:
                region_entity.add_property("type", "region", "text")
            KnowledgeExtractor._find_or_create_relation(
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
                link_entity_id = f"link_{_sanitize_id(domain)}"
                link_entity = KnowledgeExtractor._find_or_create_entity(
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
                KnowledgeExtractor._find_or_create_relation(
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

        account_lang = _normalize_language(meta_copy.get("language"))
        if process_language_data and account_lang is not None and "language" not in existing:
            try:
                author.add_property(
                    "language", account_lang, PropertyType.LANGUAGE
                )
            except (ValidationError, ValueError) as exc:
                logger.warning(
                    "Failed to enrich language for entity %s: %s",
                    author.id,
                    exc,
                )

        audience_locations = meta_copy.get("audience_locations")
        if isinstance(audience_locations, list):
            for location_str in audience_locations:
                location_val = str(location_str).strip()
                if not location_val:
                    continue
                loc_id = f"audience_loc_{_sanitize_id(location_val)}"
                loc_entity = KnowledgeExtractor._find_or_create_entity(
                    entities,
                    entity_id=loc_id,
                    label="Place",
                    name=location_val,
                )
                existing_loc = {p.key for p in loc_entity.properties}
                if "type" not in existing_loc:
                    loc_entity.add_property("type", "region", "text")
                KnowledgeExtractor._find_or_create_relation(
                    relations,
                    source_id=source_node_id,
                    relation_type="HAS_AUDIENCE_IN",
                    target_id=loc_id,
                )

    @staticmethod
    def _enrich_publication_node(
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
        source_node_id: str,
        post_id: int,
        post_metrics: dict[str, int | None],
        raw_metadata: dict[str, Any],
        platform: str | None = None,
        *,
        author_subscribers: int | None = None,
        process_language_data: bool = False,
    ) -> None:
        pub = KnowledgeExtractor._find_entity(entities, source_node_id)
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
                cleaned_tag = _clean_hashtag(str(tag))
                if not cleaned_tag:
                    continue
                tag_entity_id = f"hashtag_{cleaned_tag}"
                tag_entity = KnowledgeExtractor._find_or_create_entity(
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
                KnowledgeExtractor._find_or_create_relation(
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
                loc_id = f"loc_{_sanitize_id(loc_name)}"
                loc_display = loc_name
            elif has_coords:
                loc_id = f"loc_{lat}_{lng}"
                loc_display = f"{lat}, {lng}"
            else:
                loc_id = None
                loc_display = None

            if loc_id is not None and loc_display is not None:
                loc_entity = KnowledgeExtractor._find_or_create_entity(
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
                KnowledgeExtractor._find_or_create_relation(
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
                audio_entity = KnowledgeExtractor._find_or_create_entity(
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
                KnowledgeExtractor._find_or_create_relation(
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
                audio_entity = KnowledgeExtractor._find_or_create_entity(
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
                KnowledgeExtractor._find_or_create_relation(
                    relations,
                    source_id=source_node_id,
                    relation_type="USES_AUDIO",
                    target_id=audio_entity_id,
                )

        pub_lang = _normalize_language(raw_metadata.get("language"))
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

    async def process_post(
        self,
        post_id: int,
        text: str,
        author_id: int,
        post_metrics: dict[str, int | None],
        raw_metadata: dict[str, Any],
        graph_repo: Any,
        qdrant: Any | None = None,
        platform: str | None = None,
        account_metadata: dict[str, Any] | None = None,
        platform_content_id: str = "",
    ) -> None:
        process_language = getattr(self.settings, "process_language_data", False)

        logger.info("Processing post_id=%d for knowledge extraction", post_id)

        if account_metadata is not None:
            account_status = account_metadata.get("status")
            if account_status is not None and account_status != "parsed":
                logger.warning(
                    "Account metadata status is '%s' (not 'parsed') for "
                    "post_id=%d, skipping extraction",
                    account_status,
                    post_id,
                )
                return

        result = await self._call_llm(
            text=text,
            author_id=author_id,
            post_id=post_id,
            metadata=raw_metadata if raw_metadata else None,
            platform=platform,
        )

        platform_slug = platform.lower() if platform else "unknown"
        author_node_id = f"actor_{platform_slug}_{author_id}"
        clean_content_id = re.sub(
            r"[^a-z0-9_]", "_", str(platform_content_id).strip().lower()
        )
        if clean_content_id:
            if platform_slug == "telegram":
                pub_node_id = (
                    f"event_publication_{platform_slug}_{author_id}_{clean_content_id}"
                )
            else:
                pub_node_id = (
                    f"event_publication_{platform_slug}_{clean_content_id}"
                )
            pub_display_name = f"Publication {platform_content_id}"
        else:
            pub_node_id = f"event_publication_{platform_slug}_{post_id}"
            pub_display_name = f"Publication {post_id}"

        llm_author_ids: set[str] = set()
        llm_pub_ids: set[str] = set()
        for entity in result.entities:
            for prop in entity.properties:
                if prop.key == "type" and prop.value == "author":
                    llm_author_ids.add(entity.id)
                    break
                if prop.key == "type" and prop.value == "publication":
                    llm_pub_ids.add(entity.id)
                    break
        result.entities = [
            e
            for e in result.entities
            if e.id not in llm_author_ids and e.id not in llm_pub_ids
        ]
        for rel in result.relations:
            if rel.source_id in llm_author_ids:
                rel.source_id = author_node_id
            if rel.target_id in llm_author_ids:
                rel.target_id = author_node_id
            if rel.source_id in llm_pub_ids:
                rel.source_id = pub_node_id
            if rel.target_id in llm_pub_ids:
                rel.target_id = pub_node_id
        if llm_author_ids or llm_pub_ids:
            logger.info(
                "Remapped LLM IDs for post_id=%d: "
                "author_ids=%s, pub_ids=%s",
                post_id,
                llm_author_ids,
                llm_pub_ids,
            )

        author_entity = self._find_or_create_entity(
            result.entities,
            entity_id=author_node_id,
            label="Actor",
            name=f"Author {author_id}",
        )

        author_prop_keys = {p.key for p in author_entity.properties}
        if "platform" not in author_prop_keys:
            author_entity.add_property(
                "platform", platform or "unknown", "text"
            )
        if "platform_id" not in author_prop_keys:
            author_entity.add_property("platform_id", str(author_id), "text")

        if account_metadata:
            self._enrich_author_node(
                result.entities,
                result.relations,
                author_node_id,
                account_metadata,
                platform=platform,
                process_language_data=process_language,
            )

        self._find_or_create_entity(
            result.entities,
            entity_id=pub_node_id,
            label="Event",
            name=pub_display_name,
        )

        author_subscribers: int | None = None
        if account_metadata is not None:
            raw_subscribers = account_metadata.get("subscribers_count")
            if isinstance(raw_subscribers, (int, float)):
                author_subscribers = int(raw_subscribers)

        self._enrich_publication_node(
            result.entities,
            result.relations,
            pub_node_id,
            post_id,
            post_metrics,
            raw_metadata,
            platform,
            author_subscribers=author_subscribers,
            process_language_data=process_language,
        )

        self._find_or_create_relation(
            result.relations,
            source_id=author_node_id,
            relation_type="POSTED",
            target_id=pub_node_id,
        )

        coauthors = raw_metadata.get("coauthors")
        if isinstance(coauthors, list):
            for coauthor in coauthors:
                coauthor_name = str(coauthor).strip()
                if not coauthor_name:
                    continue
                coauthor_slug = re.sub(
                    r"[^a-z0-9_]", "_", coauthor_name.lower()
                )
                coauthor_node_id = f"actor_{platform_slug}_{coauthor_slug}"
                coauthor_entity = self._find_or_create_entity(
                    result.entities,
                    entity_id=coauthor_node_id,
                    label="Actor",
                    name=coauthor_name,
                )
                coauthor_prop_keys = {p.key for p in coauthor_entity.properties}
                if "platform" not in coauthor_prop_keys:
                    coauthor_entity.add_property(
                        "platform", platform or "unknown", "text"
                    )
                if "platform_id" not in coauthor_prop_keys:
                    coauthor_entity.add_property(
                        "platform_id", coauthor_slug, "text"
                    )
                self._find_or_create_relation(
                    result.relations,
                    source_id=pub_node_id,
                    relation_type="COAUTHOR",
                    target_id=coauthor_node_id,
                )

        tagged_users = raw_metadata.get("tagged_users")
        if isinstance(tagged_users, list):
            for tagged_user in tagged_users:
                tagged_name = str(tagged_user).strip()
                if not tagged_name:
                    continue
                tagged_slug = re.sub(
                    r"[^a-z0-9_]", "_", tagged_name.lower()
                )
                tagged_node_id = f"actor_{platform_slug}_{tagged_slug}"
                tagged_entity = self._find_or_create_entity(
                    result.entities,
                    entity_id=tagged_node_id,
                    label="Actor",
                    name=tagged_name,
                )
                tagged_prop_keys = {p.key for p in tagged_entity.properties}
                if "platform" not in tagged_prop_keys:
                    tagged_entity.add_property(
                        "platform", platform or "unknown", "text"
                    )
                if "platform_id" not in tagged_prop_keys:
                    tagged_entity.add_property(
                        "platform_id", tagged_slug, "text"
                    )
                self._find_or_create_relation(
                    result.relations,
                    source_id=pub_node_id,
                    relation_type="TAGGED",
                    target_id=tagged_node_id,
                )

        topic_target_ids: set[str] = set()
        for relation in result.relations:
            if relation.relation_type == "ABOUT":
                target_entity = self._find_entity(
                    result.entities, relation.target_id
                )
                if (
                    target_entity is not None
                    and target_entity.label == "Concept"
                ):
                    topic_target_ids.add(relation.target_id)

        for topic_id in topic_target_ids:
            self._find_or_create_relation(
                result.relations,
                source_id=author_node_id,
                relation_type="COVERS_TOPIC",
                target_id=topic_id,
            )

        current_ts = int(time.time())
        for entity in result.entities:
            entity_prop_keys = {p.key for p in entity.properties}
            if "last_modified_at" not in entity_prop_keys:
                entity.add_property("last_modified_at", current_ts, "numeric")

        for relation in result.relations:
            rel_prop_keys = {p.key for p in relation.properties}
            if "confidence" not in rel_prop_keys:
                relation.add_property("confidence", 0.5, "numeric")

        await graph_repo.save_extraction_result(post_id, result)

        if qdrant is not None:
            try:
                await qdrant.upsert_entities(result.entities)
            except Exception as exc:
                logger.error(
                    "Qdrant sync failed for post_id=%d: %s",
                    post_id,
                    exc,
                    exc_info=True,
                )

        logger.info(
            "Completed extraction for post_id=%d: %d entities, %d relations",
            post_id,
            len(result.entities),
            len(result.relations),
        )
