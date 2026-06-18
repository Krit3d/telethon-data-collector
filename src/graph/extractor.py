import asyncio
import json
import logging
import re
import time
from typing import Any

import aiohttp
from pydantic import ValidationError

from src.config.config import Settings
from src.graph.schema import (
    ExtractedEntity,
    ExtractedRelation,
    OpenSPGExtractionResult,
    get_open_spg_llm_prompt,
)
from src.graph.utils import _repair_json

logger = logging.getLogger(__name__)

_DOMAIN_ENTITY_MAPPING = (
    "\n\nADDITIONAL DOMAIN ENTITY CLASSIFICATION RULES:\n"
    "You MUST classify each extracted entity using a 'type' marker property in its properties list.\n"
    "This property tells the system what domain-specific category the entity belongs to.\n\n"
    "Classification mapping (domain concept -> OpenSPG label + type marker property):\n"
    "  - Author (person, blogger, content creator, journalist) -> label: Actor, "
    'add property: {"key": "type", "value": "author", "type": "text"}\n'
    "  - Brand (company, product line, trademark, startup) -> label: Actor, "
    'add property: {"key": "type", "value": "brand", "type": "text"}\n'
    "  - Product (specific product, gadget, software, app) -> label: Entity, "
    'add property: {"key": "type", "value": "product", "type": "text"}\n'
    "  - Topic (subject, theme, discussion point, trend) -> label: Entity, "
    'add property: {"key": "type", "value": "topic", "type": "text"}\n'
    "  - Hashtag (trending tag, keyword tag, campaign tag) -> label: Entity, "
    'add property: {"key": "type", "value": "hashtag", "type": "text"}\n'
    "  - Publication (article, post, news piece, announcement, report) -> label: Event, "
    'add property: {"key": "type", "value": "publication", "type": "text"}\n'
    "  - Collaboration (joint venture, co-authored post, advertising partnership, brand collaboration) -> label: Event, "
    'add property: {"key": "type", "value": "collaboration", "type": "text"}\n'
    "  - Region (geographic area, country, city, district, state) -> label: Place, "
    'add property: {"key": "type", "value": "region", "type": "text"}\n\n'
    "Every extracted entity MUST include the 'type' marker property. "
    "If an entity does not fit any of the above categories, use the most appropriate "
    "OpenSPG label (Actor/Entity/Event/Place) and set type to 'other'."
)

_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 2.0
_RATE_LIMIT_COOLDOWN = 60.0
_REQUEST_TIMEOUT = 120


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
                                    "domain classification (author, brand, product, topic, hashtag, "
                                    "publication, collaboration, region, or other)."
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
    def _enrich_author_node(
        author: ExtractedEntity,
        account_metadata: dict[str, Any],
    ) -> None:
        existing = {p.key for p in author.properties}

        base_field_map: dict[str, tuple[str, str]] = {
            "follower_count": ("numeric", "follower_count"),
            "subscribers_count": ("numeric", "follower_count"),
            "engagement_rate": ("numeric", "engagement_rate"),
            "region": ("location", "region"),
            "handle": ("text", "handle"),
            "username": ("text", "handle"),
            "title": ("text", "display_name"),
        }

        for src_key, (prop_type, target_key) in base_field_map.items():
            value = account_metadata.get(src_key)
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
            "category": "category",
            "website": "website",
            "link_in_bio": "link_in_bio",
            "profile_url": "profile_url",
        }

        for src_key, target_key in profile_field_map.items():
            value = account_metadata.get(src_key)
            if value is not None and target_key not in existing:
                try:
                    author.add_property(target_key, str(value), "text")
                except (ValidationError, ValueError) as exc:
                    logger.warning(
                        "Failed to enrich profile property %s for entity %s: %s",
                        target_key,
                        author.id,
                        exc,
                    )

        contacts = account_metadata.get("contacts")
        if isinstance(contacts, dict):
            contact_list_keys = (
                "emails",
                "phones",
                "telegram_handles",
                "telegram_channels",
                "telegram_personal",
                "advertising_emails",
                "advertising_telegrams",
            )
            for ckey in contact_list_keys:
                cval = contacts.get(ckey)
                if isinstance(cval, list) and cval and ckey not in existing:
                    joined = ", ".join(str(item) for item in cval)
                    try:
                        author.add_property(ckey, joined, "text")
                    except (ValidationError, ValueError) as exc:
                        logger.warning(
                            "Failed to enrich contact property %s for entity %s: %s",
                            ckey,
                            author.id,
                            exc,
                        )

        geo_data = account_metadata.get("geo_data")
        if isinstance(geo_data, dict):
            coords = geo_data.get("coordinates")
            if (
                isinstance(coords, list)
                and len(coords) == 2
                and all(isinstance(c, float) for c in coords)
                and "coordinates" not in existing
            ):
                try:
                    author.add_property("coordinates", coords, "geo")
                except (ValidationError, ValueError) as exc:
                    logger.warning(
                        "Failed to enrich geo property coordinates for entity %s: %s",
                        author.id,
                        exc,
                    )

            geo_text_keys: dict[str, str] = {
                "city": "city",
                "country": "country",
            }
            for gkey, gprop in geo_text_keys.items():
                gval = geo_data.get(gkey)
                if isinstance(gval, str) and gval and gprop not in existing:
                    try:
                        author.add_property(gprop, gval, "text")
                    except (ValidationError, ValueError) as exc:
                        logger.warning(
                            "Failed to enrich geo property %s for entity %s: %s",
                            gprop,
                            author.id,
                            exc,
                        )

        ext_links = account_metadata.get("external_links")
        if isinstance(ext_links, list) and ext_links and "external_links" not in existing:
            joined = ", ".join(str(link) for link in ext_links)
            try:
                author.add_property("external_links", joined, "text")
            except (ValidationError, ValueError) as exc:
                logger.warning(
                    "Failed to enrich external_links for entity %s: %s",
                    author.id,
                    exc,
                )

        ext_platforms = account_metadata.get("external_platforms")
        if isinstance(ext_platforms, dict) and ext_platforms and "external_platforms" not in existing:
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

    @staticmethod
    def _enrich_publication_node(
        pub: ExtractedEntity,
        post_metrics: dict[str, int | None],
        raw_metadata: dict[str, Any],
    ) -> None:
        existing = {p.key for p in pub.properties}

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

        if "video_url" in raw_metadata and "video_url" not in existing:
            try:
                pub.add_property("video_url", raw_metadata["video_url"], "text")
            except (ValidationError, ValueError) as exc:
                logger.warning(
                    "Failed to enrich property %s for entity %s: %s",
                    "video_url",
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
                    "Failed to enrich property %s for entity %s: %s",
                    "published_at",
                    pub.id,
                    exc,
                )

        transcription = raw_metadata.get("transcription")
        if transcription and "transcript" not in existing:
            try:
                pub.add_property("transcript", transcription, "text")
            except (ValidationError, ValueError) as exc:
                logger.warning(
                    "Failed to enrich property %s for entity %s: %s",
                    "transcript",
                    pub.id,
                    exc,
                )

        text_field_map: dict[str, str] = {
            "music_title": "music_title",
            "music_author": "music_author",
            "accessibility_caption": "accessibility_caption",
        }
        for src_key, target_key in text_field_map.items():
            value = raw_metadata.get(src_key)
            if value is not None and target_key not in existing:
                try:
                    pub.add_property(target_key, str(value), "text")
                except (ValidationError, ValueError) as exc:
                    logger.warning(
                        "Failed to enrich property %s for entity %s: %s",
                        target_key,
                        pub.id,
                        exc,
                    )

        hashtags = raw_metadata.get("hashtags")
        if isinstance(hashtags, list) and hashtags and "hashtags" not in existing:
            joined = ", ".join(str(tag) for tag in hashtags)
            try:
                pub.add_property("hashtags", joined, "text")
            except (ValidationError, ValueError) as exc:
                logger.warning(
                    "Failed to enrich hashtags for entity %s: %s",
                    pub.id,
                    exc,
                )

        geo_data = raw_metadata.get("geo_data")
        if isinstance(geo_data, dict):
            loc_name = geo_data.get("name")
            if isinstance(loc_name, str) and loc_name and "location_name" not in existing:
                try:
                    pub.add_property("location_name", loc_name, "text")
                except (ValidationError, ValueError) as exc:
                    logger.warning(
                        "Failed to enrich location_name for entity %s: %s",
                        pub.id,
                        exc,
                    )

            lat = geo_data.get("lat") or geo_data.get("latitude")
            lng = geo_data.get("lng") or geo_data.get("longitude")
            if (
                lat is not None
                and lng is not None
                and isinstance(lat, (int, float))
                and isinstance(lng, (int, float))
                and "coordinates" not in existing
            ):
                try:
                    pub.add_property("coordinates", [float(lat), float(lng)], "geo")
                except (ValidationError, ValueError) as exc:
                    logger.warning(
                        "Failed to enrich geo coordinates for entity %s: %s",
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
        logger.info("Processing post_id=%d for knowledge extraction", post_id)

        result = await self._call_llm(
            text=text,
            author_id=author_id,
            post_id=post_id,
            metadata=raw_metadata if raw_metadata else None,
            platform=platform,
        )

        platform_slug = platform.lower() if platform else "unknown"
        author_node_id = f"actor_{platform_slug}_{author_id}"
        clean_content_id = re.sub(r"[^a-z0-9_]", "_", str(platform_content_id).strip().lower())
        if clean_content_id:
            if platform_slug == "telegram":
                pub_node_id = f"event_publication_{platform_slug}_{author_id}_{clean_content_id}"
            else:
                pub_node_id = f"event_publication_{platform_slug}_{clean_content_id}"
            pub_display_name = f"Publication {platform_content_id}"
        else:
            pub_node_id = f"event_publication_{platform_slug}_{post_id}"
            pub_display_name = f"Publication {post_id}"

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
            self._enrich_author_node(author_entity, account_metadata)

        pub_entity = self._find_or_create_entity(
            result.entities,
            entity_id=pub_node_id,
            label="Event",
            name=pub_display_name,
        )

        self._enrich_publication_node(pub_entity, post_metrics, raw_metadata)

        coauthors = raw_metadata.get("coauthors")
        if isinstance(coauthors, list):
            for coauthor in coauthors:
                coauthor_name = str(coauthor).strip()
                if not coauthor_name:
                    continue
                coauthor_slug = re.sub(r"[^a-z0-9_]", "_", coauthor_name.lower())
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
                    coauthor_entity.add_property("platform_id", coauthor_slug, "text")
                result.relations.append(
                    ExtractedRelation(
                        source_id=pub_node_id,
                        relation_type="COAUTHOR",
                        target_id=coauthor_node_id,
                    )
                )

        tagged_users = raw_metadata.get("tagged_users")
        if isinstance(tagged_users, list):
            for tagged_user in tagged_users:
                tagged_name = str(tagged_user).strip()
                if not tagged_name:
                    continue
                tagged_slug = re.sub(r"[^a-z0-9_]", "_", tagged_name.lower())
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
                    tagged_entity.add_property("platform_id", tagged_slug, "text")
                result.relations.append(
                    ExtractedRelation(
                        source_id=pub_node_id,
                        relation_type="TAGGED",
                        target_id=tagged_node_id,
                    )
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
