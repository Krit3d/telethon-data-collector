import asyncio
import json
import logging
import time
from typing import Any

import aiohttp
from pydantic import ValidationError

from src.config.config import Settings
from src.graph.schema import (
    ExtractedEntity,
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
    def _enrich_author_node(
        author: ExtractedEntity,
        account_metadata: dict[str, Any],
    ) -> None:
        field_map: dict[str, tuple[str, str]] = {
            "follower_count": ("numeric", "follower_count"),
            "subscribers_count": ("numeric", "follower_count"),
            "engagement_rate": ("numeric", "engagement_rate"),
            "region": ("location", "region"),
            "handle": ("text", "handle"),
            "username": ("text", "handle"),
            "title": ("text", "display_name"),
        }

        existing = {p.key for p in author.properties}

        for src_key, (prop_type, target_key) in field_map.items():
            value = account_metadata.get(src_key)
            if value is not None and target_key not in existing:
                try:
                    author.add_property(target_key, value, prop_type)
                except (ValidationError, ValueError) as exc:
                    logger.warning(
                        "Failed to enrich property %s for entity %s: %s",
                        target_key,
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
        pub_node_id = f"event_publication_{platform_slug}_{post_id}"

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
            name=f"Publication {post_id}",
        )

        self._enrich_publication_node(pub_entity, post_metrics, raw_metadata)

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
