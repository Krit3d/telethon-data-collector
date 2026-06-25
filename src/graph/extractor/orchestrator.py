import asyncio
import logging
import re
import time
from typing import Any

from src.config.config import Settings
from src.graph.schema import decode_unicode_escapes
from src.graph.extractor.client import LLMClient
from src.graph.extractor.enrich_author import enrich_author_node
from src.graph.extractor.enrich_pub import enrich_publication_node
from src.graph.extractor.extraction_helpers import (
    find_entity,
    find_or_create_entity,
    find_or_create_relation,
)

logger = logging.getLogger(__name__)


def _resolve_author_name(
    account_metadata: dict[str, Any] | None,
    author_id: int,
) -> str:
    if account_metadata:
        for key in ("title", "display_name", "username"):
            value = account_metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return f"Author {author_id}"


def _resolve_publication_name(
    account_metadata: dict[str, Any] | None,
    platform_content_id: str,
    post_id: int,
) -> str:
    if platform_content_id:
        return platform_content_id
    if account_metadata:
        for key in ("title", "display_name", "username"):
            value = account_metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return f"Post {post_id}"


class KnowledgeExtractor:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._llm_client = LLMClient(settings)
        self._write_semaphore = asyncio.Semaphore(self.settings.graph_write_concurrency)

    async def close(self) -> None:
        await self._llm_client.close()
        logger.debug("KnowledgeExtractor: resources released")

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
        account_id: int | None = None,
        cached_result: Any | None = None,
    ) -> Any | None:
        process_language = getattr(self.settings, "process_language_data", False)

        logger.info("Processing post_id=%d for knowledge extraction", post_id)

        _author_str = str(author_id)
        if len(_author_str) > 100 or " " in _author_str or "\n" in _author_str:
            logger.warning(
                "Corrupted author_id detected for post_id=%d (len=%d), "
                "falling back to account_id=%s",
                post_id,
                len(_author_str),
                account_id,
            )
            author_id = int(account_id) if account_id is not None else 0

        if account_metadata is not None:
            account_status = account_metadata.get("status")
            if account_status is not None and account_status not in ("parsed", "verified"):
                logger.warning(
                    "Account metadata status is '%s' (not 'parsed') for "
                    "post_id=%d, skipping extraction",
                    account_status,
                    post_id,
                )
                return None

        cleaned_raw_metadata = raw_metadata.copy() if raw_metadata else None
        if cleaned_raw_metadata is not None:
            cleaned_raw_metadata.pop("category", None)
            cleaned_raw_metadata.pop("language", None)

        cleaned_account_metadata = account_metadata.copy() if account_metadata else None
        if cleaned_account_metadata is not None:
            cleaned_account_metadata.pop("category", None)
            cleaned_account_metadata.pop("language", None)

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
        else:
            pub_node_id = f"event_publication_{platform_slug}_{post_id}"

        pub_display_name = _resolve_publication_name(
            account_metadata, platform_content_id, post_id
        )

        if cached_result is not None:
            result = cached_result
            logger.info("Using cached extraction result for post_id=%d", post_id)
        else:
            result = await self._llm_client.call_llm(
                text=text,
                author_id=author_id,
                post_id=post_id,
                pub_node_id=pub_node_id,
                metadata=cleaned_raw_metadata if cleaned_raw_metadata else None,
                platform=platform,
            )

        _FILTERED_PROPERTY_KEYS = {"category", "language"}
        for entity in result.entities:
            entity.properties = [
                p for p in entity.properties
                if p.key not in _FILTERED_PROPERTY_KEYS
            ]

        fallback_author_ids: set[str] = set()
        fallback_pub_ids: set[str] = set()
        for entity in result.entities:
            if entity.id in (author_node_id, pub_node_id):
                continue
            for prop in entity.properties:
                if prop.key == "type" and prop.value == "author":
                    fallback_author_ids.add(entity.id)
                    break
                if prop.key == "type" and prop.value == "publication":
                    fallback_pub_ids.add(entity.id)
                    break
        llm_fallback_ids = fallback_author_ids | fallback_pub_ids
        result.entities = [
            e for e in result.entities if e.id not in llm_fallback_ids
        ]
        for rel in result.relations:
            if rel.source_id in fallback_author_ids:
                rel.source_id = author_node_id
            if rel.target_id in fallback_author_ids:
                rel.target_id = author_node_id
            if rel.source_id in fallback_pub_ids:
                rel.source_id = pub_node_id
            if rel.target_id in fallback_pub_ids:
                rel.target_id = pub_node_id
        if llm_fallback_ids:
            logger.info(
                "Safety remap for post_id=%d: author=%s, pub=%s",
                post_id,
                fallback_author_ids,
                fallback_pub_ids,
            )

        author_display_name = _resolve_author_name(account_metadata, author_id)

        author_entity = find_or_create_entity(
            result.entities,
            entity_id=author_node_id,
            label="Actor",
            name=author_display_name,
        )
        author_entity.name = decode_unicode_escapes(author_display_name)

        author_prop_keys = {p.key for p in author_entity.properties}
        if "platform" not in author_prop_keys:
            author_entity.add_property(
                "platform", platform or "unknown", "text"
            )
        if "platform_id" not in author_prop_keys:
            author_entity.add_property("platform_id", str(author_id), "text")

        if cleaned_account_metadata:
            enrich_author_node(
                result.entities,
                result.relations,
                author_node_id,
                cleaned_account_metadata,
                platform=platform,
                process_language_data=process_language,
            )

        pub_entity = find_or_create_entity(
            result.entities,
            entity_id=pub_node_id,
            label="Event",
            name=pub_display_name,
        )
        pub_entity.name = decode_unicode_escapes(pub_display_name)

        author_subscribers: int | None = None
        if account_metadata is not None:
            raw_subscribers = account_metadata.get("subscribers_count")
            if isinstance(raw_subscribers, (int, float)):
                author_subscribers = int(raw_subscribers)

        enrich_publication_node(
            result.entities,
            result.relations,
            pub_node_id,
            post_id,
            post_metrics,
            cleaned_raw_metadata or {},
            platform,
            author_subscribers=author_subscribers,
        )

        find_or_create_relation(
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
                coauthor_entity = find_or_create_entity(
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
                find_or_create_relation(
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
                tagged_entity = find_or_create_entity(
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
                find_or_create_relation(
                    result.relations,
                    source_id=pub_node_id,
                    relation_type="TAGGED",
                    target_id=tagged_node_id,
                )

        topic_target_ids: set[str] = set()
        for relation in result.relations:
            if relation.relation_type in ("ABOUT", "DISCUSSES"):
                target_entity = find_entity(
                    result.entities, relation.target_id
                )
                if (
                    target_entity is not None
                    and target_entity.label == "Entity"
                ):
                    topic_target_ids.add(relation.target_id)

        for topic_id in topic_target_ids:
            find_or_create_relation(
                result.relations,
                source_id=author_node_id,
                relation_type="COVERS_TOPIC",
                target_id=topic_id,
            )

        _PLACEHOLDER_NAMES = frozenset({
            "", "#", "other", "unknown", "none", "null", "undefined", "n_a",
        })
        pre_filter_count = len(result.entities)
        kept_entities = []
        discarded_ids: set[str] = set()
        for entity in result.entities:
            trimmed_name = entity.name.strip()
            if len(trimmed_name) < 2:
                discarded_ids.add(entity.id)
                continue
            if trimmed_name.lower() in _PLACEHOLDER_NAMES:
                discarded_ids.add(entity.id)
                continue
            kept_entities.append(entity)
        result.entities = kept_entities
        if discarded_ids:
            result.relations = [
                r for r in result.relations
                if r.source_id not in discarded_ids
                and r.target_id not in discarded_ids
            ]
            logger.info(
                "Garbage filtration for post_id=%d: discarded %d entities, %d relations removed",
                post_id,
                len(discarded_ids),
                pre_filter_count - len(result.entities),
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

        qdrant_task: asyncio.Task | None = None
        if qdrant is not None:
            qdrant_task = asyncio.create_task(
                qdrant.upsert_entities(result.entities)
            )

        async with self._write_semaphore:
            await graph_repo.save_extraction_result(post_id, result)

        if qdrant_task is not None:
            try:
                await qdrant_task
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
        return result
