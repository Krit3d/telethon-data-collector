import logging
import re
import time
from typing import Any

from src.config.config import Settings
from src.graph.extractor.client import LLMClient
from src.graph.extractor.enrich_author import enrich_author_node
from src.graph.extractor.enrich_pub import enrich_publication_node
from src.graph.extractor.extraction_helpers import (
    find_entity,
    find_or_create_entity,
    find_or_create_relation,
)

logger = logging.getLogger(__name__)


class KnowledgeExtractor:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._llm_client = LLMClient(settings)

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

        result = await self._llm_client.call_llm(
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

        author_entity = find_or_create_entity(
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
            enrich_author_node(
                result.entities,
                result.relations,
                author_node_id,
                account_metadata,
                platform=platform,
                process_language_data=process_language,
            )

        find_or_create_entity(
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

        enrich_publication_node(
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
