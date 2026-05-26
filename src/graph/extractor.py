"""Knowledge graph extractor for OpenSPG-based extraction pipeline."""

import asyncio
import json
import logging
import time
from typing import Any

import aiohttp
from pydantic import BaseModel, ValidationError

from src.config.config import Settings
from src.db.graph_repo import GraphRepository
from src.embeddings.qdrant_service import QdrantService
from src.graph.schema import (
    OpenSPGExtractionResult,
    get_open_spg_llm_prompt,
)
from src.graph.utils import (
    _repair_json,
    _sanitize_key,
    _convert_to_dict,
    _merge_metadata_into_properties,
)

logger = logging.getLogger(__name__)


class KnowledgeExtractor:
    """Extracts knowledge triples (nodes and edges) from text content using LLM."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the extractor with configuration settings.

        Args:
            settings: Application settings containing LLM configuration.
        """
        self.settings = settings
        self._session: aiohttp.ClientSession | None = None

    async def _call_llm(
        self,
        text: str,
        author_id: int,
        post_id: int,
        metadata: dict | None = None,
    ) -> OpenSPGExtractionResult | None:
        """Call the LLM API to extract knowledge triples from text with retry logic.

        Args:
            text: Input text to analyze.
            author_id: Telegram user ID of the post author (used for author node).
            post_id: Database ID of the post (used for logging and context).
            metadata: Optional pre-collected metadata to pass to the LLM prompt.
                      Fields like 'language', 'location', 'geo' in this metadata
                      will be excluded from LLM extraction via prompt instructions.

        Returns:
            OpenSPGExtractionResult containing extracted entities and relations.
            Returns None on complete failure (all retries exhausted).
        """
        if not self.settings.llm_api_key:
            logger.warning("LLM API key not configured, skipping extraction")
            return None

        # Lazy initialization of HTTP session
        if self._session is None:
            self._session = aiohttp.ClientSession()

        # Construct the OpenSPG prompt with optional metadata
        # The prompt now includes explicit instructions to skip pre-extracted fields
        prompt = get_open_spg_llm_prompt(text, author_id, metadata)

        max_retries = 2
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                async with self._session.post(
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
                                "content": "You are a highly meticulous OpenSPG knowledge extraction engine. Even for short texts, identify at least the author and any mentioned concepts, locations, or dates. Never skip entities if they are present.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 4096,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            "LLM API request failed: status=%d, response=%s",
                            response.status,
                            error_text,
                        )
                        # Specific handling for rate limit (429) errors
                        if response.status == 429:
                            if attempt < max_retries:
                                cooldown = (
                                    60  # 60-second cooldown for rate limits
                                )
                                logger.warning(
                                    "Rate limit hit (429). Cooling down for %d seconds before retry %d/%d...",
                                    cooldown,
                                    attempt + 1,
                                    max_retries,
                                )
                                await asyncio.sleep(cooldown)
                                continue
                            else:
                                logger.error(
                                    "Rate limit error persisted after %d retries for post_id=%s",
                                    max_retries,
                                    post_id,
                                )
                                return None
                        if attempt < max_retries:
                            continue
                        return None

                    data = await response.json()
                    content = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )

                    # Log raw content length for debugging truncation issues
                    logger.debug(
                        "LLM response received for post_id=%s: content_length=%d chars",
                        post_id,
                        len(content),
                    )

                    if not content:
                        logger.warning("LLM returned empty content")
                        if attempt < max_retries:
                            continue
                        return None

                    # Try to repair JSON if malformed
                    repaired_content = _repair_json(content)
                    if repaired_content != content:
                        logger.info(
                            "Applied JSON repair for post_id=%s", post_id
                        )
                        content = repaired_content

                    # Parse the JSON response
                    try:
                        parsed_json = json.loads(content)
                    except json.JSONDecodeError as e:
                        if attempt < max_retries:
                            logger.warning(
                                "Failed to decode LLM response as JSON (attempt %d/%d) for post_id=%s: %s\n"
                                "First 500 chars: %s\n"
                                "Last 500 chars: %s",
                                attempt + 1,
                                max_retries,
                                post_id,
                                e,
                                content[:500],
                                (
                                    content[-500:]
                                    if len(content) > 500
                                    else content
                                ),
                            )
                            # Modify prompt for retry
                            prompt = (
                                get_open_spg_llm_prompt(text, author_id, metadata)
                                + "\n\nIMPORTANT: Your previous response was truncated. Please provide a more concise JSON, focusing only on the most important entities."
                            )
                            continue
                        else:
                            logger.error(
                                "Failed to decode LLM response as JSON after %d retries for post_id=%s: %s\n"
                                "First 500 chars: %s\n"
                                "Last 500 chars: %s",
                                max_retries,
                                post_id,
                                e,
                                content[:500],
                                (
                                    content[-500:]
                                    if len(content) > 500
                                    else content
                                ),
                            )
                            # Raise exception to skip post and retry later
                            raise Exception(
                                f"Failed to decode LLM response as JSON after {max_retries} retries for post_id={post_id}"
                            ) from e

                    # Parse and validate the JSON response directly
                    try:
                        parsed_json = json.loads(content)
                        open_spg_result = (
                            OpenSPGExtractionResult.model_validate(parsed_json)
                        )
                    except (json.JSONDecodeError, ValidationError) as e:
                        if attempt < max_retries:
                            logger.warning(
                                "Failed to parse/validate LLM response (attempt %d/%d) for post_id=%s: %s\n"
                                "First 500 chars: %s\n"
                                "Last 500 chars: %s",
                                attempt + 1,
                                max_retries,
                                post_id,
                                e,
                                content[:500],
                                (
                                    content[-500:]
                                    if len(content) > 500
                                    else content
                                ),
                            )
                            # Modify prompt for retry
                            prompt = (
                                get_open_spg_llm_prompt(text, author_id, metadata)
                                + "\n\nIMPORTANT: Your previous response was truncated or invalid. Ensure you return ONLY valid JSON with the exact structure specified in the prompt. Limit to 5-7 most important entities."
                            )
                            continue
                        else:
                            logger.error(
                                "Failed to parse/validate LLM response after %d retries for post_id=%s: %s\n"
                                "First 500 chars: %s\n"
                                "Last 500 chars: %s",
                                max_retries,
                                post_id,
                                e,
                                content[:500],
                                (
                                    content[-500:]
                                    if len(content) > 500
                                    else content
                                ),
                            )
                            # Raise exception to skip post and retry later
                            raise Exception(
                                f"Failed to parse/validate LLM response after {max_retries} retries for post_id={post_id}"
                            ) from e

                    logger.info(
                        "LLM extraction successful: %d entities, %d relations",
                        len(open_spg_result.entities),
                        len(open_spg_result.relations),
                    )
                    return open_spg_result

            except aiohttp.ClientError as e:
                last_error = e
                logger.error(
                    "LLM API request failed with network error (attempt %d/%d) for post_id=%s: %s",
                    attempt + 1,
                    max_retries + 1,
                    post_id,
                    e,
                )
                if attempt < max_retries:
                    continue
            except TimeoutError as e:
                last_error = e
                logger.error(
                    "LLM API request timed out (attempt %d/%d) for post_id=%s: %s",
                    attempt + 1,
                    max_retries + 1,
                    post_id,
                    e,
                )
                if attempt < max_retries:
                    continue
            except Exception as e:
                last_error = e
                logger.error(
                    "Unexpected error during LLM extraction (attempt %d/%d) for post_id=%s: %s",
                    attempt + 1,
                    max_retries + 1,
                    post_id,
                    e,
                    exc_info=True,
                )
                if attempt < max_retries:
                    continue

        # If we exhausted all retries, return None
        return None

    async def extract_triplets(
        self,
        text: str,
        author_id: int,
        post_id: int,
        metadata: dict | None = None,
    ) -> OpenSPGExtractionResult | None:
        """
        Extract knowledge triplets from the given text using LLM.

        Args:
            text: Input text to analyze.
            author_id: Telegram user ID of the post author.
            post_id: Database ID of the post (for logging).
            metadata: Optional pre-collected metadata to pass to the LLM.
                      Fields in metadata (language, location, geo) will be
                      excluded from LLM extraction via prompt instructions.

        Returns:
            OpenSPGExtractionResult containing extracted entities and relations.
            Returns None if extraction failed completely.
        """
        logger.debug("Extracting triplets from text: %s", text[:100])
        return await self._call_llm(text, author_id, post_id, metadata)

    async def close(self) -> None:
        """Close the aiohttp session and clean up resources."""
        if self._session is not None:
            await self._session.close()
            self._session = None
            logger.debug("KnowledgeExtractor: aiohttp session closed")

    async def process_post(
        self,
        post_id: int,
        text: str,
        author_id: int,
        graph_repo: GraphRepository,
        qdrant: QdrantService | None = None,
        post_metrics: dict | BaseModel | None = None,
        raw_metadata: dict | BaseModel | None = None,
    ) -> None:
        """
        Process a single post: extract knowledge triples and persist to AGE graph.

        This method handles the complete pipeline:
        1. Upsert Content node with merged metrics and raw_metadata
        2. Upsert Actor node for the author
        3. Create POSTED relationship
        4. Extract knowledge via LLM (excluding pre-extracted fields)
        5. Upsert extracted entities and relations to graph
        6. Sync to Qdrant for vector search

        Args:
            post_id: Database ID of the content.
            text: Text content of the content.
            author_id: Telegram user ID of the content author.
            graph_repo: GraphRepository instance for AGE graph persistence operations.
            qdrant: Optional QdrantService for syncing entities to vector store.
            post_metrics: Optional dictionary or Pydantic model containing content metrics (views, reactions, etc.).
            raw_metadata: Optional dictionary or Pydantic model containing pre-extracted metadata
                         (language, geo, location, etc.). This metadata will be merged into the Content node
                         and excluded from LLM extraction.
        """
        logger.info("Processing post id=%s for knowledge extraction", post_id)

        # Step A: Standardize the Content node ID
        content_node_id = f"content_{post_id}"

        # Step B: Create/Upsert the Content node with merged metrics and metadata
        try:
            # Start with base properties
            content_properties: dict[str, Any] = {
                "id": content_node_id,
                "post_id": post_id,
                "author_id": author_id,
            }

            # Merge post_metrics (views, comments, reactions) into properties
            # Handle both dict and Pydantic model inputs
            content_properties = _merge_metadata_into_properties(content_properties, post_metrics)

            # Merge raw_metadata into properties
            # This ensures all pre-extracted metadata is stored in the Content node
            content_properties = _merge_metadata_into_properties(content_properties, raw_metadata)

            # Upsert the Content node with all properties
            await graph_repo.upsert_graph_node(
                label="Content",
                properties=content_properties,
                merge_key="id",
            )
            logger.debug("Upserted Content node: id=%s", content_node_id)
        except Exception as e:
            logger.error(
                "Failed to upsert Content node (post_id=%s): %s",
                post_id,
                e,
                exc_info=True,
            )
            raise

        # Step C: Create/Upsert the Actor node for the channel/author
        actor_node_id = f"actor_{author_id}"
        try:
            await graph_repo.upsert_graph_node(
                label="Actor",
                properties={
                    "id": actor_node_id,
                    "name": f"Channel {author_id}",  # Baseline fallback name
                    "author_id": author_id,
                },
                merge_key="id",
            )
            logger.debug("Upserted Actor node: id=%s", actor_node_id)
        except Exception as e:
            logger.error(
                "Failed to upsert Actor node (author_id=%s): %s",
                author_id,
                e,
                exc_info=True,
            )
            raise

        # Step D: Create/Upsert the POSTED relationship
        try:
            await graph_repo.upsert_graph_edge(
                start_label="Actor",
                start_merge_key="id",
                start_merge_val=actor_node_id,
                edge_label="POSTED",
                end_label="Content",
                end_merge_key="id",
                end_merge_val=content_node_id,
                edge_properties={},
            )
            logger.debug(
                "Created POSTED relationship: Actor(%s)-[:POSTED]->Content(%s)",
                actor_node_id,
                content_node_id,
            )
        except Exception as e:
            logger.error(
                "Failed to create POSTED relationship (author_id=%s, post_id=%s): %s",
                author_id,
                post_id,
                e,
            )
            # Do not raise - continue with extraction

        # Step E: Call LLM extraction with raw_metadata
        # The LLM prompt will instruct the model to skip pre-extracted fields
        # Convert raw_metadata to dict if it's a Pydantic model
        raw_metadata_dict = _convert_to_dict(raw_metadata) if raw_metadata else None
        result = await self.extract_triplets(text, author_id, post_id, raw_metadata_dict)

        if result is None or (not result.entities and not result.relations):
            logger.warning(
                "Empty extraction for content id=%s. Text snippet: %s. Core Content skeleton is saved.",
                post_id,
                text[:100],
            )
            return

        # Step F: Process extracted entities and relations
        # Detailed logging after extraction (before upserting)
        logger.info(
            "Extracted %d entities and %d relations from content %d",
            len(result.entities),
            len(result.relations),
            post_id,
        )

        # Add last_modified_at timestamp to all entities (for incremental updates tracking)
        current_timestamp = int(time.time())
        for entity in result.entities:
            entity.add_property(
                "last_modified_at", current_timestamp, "numeric"
            )

        # Add default confidence property to relations if not present
        for relation in result.relations:
            props_dict = relation.get_property_dict()
            if "confidence" not in props_dict:
                relation.add_property("confidence", 0.5, "numeric")

        # Build entity ID -> label mapping for relation upserts
        entity_id_to_label: dict[str, str] = {}
        for entity in result.entities:
            entity_id_to_label[entity.id] = entity.label

        # Upsert entities to AGE graph
        for entity in result.entities:
            try:
                props = {
                    "id": entity.id,
                    "name": entity.name,
                    **entity.get_property_dict(),
                }
                await graph_repo.upsert_graph_node(
                    label=entity.label,
                    properties=props,
                    merge_key="id",
                )
                logger.debug(
                    "Upserted entity: label=%s, id=%s",
                    entity.label,
                    entity.id,
                )
            except Exception as e:
                logger.error(
                    "Failed to upsert entity (post_id=%s, label=%s): %s",
                    post_id,
                    entity.label,
                    e,
                )
                raise

        # Create MENTIONS relationship: (Content)-[:MENTIONS]->(Entity) for each extracted entity
        for entity in result.entities:
            try:
                await graph_repo.upsert_graph_edge(
                    start_label="Content",
                    start_merge_key="id",
                    start_merge_val=content_node_id,  # Use standardized Content node ID
                    edge_label="MENTIONS",
                    end_label=entity.label,
                    end_merge_key="id",
                    end_merge_val=entity.id,
                    edge_properties={},
                )
                logger.debug(
                    "Created MENTIONS relationship: Content(%s)-[:MENTIONS]->%s(%s)",
                    content_node_id,
                    entity.label,
                    entity.id,
                )
            except Exception as e:
                logger.error(
                    "Failed to create MENTIONS relationship (post_id=%s, entity_id=%s): %s",
                    post_id,
                    entity.id,
                    e,
                )
                # Do not raise - continue with other operations

        # Upsert relations to AGE graph
        for relation in result.relations:
            start_label = entity_id_to_label.get(relation.source_id, "Entity")
            end_label = entity_id_to_label.get(relation.target_id, "Entity")
            try:
                edge_props = relation.get_property_dict()
                await graph_repo.upsert_graph_edge(
                    start_label=start_label,
                    start_merge_key="id",
                    start_merge_val=relation.source_id,
                    edge_label=relation.relation_type,
                    end_label=end_label,
                    end_merge_key="id",
                    end_merge_val=relation.target_id,
                    edge_properties=edge_props,
                )
                logger.debug(
                    "Upserted relation: %s(%s)-%s->%s(%s)",
                    start_label,
                    relation.source_id,
                    relation.relation_type,
                    end_label,
                    relation.target_id,
                )
            except Exception as e:
                logger.error(
                    "Failed to upsert relation (post_id=%s, relation=%s): %s",
                    post_id,
                    relation.relation_type,
                    e,
                )
                raise

        # Sync entities to Qdrant (if service is available)
        if qdrant is not None:
            try:
                await qdrant.upsert_entities(result.entities)
            except Exception as e:
                logger.error(
                    "Failed to sync entities to Qdrant (post_id=%s): %s",
                    post_id,
                    e,
                    exc_info=True,
                )
                # Do not raise - Qdrant failure should not crash the pipeline

        logger.info(
            "Completed processing content id=%s: %d entities, %d relations",
            post_id,
            len(result.entities),
            len(result.relations),
        )
