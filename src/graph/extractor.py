"""Knowledge graph extractor for OpenSPG-based extraction pipeline."""

import json
import logging
import re
from typing import Any

import aiohttp
from pydantic import ValidationError

from src.config.config import Settings
from src.db.database import Database
from src.embeddings.qdrant_service import QdrantService
from src.graph.schema import (
    ExtractionResult,
    OpenSPGExtractionResult,
    Property,
    PropertyType,
    SPGEdge,
    SPGNode,
    get_open_spg_llm_prompt,
    open_spg_result_to_extraction_result,
)

logger = logging.getLogger(__name__)


def _repair_json(content: str) -> str:
    """Attempt to repair malformed or truncated JSON.
    
    Uses simple heuristics to fix common issues:
    - Unclosed braces/brackets
    - Trailing incomplete tokens
    - Truncated strings
    
    Args:
        content: The potentially malformed JSON string.
        
    Returns:
        Repaired JSON string if successful, otherwise original content.
    """
    original = content
    
    # Strip whitespace from ends
    content = content.strip()
    
    # Try to find the last complete JSON object by counting braces
    brace_count = 0
    bracket_count = 0
    in_string = False
    escape_next = False
    last_valid_pos = -1
    
    for i, char in enumerate(content):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\' and in_string:
            escape_next = True
            continue
            
        if char == '"':
            in_string = not in_string
            continue
            
        if in_string:
            continue
            
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and bracket_count == 0:
                last_valid_pos = i
        elif char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if brace_count == 0 and bracket_count == 0:
                last_valid_pos = i
    
    # If we have unclosed braces, try to close them
    if brace_count > 0 or bracket_count > 0:
        # Find position where we had a complete object
        if last_valid_pos > 0:
            content = content[:last_valid_pos + 1]
            # Add missing closing braces/brackets
            content += ']' * bracket_count
            content += '}' * brace_count
        else:
            # Could not find a complete object, return original
            return original
    
    # Validate that the repaired JSON is parseable
    try:
        json.loads(content)
        return content
    except json.JSONDecodeError:
        return original


def _convert_properties_dict_to_list(properties_list: list[dict[str, Any]]) -> list[Property]:
    """Convert a list of property objects to a list of Property objects.

    The LLM now returns properties as a list of objects with explicit type:
    [{"key": "prop_name", "value": <any>, "type": "text" | "numeric" | "geo" | "language" | "location"}]

    Args:
        properties_list: List of dictionaries, each containing 'key', 'value', and 'type' fields.

    Returns:
        List of validated Property objects with types mapped to PropertyType enum.
    """
    properties = []
    for prop_data in properties_list:
        try:
            # Extract fields from the property object
            key = prop_data.get("key")
            value = prop_data.get("value")
            type_str = prop_data.get("type")

            if key is None or value is None or type_str is None:
                logger.warning(
                    "Skipping invalid property: missing required fields (key, value, type): %s",
                    prop_data,
                )
                continue

            # Convert type string to PropertyType enum (handles both string and enum)
            try:
                prop_type = PropertyType(type_str)
            except ValueError:
                logger.warning(
                    "Invalid property type '%s' for key '%s', defaulting to TEXT",
                    type_str,
                    key,
                )
                prop_type = PropertyType.TEXT

            # Create Property object - validation happens in the model
            property_obj = Property(key=key, value=value, type=prop_type)
            properties.append(property_obj)

        except Exception as e:
            logger.warning(
                "Failed to parse property %s: %s. Skipping.",
                prop_data,
                e,
            )
            continue

    return properties


class KnowledgeExtractor:
    """Extracts knowledge triples (nodes and edges) from text content using LLM."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the extractor with configuration settings.

        Args:
            settings: Application settings containing LLM configuration.
        """
        self.settings = settings
        self._session: aiohttp.ClientSession | None = None

    async def _call_llm(self, text: str, author_id: int, post_id: int) -> ExtractionResult:
        """Call the LLM API to extract knowledge triples from text with retry logic.

        Args:
            text: Input text to analyze.
            author_id: Telegram user ID of the post author (used for author node).
            post_id: Database ID of the post (used for logging and context).

        Returns:
            ExtractionResult containing extracted nodes and edges.
            Returns empty result on any error (timeout, validation, etc.).
        """
        if not self.settings.llm_api_key:
            logger.warning("LLM API key not configured, skipping extraction")
            return ExtractionResult(nodes=[], edges=[])

        # Lazy initialization of HTTP session
        if self._session is None:
            self._session = aiohttp.ClientSession()

        # Construct the OpenSPG prompt
        prompt = get_open_spg_llm_prompt(text, author_id)

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
                        "max_tokens": 4000,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    if response.status != 200:
                        logger.error(
                            "LLM API request failed: status=%d, response=%s",
                            response.status,
                            await response.text(),
                        )
                        if attempt < max_retries:
                            continue
                        return ExtractionResult(nodes=[], edges=[])

                    data = await response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                    if not content:
                        logger.warning("LLM returned empty content")
                        if attempt < max_retries:
                            continue
                        return ExtractionResult(nodes=[], edges=[])

                    # Try to repair JSON if malformed
                    repaired_content = _repair_json(content)
                    if repaired_content != content:
                        logger.info("Applied JSON repair for post_id=%s", post_id)
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
                                content[-500:] if len(content) > 500 else content,
                            )
                            # Modify prompt for retry
                            prompt = get_open_spg_llm_prompt(text, author_id) + "\n\nIMPORTANT: Your previous response was truncated. Please provide a more concise JSON, focusing only on the most important entities."
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
                                content[-500:] if len(content) > 500 else content,
                            )
                            return ExtractionResult(nodes=[], edges=[])

                    # Convert flat properties dict to OpenSPG format
                    # LLM now returns: {"entities": [{...}], "relations": [{...}]}
                    # where each entity has "properties" as a simple dict
                    entities_data = parsed_json.get("entities", [])
                    relations_data = parsed_json.get("relations", [])

                    # Transform entities: convert properties list to list of Property objects
                    transformed_entities = []
                    for entity_data in entities_data:
                        props_list = entity_data.get("properties", [])
                        properties_list = _convert_properties_dict_to_list(props_list)

                        transformed_entity = {
                            "id": entity_data["id"],
                            "label": entity_data["label"],
                            "name": entity_data["name"],
                            "properties": properties_list,
                        }
                        transformed_entities.append(transformed_entity)

                    # Transform relations: convert properties list to list of Property objects
                    transformed_relations = []
                    for relation_data in relations_data:
                        props_list = relation_data.get("properties", [])
                        properties_list = _convert_properties_dict_to_list(props_list)

                        transformed_relation = {
                            "source_id": relation_data["source_id"],
                            "relation_type": relation_data["relation_type"],
                            "target_id": relation_data["target_id"],
                            "properties": properties_list,
                        }
                        transformed_relations.append(transformed_relation)

                    # Build the OpenSPG result JSON with transformed data
                    open_spg_json = {
                        "entities": transformed_entities,
                        "relations": transformed_relations,
                    }

                    # Validate the transformed result against the OpenSPG schema
                    try:
                        open_spg_result = OpenSPGExtractionResult.model_validate(open_spg_json)
                    except ValidationError as e:
                        if attempt < max_retries:
                            logger.warning(
                                "Failed to validate transformed OpenSPG result (attempt %d/%d) for post_id=%s: %s\n"
                                "First 500 chars of transformed JSON: %s",
                                attempt + 1,
                                max_retries,
                                post_id,
                                e,
                                json.dumps(open_spg_json)[:500],
                            )
                            # Modify prompt for retry
                            prompt = get_open_spg_llm_prompt(text, author_id) + "\n\nIMPORTANT: Your previous response was truncated. Please provide a more concise JSON, focusing only on the most important entities."
                            continue
                        else:
                            logger.error(
                                "Failed to validate transformed OpenSPG result after %d retries for post_id=%s: %s\n"
                                "First 500 chars of transformed JSON: %s",
                                max_retries,
                                post_id,
                                e,
                                json.dumps(open_spg_json)[:500],
                            )
                            return ExtractionResult(nodes=[], edges=[])

                    # Convert OpenSPG result to legacy ExtractionResult for backward compatibility
                    extraction_result = open_spg_result_to_extraction_result(open_spg_result)
                    nodes = extraction_result.nodes
                    edges = extraction_result.edges

                    logger.info(
                        "LLM extraction successful: %d nodes, %d edges",
                        len(nodes),
                        len(edges),
                    )
                    return ExtractionResult(nodes=nodes, edges=edges)

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
        
        # If we exhausted all retries, return empty result
        return ExtractionResult(nodes=[], edges=[])

    async def extract_triplets(
        self, text: str, author_id: int, post_id: int
    ) -> ExtractionResult:
        """
        Extract knowledge triples from the given text using LLM.

        Args:
            text: Input text to analyze.
            author_id: Telegram user ID of the post author.
            post_id: Database ID of the post (for logging).

        Returns:
            ExtractionResult containing extracted nodes and edges.
        """
        logger.debug("Extracting triples from text: %s", text[:100])
        return await self._call_llm(text, author_id, post_id)

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
        db: Database,
        qdrant: QdrantService | None = None,
    ) -> None:
        """
        Process a single post: extract knowledge triples and persist to AGE graph.

        Args:
            post_id: Database ID of the post.
            text: Text content of the post.
            author_id: Telegram user ID of the post author.
            db: Database instance for persistence operations.
            qdrant: Optional QdrantService for syncing entities to vector store.
        """

        logger.info("Processing post id=%s for knowledge extraction", post_id)

        # Extract triples from text
        result = await self.extract_triplets(text, author_id, post_id)

        if not result.nodes and not result.edges:
            logger.warning(
                "Empty extraction for post id=%s. Text snippet: %s",
                post_id,
                text[:100],
            )
            return

        # Build node ID -> label mapping for edge upserts
        node_id_to_label: dict[str, str] = {}
        for node in result.nodes:
            node_id = node.properties.get("id")
            if node_id:
                node_id_to_label[str(node_id)] = node.label

        # Upsert nodes to AGE graph
        for node in result.nodes:
            try:
                await db.upsert_graph_node(
                    label=node.label,
                    properties=node.properties,
                    merge_key="id",
                )
                logger.debug(
                    "Upserted node: label=%s, id=%s",
                    node.label,
                    node.properties.get("id"),
                )
            except Exception as e:
                logger.error(
                    "Failed to upsert node (post_id=%s, label=%s): %s",
                    post_id,
                    node.label,
                    e,
                )
                raise

        # Upsert edges to AGE graph
        for edge in result.edges:
            start_label = node_id_to_label.get(edge.start_node_id, "Entity")
            end_label = node_id_to_label.get(edge.end_node_id, "Entity")
            try:
                await db.upsert_graph_edge(
                    start_label=start_label,
                    start_merge_key="id",
                    start_merge_val=edge.start_node_id,
                    edge_label=edge.edge_label,
                    end_label=end_label,
                    end_merge_key="id",
                    end_merge_val=edge.end_node_id,
                    edge_properties=edge.properties,
                )
                logger.debug(
                    "Upserted edge: %s(%s)-%s->%s(%s)",
                    start_label,
                    edge.start_node_id,
                    edge.edge_label,
                    end_label,
                    edge.end_node_id,
                )
            except Exception as e:
                logger.error(
                    "Failed to upsert edge (post_id=%s, edge=%s): %s",
                    post_id,
                    edge.edge_label,
                    e,
                )
                raise

        # Sync entities to Qdrant (if service is available)
        if qdrant is not None:
            try:
                await qdrant.upsert_entities(result.nodes)
            except Exception as e:
                logger.error(
                    "Failed to sync entities to Qdrant (post_id=%s): %s",
                    post_id,
                    e,
                    exc_info=True,
                )
                # Do not raise - Qdrant failure should not crash the pipeline

        logger.info(
            "Completed processing post id=%s: %d nodes, %d edges",
            post_id,
            len(result.nodes),
            len(result.edges),
        )
