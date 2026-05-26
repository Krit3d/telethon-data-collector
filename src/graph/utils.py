"""Utility functions for knowledge graph extraction and processing."""

import json
import logging
import re
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _repair_json(content: str) -> str:
    """Attempt to repair malformed or truncated JSON.

    Uses multiple strategies to fix common issues:
    - Strip markdown code blocks (```json ... ```)
    - Unclosed braces/brackets at root level
    - Truncated objects inside entities/relations lists (discards partial objects)
    - Trailing incomplete tokens
    - Fallback: find last complete closing brace/bracket and close root

    Args:
        content: The potentially malformed JSON string.

    Returns:
        Repaired JSON string if successful, otherwise original content.
    """
    original = content

    content = content.strip()
    if content.startswith("```json") and content.endswith("```"):
        content = content[7:-3].strip()
    elif content.startswith("```") and content.endswith("```"):
        content = content[3:-3].strip()

    if not content:
        return original

    # First, try simple parsing to see if it is already valid
    try:
        json.loads(content)
        return content
    except json.JSONDecodeError:
        pass  # Proceed with repair strategies

    # Pre-check: Detect if the last character is inside an unclosed string
    # (common truncation pattern: "... "key": "value" <-- missing closing quote)
    in_string = False
    escape_next = False
    for i, char in enumerate(content):
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string

    # If we end inside a string, add a closing quote before further repairs
    if in_string:
        content = content + '"'
        logger.debug("Added closing quote for unclosed string in JSON")
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            pass  # Continue with other strategies

    # Strategy 1: Simple brace/bracket matching for root-level truncation
    brace_count = 0
    bracket_count = 0
    in_string = False
    escape_next = False
    last_valid_pos = -1

    for i, char in enumerate(content):
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and bracket_count == 0:
                last_valid_pos = i
        elif char == "[":
            bracket_count += 1
        elif char == "]":
            bracket_count -= 1
            if brace_count == 0 and bracket_count == 0:
                last_valid_pos = i

    if brace_count > 0 or bracket_count > 0:
        if last_valid_pos > 0:
            truncated = content[: last_valid_pos + 1]
            truncated += "]" * bracket_count
            truncated += "}" * brace_count
            try:
                json.loads(truncated)
                logger.debug("JSON repaired using root-level brace closure")
                return truncated
            except json.JSONDecodeError:
                pass  # Fall through to array-level repair

    # Strategy 2: Handle truncation inside entities/relations arrays
    # Find the start positions of entities and relations arrays
    entities_key_match = re.search(r'"entities"\s*:\s*\[', content)
    relations_key_match = re.search(r'"relations"\s*:\s*\[', content)

    if entities_key_match or relations_key_match:
        truncate_point = last_valid_pos if last_valid_pos > 0 else len(content)

        # Find the last complete entity
        last_complete_entity_end = -1
        if entities_key_match:
            entities_array_start = entities_key_match.end()
            search_limit = min(truncate_point, len(content))
            pos = entities_array_start
            while pos < search_limit:
                brace_pos = content.find("}", pos, search_limit)
                if brace_pos == -1:
                    break
                depth = 0
                i = entities_array_start
                while i <= brace_pos:
                    c = content[i]
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            after = brace_pos + 1
                            while (
                                after < search_limit
                                and content[after] in " \t\n\r"
                            ):
                                after += 1
                            if after >= search_limit:
                                last_complete_entity_end = brace_pos
                                break
                            if content[after] in (",", "]"):
                                last_complete_entity_end = brace_pos
                                pos = brace_pos + 1
                                if content[after] == "]":
                                    break
                                continue
                    i += 1
                break

        # Find the last complete relation
        last_complete_relation_end = -1
        if relations_key_match:
            relations_array_start = relations_key_match.end()
            search_limit = min(truncate_point, len(content))
            pos = relations_array_start
            while pos < search_limit:
                brace_pos = content.find("}", pos, search_limit)
                if brace_pos == -1:
                    break
                depth = 0
                i = relations_array_start
                while i <= brace_pos:
                    c = content[i]
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            after = brace_pos + 1
                            while (
                                after < search_limit
                                and content[after] in " \t\n\r"
                            ):
                                after += 1
                            if after >= search_limit:
                                last_complete_relation_end = brace_pos
                                break
                            if content[after] in (",", "]"):
                                last_complete_relation_end = brace_pos
                                pos = brace_pos + 1
                                if content[after] == "]":
                                    break
                                continue
                    i += 1
                break

        if last_complete_entity_end > 0 or last_complete_relation_end > 0:
            repaired = "{"
            if entities_key_match:
                repaired += content[
                    entities_key_match.start() : entities_array_start
                ]
                if last_complete_entity_end > 0:
                    repaired += content[
                        entities_array_start : last_complete_entity_end + 1
                    ]
                repaired += "]"
            if relations_key_match:
                if entities_key_match:
                    repaired += ","
                repaired += content[
                    relations_key_match.start() : relations_array_start
                ]
                if last_complete_relation_end > 0:
                    repaired += content[
                        relations_array_start : last_complete_relation_end + 1
                    ]
                repaired += "]"
            repaired += "}"

            try:
                parsed = json.loads(repaired)
                if isinstance(parsed.get("entities"), list) and isinstance(
                    parsed.get("relations"), list
                ):
                    logger.debug(
                        "JSON repaired using array truncation strategy"
                    )
                    return repaired
            except json.JSONDecodeError:
                pass

    # Strategy 3 (Fallback): Find the last complete closing brace or bracket
    # that closes the root object/array and truncate there
    if last_valid_pos > 0:
        # Try to close the root object with just a single }
        fallback = content[: last_valid_pos + 1]
        # Count if we need to add closing brackets/braces
        if brace_count > 0:
            fallback += "}" * brace_count
        if bracket_count > 0:
            fallback += "]" * bracket_count
        try:
            parsed = json.loads(fallback)
            logger.debug(
                "JSON repaired using fallback brace closure at last_valid_pos"
            )
            return fallback
        except json.JSONDecodeError:
            pass

    return original


def _sanitize_key(key: Any) -> str:
    """Sanitize dictionary keys to be Cypher-safe snake_case.

    Converts the key to string, replaces spaces and hyphens with underscores,
    removes special characters, and converts to lowercase.
    Handles keys like 'geo_lat' or 'geo_long' to ensure they remain
    as valid Cypher identifiers (alphanumeric and underscores only).

    Args:
        key: The original key value.

    Returns:
        Sanitized key string.
    """
    if not isinstance(key, str):
        key = str(key)
    # Convert hyphens and spaces to underscores
    key = key.replace("-", "_").replace(" ", "_")
    # Remove non-alphanumeric and non-underscore characters
    key = re.sub(r'[^A-Za-z0-9_]', '', key)
    # Convert to lowercase
    key = key.lower()
    return key if key else "unknown_property"


def _convert_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a Pydantic model or other object to a dictionary safely.

    Handles Pydantic models (v1 and v2), dataclasses, and falls back
    to dict conversion or empty dict if conversion fails.

    Args:
        obj: The object to convert.

    Returns:
        Dictionary representation of the object, or empty dict if conversion fails.
    """
    if obj is None:
        return {}
    
    # Handle Pydantic models (v2 and v1 compatibility)
    if isinstance(obj, BaseModel):
        try:
            return obj.model_dump(exclude_none=True)
        except AttributeError:
            # Fallback for Pydantic v1
            try:
                return obj.dict(exclude_none=True)
            except AttributeError:
                logger.warning("Failed to convert Pydantic model to dict")
                return {}
    
    # Handle dictionaries directly
    if isinstance(obj, dict):
        return obj
    
    # Handle objects with __dict__ attribute (dataclasses, regular objects)
    if hasattr(obj, '__dict__'):
        try:
            return dict(obj.__dict__)
        except (TypeError, ValueError):
            pass
    
    logger.warning("Cannot convert object of type %s to dict", type(obj).__name__)
    return {}


def _merge_metadata_into_properties(
    base_props: dict[str, Any],
    metadata: dict[str, Any] | BaseModel | None,
) -> dict[str, Any]:
    """Merge metadata dictionary or Pydantic model into base properties with key sanitization.

    This function handles both dictionary and Pydantic model inputs for metadata.
    All keys are sanitized to be Cypher-safe snake_case identifiers.
    Existing keys in base_props are overwritten by metadata values.

    Args:
        base_props: Base properties dictionary.
        metadata: Metadata dictionary or Pydantic model to merge (keys will be sanitized).
                  Can be None, in which case base_props is returned unchanged.

    Returns:
        Merged properties dictionary with sanitized keys.
    """
    if metadata is None:
        return base_props

    # Convert Pydantic models or other objects to dict
    metadata_dict = _convert_to_dict(metadata)
    
    if not metadata_dict:
        return base_props

    merged = base_props.copy()
    for key, value in metadata_dict.items():
        sanitized_key = _sanitize_key(key)
        # Only merge if value is not None (skip None values to keep existing ones)
        if value is not None:
            merged[sanitized_key] = value
        elif sanitized_key not in merged:
            # Only set None if the key doesn't exist in merged
            merged[sanitized_key] = None
    return merged
