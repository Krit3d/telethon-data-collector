from __future__ import annotations

import re
import uuid
from enum import Enum
import json
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.graph.utils import sanitize_key


def decode_unicode_escapes(val: str) -> str:
    try:
        decoded = re.sub(
            r"\\u([0-9a-fA-F]{4})",
            lambda m: chr(int(m.group(1), 16)),
            val,
        )
        return decoded.replace("\\", "/")
    except Exception:
        return val


_CORRUPTED_VALUES = frozenset({"un", "unknown", "null", "none", "undefined", ""})


class PropertyType(str, Enum):
    TEXT = "text"
    NUMERIC = "numeric"
    GEO = "geo"
    LANGUAGE = "language"
    LOCATION = "location"
    CATEGORY = "category"


class Property(BaseModel):
    key: str = Field(..., description="Property name/key")
    value: Any | None = Field(
        default=None, description="Property value (validated based on type)"
    )
    type: PropertyType = Field(
        ..., description="Property type category from PropertyType enum"
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def convert_type_string(cls, data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(data, dict) and "type" in data:
            type_value = data["type"]
            if isinstance(type_value, str):
                try:
                    data["type"] = PropertyType(type_value)
                except ValueError:
                    data["type"] = PropertyType.TEXT
        return data

    @model_validator(mode="before")
    @classmethod
    def coerce_value_types(cls, data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(data, dict):
            prop_type = data.get("type")
            value = data.get("value")

            if prop_type is None:
                return data

            if value is None:
                return data

            if isinstance(value, bool):
                if prop_type == PropertyType.TEXT:
                    data["value"] = "true" if value else "false"
                elif prop_type == PropertyType.NUMERIC:
                    data["value"] = 1 if value else 0
                else:
                    data["value"] = str(value)
                    data["type"] = PropertyType.TEXT
                return data

            if prop_type == PropertyType.TEXT and isinstance(value, str):
                if value.strip().lower() in _CORRUPTED_VALUES:
                    data["value"] = None
                    return data

            if prop_type in (PropertyType.LANGUAGE, PropertyType.CATEGORY):
                data["value"] = None
                return data

            if prop_type == PropertyType.TEXT and not isinstance(value, str):
                data["value"] = str(value)
                return data

            if prop_type == PropertyType.NUMERIC and isinstance(value, str):
                stripped = value.strip()
                try:
                    if stripped.isdigit() or (
                        stripped.startswith("-") and stripped[1:].isdigit()
                    ):
                        data["value"] = int(stripped)
                    else:
                        data["value"] = float(stripped)
                except (ValueError, TypeError):
                    pass

            value_after = data.get("value")
            if isinstance(value_after, str):
                data["value"] = decode_unicode_escapes(value_after)

        return data

    @field_validator("key", mode="before")
    @classmethod
    def validate_and_sanitize_key(cls, v: Any) -> str:
        return sanitize_key(v)

    @model_validator(mode="after")
    def validate_value_against_type(self) -> Property:
        prop_type = self.type
        value = self.value

        if value is None:
            return self

        if prop_type == PropertyType.LANGUAGE:
            if not (
                isinstance(value, str) and len(value) == 2 and value.isalpha()
            ):
                self.value = None
                return self

        if prop_type == PropertyType.NUMERIC:
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Property '{self.key}' with type NUMERIC must have an int or float value, got {type(value).__name__}"
                )

        elif prop_type == PropertyType.GEO:
            if not (
                isinstance(value, list)
                and len(value) == 2
                and all(isinstance(coord, (int, float)) for coord in value)
            ):
                raise ValueError(
                    f"Property '{self.key}' with type GEO must be a list of two floats, got {value!r}"
                )

        elif prop_type == PropertyType.TEXT:
            if not isinstance(value, str):
                raise ValueError(
                    f"Property '{self.key}' with type TEXT must be a string, got {type(value).__name__}"
                )

        elif prop_type == PropertyType.LOCATION:
            if not isinstance(value, str):
                raise ValueError(
                    f"Property '{self.key}' with type LOCATION must be a string, got {type(value).__name__}"
                )

        elif prop_type == PropertyType.CATEGORY:
            if not isinstance(value, str):
                self.value = str(value)

        return self


class ExtractedEntity(BaseModel):
    id: str = Field(
        ...,
        description="Unique standardized identifier (e.g., 'person_pavel_durov')",
    )
    label: str = Field(
        ...,
        description="Entity type (e.g., 'Person', 'Organization', 'Location')",
    )
    name: str = Field(..., description="Display name for the entity")
    properties: list[Property] = Field(
        default_factory=list,
        description="List of typed properties (age, coordinates, language, etc.)",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("id", mode="before")
    @classmethod
    def sanitize_id(cls, v: Any) -> str:
        value = str(v).strip()
        value = re.sub(r"[^a-zA-Z0-9_]", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")
        if not value:
            value = f"entity_{uuid.uuid4().hex[:8]}"
        if value[0].isdigit():
            value = f"ent_{value}"
        return value

    @field_validator("label", mode="before")
    @classmethod
    def sanitize_label(cls, v: Any) -> str:
        value = str(v).strip()
        value = re.sub(r"[^a-zA-Z0-9_]", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")
        if not value:
            value = "Entity"
        if value[0].isdigit():
            value = f"Label_{value}"
        else:
            value = value[0].upper() + value[1:]
        return value

    @model_validator(mode="before")
    @classmethod
    def sanitize_entity_name(cls, data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(data, dict) and "name" in data and isinstance(data["name"], str):
            name = data["name"]
            name = name.lstrip("#").strip()
            name = decode_unicode_escapes(name)
            name = re.sub(r"\[.*?\]", "", name)
            name = re.sub(r"\s+", " ", name).strip()
            data["name"] = name
        return data

    def get_property_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for prop in self.properties:
            if prop.value is None:
                continue
            if str(prop.value).strip().lower() in _CORRUPTED_VALUES:
                continue
            result[prop.key] = prop.value
        return result

    def add_property(
        self, key: str, value: Any, type: str | PropertyType
    ) -> None:
        if isinstance(type, str):
            type = PropertyType(type)
        self.properties.append(Property(key=key, value=value, type=type))


class ExtractedRelation(BaseModel):
    source_id: str = Field(..., description="ID of the source entity")
    relation_type: str = Field(
        ...,
        description="Strict relationship type (e.g., 'LOCATED_IN', 'WORKS_AT', 'DISCUSSES')",
    )
    target_id: str = Field(..., description="ID of the target entity")
    properties: list[Property] = Field(
        default_factory=list,
        description="Optional list of typed relation properties",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("source_id", "target_id", mode="before")
    @classmethod
    def sanitize_relation_id(cls, v: Any) -> str:
        value = str(v).strip()
        value = re.sub(r"[^a-zA-Z0-9_]", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")
        if not value:
            value = f"entity_{uuid.uuid4().hex[:8]}"
        if value[0].isdigit():
            value = f"ent_{value}"
        return value

    @field_validator("relation_type", mode="before")
    @classmethod
    def sanitize_relation_type(cls, v: Any) -> str:
        value = str(v).strip()
        value = re.sub(r"[^a-zA-Z0-9_]", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")
        if not value:
            value = "RELATED_TO"
        if value[0].isdigit():
            value = f"REL_{value}"
        return value.upper()

    def get_property_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for prop in self.properties:
            if prop.value is None:
                continue
            if str(prop.value).strip().lower() in _CORRUPTED_VALUES:
                continue
            result[prop.key] = prop.value
        return result

    def add_property(
        self, key: str, value: Any, type: str | PropertyType
    ) -> None:
        if isinstance(type, str):
            type = PropertyType(type)
        self.properties.append(Property(key=key, value=value, type=type))


class OpenSPGExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(
        default_factory=list, description="List of extracted entity nodes"
    )
    relations: list[ExtractedRelation] = Field(
        default_factory=list, description="List of extracted relation edges"
    )

    model_config = ConfigDict(extra="forbid")


class SPGNode(BaseModel):
    label: str
    properties: dict[str, Any]

    model_config = ConfigDict(extra="allow")


class SPGEdge(BaseModel):
    start_node_id: str
    edge_label: str
    end_node_id: str
    properties: dict[str, Any] = {}

    model_config = ConfigDict(extra="allow")


class ExtractionResult(BaseModel):
    nodes: list[SPGNode]
    edges: list[SPGEdge]

    model_config = ConfigDict(extra="allow")


class LLMNode(BaseModel):
    id: str
    label: str
    name: str
    description: str

    model_config = ConfigDict(extra="allow")


class LLMEdge(BaseModel):
    source_id: str
    relation_type: str
    target_id: str
    properties: dict[str, Any] = {}

    model_config = ConfigDict(extra="allow")


class LLMExtractionResult(BaseModel):
    nodes: list[LLMNode]
    edges: list[LLMEdge]

    model_config = ConfigDict(extra="allow")


def entity_to_spg_node(entity: ExtractedEntity) -> SPGNode:
    prop_dict = entity.get_property_dict()
    spg_properties = {
        "id": entity.id,
        "name": entity.name,
        **prop_dict,
    }
    return SPGNode(label=entity.label, properties=spg_properties)


def relation_to_spg_edge(relation: ExtractedRelation) -> SPGEdge:
    prop_dict = relation.get_property_dict()
    return SPGEdge(
        start_node_id=relation.source_id,
        edge_label=relation.relation_type,
        end_node_id=relation.target_id,
        properties=prop_dict,
    )


def open_spg_result_to_extraction_result(
    open_spg_result: OpenSPGExtractionResult,
) -> ExtractionResult:
    nodes = [entity_to_spg_node(entity) for entity in open_spg_result.entities]
    edges = [
        relation_to_spg_edge(relation) for relation in open_spg_result.relations
    ]
    return ExtractionResult(nodes=nodes, edges=edges)


def get_open_spg_llm_prompt(
    text: str,
    pub_node_id: str,
    author_id: str | int | None = None,
    platform: str | None = None,
    metadata: dict | None = None,
) -> str:
    author_instruction = ""
    if author_id is not None:
        platform_slug = platform.lower() if platform else "unknown"
        actor_id = f"actor_{platform_slug}_{author_id}"
        author_instruction = f"""5. If the text EXPLICITLY mentions or references the author by name (platform ID: {author_id}), OR if the author performs specific actions described in the text, create an Actor node with id "{actor_id}" and name based on the author's displayed name (if available) or "Author {author_id}". The Actor node MUST always include the following properties: {{'key': 'platform', 'value': '{platform or 'unknown'}', 'type': 'text'}} and {{'key': 'platform_id', 'value': '{author_id}', 'type': 'text'}}. If the author is only implied or not directly mentioned, DO NOT create an author node.
6. """

    metadata_instruction = ""
    metadata_json_str = ""
    if metadata is not None and len(metadata) > 0:
        cleaned_metadata = {k: v for k, v in metadata.items() if k not in ("category", "language")}
        metadata_json_str = json.dumps(cleaned_metadata, ensure_ascii=False, indent=2)

        excluded_fields = []
        if "location" in cleaned_metadata:
            excluded_fields.append("'location' (human-readable location)")
        if "geo" in cleaned_metadata or "geo_lat" in cleaned_metadata or "geo_long" in cleaned_metadata:
            excluded_fields.append("'geo'/'geo_lat'/'geo_long' (coordinates)")

        excluded_text = ", ".join(excluded_fields) if excluded_fields else "none"

        metadata_instruction = f"""
PRE-EXTRACTED METADATA (DO NOT RE-EXTRACT):
The following metadata has been pre-collected from the post and is already available in the database:
{metadata_json_str}

CRITICAL INSTRUCTION FOR PRE-EXTRACTED METADATA:
- The following fields are ALREADY PRESENT in the pre-extracted metadata: {excluded_text}
- STRICTLY FORBIDDEN: DO NOT extract or include any of these pre-extracted fields in your output.
- Focus strictly on extracting NEW business logic entities (Actors, Entities, Events, Places) and their relations from the text.
- The pre-extracted metadata will be merged into the Content node automatically; you do NOT need to include these as properties in your extraction output.
- If you see information in the text that matches pre-extracted metadata (e.g., location, coordinates), IGNORE it completely and do NOT add it to your extraction.
"""

    prompt = f"""You are an OpenSPG (Semantic Parsing Graph) incremental knowledge accumulation engine. Your task is to extract or update knowledge entities and relationships from the provided text, treating the graph as a persistent state that grows incrementally across any domain (IT, politics, lifestyle, science, etc.).

CRITICAL OpenSPG DOCTRINE:
- You are building a KNOWLEDGE GRAPH that accumulates facts over time.
- If a post mentions that a blogger has a second child, you must extract a property {{'key': 'children_count', 'value': 2, 'type': 'numeric'}} for that Actor node, so the graph updates its state incrementally.
- Every entity may have multiple properties added across different texts. Do not replace existing properties; only add new ones or update numeric/text values when new information is provided.
- Use ONLY the four core universal labels: Actor (persons, organizations, groups, social media accounts, channels — the universal node type for ALL social profiles across Telegram, Instagram, YouTube, TikTok, and Threads), Entity (abstract concepts, products, technologies, objects), Event (occurrences, meetings, incidents), Place (geographic locations, cities, coordinates).

MAIN POST ENTITY:
The text below is from a single post/publication. You MUST create a node for this post with the EXACT id: "{pub_node_id}".
Label this node as "Event" or "Entity" as appropriate. All relations that discuss, mention, or are about the post's core content MUST use "{pub_node_id}" as their source_id or target_id.

OpenSPG Principles:
- Build a connected subgraph of entities and relationships.
- All non-entity attributes MUST be expressed as TYPED PROPERTIES with explicit type classification.
- Use canonical IDs (lowercase with underscores) and UPPER_SNAKE_CASE relation types.
- Properties are the mechanism for incremental state updates.

Text to analyze:
{text}
{metadata_instruction}
Extraction Instructions:

1. UNIVERSAL ENTITY TYPES (use EXACTLY these four labels):
   - Actor: People, organizations, groups, social media accounts, channels, and any named entity that can act or be referenced. Actor is the UNIVERSAL node type for ALL social profiles — every social media account or channel (Telegram, Instagram, YouTube, TikTok, Threads) MUST be extracted as an Actor node (e.g., "Pavel Durov", "TechCrunch Telegram channel", "UN Security Council").
     CRITICAL MANDATE: ALL social media accounts and channels across platforms (Telegram, Instagram, YouTube, TikTok, Threads) MUST include at minimum these two MANDATORY properties:
       * {{"key": "platform", "value": "<platform_name>", "type": "text"}} (one of: "telegram", "instagram", "youtube", "tiktok", "threads")
       * {{"key": "platform_id", "value": "<account_or_channel_identifier>", "type": "text"}}
   - Entity: Concepts, objects, products, technologies, ideas, or any non-place, non-event thing (e.g., "Python programming language", "Quantum Computing", "iPhone 15")
   - Event: Occurrences, meetings, incidents, or time-bound activities (e.g., "WWDC 2024", "Russian invasion of Ukraine", "product launch")
   - Place: Geographic locations, cities, countries, coordinates, or physical places (e.g., "Moscow", "Silicon Valley", "[55.7558, 37.6173]")

2. ENTITY FORMAT (strict JSON structure):
   {{
     "id": "canonical_string_id" (e.g., "actor_pavel_durov", "entity_python", "event_wwdc_2024", "place_moscow"),
     "label": "Actor" | "Entity" | "Event" | "Place" (exactly one of these four),
     "name": "Human-readable display name",
     "properties": [
       {{"key": "property_name", "value": <any>, "type": "text" | "numeric" | "geo" | "location"}}
     ]
   }}
   
   CRITICAL: "properties" is a LIST OF OBJECTS, each with 'key', 'value', and 'type' fields. NOT a dictionary.
   
   PROPERTY TYPE RULES (strict enforcement):
   - "text": string values (names, descriptions, etc.)
   - "numeric": int or float values (counts, years, quantities, IDs, timestamps)
   - "geo": array of exactly two floats [latitude, longitude] (e.g., [55.7558, 37.6173])
   - "location": human-readable location string (e.g., "Moscow, Russia", "Cupertino, CA")

   CRITICAL: Property keys MUST be in strict snake_case (e.g., 'birth_date', 'co_founder' -> 'co_founder'). No hyphens or spaces allowed in keys.
   
   EXAMPLES OF PROPERTIES:
   - Actor: [{{'key': 'platform', 'value': 'telegram', 'type': 'text'}}, {{'key': 'platform_id', 'value': '123456', 'type': 'text'}}, {{'key': 'nationality', 'value': 'Russian', 'type': 'text'}}]
   - Entity: [{{'key': 'domain', 'value': 'programming_language', 'type': 'text'}}, {{'key': 'release_year', 'value': 1991, 'type': 'numeric'}}]
   - Event: [{{'key': 'timestamp', 'value': 1712345678, 'type': 'numeric'}}, {{'key': 'location', 'value': 'San Francisco', 'type': 'location'}}]
   - Place: [{{'key': 'coordinates', 'value': [55.7558, 37.6173], 'type': 'geo'}}, {{'key': 'country', 'value': 'Russia', 'type': 'text'}}]

3. RELATIONSHIPS:
   - relation_type: MUST be UPPER_SNAKE_CASE (e.g., LOCATED_IN, WORKS_AT, DISCUSSES, MENTIONS, CREATED, PART_OF, ATTENDED, INFLUENCED_BY)
   - Connect entities via source_id and target_id
   - "properties": list of typed property objects (can include metadata like confidence, time_period, role)
   
   RELATIONSHIP FORMAT:
   {{
     "source_id": "entity_id",
     "relation_type": "UPPER_SNAKE_CASE",
     "target_id": "entity_id",
     "properties": [{{"key": "...", "value": <any>, "type": "text" | "numeric" | "geo" | "location"}}]
   }}

{author_instruction}4. INCREMENTAL AGGREGATION STRATEGY:
   - Extract ALL entities and relationships mentioned in the text, even if they seem obvious.
   - For each entity, include at least 1-2 relevant properties beyond the name.
   - If an entity already exists in the graph with properties, you are ADDING to its knowledge. Do not worry about deduplication; the system handles merging by canonical ID.
   - CRITICAL UPDATE RULE: If you detect that a property (like a counter, status, location, or any other attribute) has CHANGED from a previous state, you MUST return the NEW value for that property with the SAME property key. The system will merge by replacing the old value with the new one. For example, if "John now has 2 children" and previously he had 1, extract {{'key': 'children_count', 'value': 2, 'type': 'numeric'}}.
   - Numeric properties can be updated (e.g., if children_count increases, extract the new value).
   - Text properties can be extended (e.g., add new 'alternate_name' or 'description').
   - Always prefer extracting concrete, factual properties over vague ones.
   - MANDATORY PROPERTY EXTRACTION: ALWAYS extract the following property types if present in the text AND NOT already provided in pre-extracted metadata:
     * Location: human-readable location string (e.g., "Moscow, Russia") with type "location" - SKIP if already in metadata
     * Geo: coordinates as [latitude, longitude] array with type "geo" - SKIP if already in metadata
     * Text: any textual data (names, descriptions, etc.) with type "text"
     * Numeric: counts, ages, years, quantities with type "numeric"
     These properties are essential for maintaining a rich, queryable knowledge graph state.

5. LENGTH CONSTRAINT:
   - To avoid JSON truncation, be extremely concise. Extract a maximum of 5-7 most important entities per post.
   - Only include entities and relations that are directly and explicitly mentioned in the text.
   - Omit less important or peripheral entities to stay within the limit.

6. PROPERTY VALUE QUALITY (MANDATORY):
   - NEVER output properties with values like "unknown", "un", "none", "null", "undefined", or empty strings ("").
   - If the exact factual value of a property is NOT known or is uncertain, COMPLETELY OMIT that property from the "properties" list. Do NOT use placeholder or corrupted values.
   - Corrupted placeholder values cause downstream parse failures and pollute the knowledge graph with garbage data.
   - Only include properties where you can provide a CONCRETE, FACTUAL, verifiable value.

7. JSON FORMAT COMPLIANCE (MANDATORY):
   - The JSON structure MUST be 100% compliant with the OpenSPG schema defined above.
   - Absolutely NO markdown wrapping (no ```json or ``` delimiters around the output).
   - No trailing commas anywhere in the JSON object or arrays.
   - No formatting errors — every opening bracket/brace must have a matching closing counterpart.
   - Any JSON parse failure will corrupt the entire extraction result for this post.
   - Your ENTIRE response must be a single valid JSON object and nothing else.

8. OUTPUT FORMAT:
Return ONLY a valid JSON object with exactly this structure. No markdown, no code blocks, no explanations:
{{
  "entities": [ ... ],
  "relations": [ ... ]
}}

TOPIC COVERAGE RULE (MANDATORY for Knowledge Augmented Generation):
Any key concepts, technologies, products, or subjects discussed in the text MUST be represented as Entity nodes. The main subject or publication of the text MUST connect to these Entity nodes using the ABOUT relationship. For example, if a post discusses Python programming, you MUST create an Entity node for "Python" and connect the publication to it via ABOUT. The ABOUT relation is the PRIMARY connection between a post/publication and its core subjects — always prefer ABOUT over DISCUSSES for the main topic link.

Example for a tech blog post:
{{
  "entities": [
    {{
      "id": "actor_john_doe",
      "label": "Actor",
      "name": "John Doe",
      "properties": [
        {{"key": "platform", "value": "telegram", "type": "text"}},
        {{"key": "platform_id", "value": "987654321", "type": "text"}},
        {{"key": "role", "value": "software_engineer", "type": "text"}}
      ]
    }},
    {{
      "id": "entity_python",
      "label": "Entity",
      "name": "Python",
      "properties": [
        {{"key": "domain", "value": "programming_language", "type": "text"}},
        {{"key": "release_year", "value": 1991, "type": "numeric"}}
      ]
    }},
    {{
      "id": "event_pycon_2024",
      "label": "Event",
      "name": "PyCon 2024",
      "properties": [
        {{"key": "timestamp", "value": 1712345678, "type": "numeric"}},
        {{"key": "location", "value": "Pittsburgh, PA", "type": "location"}}
      ]
    }},
    {{
      "id": "place_pittsburgh",
      "label": "Place",
      "name": "Pittsburgh",
      "properties": [
        {{"key": "coordinates", "value": [40.4406, -79.9959], "type": "geo"}},
        {{"key": "country", "value": "USA", "type": "text"}}
      ]
    }}
  ],
  "relations": [
    {{
      "source_id": "event_pycon_2024",
      "relation_type": "ABOUT",
      "target_id": "entity_python"
    }},
    {{
      "source_id": "actor_john_doe",
      "relation_type": "ATTENDED",
      "target_id": "event_pycon_2024",
      "properties": [
        {{"key": "role", "value": "speaker", "type": "text"}}
      ]
    }},
    {{
      "source_id": "event_pycon_2024",
      "relation_type": "LOCATED_IN",
      "target_id": "place_pittsburgh"
    }},
    {{
      "source_id": "actor_john_doe",
      "relation_type": "USES",
      "target_id": "entity_python"
    }}
  ]
}}"""
    return prompt
