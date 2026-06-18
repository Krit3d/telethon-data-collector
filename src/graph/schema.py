from __future__ import annotations

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


class PropertyType(str, Enum):
    TEXT = "text"
    NUMERIC = "numeric"
    GEO = "geo"
    LANGUAGE = "language"
    LOCATION = "location"


class Property(BaseModel):
    key: str = Field(..., description="Property name/key")
    value: Any = Field(
        ..., description="Property value (validated based on type)"
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

            if prop_type is None or value is None:
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

        return data

    @field_validator("key", mode="before")
    @classmethod
    def validate_and_sanitize_key(cls, v: Any) -> str:
        return sanitize_key(v)

    @model_validator(mode="after")
    def validate_value_against_type(self) -> Property:
        prop_type = self.type
        value = self.value

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

        elif prop_type == PropertyType.LANGUAGE:
            if not (
                isinstance(value, str) and len(value) == 2 and value.isalpha()
            ):
                raise ValueError(
                    f"Property '{self.key}' with type LANGUAGE must be a 2-letter alphabetic string, got {value!r}"
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

    def get_property_dict(self) -> dict[str, Any]:
        return {prop.key: prop.value for prop in self.properties}

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

    def get_property_dict(self) -> dict[str, Any]:
        return {prop.key: prop.value for prop in self.properties}

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
        metadata_json_str = json.dumps(metadata, ensure_ascii=False, indent=2)

        excluded_fields = []
        if "language" in metadata:
            excluded_fields.append("'language' (2-letter code)")
        if "location" in metadata:
            excluded_fields.append("'location' (human-readable location)")
        if "geo" in metadata or "geo_lat" in metadata or "geo_long" in metadata:
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
- If you see information in the text that matches pre-extracted metadata (e.g., language, location, coordinates), IGNORE it completely and do NOT add it to your extraction.
"""

    prompt = f"""You are an OpenSPG (Semantic Parsing Graph) incremental knowledge accumulation engine. Your task is to extract or update knowledge entities and relationships from the provided text, treating the graph as a persistent state that grows incrementally across any domain (IT, politics, lifestyle, science, etc.).

CRITICAL OpenSPG DOCTRINE:
- You are building a KNOWLEDGE GRAPH that accumulates facts over time.
- If a post mentions that a blogger has a second child, you must extract a property {{'key': 'children_count', 'value': 2, 'type': 'numeric'}} for that Actor node, so the graph updates its state incrementally.
- Every entity may have multiple properties added across different texts. Do not replace existing properties; only add new ones or update numeric/text values when new information is provided.
- Use ONLY the four core universal labels: Actor (persons, organizations, groups, social media accounts, channels — the universal node type for ALL social profiles across Telegram, Instagram, YouTube, TikTok, and Threads), Entity (abstract concepts, products, technologies, objects), Event (occurrences, meetings, incidents), Place (geographic locations, cities, coordinates).

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
       {{"key": "property_name", "value": <any>, "type": "text" | "numeric" | "geo" | "language" | "location"}}
     ]
   }}
   
   CRITICAL: "properties" is a LIST OF OBJECTS, each with 'key', 'value', and 'type' fields. NOT a dictionary.
   
   PROPERTY TYPE RULES (strict enforcement):
   - "text": string values (names, descriptions, languages, etc.)
   - "numeric": int or float values (counts, years, quantities, IDs, timestamps)
   - "geo": array of exactly two floats [latitude, longitude] (e.g., [55.7558, 37.6173])
   - "language": 2-letter language code string (e.g., "en", "ru", "zh")
   - "location": human-readable location string (e.g., "Moscow, Russia", "Cupertino, CA")

   CRITICAL: Property keys MUST be in strict snake_case (e.g., 'birth_date', 'co_founder' -> 'co_founder'). No hyphens or spaces allowed in keys.
   
   EXAMPLES OF PROPERTIES:
   - Actor: [{{'key': 'platform', 'value': 'telegram', 'type': 'text'}}, {{'key': 'platform_id', 'value': '123456', 'type': 'text'}}, {{'key': 'nationality', 'value': 'Russian', 'type': 'text'}}]
   - Entity: [{{'key': 'category', 'value': 'programming_language', 'type': 'text'}}, {{'key': 'release_year', 'value': 1991, 'type': 'numeric'}}]
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
     "properties": [{{"key": "...", "value": <any>, "type": "text" | "numeric" | "geo" | "language" | "location"}}]
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
     * Language: 2-letter language code (e.g., "en", "ru", "zh") with type "language" - SKIP if already in metadata
     * Location: human-readable location string (e.g., "Moscow, Russia") with type "location" - SKIP if already in metadata
     * Geo: coordinates as [latitude, longitude] array with type "geo" - SKIP if already in metadata
     * Text: any textual data (names, descriptions, etc.) with type "text"
     * Numeric: counts, ages, years, quantities with type "numeric"
     These properties are essential for maintaining a rich, queryable knowledge graph state.

5. LENGTH CONSTRAINT:
   - To avoid JSON truncation, be extremely concise. Extract a maximum of 5-7 most important entities per post.
   - Only include entities and relations that are directly and explicitly mentioned in the text.
   - Omit less important or peripheral entities to stay within the limit.

6. OUTPUT FORMAT:
Return ONLY a valid JSON object with exactly this structure:
{{
  "entities": [ ... ],
  "relations": [ ... ]
}}

Do not include any explanatory text, markdown formatting, or code block delimiters. The response must be pure JSON.

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
        {{"key": "category", "value": "programming_language", "type": "text"}},
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
