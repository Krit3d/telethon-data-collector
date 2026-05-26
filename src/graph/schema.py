"""Pydantic models for OpenSPG (Semantic Parsing Graph) schema elements.

OpenSPG supports dynamic knowledge extraction with typed property accumulation
for flexible entity and relationship attributes.
"""

from enum import Enum
import json
import re
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class PropertyType(str, Enum):
    """Enumeration of supported property types in OpenSPG.

    All properties in the system must conform to one of these strict types
    to ensure type safety and universal domain support (IT, politics, etc.).
    """

    TEXT = "text"
    NUMERIC = "numeric"
    GEO = "geo"
    LANGUAGE = "language"
    LOCATION = "location"


class Property(BaseModel):
    """Represents a typed dynamic property with strict validation.

    Properties are used to store attributes on entities and relations.
    Each property has a key, a value, and a type from the PropertyType enum.
    The value is validated against the type to ensure data integrity.
    """

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
        """Convert string type to PropertyType enum for backward compatibility.

        This validator allows passing type as a string (e.g., "text") which
        is automatically converted to the corresponding PropertyType enum value.
        If the string is not a valid PropertyType member, falls back to
        PropertyType.TEXT to prevent Pydantic ValidationError.

        Args:
            data: Raw input data before model validation.

        Returns:
            Modified data with type converted to PropertyType enum if it was a string,
            or PropertyType.TEXT if the string was invalid.
        """
        if isinstance(data, dict) and "type" in data:
            type_value = data["type"]
            if isinstance(type_value, str):
                try:
                    data["type"] = PropertyType(type_value)
                except ValueError:
                    # Fallback to TEXT for invalid type strings to ensure resilience
                    data["type"] = PropertyType.TEXT
        return data

    @model_validator(mode="before")
    @classmethod
    def coerce_value_types(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Coerce value types to match PropertyType expectations for resilience.

        This validator automatically converts common type mismatches caused by LLM output:
        - Boolean values: handled based on prop_type:
          - TEXT: converts to "true" or "false" strings
          - NUMERIC: converts to 1 or 0 integers
          - Other types: falls back to string conversion and sets type to TEXT
        - TEXT: converts non-string values (int, float) to strings
        - NUMERIC: converts numeric strings to int or float

        This ensures the extractor does not crash on minor type mistakes while
        maintaining strict validation for GEO and LANGUAGE types.

        Args:
            data: Raw input data before model validation.

        Returns:
            Modified data with coerced values where appropriate.
        """

        if isinstance(data, dict):
            prop_type = data.get("type")
            value = data.get("value")

            if prop_type is None or value is None:
                return data

            # Handle boolean values first to avoid misclassification as integers
            if isinstance(value, bool):
                if prop_type == PropertyType.TEXT:
                    data["value"] = "true" if value else "false"
                elif prop_type == PropertyType.NUMERIC:
                    data["value"] = 1 if value else 0
                else:
                    # Fallback: convert to string and set type to TEXT
                    data["value"] = str(value)
                    data["type"] = PropertyType.TEXT
                return data

            # Handle TEXT type: convert any non-string to string
            if prop_type == PropertyType.TEXT and not isinstance(value, str):
                data["value"] = str(value)
                return data

            # Handle NUMERIC type: convert string numbers to numeric
            if prop_type == PropertyType.NUMERIC and isinstance(value, str):
                stripped = value.strip()
                try:
                    # Try to convert to int first for whole numbers (including negative)
                    if stripped.isdigit() or (
                        stripped.startswith("-") and stripped[1:].isdigit()
                    ):
                        data["value"] = int(stripped)
                    else:
                        # Try float for decimal numbers or scientific notation
                        data["value"] = float(stripped)
                except (ValueError, TypeError):
                    # If conversion fails, keep original and let validation handle error
                    pass

        return data

    @field_validator("key", mode="before")
    @classmethod
    def sanitize_key(cls, v: Any) -> str:
        """Sanitize property keys to ensure they are Cypher-safe snake_case.

        Converts hyphens and spaces to underscores, and strips special characters.
        """

        if not isinstance(v, str):
            v = str(v)
        # Convert hyphens and spaces to underscores
        v = v.replace("-", "_").replace(" ", "_")
        # Remove non-alphanumeric and non-underscore characters
        v = re.sub(r"[^A-Za-z0-9_]", "", v)
        # Convert to lowercase
        v = v.lower()
        return v if v else "unknown_property"

    @model_validator(mode="after")
    def validate_value_against_type(self) -> "Property":
        """Validate that the property value matches the expected type.

        Enforces strict type constraints:
        - NUMERIC: value must be int or float
        - GEO: value must be a list of exactly two floats
        - LANGUAGE: value must be a 2-letter string
        - TEXT: any string value accepted
        - LOCATION: any string value accepted (human-readable location)

        Returns:
            The validated Property instance.

        Raises:
            ValueError: If the value does not match the type constraints.
        """

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
    """Represents an extracted entity node in OpenSPG format.

    Entities are the nodes in the knowledge graph. They have a unique standardized
    ID, a type label, a display name, and a list of typed properties.
    """

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
        """Convert properties list to a dictionary keyed by property key.

        Returns:
            Dictionary mapping property keys to their values.
        """
        return {prop.key: prop.value for prop in self.properties}

    def add_property(
        self, key: str, value: Any, type: str | PropertyType
    ) -> None:
        """Add a typed property to the entity.

        Args:
            key: Property name.
            value: Property value (will be validated against type).
            type: Property type - either a PropertyType enum member or its string value.
        """
        # Convert string to PropertyType enum for backward compatibility
        if isinstance(type, str):
            type = PropertyType(type)
        self.properties.append(Property(key=key, value=value, type=type))


class ExtractedRelation(BaseModel):
    """Represents an extracted relationship/edge in OpenSPG format.

    Relations connect entities and can have their own typed properties
    to provide context (e.g., 'since_year', 'confidence', 'role').
    """

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
        """Convert properties list to a dictionary keyed by property key.

        Returns:
            Dictionary mapping property keys to their values.
        """
        return {prop.key: prop.value for prop in self.properties}

    def add_property(
        self, key: str, value: Any, type: str | PropertyType
    ) -> None:
        """Add a typed property to the relation.

        Args:
            key: Property name.
            value: Property value (will be validated against type).
            type: Property type - either a PropertyType enum member or its string value.
        """
        # Convert string to PropertyType enum for backward compatibility
        if isinstance(type, str):
            type = PropertyType(type)
        self.properties.append(Property(key=key, value=value, type=type))


class OpenSPGExtractionResult(BaseModel):
    """Container for the result of an OpenSPG knowledge extraction operation.

    Holds all extracted entities and relations from a text extraction.
    """

    entities: list[ExtractedEntity] = Field(
        default_factory=list, description="List of extracted entity nodes"
    )
    relations: list[ExtractedRelation] = Field(
        default_factory=list, description="List of extracted relation edges"
    )

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Backward Compatibility Layer
# ---------------------------------------------------------------------------
# The original SPGNode and SPGEdge classes are kept for backward compatibility
# with existing code (database operations, Qdrant service). New code should use
# ExtractedEntity and ExtractedRelation. Conversion functions are provided to
# migrate between formats.


class SPGNode(BaseModel):
    """Represents a graph node in the knowledge graph.

    DEPRECATED: Use ExtractedEntity for new code. Kept for backward compatibility.
    """

    label: str
    properties: dict[str, Any]

    model_config = ConfigDict(extra="allow")


class SPGEdge(BaseModel):
    """Represents a graph edge/relationship in the knowledge graph.

    DEPRECATED: Use ExtractedRelation for new code. Kept for backward compatibility.
    """

    start_node_id: str
    edge_label: str
    end_node_id: str
    properties: dict[str, Any] = {}

    model_config = ConfigDict(extra="allow")


class ExtractionResult(BaseModel):
    """Container for the result of a knowledge extraction operation.

    DEPRECATED: Use OpenSPGExtractionResult for new code. Kept for backward compatibility.
    """

    nodes: list[SPGNode]
    edges: list[SPGEdge]

    model_config = ConfigDict(extra="allow")


class LLMNode(BaseModel):
    """Schema for nodes as returned by the LLM extraction.

    DEPRECATED: Use ExtractedEntity for new code. Kept for backward compatibility.
    """

    id: str
    label: str
    name: str
    description: str

    model_config = ConfigDict(extra="allow")


class LLMEdge(BaseModel):
    """Schema for edges as returned by the LLM extraction.

    DEPRECATED: Use ExtractedRelation for new code. Kept for backward compatibility.
    """

    source_id: str
    relation_type: str
    target_id: str
    properties: dict[str, Any] = {}

    model_config = ConfigDict(extra="allow")


class LLMExtractionResult(BaseModel):
    """Schema for the complete LLM extraction result.

    DEPRECATED: Use OpenSPGExtractionResult for new code. Kept for backward compatibility.
    """

    nodes: list[LLMNode]
    edges: list[LLMEdge]

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Conversion Functions (Backward Compatibility)
# ---------------------------------------------------------------------------


def entity_to_spg_node(entity: ExtractedEntity) -> SPGNode:
    """Convert an ExtractedEntity to a legacy SPGNode.

    The SPGNode stores all entity data in the properties dict, including
    the standardized ID, name, and label as metadata.

    Args:
        entity: The ExtractedEntity to convert.

    Returns:
        A new SPGNode instance with properties merged into a dictionary.
    """
    prop_dict = entity.get_property_dict()
    spg_properties = {
        "id": entity.id,
        "name": entity.name,
        **prop_dict,
    }
    return SPGNode(label=entity.label, properties=spg_properties)


def relation_to_spg_edge(relation: ExtractedRelation) -> SPGEdge:
    """Convert an ExtractedRelation to a legacy SPGEdge.

    Args:
        relation: The ExtractedRelation to convert.

    Returns:
        A new SPGEdge instance with properties merged into a dictionary.
    """
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
    """Convert an OpenSPGExtractionResult to a legacy ExtractionResult.

    This allows existing code that expects SPGNode/SPGEdge to work with
    the new OpenSPG models.

    Args:
        open_spg_result: The OpenSPGExtractionResult to convert.

    Returns:
        A new ExtractionResult containing SPGNode and SPGEdge objects.
    """
    nodes = [entity_to_spg_node(entity) for entity in open_spg_result.entities]
    edges = [
        relation_to_spg_edge(relation) for relation in open_spg_result.relations
    ]
    return ExtractionResult(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# LLM Prompt Helper
# ---------------------------------------------------------------------------


def get_open_spg_llm_prompt(
    text: str, author_id: int | None = None, metadata: dict | None = None
) -> str:
    """Generate the OpenSPG extraction prompt for LLM.

    This prompt instructs the LLM to output structured OpenSPG data with
    typed properties for dynamic attribute accumulation.

    Args:
        text: The input text to extract from.
        author_id: Optional Telegram user ID to assign to an Author entity.
        metadata: Optional pre-collected metadata dictionary (e.g., language, geo, location).
                  If provided, the LLM will be instructed not to re-extract these fields.

    Returns:
        Formatted prompt string for the LLM.
    """
    author_instruction = ""
    if author_id is not None:
        author_instruction = f"""5. If the text EXPLICITLY mentions or references the author by name (Telegram user ID: {author_id}), OR if the author performs specific actions described in the text, create an Actor node with id "actor_{author_id}" and name based on the author's displayed name (if available) or "Author {author_id}". Include a property {{'key': 'telegram_id', 'value': {author_id}, 'type': 'numeric'}}. If the author is only implied or not directly mentioned, DO NOT create an author node.
6. """

    # Build metadata instruction section if metadata is provided
    metadata_instruction = ""
    metadata_json_str = ""
    if metadata is not None and len(metadata) > 0:
        metadata_json_str = json.dumps(metadata, ensure_ascii=False, indent=2)
        
        # Check which fields are present in metadata to customize instructions
        excluded_fields = []
        if "language" in metadata:
            excluded_fields.append("'language' (2-letter code)")
        if "location" in metadata:
            excluded_fields.append("'location' (human-readable location)")
        # Handle both 'geo' (list) and separate 'geo_lat'/'geo_long' keys
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
- Use ONLY the four core universal labels: Actor (persons, organizations, any named individual/group), Entity (abstract concepts, products, technologies, objects), Event (occurrences, meetings, incidents), Place (geographic locations, cities, coordinates).

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
   - Actor: People, organizations, groups, or any named entity that can act or be referenced (e.g., "Pavel Durov", "Telegram", "UN Security Council")
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
   - Actor: [{{'key': 'telegram_id', 'value': 123456, 'type': 'numeric'}}, {{'key': 'nationality', 'value': 'Russian', 'type': 'text'}}]
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
        {{"key": "telegram_id", "value": 987654321, "type": "numeric"}},
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
