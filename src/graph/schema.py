"""Pydantic models for OpenSPG (Semantic Parsing Graph) schema elements.

OpenSPG supports dynamic knowledge extraction with typed property accumulation
for flexible entity and relationship attributes.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Property(BaseModel):
    """Represents a dynamic property with type information.

    OpenSPG allows accumulating various types of properties on entities and relations:
    - string: Textual data
    - number: Numeric data (int, float)
    - geo: Geospatial coordinates
    - language: Language codes
    - etc.
    """

    key: str = Field(..., description="Property name/key")
    value: Any = Field(..., description="Property value (any JSON-serializable type)")
    type: str = Field(
        ..., description="Property type category: 'string', 'number', 'geo', 'language', etc."
    )

    model_config = ConfigDict(extra="forbid")


class ExtractedEntity(BaseModel):
    """Represents an extracted entity node in OpenSPG format.

    Entities are the nodes in the knowledge graph. They have a unique standardized
    ID, a type label, a display name, and a list of dynamic properties.
    """

    id: str = Field(..., description="Unique standardized identifier (e.g., 'person_pavel_durov')")
    label: str = Field(..., description="Entity type (e.g., 'Person', 'Organization', 'Location')")
    name: str = Field(..., description="Display name for the entity")
    properties: list[Property] = Field(
        default_factory=list, description="List of dynamic properties (age, coordinates, language, etc.)"
    )

    model_config = ConfigDict(extra="forbid")

    def get_property_dict(self) -> dict[str, Any]:
        """Convert properties list to a dictionary keyed by property key.

        Returns:
            Dictionary mapping property keys to their values.
        """
        return {prop.key: prop.value for prop in self.properties}

    def add_property(self, key: str, value: Any, type: str) -> None:
        """Add a property to the entity.

        Args:
            key: Property name.
            value: Property value.
            type: Property type category.
        """
        self.properties.append(Property(key=key, value=value, type=type))


class ExtractedRelation(BaseModel):
    """Represents an extracted relationship/edge in OpenSPG format.

    Relations connect entities and can have their own dynamic properties
    to provide context (e.g., 'since_year', 'confidence', 'role').
    """

    source_id: str = Field(..., description="ID of the source entity")
    relation_type: str = Field(
        ..., description="Strict relationship type (e.g., 'LOCATED_IN', 'WORKS_AT', 'DISCUSSES')"
    )
    target_id: str = Field(..., description="ID of the target entity")
    properties: list[Property] = Field(
        default_factory=list, description="Optional list of relation properties"
    )

    model_config = ConfigDict(extra="forbid")

    def get_property_dict(self) -> dict[str, Any]:
        """Convert properties list to a dictionary keyed by property key.

        Returns:
            Dictionary mapping property keys to their values.
        """
        return {prop.key: prop.value for prop in self.properties}

    def add_property(self, key: str, value: Any, type: str) -> None:
        """Add a property to the relation.

        Args:
            key: Property name.
            value: Property value.
            type: Property type category.
        """
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


def open_spg_result_to_extraction_result(open_spg_result: OpenSPGExtractionResult) -> ExtractionResult:
    """Convert an OpenSPGExtractionResult to a legacy ExtractionResult.

    This allows existing code that expects SPGNode/SPGEdge to work with
    the new OpenSPG models.

    Args:
        open_spg_result: The OpenSPGExtractionResult to convert.

    Returns:
        A new ExtractionResult containing SPGNode and SPGEdge objects.
    """
    nodes = [entity_to_spg_node(entity) for entity in open_spg_result.entities]
    edges = [relation_to_spg_edge(relation) for relation in open_spg_result.relations]
    return ExtractionResult(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# LLM Prompt Helper
# ---------------------------------------------------------------------------

def get_open_spg_llm_prompt(text: str, author_id: int | None = None) -> str:
    """Generate the OpenSPG extraction prompt for LLM.

    This prompt instructs the LLM to output structured OpenSPG data with
    typed properties for dynamic attribute accumulation.

    Args:
        text: The input text to extract from.
        author_id: Optional Telegram user ID to assign to an Author entity.

    Returns:
        Formatted prompt string for the LLM.
    """
    author_instruction = ""
    if author_id is not None:
        author_instruction = f"""5. If the text mentions or references the author (Telegram user ID: {author_id}), create a Person node with id "person_{author_id}" and name "Author {author_id}". Include a 'telegram_id' property with type 'number' and value {author_id}. This author node serves as the root for the knowledge subgraph.
6. """

    prompt = f"""You are an OpenSPG (Semantic Parsing Graph) engine. Your task is to perform dynamic incremental subgraph extraction from the provided text. Follow these OpenSPG paradigms strictly:

OpenSPG Principles:
- Build a connected subgraph of entities and relationships.
- Separate data types: classify all non-entity attributes as typed properties (text, numeric, geo, language).
- Use canonical IDs and uppercase relation types.

Text to analyze:
{text}

Extraction Instructions:

1. ENTITY TYPES (Primary OpenSPG Categories):
   - Person: Individuals (including the author if mentioned)
   - Location: Geographic places, cities, countries, coordinates
   - Organization: Companies, institutions, agencies
   - Event: Occurrences, meetings, incidents
   - IT_Concept: Technologies, software, programming concepts, digital products

2. FORMAT FOR EACH ENTITY:
   {{
     "id": "canonical_string_id" (e.g., "person_john_doe", "loc_paris", "org_telegram"),
     "label": "EntityType" (exact: Person, Location, Organization, Event, IT_Concept),
     "name": "Human-readable display name",
     "properties": [
       {{"key": "property_name", "value": <any JSON-serializable value>, "type": "string|number|geo|language"}}
     ]
   }}

3. PROPERTY TYPE CLASSIFICATION:
   - "string": General text, descriptions, names, titles
   - "number": Numeric data (integers, floats), counts, years, ages, IDs
   - "geo": Geographic coordinates (lat/lon), addresses, places (use for Location entities' coordinates)
   - "language": Language codes (e.g., "en", "ru", "fr")

4. RELATIONSHIPS:
   - relation_type: MUST be UPPER_SNAKE_CASE (e.g., LOCATED_IN, WORKS_AT, DISCUSSES, MENTIONS, CREATED, PART_OF)
   - Connect entities via source_id and target_id
   - Optional properties array for relationship metadata (e.g., confidence, time_period)

{author_instruction}5. OUTPUT FORMAT:
Return ONLY a valid JSON object with exactly this structure:
{{
  "entities": [ ... ],
  "relations": [ ... ]
}}

Do not include any explanatory text, markdown formatting, or code block delimiters. The response must be pure JSON.

Example:
{{
  "entities": [
    {{
      "id": "person_elon_musk",
      "label": "Person",
      "name": "Elon Musk",
      "properties": [
        {{"key": "nationality", "value": "South African/American", "type": "string"}},
        {{"key": "birth_year", "value": 1971, "type": "number"}}
      ]
    }},
    {{
      "id": "org_tesla",
      "label": "Organization",
      "name": "Tesla Inc.",
      "properties": [
        {{"key": "industry", "value": "Electric Vehicles", "type": "string"}},
        {{"key": "founded", "value": 2003, "type": "number"}}
      ]
    }},
    {{
      "id": "loc_palo_alto",
      "label": "Location",
      "name": "Palo Alto",
      "properties": [
        {{"key": "coordinates", "value": [37.4419, -122.1430], "type": "geo"}}
      ]
    }}
  ],
  "relations": [
    {{
      "source_id": "person_elon_musk",
      "relation_type": "WORKS_AT",
      "target_id": "org_tesla",
      "properties": [
        {{"key": "role", "value": "CEO", "type": "string"}}
      ]
    }},
    {{
      "source_id": "org_tesla",
      "relation_type": "LOCATED_IN",
      "target_id": "loc_palo_alto"
    }}
  ]
}}"""
    return prompt
