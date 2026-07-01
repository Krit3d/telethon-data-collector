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
        def _replace_unicode(m: re.Match) -> str:
            cp = int(m.group(1), 16)
            if 0xD800 <= cp <= 0xDFFF:
                return ""
            return chr(cp)

        decoded = re.sub(
            r"\\u([0-9a-fA-F]{4})",
            _replace_unicode,
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

    model_config = ConfigDict(extra="ignore")

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
                    data["type"] = PropertyType.TEXT

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

    model_config = ConfigDict(extra="ignore")

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
        if len(value) < 5:
            value = f"{value}_{uuid.uuid4().hex[:6]}"
        return value

    @field_validator("label", mode="before")
    @classmethod
    def sanitize_label(cls, v: Any) -> str:
        value = str(v).strip()
        if not value:
            return "Entity"
        value = re.sub(r"[^a-zA-Z0-9_]", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")
        if not value:
            return "Entity"
        if value[0].isdigit():
            value = f"Label_{value}"
        else:
            value = value[0].upper() + value[1:]
        if value in ("Actor", "Person", "Brand", "Organization", "Company", "Author"):
            return "Actor"
        if value in ("Place", "Location", "City", "Country", "Geo"):
            return "Place"
        if value in ("Event", "Show", "Publication", "Incident"):
            return "Event"
        return "Entity"

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

    model_config = ConfigDict(extra="ignore")

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

    model_config = ConfigDict(extra="ignore")


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
    metadata_str = json.dumps(metadata, ensure_ascii=False, indent=2) if metadata else "NONE"
    actor_id = f"actor_{platform.lower()}_{author_id}" if platform and author_id is not None else "NONE"
    platform_val = platform or "unknown"

    lines: list[str] = [
        "Input:",
        f"TEXT: {text}",
        f"PUB_NODE_ID: {pub_node_id}",
        f"AUTHOR: {actor_id}",
        f"META: {metadata_str}",
        "",
        "Instructions:",
        f"1. Create Actor node for {actor_id} (properties: platform={platform_val}, platform_id={author_id if author_id is not None else 'N/A'}).",
        f"2. Extract at most 3-4 key semantic concept entities per post (excluding the author Actor node) — Topics, Events, Places, Organizations, or Brands relevant to the content. Map each to the appropriate label. Do NOT extract individual hashtags as separate topic nodes.",
        "3. If the post contains a large block of hashtags, consolidate them into 2-3 high-level concept entities (e.g., group #portraitmood, #portrait_shots, #make_portraits into a single Entity node named 'Portrait Photography').",
        "4. For every relation, include a property: {\"key\": \"sentiment\", \"value\": \"positive|negative|neutral\", \"type\": \"text\"}.",
        "5. Strictly ignore minor credit lists (makeup, photographers, assistants, production crew). These waste tokens and cause JSON truncation.",
        "",
        "The very first character of your response must be '<think>'. Do not output anything before '<think>' — no introductions, no formatting, no numbered lists, no whitespace.",
        "Write no more than 3 concise sentences of analysis inside <think>...</think> tags, then output the JSON.",
        "",
        "Output format:",
        "<think>",
        "[reasoning about entities and relations]",
        "</think>",
        "```json",
        "{",
        '  "entities": [',
        "    {",
        '      "id": "actor_telegram_12345",',
        '      "label": "Actor",',
        '      "name": "Author Name",',
        "      \"properties\": [",
        '        {"key": "platform", "value": "telegram", "type": "text"},',
        '        {"key": "platform_id", "value": "12345", "type": "text"}',
        "      ]",
        "    },",
        "    {",
        '      "id": "topic_ai",',
        '      "label": "Entity",',
        '      "name": "Artificial Intelligence",',
        '      "properties": [{"key": "type", "value": "topic", "type": "text"}]',
        "    }",
        "  ],",
        '  "relations": [',
        "    {",
        '      "source_id": "actor_telegram_12345",',
        '      "relation_type": "MENTIONS",',
        '      "target_id": "' + pub_node_id + '",',
        '      "properties": [{"key": "sentiment", "value": "positive", "type": "text"}]',
        "    },",
        "    {",
        '      "source_id": "topic_ai",',
        '      "relation_type": "ABOUT",',
        '      "target_id": "' + pub_node_id + '",',
        '      "properties": []',
        "    }",
        "  ]",
        "}",
        "```",
        "",
        'Forbidden values: empty strings, "unknown", "null", or missing values in id, name, label, or key fields.',
    ]

    return "\n".join(lines)
