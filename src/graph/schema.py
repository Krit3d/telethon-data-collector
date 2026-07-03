from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.graph.utils import is_garbage_value, sanitize_id, sanitize_key


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


_ALLOWED_RELATION_TYPES = frozenset({
    "MENTIONS", "ABOUT", "USES_HASHTAG", "POSTED", "COVERS_TOPIC",
    "COAUTHOR", "TAGGED_AT", "HAS_CONTACT", "BASED_IN", "LOCATED_IN",
    "WORKS_AT", "RELATED_TO",
})


def _generate_deterministic_id(label: str, name: str) -> str:
    sanitized = sanitize_id(name)
    if not sanitized:
        sanitized = "unknown"
    if sanitized[0].isdigit():
        sanitized = f"ent_{sanitized}"
    label_prefix = label.lower().strip()
    if label_prefix not in ("actor", "place", "event", "entity", "topic"):
        label_prefix = "entity"
    return f"{label_prefix}_{sanitized}"


def _sanitize_label_text(v: Any) -> str:
    value = str(v).strip()
    if not value:
        return "Entity"
    value = re.sub(r"[^a-zA-Z0-9_]", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        return "Entity"
    if value[0].isdigit():
        value = f"Label_{value}"
    value_lower = value.lower()
    if value_lower in (
        "actor", "person", "brand", "organization", "company",
        "author", "ctor", "user", "creator", "shoooter", "shooter",
    ):
        return "Actor"
    if value_lower in (
        "place", "location", "city", "country", "geo", "region", "address",
    ):
        return "Place"
    if value_lower in (
        "event", "show", "publication", "incident", "wedding", "post",
    ):
        return "Event"
    if value_lower in ("ntity", "name", "entity_name"):
        return "Entity"
    return "Entity"


def _sanitize_id_text(v: Any) -> str:
    value = sanitize_id(str(v))
    if not value:
        return "entity_unknown"
    if value[0].isdigit():
        value = f"ent_{value}"
    return value




class PropertyType(str, Enum):
    TEXT = "text"
    NUMERIC = "numeric"
    GEO = "geo"
    LANGUAGE = "language"
    LOCATION = "location"
    CATEGORY = "category"


class Property(BaseModel):
    key: str
    value: Any | None = None
    type: PropertyType

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
                if is_garbage_value(value):
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
                self.value = str(value)
                self.type = PropertyType.TEXT
            return self

        if prop_type == PropertyType.GEO:
            if not (
                isinstance(value, list)
                and len(value) == 2
                and all(isinstance(c, (int, float)) for c in value)
            ):
                self.value = None
            return self

        if prop_type == PropertyType.TEXT:
            if not isinstance(value, str):
                self.value = str(value)
            if isinstance(self.value, str) and is_garbage_value(self.value):
                self.value = None
            return self

        if prop_type == PropertyType.LOCATION:
            if not isinstance(value, str):
                self.value = str(value)
            return self

        if prop_type == PropertyType.CATEGORY:
            if not isinstance(value, str):
                self.value = str(value)
            return self

        return self


class ExtractedEntity(BaseModel):
    id: str
    label: str
    name: str
    properties: list[Property] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @field_validator("id", mode="before")
    @classmethod
    def sanitize_id(cls, v: Any) -> str:
        return _sanitize_id_text(v)

    @field_validator("label", mode="before")
    @classmethod
    def sanitize_label(cls, v: Any) -> str:
        return _sanitize_label_text(v)

    @model_validator(mode="before")
    @classmethod
    def sanitize_entity_name(cls, data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(data, dict):
            if "name" in data and isinstance(data["name"], str):
                name = data["name"]
                name = name.lstrip("#").strip()
                name = decode_unicode_escapes(name)
                name = re.sub(r"\[.*?\]", "", name)
                name = re.sub(r"\s+", " ", name).strip()
                if is_garbage_value(name):
                    name = "Unnamed Entity"
                data["name"] = name
            if "id" in data and isinstance(data["id"], str):
                if is_garbage_value(data["id"]):
                    label_raw = str(data.get("label", "Entity"))
                    name_raw = data.get("name", "")
                    if is_garbage_value(name_raw) or name_raw == "Unnamed Entity":
                        name_raw = "unknown"
                    data["id"] = _generate_deterministic_id(label_raw, name_raw)
        return data

    @model_validator(mode="after")
    def align_id_prefix_with_label(self) -> ExtractedEntity:
        if self.label == "Actor" and not self.id.startswith("actor_"):
            self.id = f"actor_{self.id}"
        elif self.label == "Place" and not self.id.startswith("place_"):
            self.id = f"place_{self.id}"
        elif self.label == "Event" and not self.id.startswith("event_"):
            self.id = f"event_{self.id}"
        elif self.label == "Entity" and not self.id.startswith("topic_") and not self.id.startswith("entity_"):
            self.id = f"topic_{self.id}"
        return self

    @model_validator(mode="after")
    def filter_properties(self) -> ExtractedEntity:
        self.properties = [
            p for p in self.properties
            if p.value is not None
            and not (isinstance(p.value, str) and is_garbage_value(p.value))
        ]
        return self

    def get_property_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for prop in self.properties:
            if prop.value is None:
                continue
            if isinstance(prop.value, str) and is_garbage_value(prop.value):
                continue
            result[prop.key] = prop.value
        return result

    def add_property(self, key: str, value: Any, type: str | PropertyType) -> None:
        if isinstance(type, PropertyType):
            resolved_type = type
        else:
            try:
                resolved_type = PropertyType(type)
            except ValueError:
                resolved_type = PropertyType.TEXT
        self.properties.append(Property(key=key, value=value, type=resolved_type))


class ExtractedRelation(BaseModel):
    source_id: str
    relation_type: str
    target_id: str
    properties: list[Property] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @field_validator("source_id", "target_id", mode="before")
    @classmethod
    def sanitize_relation_id(cls, v: Any) -> str:
        return _sanitize_id_text(v)

    @field_validator("relation_type", mode="before")
    @classmethod
    def sanitize_relation_type(cls, v: Any) -> str:
        value = str(v).strip()
        value = re.sub(r"[^a-zA-Z0-9_]", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")
        if not value:
            return "RELATED_TO"
        if value[0].isdigit():
            value = f"REL_{value}"
        value = value.upper()
        if value not in _ALLOWED_RELATION_TYPES:
            return "RELATED_TO"
        return value

    @model_validator(mode="after")
    def filter_properties(self) -> ExtractedRelation:
        self.properties = [
            p for p in self.properties
            if p.value is not None
            and not (isinstance(p.value, str) and is_garbage_value(p.value))
        ]
        return self

    def get_property_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for prop in self.properties:
            if prop.value is None:
                continue
            if isinstance(prop.value, str) and is_garbage_value(prop.value):
                continue
            result[prop.key] = prop.value
        return result

    def add_property(self, key: str, value: Any, type: str | PropertyType) -> None:
        if isinstance(type, PropertyType):
            resolved_type = type
        else:
            try:
                resolved_type = PropertyType(type)
            except ValueError:
                resolved_type = PropertyType.TEXT
        self.properties.append(Property(key=key, value=value, type=resolved_type))


class OpenSPGExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def filter_bad_entities_and_relations(self) -> OpenSPGExtractionResult:
        valid_ids: set[str] = set()
        filtered_entities: list[ExtractedEntity] = []
        for entity in self.entities:
            if is_garbage_value(entity.name) or entity.name == "Unnamed Entity":
                continue
            if not entity.id:
                continue
            filtered_entities.append(entity)
            valid_ids.add(entity.id)
        self.entities = filtered_entities

        filtered_relations: list[ExtractedRelation] = []
        for relation in self.relations:
            if relation.source_id == relation.target_id:
                continue
            filtered_relations.append(relation)
        self.relations = filtered_relations
        return self


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

    @field_validator("id", mode="before")
    @classmethod
    def sanitize_llm_node_id(cls, v: Any) -> str:
        return _sanitize_id_text(v)

    @field_validator("label", mode="before")
    @classmethod
    def sanitize_llm_node_label(cls, v: Any) -> str:
        return _sanitize_label_text(v)

    @model_validator(mode="before")
    @classmethod
    def sanitize_llm_node_name(cls, data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(data, dict):
            if "name" in data and isinstance(data["name"], str):
                name = data["name"]
                name = name.lstrip("#").strip()
                name = decode_unicode_escapes(name)
                name = re.sub(r"\[.*?\]", "", name)
                name = re.sub(r"\s+", " ", name).strip()
                if is_garbage_value(name):
                    name = "Unnamed Entity"
                data["name"] = name
            if "id" in data and isinstance(data["id"], str):
                if is_garbage_value(data["id"]):
                    label_raw = str(data.get("label", "Entity"))
                    name_raw = data.get("name", "")
                    if is_garbage_value(name_raw) or name_raw == "Unnamed Entity":
                        name_raw = "unknown"
                    data["id"] = _generate_deterministic_id(label_raw, name_raw)
        return data


class LLMEdge(BaseModel):
    source_id: str
    relation_type: str
    target_id: str
    properties: dict[str, Any] = {}

    model_config = ConfigDict(extra="allow")

    @field_validator("source_id", "target_id", mode="before")
    @classmethod
    def sanitize_llm_edge_id(cls, v: Any) -> str:
        return _sanitize_id_text(v)

    @field_validator("relation_type", mode="before")
    @classmethod
    def sanitize_llm_edge_relation_type(cls, v: Any) -> str:
        value = str(v).strip()
        value = re.sub(r"[^a-zA-Z0-9_]", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")
        if not value:
            return "RELATED_TO"
        if value[0].isdigit():
            value = f"REL_{value}"
        value = value.upper()
        if value not in _ALLOWED_RELATION_TYPES:
            return "RELATED_TO"
        return value


class LLMExtractionResult(BaseModel):
    nodes: list[LLMNode]
    edges: list[LLMEdge]

    model_config = ConfigDict(extra="allow")


def entity_to_spg_node(entity: ExtractedEntity) -> SPGNode:
    prop_dict = entity.get_property_dict()
    spg_properties: dict[str, Any] = {
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
        "4. Strictly and only for 'MENTIONS' relations, include a sentiment property: {\"key\": \"sentiment\", \"value\": \"positive|negative|neutral\", \"type\": \"text\"}. Do NOT generate any properties (including sentiment) for other relation types.",
        "5. Strictly ignore minor credit lists (makeup, photographers, assistants, production crew). These waste tokens and cause JSON truncation.",
        "",
        "The first character of your response must be the opening brace '{' of a raw JSON object.",
        "Respond with a single raw JSON object containing exactly three keys: \"thinking\", \"entities\", and \"relations\".",
        "Write your step-by-step reasoning inside the \"thinking\" string property first, before outputting the arrays.",
        "No conversational text, headers, or markdown wrappers are allowed.",
        "",
        "Example JSON structure (fill in your own values):",
        "{",
        '  "thinking": "Brief step-by-step reasoning here",',
        '  "entities": [',
        "    {",
        '      "id": "actor_telegram_12345",',
        '      "label": "Actor",',
        '      "name": "Author Name",',
        '      "properties": [',
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
        "",
        'Forbidden values: empty strings, "unknown", "null", or missing values in id, name, label, or key fields.',
    ]

    return "\n".join(lines)
