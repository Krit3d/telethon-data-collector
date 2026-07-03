from typing import Any

from src.graph.schema import ExtractedEntity, ExtractedRelation
from src.graph.utils import sanitize_id


def clean_telegram_link(val: str) -> str:
    val = val.strip()
    if val.startswith("@"):
        return f"https://t.me/{val[1:]}"
    if val.startswith("+") and not val.startswith("https://"):
        after_plus = val[1:]
        if after_plus.isdigit():
            return val
        if any(c.isalpha() for c in after_plus):
            return f"https://t.me/{val}"
    return val


def clean_hashtag(tag: str) -> str:
    cleaned = tag.strip().lstrip("#").strip()
    return sanitize_id(cleaned)


def normalize_language(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip().lower()
    if len(stripped) == 2 and stripped.isalpha():
        return stripped
    return None


def find_entity(
    entities: list[ExtractedEntity],
    entity_id: str,
) -> ExtractedEntity | None:
    for entity in entities:
        if entity.id == entity_id:
            return entity
    return None


def find_or_create_entity(
    entities: list[ExtractedEntity],
    entity_id: str,
    label: str,
    name: str,
) -> ExtractedEntity:
    for entity in entities:
        if entity.id == entity_id:
            return entity
    created = ExtractedEntity(
        id=entity_id,
        label=label,
        name=name,
        properties=[],
    )
    entities.append(created)
    return created


def find_or_create_relation(
    relations: list[ExtractedRelation],
    source_id: str,
    relation_type: str,
    target_id: str,
) -> ExtractedRelation:
    for rel in relations:
        if (
            rel.source_id == source_id
            and rel.relation_type == relation_type
            and rel.target_id == target_id
        ):
            return rel
    created = ExtractedRelation(
        source_id=source_id,
        relation_type=relation_type,
        target_id=target_id,
    )
    relations.append(created)
    return created
