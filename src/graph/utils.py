from __future__ import annotations

import json
import re
import unicodedata
import uuid
from collections.abc import Mapping
from typing import Any

from src.graph.ontology import EntityType

_NON_ALNUM_WS_RE = re.compile(r"[^\w\s]")
_MULTI_WS_RE = re.compile(r"\s+")
_HASHTAG_RE = re.compile(r"#\w+")
_HASHTAG_EXTRACT_RE = re.compile(r"#([\w_]+)")
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)

_SYSTEM_GARBAGE: frozenset[str] = frozenset({
    "name", "unknown", "null", "none", "undefined",
    "n a", "nan", "dummy", "test", "no name", "blank",
    "deleted", "removed", "sample", "other", "id",
    "string", "no data", "unknown author", "unknown person",
    "не указано", "неизвестно", "без названия",
    "информация", "не определено", "отсутствует",
})

_ACTOR_GARBAGE: frozenset[str] = frozenset({
    "actor", "author", "channel", "account",
    "актор", "автор", "канал", "аккаунт", "сообщество", "блог", "профиль",
})

_POST_GARBAGE: frozenset[str] = frozenset({
    "post", "posts", "publication", "article", "content", "reels", "shorts", "video",
    "пост", "посты", "публикация", "публикации", "запись", "статья",
    "новость", "новости", "контент", "видео", "рилс",
})

_ENTITY_GARBAGE: frozenset[str] = frozenset({
    "entity", "entities", "object", "term", "person",
    "сущность", "сущности", "объект", "термин", "понятие", "персоналия",
})

_ORGANIZATION_GARBAGE: frozenset[str] = frozenset({
    "organization", "company", "brand", "agency", "media",
    "организация", "организации", "компания", "компании",
    "бренд", "агентство", "сми", "фирма", "предприятие",
})

_PRODUCT_GARBAGE: frozenset[str] = frozenset({
    "product", "products", "service", "software", "app", "gadget", "course",
    "продукт", "продукты", "сервис", "товар", "товары",
    "услуга", "услуги", "приложение", "софт", "гаджет", "курс",
})

_EVENT_GARBAGE: frozenset[str] = frozenset({
    "event", "events", "incident", "conference", "festival", "release", "trend",
    "событие", "события", "мероприятие", "мероприятия",
    "инфоповод", "конференция", "фестиваль", "релиз", "тренд", "ивент",
})

_MICROCONCEPT_GARBAGE: frozenset[str] = frozenset({
    "microconcept", "micro concept", "topic", "theme", "tag",
    "микроконцепт", "тема", "тематика", "топик", "тег",
})

_CONCEPT_GARBAGE: frozenset[str] = frozenset({
    "concept", "concepts", "category", "taxonomy",
    "концепт", "категория", "категории", "рубрика", "таксономия",
})

_HASHTAG_GARBAGE: frozenset[str] = frozenset({
    "hashtag", "hashtags", "tag",
    "хештег", "хэштег", "метки",
})

_GARBAGE_WORDS: frozenset[str] = (
    _SYSTEM_GARBAGE
    | _ACTOR_GARBAGE
    | _POST_GARBAGE
    | _ENTITY_GARBAGE
    | _ORGANIZATION_GARBAGE
    | _PRODUCT_GARBAGE
    | _EVENT_GARBAGE
    | _MICROCONCEPT_GARBAGE
    | _CONCEPT_GARBAGE
    | _HASHTAG_GARBAGE
)

_LABEL_GARBAGE_MAP: dict[EntityType, frozenset[str]] = {
    EntityType.Actor: _ACTOR_GARBAGE,
    EntityType.Post: _POST_GARBAGE,
    EntityType.Entity: _ENTITY_GARBAGE,
    EntityType.Organization: _ORGANIZATION_GARBAGE,
    EntityType.Product: _PRODUCT_GARBAGE,
    EntityType.Event: _EVENT_GARBAGE,
    EntityType.MicroConcept: _MICROCONCEPT_GARBAGE,
    EntityType.Concept: _CONCEPT_GARBAGE,
    EntityType.Hashtag: _HASHTAG_GARBAGE,
}

_ENTITY_TYPE_LABELS: dict[str, str] = {v.value.lower(): v.value for v in EntityType}


def _resolve_label(label: EntityType | str) -> str:
    if isinstance(label, EntityType):
        return label.value
    stripped = str(label).strip()
    return _ENTITY_TYPE_LABELS.get(stripped.lower(), stripped)


def clean_name_lower(name: str) -> str:
    name = unicodedata.normalize("NFKC", name)
    name = _NON_ALNUM_WS_RE.sub(" ", name)
    name = name.replace("_", " ")
    name = _MULTI_WS_RE.sub(" ", name)
    return name.strip().lower()


def clean_identifier(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("#", "").replace("@", "")
    text = text.replace(" ", "_").replace("-", "_")
    text = re.sub(r"[^\w]", "", text)
    return text.lower()


def is_garbage_value(val: str | None, label: EntityType | str | None = None) -> bool:
    if val is None:
        return True
    stripped = val.strip()
    if len(stripped) < 2 or stripped.isdigit():
        return True
    if _URL_RE.search(stripped):
        return True
    cleaned = clean_name_lower(stripped)
    if cleaned in _GARBAGE_WORDS:
        return True
    if label is not None:
        resolved = _resolve_label(label)
        try:
            entity_type = EntityType(resolved)
            label_garbage = _LABEL_GARBAGE_MAP.get(entity_type)
            if label_garbage is not None and cleaned in label_garbage:
                return True
        except ValueError:
            pass
    return False


def generate_uuid5(namespace_str: str, cleaned_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace_str}:{cleaned_key}"))


def build_node_id(
    label: EntityType | str,
    key: str,
    platform: str | None = None,
    account_id: int | None = None,
    content_id: int | None = None,
) -> str:
    label_str = _resolve_label(label)
    match label_str:
        case "Actor":
            if platform is None or account_id is None:
                raise ValueError("platform and account_id are required for Actor")
            return f"actor_{platform}_{account_id}"
        case "Post":
            if platform is None or account_id is None or content_id is None:
                raise ValueError("platform, account_id, and content_id are required for Post")
            return f"event_publication_{platform}_{account_id}_{content_id}"
        case "Concept":
            return f"concept_{key.strip()}"
        case "Hashtag":
            return f"hashtag_{clean_identifier(key)}"
        case "Entity":
            return f"entity_{generate_uuid5('entity', clean_name_lower(key))}"
        case "Organization":
            return f"organization_{generate_uuid5('organization', clean_name_lower(key))}"
        case "Product":
            return f"product_{generate_uuid5('product', clean_name_lower(key))}"
        case "Event":
            return f"event_{generate_uuid5('event', clean_name_lower(key))}"
        case "MicroConcept":
            return f"microconcept_{generate_uuid5('microconcept', clean_name_lower(key))}"
    raise ValueError(f"Unsupported node label for ID generation: {label_str}")


def format_bge_representation(label: str, name: str, subtype: str | None = None) -> str:
    if subtype:
        return f"{label}: {name} ({subtype})"
    return f"{label}: {name}"


def extract_hashtags(text: str) -> list[str]:
    return _HASHTAG_RE.findall(text)


def extract_raw_hashtags(
    text: str | None,
    raw_metadata_hashtags: list[str] | list[dict[str, Any]] | None = None,
    author_bio: str | None = None,
    author_title: str | None = None,
) -> list[str]:
    result: set[str] = set()

    def _collect(source: str | None) -> None:
        if not source:
            return
        for match in _HASHTAG_EXTRACT_RE.finditer(source):
            raw = match.group(1)
            cleaned = clean_identifier(raw)
            if cleaned and not is_garbage_value(cleaned, EntityType.Hashtag):
                result.add(cleaned)

    _collect(text)
    _collect(author_bio)
    _collect(author_title)

    if raw_metadata_hashtags:
        for item in raw_metadata_hashtags:
            if isinstance(item, str):
                cleaned = clean_identifier(item)
                if cleaned and not is_garbage_value(cleaned, EntityType.Hashtag):
                    result.add(cleaned)
            elif isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str):
                        cleaned = clean_identifier(v)
                        if cleaned and not is_garbage_value(cleaned, EntityType.Hashtag):
                            result.add(cleaned)

    return sorted(result)


_PRIMITIVE_TYPES: tuple[type, ...] = (int, float, str, bool)


def sanitize_properties(props: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for k, v in props.items():
        if v is None:
            continue
        if isinstance(v, list | tuple):
            if not v:
                continue
            if all(isinstance(x, _PRIMITIVE_TYPES) for x in v):
                result[k] = list(v)
            else:
                result[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, _PRIMITIVE_TYPES):
            result[k] = v
        elif isinstance(v, Mapping):
            result[k] = json.dumps(dict(v), ensure_ascii=False)
        elif hasattr(v, 'model_dump'):
            result[k] = json.dumps(v.model_dump(), ensure_ascii=False)
        elif hasattr(v, 'dict'):
            result[k] = json.dumps(v.dict(), ensure_ascii=False)
        else:
            result[k] = str(v)
    return result

