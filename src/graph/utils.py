from __future__ import annotations

import re
import unicodedata
import uuid
from typing import Any

from src.graph.ontology import EntityType


_NON_ALNUM_WS_RE = re.compile(r"[^\w\s]")
_MULTI_WS_RE = re.compile(r"\s+")
_HASHTAG_RE = re.compile(r"#\w+")
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

_PLACE_GARBAGE: frozenset[str] = frozenset({
    "place", "places", "location", "city", "country", "region", "venue",
    "место", "места", "локация", "локации", "город", "страна", "регион",
    "геометка", "местоположение",
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

_TONE_GARBAGE: frozenset[str] = frozenset({
    "tone", "style", "sentiment",
    "тон", "тональность", "стиль", "подача",
})

_LANGUAGE_GARBAGE: frozenset[str] = frozenset({
    "language", "lang",
    "язык", "язык контента",
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
    | _PLACE_GARBAGE
    | _ORGANIZATION_GARBAGE
    | _PRODUCT_GARBAGE
    | _EVENT_GARBAGE
    | _MICROCONCEPT_GARBAGE
    | _CONCEPT_GARBAGE
    | _TONE_GARBAGE
    | _LANGUAGE_GARBAGE
    | _HASHTAG_GARBAGE
)

_LABEL_GARBAGE_MAP: dict[EntityType, frozenset[str]] = {
    EntityType.Actor: _ACTOR_GARBAGE,
    EntityType.Post: _POST_GARBAGE,
    EntityType.Entity: _ENTITY_GARBAGE,
    EntityType.Place: _PLACE_GARBAGE,
    EntityType.Organization: _ORGANIZATION_GARBAGE,
    EntityType.Product: _PRODUCT_GARBAGE,
    EntityType.Event: _EVENT_GARBAGE,
    EntityType.MicroConcept: _MICROCONCEPT_GARBAGE,
    EntityType.Concept: _CONCEPT_GARBAGE,
    EntityType.Tone: _TONE_GARBAGE,
    EntityType.Language: _LANGUAGE_GARBAGE,
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
    country_code: str | None = None,
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
        case "Place":
            if country_code is not None:
                place_key = f"{clean_name_lower(key)}_{country_code.strip().lower()}"
            else:
                place_key = clean_name_lower(key)
            return f"place_{generate_uuid5('place', place_key)}"
        case "Concept":
            return f"concept_{key.strip()}"
        case "Tone":
            return f"tone_{clean_identifier(key)}"
        case "Language":
            return f"lang_{key.strip().lower()}"
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


def format_bge_representation(label: str, name: str, properties: dict[str, Any] | None = None) -> str:
    parts = [f"{label}: {name}"]
    if properties:
        for k in sorted(properties):
            v = properties[k]
            if v is not None and v != "":
                parts.append(f"{k}: {v}")
    return ", ".join(parts)


def extract_hashtags(text: str) -> list[str]:
    return list({clean_identifier(h) for h in _HASHTAG_RE.findall(text)})