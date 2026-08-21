from __future__ import annotations

import json
import re
import unicodedata
import uuid
from collections.abc import Mapping
from typing import Any

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
    "event", "events", "incident", "conference", "festival",
    "событие", "события", "мероприятие", "мероприятия",
    "инфоповод", "конференция", "фестиваль", "ивент",
    "соревнование", "турнир", "конкурс", "хакатон", "премия",
    "competition", "contest", "tournament", "hackathon", "award",
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

_LABEL_GARBAGE_MAP: dict[str, frozenset[str]] = {
    "Actor": _ACTOR_GARBAGE,
    "Post": _POST_GARBAGE,
    "Entity": _ENTITY_GARBAGE,
    "Organization": _ORGANIZATION_GARBAGE,
    "Product": _PRODUCT_GARBAGE,
    "Event": _EVENT_GARBAGE,
    "MicroConcept": _MICROCONCEPT_GARBAGE,
    "Concept": _CONCEPT_GARBAGE,
    "Hashtag": _HASHTAG_GARBAGE,
}


def _resolve_label(label: str | Any) -> str:
    return getattr(label, "value", str(label)).strip()


def clean_name_lower(name: str) -> str:
    name = unicodedata.normalize("NFKC", name)
    name = _NON_ALNUM_WS_RE.sub(" ", name)
    name = name.replace("_", " ")
    name = _MULTI_WS_RE.sub(" ", name)
    return name.strip().lower()


def is_author_entity(name: str, author_title: str, author_handle: str | None = None) -> bool:
    name_clean = clean_name_lower(name)
    title_clean = clean_name_lower(author_title) if author_title else ''
    handle_clean = clean_name_lower(author_handle) if author_handle else ''
    if not name_clean:
        return False
    if name_clean == title_clean or (handle_clean and name_clean == handle_clean):
        return True
    if handle_clean and len(name_clean) >= 4 and name_clean == handle_clean.replace('_', ''):
        return True
    if title_clean:
        name_tokens = name_clean.split()
        title_tokens = set(title_clean.split())
        if len(name_tokens) >= 2 and set(name_tokens) <= title_tokens:
            return True
    return False


_LOWER_BRANDS: dict[str, str] = {
    "iphone": "iPhone",
    "ipad": "iPad",
    "ios": "iOS",
    "macos": "macOS",
    "ebay": "eBay",
    "imac": "iMac",
    "chatgpt": "ChatGPT",
    "openai": "OpenAI",
    "youtube": "YouTube",
    "github": "GitHub",
    "playstation": "PlayStation",
    "postgresql": "PostgreSQL",
    "graphql": "GraphQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "tiktok": "TikTok",
    "linkedin": "LinkedIn",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
}

_PROPER_NOUN_LABELS: frozenset[str] = frozenset({"Actor", "Organization", "Event"})


def format_display_name(name: str, label: str | Any | None = None, is_person: bool = False) -> str:
    name = name.strip().strip("\"'«»„“”‘’`")
    if not name:
        return name

    has_upper = any(char.isupper() for char in name)
    has_lower = any(char.islower() for char in name)

    if has_upper and has_lower:
        return name

    if has_upper and not has_lower:
        if len(name) <= 4:
            return name
        name = name.title()

    words = name.split()

    for i, word in enumerate(words):
        lower_word = word.lower()
        if lower_word in _LOWER_BRANDS:
            words[i] = _LOWER_BRANDS[lower_word]

    if not has_upper:
        first_lower = words[0].lower()
        if first_lower in _LOWER_BRANDS:
            if len(words) > 1:
                words[1:] = map(str.title, words[1:])
            return " ".join(words)
        if is_person or (label is not None and _resolve_label(label) in _PROPER_NOUN_LABELS):
            return " ".join(words).title()
        return name[0].upper() + name[1:]

    return " ".join(words)


def clean_identifier(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("#", "").replace("@", "")
    text = text.replace(" ", "_").replace("-", "_")
    text = re.sub(r"[^\w]", "", text)
    return text.lower()


def is_garbage_value(val: str | None, label: str | Any | None = None) -> bool:
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
        label_garbage = _LABEL_GARBAGE_MAP.get(resolved)
        if label_garbage is not None and cleaned in label_garbage:
            return True
    return False


def generate_uuid5(namespace_str: str, cleaned_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace_str}:{cleaned_key}"))


def build_node_id(
    label: str | Any,
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
            return f"actor_{platform.strip().lower()}_{account_id}"
        case "Post":
            if platform is None or account_id is None or content_id is None:
                raise ValueError("platform, account_id, and content_id are required for Post")
            return f"event_publication_{platform.strip().lower()}_{account_id}_{content_id}"
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
            if cleaned and not is_garbage_value(cleaned, "Hashtag"):
                result.add(cleaned)

    _collect(text)
    _collect(author_bio)
    _collect(author_title)

    if raw_metadata_hashtags:
        for item in raw_metadata_hashtags:
            if isinstance(item, str):
                cleaned = clean_identifier(item)
                if cleaned and not is_garbage_value(cleaned, "Hashtag"):
                    result.add(cleaned)
            elif isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str):
                        cleaned = clean_identifier(v)
                        if cleaned and not is_garbage_value(cleaned, "Hashtag"):
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
