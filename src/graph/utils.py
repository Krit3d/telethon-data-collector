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

_REGULATORY_MARKERS: re.Pattern[str] = re.compile(
    r"\b(?:\d+[- ]?фз|фз[- ]?\d+|закон(?:\s+о\b|\s+об\b|\s+рф|\s+№|\s+\d+)?|статья\s+\d+|ст\.\s*\d+|кодекс|постановление|гост(?:\s+\d+)?|iso(?:\s+\d+)?|gdpr|санпин|снип|регламент|указ(?:\s+президента|\s+№|\s+\d+)?)\b",
    re.IGNORECASE,
)

_EVENT_TOPONYM_MARKERS: re.Pattern[str] = re.compile(
    r"(?:conf|fest|summit|forum|meetup|conference|съезд|форум|фестиваль|саммит|выставка|чемпионат|кубок|турнир|хакатон|премия|марафон|интенсив|конкурс|tournament|hackathon|award|webinar|marathon|contest)",
    re.IGNORECASE,
)

_FALSE_EVENT_TOPONYMS: frozenset[str] = frozenset({
    "россия", "russia", "сша", "usa", "германия", "germany", "франция", "france",
    "великобритания", "united kingdom", "uk", "италия", "italy", "испания", "spain",
    "китай", "china", "япония", "japan", "индия", "india", "бразилия", "brazil",
    "канада", "canada", "австралия", "australia", "южная корея", "south korea",
    "вьетнам", "vietnam", "тайланд", "thailand", "турция", "turkey", "оаэ", "uae",
    "дубай", "dubai", "москва", "moscow", "санкт петербург", "санкт-петербург", "st petersburg",
    "лондон", "london", "нью йорк", "нью-йорк", "new york", "париж", "paris", "берлин", "berlin",
    "пекин", "beijing", "токио", "tokyo", "сингапур", "singapore", "гонконг", "hong kong",
    "шанхай", "shanghai", "абу даби", "абу-даби", "abu dhabi", "доха", "doha", "рим", "rome",
    "милан", "milan", "барселона", "barcelona", "мадрид", "madrid", "амстердам", "amsterdam",
    "шереметьево", "sheremetyevo", "домодедово", "domodedovo", "внуково", "vnukovo",
    "пулково", "pulkovo", "хитроу", "heathrow", "аэропорт", "airport",
    "казахстан", "kazakhstan", "узбекистан", "uzbekistan", "беларусь", "belarus",
    "украина", "ukraine", "польша", "poland", "чехия", "czech", "австрия", "austria",
    "швейцария", "switzerland", "швеция", "sweden", "норвегия", "norway", "дания", "denmark",
    "финляндия", "finland", "нидерланды", "netherlands", "бельгия", "belgium",
    "португалия", "portugal", "греция", "greece", "египет", "egypt", "израиль", "israel",
    "саудовская аравия", "saudi arabia", "катар", "qatar", "кувейт", "kuwait",
    "оман", "oman", "бахрейн", "bahrain", "индонезия", "indonesia", "малайзия", "malaysia",
    "филиппины", "philippines", "мексика", "mexico", "аргентина", "argentina",
    "чили", "chile", "колумбия", "colombia", "юар", "south africa", "нигерия", "nigeria",
    "кения", "kenya", "марокко", "morocco", "алжир", "algeria", "тунис", "tunisia",
    "сербия", "serbia", "хорватия", "croatia", "болгария", "bulgaria", "румыния", "romania",
    "венгрия", "hungary", "словакия", "slovakia", "словения", "slovenia", "литва", "lithuania",
    "латвия", "latvia", "эстония", "estonia", "азербайджан", "azerbaijan", "армения", "armenia",
    "грузия", "georgia", "монголия", "mongolia", "туркменистан", "turkmenistan",
    "кыргызстан", "kyrgyzstan", "таджикистан", "tajikistan",
})


def is_regulatory_entity(name: str) -> bool:
    return bool(_REGULATORY_MARKERS.search(name))


_GENERIC_EVENT_WORDS: frozenset[str] = frozenset({
    "конференция", "митап", "форум", "фестиваль", "саммит", "выставка",
    "вебинар", "съезд", "лекция", "conference", "meetup", "forum", "summit",
    "festival", "webinar", "trip", "travel", "поездка", "путешествие",
    "мастер класс", "мастер-класс", "воркшоп", "workshop",
    "интенсив", "тренинг", "training", "семинар", "seminar",
    "сходка", "музсходка",
})


def is_false_event_toponym(name: str) -> bool:
    cleaned = clean_name_lower(name)
    words = cleaned.split()
    if cleaned in _GENERIC_EVENT_WORDS:
        return True
    if cleaned in _FALSE_EVENT_TOPONYMS:
        return True
    if cleaned.startswith("г ") or cleaned.startswith("город ") or cleaned.startswith("гор "):
        space = cleaned.index(" ", 1) + 1
        rest = cleaned[space:]
        if rest in _FALSE_EVENT_TOPONYMS:
            return True
    if cleaned.startswith("в ") and len(cleaned) > 2:
        rest = cleaned[2:]
        if rest in _FALSE_EVENT_TOPONYMS:
            return True
        if rest.endswith("е"):
            base = rest[:-1]
            if base in _FALSE_EVENT_TOPONYMS:
                return True
            if base + "а" in _FALSE_EVENT_TOPONYMS:
                return True
    if _EVENT_TOPONYM_MARKERS.search(cleaned):
        return False
    if len(words) > 1 and cleaned not in _FALSE_EVENT_TOPONYMS:
        return False
    return True


_TEMPORAL_EVENT_MARKERS: re.Pattern[str] = re.compile(
    r"^(?:"
    r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
    r"\d{4}[./-]\d{2}[./-]\d{2}|"
    r"\d{4}\s*[-–]\s*\d{4}|"
    r"\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)(?:\s+\d{4})?|"
    r"(?:январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь)(?:\s+\d{4})?|"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:,?\s+\d{4})?|"
    r"\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:,?\s+\d{4})?|"
    r"\d{4}\s+год(?:а|у)?|"
    r"в\s+\d{4}\s+году|"
    r"(?:весн[ау]|лет[оа]|осень[ю]?|зим[ау])(?:\s+\d{4})?|"
    r"(?:spring|summer|autumn|fall|winter)(?:\s+\d{4})?|"
    r"понедельник|вторник|среда|четверг|пятница|суббота|воскресенье|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"вчера|сегодня|завтра|позавчера|послезавтра|"
    r"yesterday|today|tomorrow|"
    r"(?:19|20)\d{2}"
    r")$",
    re.IGNORECASE,
)


def is_false_event_temporal(name: str) -> bool:
    return bool(_TEMPORAL_EVENT_MARKERS.match(name.strip()))


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
