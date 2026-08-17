from __future__ import annotations

import asyncio
import httpx
import json
import re
import logging
import time
from typing import Any

from json_repair import repair_json
from evolution_openai import EvolutionAsyncOpenAI
from openai import AsyncOpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError
from pydantic import ValidationError

from src.config.config import Settings
from src.graph.builder.reader import PostBatchContext
from src.graph.ontology import (
    EntityType,
    ExtractedEntity,
    ExtractedPsychographics,
    ExtractedRelation,
    HashtagItem,
    OpenSPGExtractionResult,
    RelationType,
    RoleType,
    SentimentType,
    ToneType,
)
from src.graph.utils import build_node_id, clean_name_lower, extract_raw_hashtags, is_garbage_value

logger = logging.getLogger(__name__)

_TRUNCATED_TAG_RE = re.compile(r"(?:\.{2,}|…|\.$|[\-\—]$)")
_INVALID_TAG_CHARS_RE = re.compile(r"[^A-Za-z0-9\s'\-&/]")


def _ensure_dict(val: Any) -> dict[str, Any]:
    if isinstance(val, dict):
        return val
    if isinstance(val, list) and val and isinstance(val[0], dict):
        return val[0]
    return {}


def _sanitize_properties(props: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, val in props.items():
        if val is None:
            continue
        if isinstance(val, dict):
            sanitized[key] = json.dumps(val, ensure_ascii=False)
        elif isinstance(val, list):
            cleaned: list[Any] = []
            for item in val:
                if isinstance(item, (str, int, float, bool)):
                    cleaned.append(item)
                else:
                    cleaned.append(json.dumps(item, ensure_ascii=False))
            sanitized[key] = cleaned
        elif isinstance(val, (str, int, float, bool)):
            sanitized[key] = val
        else:
            sanitized[key] = str(val)
    return sanitized


def _ensure_list(val: Any) -> list[Any]:
    if isinstance(val, list):
        return val
    if isinstance(val, dict):
        return list(val.values())
    if val is not None:
        return [val]
    return []


def _build_system_prompt(author_title: str) -> str:
    return (
        "Ты — строгий и детерминированный графовый экстрактор знаний (архитектура OpenSPG/KAG). "
        "Твоя задача — извлечь структурированное представление знаний из текста поста социальной сети. "
        "Отвечай ТОЛЬКО одним валидным JSON-объектом. Любой дополнительный текст, пояснения, "
        "markdown-разметка и код до или после JSON строго запрещены. Если данных недостаточно — "
        "не выдумывай, извлекай только то, что явно присутствует в тексте.\n"
        "JSON обязан содержать РОВНО следующие ключи (все обязательны):\n"
        "  - thinking (str): краткое пошаговое рассуждение на русском языке о том, что извлечено.\n"
        "  - entities (list[dict]): извлечённые сущности.\n"
        "  - relations (list[dict]): извлечённые связи.\n"
        "  - microconcepts (list[str]): обобщающие тематические категории.\n"
        "  - psychographics (dict): психографические характеристики.\n"
        "  - is_spam_or_gambling (bool): true, если пост содержит спам, скам, гемблинг, ставки или "
        "предложения лёгкого заработка, иначе false.\n"
        "  - hashtags (list[dict]): нормализованные хэштеги.\n"
        "\n"
        "=== СЕКЦИЯ entities ===\n"
        "Каждая сущность — объект строгой формы {{name: str, label: str, entity_type: str, properties: dict}}.\n"
        "Допустимые label и строго допустимые значения enum-поля:\n"
        "  - label=Entity: entity_type строго из [technology, method, person, term, general].\n"
        "  - label=Organization: org_type строго из [company, brand, agency, media, non_profit].\n"
        "  - label=Product: product_type строго из [software, gadget, course, app, physical_good, service].\n"
        "  - label=Event: event_type строго из [conference, release, incident, festival, trend].\n"
        "Используй ТОЛЬКО перечисленные значения enum. Любое другое значение недопустимо.\n"
        f"КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО извлекать в entities:\n"
        f"  - самого автора поста '{author_title}' (в любых падежах, склонениях, формах и транслитах);\n"
        "  - саму публикацию (пост);\n"
        "  - сущности с label=Hashtag (они извлекаются автоматически). Однако если каноническая сущность, бренд, продукт, организация или событие упомянуты в тексте через хэштег (например, #iphone16 или #tesla), ты ОБЯЗАН извлечь её как каноническую сущность (Product, Organization, Entity, Event и т.д.) с нормализованным человекочитаемым именем (например, 'iPhone 16' или 'Tesla').\n"
        "  - любые служебные свойства: role, platform, author, post.\n"
        "\n"
        "=== СЕКЦИЯ relations ===\n"
        "Каждая связь — объект строгой формы {{source_id: str, source_label: str, target_id: str, "
        "target_label: str, relation_type: str, properties: dict}}.\n"
        "Допустимые relation_type: MENTIONS, ABOUT, WORKS_AT, PARTICIPATED_IN, "
        "USES_TECH, PRODUCES, RELATED_TO, COAUTHOR.\n"
        "Строгие сигнатуры связей (source -> relation -> target):\n"
        "  - MENTIONS: (Post) -> MENTIONS -> (Entity | Organization | Product | Event). "
        "Упоминание сущности в посте. Поля в properties: sentiment (str | null: positive, negative, neutral — "
        "указывается ТОЛЬКО если автор явно выражает отношение к сущности в посте) "
        "и confidence (float от 0.0 до 1.0 | null — указывается только при явной оценке).\n"
        "  - ABOUT: (Post) -> ABOUT -> (MicroConcept). Тематическая связь поста с микроконцептом. properties: weight (float от 0.0 до 1.0, по умолчанию 1.0). Калибровка weight: 1.0 — ключевая тема поста (основное содержание, >70% текста); 0.7 — вторичная тема (подробно обсуждается, 20-50% текста); 0.3 — косвенное или эпизодическое упоминание (<20% текста).\n"
        "  - WORKS_AT: (Actor) -> WORKS_AT -> (Organization). Строгое правило: properties.role принимает "
        "ОДНО значение строго из [founder, executive, employee, ambassador, advisor]. "
        "Любые другие значения запрещены.\n"
        "  - PARTICIPATED_IN: (Actor | Entity) -> PARTICIPATED_IN -> (Event). "
        "ЗАПРЕЩЕНО связывать Organization -> PARTICIPATED_IN -> Person. "
        "Только Actor или Entity -> PARTICIPATED_IN -> Event. "
        "Свойства (properties) ДОЛЖНЫ содержать поле \"role\" со строковым значением из списка: "
        "[\"speaker\", \"organizer\", \"sponsor\", \"visitor\", \"mention\"].\n"
        "  - USES_TECH: (Actor | Post) -> USES_TECH -> (Product | Entity). "
        "Использование технологии или продукта. "
        "Свойства (properties) ДОЛЖНЫ содержать поле \"proficiency\" со строковым значением из списка: "
        "[\"expert\", \"user\", \"reviewer\", \"mention\"].\n"
        "  - PRODUCES: (Organization | Actor) -> PRODUCES -> (Product). Строгое правило: "
        "Поля в properties: relation_subtype (str). Для Actor подтипы строго из: [creator, promoter, affiliate]. "
        "Для Organization подтипы строго из: [vendor, publisher, distributor, sponsor].\n"
        "  - RELATED_TO: (Entity) -> RELATED_TO -> (Entity | Organization | Product). "
        "Обобщённая семантическая связь. properties: relation_name (str), weight (float от 0.0 до 1.0, по умолчанию 1.0). Калибровка weight: 1.0 — прямая жёсткая зависимость или эквивалентность; 0.7 — сильная тематическая связь в одном контексте; 0.3 — слабая или косвенная ассоциированность.\n"
        "  - COAUTHOR: (Actor) -> COAUTHOR -> (Actor). Соавторство. Поля в properties: "
        "platform (str: instagram, telegram), post_id (str: ID публикации).\n"
        "\n"
        "=== СЕКЦИЯ microconcepts ===\n"
        "Извлеки от 1 до 3 обобщающих тематических категорий поста. Категории — атомарные обобщения темы, "
        "а не упоминания конкретных сущностей.\n"
        "Строгие правила:\n"
        "  - Каждая категория СТРОГО на английском языке.\n"
        "  - Формат: Title Case (например, Fashion, Sports Nutrition, Automotive Tuning, Self Improvement).\n"
        "  - Длина: 1-3 слова максимум.\n"
        "  - Только целые осмысленные существительные или устойчивые именные словосочетания.\n"
        "  - КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО: обрезки слов, глагольные формы, предлоги, хештеги, вёрстка "
        "и служебные знаки.\n"
        "\n"
        "=== СЕКЦИЯ psychographics ===\n"
        "Объект со строгой структурой:\n"
        "  - language (str | null): двухбуквенный ISO-код основного языка контента поста (ru, en, kz и т.д.).\n"
        "    Правила определения:\n"
        "      1. Если подпись к посту содержит 3 и более осмысленных слова -> определяй язык по подписи.\n"
        "      2. Если подпись отсутствует или короче 3 слов, но есть транскрипция (от 3 слов) -> определяй язык по транскрипции.\n"
        "      3. Если текста ни в подписи, ни в транскрипции нет (музыка, шумы, менее 3 слов) -> возвращай null.\n"
        "  - sentiment (str | null): positive, negative, neutral (или null, если текст не содержит "
        "выраженного эмоционального отношения автора).\n"
        "  - tone (str): строго из [analytical, expert, provocative, educational, entertainment, casual, "
        "hype_train, sell_courses]. Каждый текст имеет стиль передачи информации. Ты ОБЯЗАН выбрать один "
        "наиболее близкий тон из этого списка.\n"
        "  - secondary_tone (str | null): один второй по значимости тон из того же списка, или null если "
        "выраженного второго тона нет.\n"
        "  - tone_confidence (float от 0.0 до 1.0 | null): степень уверенности модели в определении основного тона (1.0 — высокая, 0.1 — очень сомнительная).\n"
        "  - score_dopamine (float): балл от 0.0 до 1.0.\n"
        "  - score_oxytocin (float): балл от 0.0 до 1.0.\n"
        "  - score_serotonin (float): балл от 0.0 до 1.0.\n"
        "  - score_cortisol (float): балл от 0.0 до 1.0.\n"
        "  - score_adrenaline (float): балл от 0.0 до 1.0.\n"
        "  - score_endorphin (float): балл от 0.0 до 1.0.\n"
        "\n"
        "=== СЕКЦИЯ hashtags ===\n"
        "Извлеки и нормализуй хэштеги поста. Формат: list of objects {{raw: str, normalized: str}}.\n"
        "- raw: сырой хэштег без символа # (например, 'нейросетидлябизнеса').\n"
        "- normalized: нормализованная строка с разделенными пробелами словами в человекочитаемом виде "
        "(например, 'нейросети для бизнеса').\n"
        "- Если хэштегов нет — возвращай пустой список []."
    )


_TONE_MAP: dict[str, ToneType] = {
    "analytical": ToneType.analytical,
    "expert": ToneType.expert,
    "provocative": ToneType.provocative,
    "educational": ToneType.educational,
    "entertainment": ToneType.entertainment,
    "casual": ToneType.casual,
    "hype_train": ToneType.hype_train,
    "sell_courses": ToneType.sell_courses,
}

_ENTITY_TYPE_PROP_KEY: dict[str, str] = {
    "Entity": "entity_type",
    "Organization": "org_type",
    "Product": "product_type",
    "Event": "event_type",
}

_LANGUAGE_NAMES: dict[str, str] = {
    "ru": "Russian",
    "en": "English",
    "kz": "Kazakh",
    "kk": "Kazakh",
    "zh": "Chinese",
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "be": "Belarusian",
    "pl": "Polish",
    "cs": "Czech",
    "sk": "Slovak",
    "bg": "Bulgarian",
    "sr": "Serbian",
    "hr": "Croatian",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "az": "Azerbaijani",
    "uz": "Uzbek",
    "ky": "Kyrgyz",
    "tg": "Tajik",
    "tk": "Turkmen",
    "mn": "Mongolian",
    "fa": "Persian",
    "ur": "Urdu",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "ne": "Nepali",
    "si": "Sinhala",
    "my": "Burmese",
    "km": "Khmer",
    "lo": "Lao",
    "tl": "Tagalog",
    "sw": "Swahili",
    "af": "Afrikaans",
    "am": "Amharic",
    "hy": "Armenian",
    "ka": "Georgian",
    "is": "Icelandic",
    "ga": "Irish",
    "cy": "Welsh",
    "sq": "Albanian",
    "mk": "Macedonian",
    "sl": "Slovenian",
    "bs": "Bosnian",
    "ca": "Catalan",
    "eu": "Basque",
    "gl": "Galician",
    "ps": "Pashto",
    "ku": "Kurdish",
    "ht": "Haitian Creole",
    "jw": "Javanese",
    "su": "Sundanese",
    "yo": "Yoruba",
    "ig": "Igbo",
    "zu": "Zulu",
    "xh": "Xhosa",
    "sn": "Shona",
    "rw": "Kinyarwanda",
    "mg": "Malagasy",
    "so": "Somali",
    "ti": "Tigrinya",
    "tt": "Tatar",
    "ba": "Bashkir",
    "cv": "Chuvash",
    "os": "Ossetian",
    "ce": "Chechen",
}

_LANGUAGE_ALIASES: dict[str, str] = {
    "русский": "ru",
    "russian": "ru",
    "rus": "ru",
    "казахский": "kk",
    "kazakh": "kk",
    "kaz": "kk",
    "kz": "kk",
    "qazaq": "kk",
    "английский": "en",
    "english": "en",
    "eng": "en",
    "узбекский": "uz",
    "uzbek": "uz",
    "украинский": "uk",
    "ukrainian": "uk",
    "белорусский": "be",
    "belarusian": "be",
    "турецкий": "tr",
    "turkish": "tr",
    "немецкий": "de",
    "german": "de",
    "deu": "de",
    "французский": "fr",
    "french": "fr",
    "fra": "fr",
    "испанский": "es",
    "spanish": "es",
    "китайский": "zh",
    "chinese": "zh",
    "zho": "zh",
    "chi": "zh",
}


def _normalize_language_code(raw: Any) -> str | None:
    code = str(raw or "").lower().strip()
    if not code:
        return None
    code = re.split(r"[-_]", code, maxsplit=1)[0].strip()
    code = _LANGUAGE_ALIASES.get(code, code)
    return code if code in _LANGUAGE_NAMES else None


def repair_and_load_json(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group()
    try:
        result = json.loads(cleaned)
        return _ensure_dict(result)
    except json.JSONDecodeError:
        pass
    try:
        repaired = repair_json(cleaned, return_objects=True)
        result = _ensure_dict(repaired)
        if result:
            return result
    except Exception:
        pass
    raise ValueError("Failed to parse LLM response as JSON after all repair attempts")


def _build_entity_id(label: str, name: str, properties: dict[str, Any] | None = None) -> str:
    try:
        entity_type = EntityType(label)
        return build_node_id(entity_type, name)
    except (ValueError, KeyError):
        return build_node_id(EntityType.Entity, name)


def _safe_float_score(raw: Any) -> float:
    try:
        value = float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return min(max(value, 0.0), 1.0)


def _parse_psychographics(data: Any) -> ExtractedPsychographics:
    dict_data = _ensure_dict(data)
    raw_sentiment = dict_data.get("sentiment")
    sentiment_str = raw_sentiment.strip().lower() if isinstance(raw_sentiment, str) else ""
    sentiment = SentimentType(sentiment_str) if sentiment_str in ("positive", "negative", "neutral") else None
    raw_tone = dict_data.get("tone")
    primary_tone = _TONE_MAP.get(str(raw_tone).lower().strip()) if raw_tone else None
    raw_secondary = dict_data.get("secondary_tone")
    secondary_tone = _TONE_MAP.get(str(raw_secondary).lower().strip()) if raw_secondary else None
    score_dopamine = _safe_float_score(dict_data.get("score_dopamine"))
    score_oxytocin = _safe_float_score(dict_data.get("score_oxytocin"))
    score_serotonin = _safe_float_score(dict_data.get("score_serotonin"))
    score_cortisol = _safe_float_score(dict_data.get("score_cortisol"))
    score_adrenaline = _safe_float_score(dict_data.get("score_adrenaline"))
    score_endorphin = _safe_float_score(dict_data.get("score_endorphin"))
    tone_confidence: float | None = None
    raw_tone_confidence = dict_data.get("tone_confidence")
    if raw_tone_confidence is not None:
        try:
            parsed_tone_confidence = float(raw_tone_confidence)
        except (TypeError, ValueError):
            parsed_tone_confidence = None
        if parsed_tone_confidence is not None and 0.0 <= parsed_tone_confidence <= 1.0:
            tone_confidence = parsed_tone_confidence
    return ExtractedPsychographics(
        language=_normalize_language_code(dict_data.get("language")),
        sentiment=sentiment,
        primary_tone=primary_tone,
        secondary_tone=secondary_tone,
        tone_confidence=tone_confidence,
        primary_hormone=None,
        secondary_hormone=None,
        score_dopamine=score_dopamine,
        score_oxytocin=score_oxytocin,
        score_serotonin=score_serotonin,
        score_cortisol=score_cortisol,
        score_adrenaline=score_adrenaline,
        score_endorphin=score_endorphin,
    )


def _parse_entities(raw_entities: Any) -> tuple[list[ExtractedEntity], dict[str, str]]:
    result: list[ExtractedEntity] = []
    id_map: dict[str, str] = {}
    raw_list = _ensure_list(raw_entities)
    for raw in raw_list:
        if isinstance(raw, str):
            raw = {"name": raw, "label": "Entity"}
        elif isinstance(raw, dict):
            raw = _ensure_dict(raw)
        else:
            continue
        name = str(raw.get("name", "")).strip()
        name = name.lstrip("#@").strip()
        name = re.sub(r"\s+", " ", name).strip()
        if not name or len(name) < 2:
            continue
        raw_label = str(raw.get("label", "Entity")).strip()
        try:
            label = EntityType(raw_label)
        except ValueError:
            label = EntityType.Entity
        if raw_label == "Hashtag" or label == EntityType.Hashtag:
            continue
        raw_properties = raw.get("properties")
        properties: dict[str, Any] = _sanitize_properties(dict(raw_properties)) if isinstance(raw_properties, dict) else {}
        entity_id = _build_entity_id(label.value, name, properties)
        prop_key = _ENTITY_TYPE_PROP_KEY.get(label.value, "entity_type")
        raw_entity_type = raw.get("entity_type") or raw.get(prop_key)
        if raw_entity_type is None:
            raw_entity_type = properties.get("entity_type") or properties.get(prop_key)
        if raw_entity_type:
            if "entity_type" in properties and prop_key != "entity_type":
                del properties["entity_type"]
            properties[prop_key] = str(raw_entity_type).lower().strip()
        result.append(ExtractedEntity(
            id=entity_id,
            name=name,
            name_lower=clean_name_lower(name),
            label=label,
            properties=properties,
        ))
        id_map[name] = entity_id
        id_map[name.lower()] = entity_id
    return result, id_map


def _parse_microconcepts(raw_microconcepts: Any) -> tuple[list[ExtractedEntity], dict[str, str]]:
    result: list[ExtractedEntity] = []
    id_map: dict[str, str] = {}
    raw_list = _ensure_list(raw_microconcepts)
    if not raw_list:
        return result, id_map
    seen_name_lower: set[str] = set()
    for item in raw_list:
        if isinstance(item, dict):
            tag = item.get("name") or item.get("tag")
            if tag is None:
                continue
            name = str(tag).strip().strip("\"'«»„“”‘’`").strip().lstrip("#@").strip()
        else:
            name = str(item).strip().strip("\"'«»„“”‘’`").strip().lstrip("#@").strip()
        if len(name) < 2:
            continue
        if _TRUNCATED_TAG_RE.search(name):
            continue
        if _INVALID_TAG_CHARS_RE.search(name):
            continue
        name = name.title()
        name_lower = clean_name_lower(name)
        if not name_lower or name_lower in seen_name_lower:
            continue
        seen_name_lower.add(name_lower)
        entity_id = build_node_id(EntityType.MicroConcept, name_lower)
        result.append(ExtractedEntity(
            id=entity_id,
            name=name,
            name_lower=name_lower,
            label=EntityType.MicroConcept,
            properties={"is_classified": False},
        ))
        id_map[name] = entity_id
        id_map[name.lower()] = entity_id
    return result, id_map


def _parse_relations(
    raw_relations: Any,
    global_id_map: dict[str, str],
    pub_node_id: str,
    author_node_id: str,
    context: PostBatchContext,
) -> list[ExtractedRelation]:
    result: list[ExtractedRelation] = []
    raw_list = _ensure_list(raw_relations)
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        raw_source_id = str(raw.get("source_id", "")).strip()
        raw_target_id = str(raw.get("target_id", "")).strip()
        if not raw_source_id or not raw_target_id:
            continue
        source_id = global_id_map.get(raw_source_id, global_id_map.get(raw_source_id.lower(), raw_source_id))
        target_id = global_id_map.get(raw_target_id, global_id_map.get(raw_target_id.lower(), raw_target_id))
        if source_id == target_id:
            continue
        if source_id == pub_node_id:
            source_label = EntityType.Post
        elif source_id == author_node_id:
            source_label = EntityType.Actor
        else:
            raw_source_label = str(raw.get("source_label", "Entity")).strip()
            try:
                source_label = EntityType(raw_source_label)
            except ValueError:
                source_label = EntityType.Entity
        if target_id == pub_node_id:
            target_label = EntityType.Post
        elif target_id == author_node_id:
            target_label = EntityType.Actor
        else:
            raw_target_label = str(raw.get("target_label", "Entity")).strip()
            try:
                target_label = EntityType(raw_target_label)
            except ValueError:
                target_label = EntityType.Entity
        raw_rel_type = str(raw.get("relation_type", "")).strip().upper()
        try:
            relation_type = RelationType(raw_rel_type)
        except ValueError:
            continue
        raw_properties = raw.get("properties")
        properties: dict[str, Any] = _sanitize_properties(dict(raw_properties)) if isinstance(raw_properties, dict) else {}
        if relation_type == RelationType.ABOUT:
            raw_weight = properties.get("weight")
            try:
                weight_val = float(raw_weight) if raw_weight is not None else 1.0
            except (TypeError, ValueError):
                weight_val = 1.0
            properties["weight"] = weight_val if weight_val > 0.0 else 1.0
        elif relation_type == RelationType.RELATED_TO:
            if "weight" not in properties:
                properties["weight"] = 1.0
            raw_relation_name = properties.get("relation_name")
            if not isinstance(raw_relation_name, str) or not raw_relation_name.strip():
                properties["relation_name"] = "related_to"
        if relation_type == RelationType.WORKS_AT and "role" in properties:
            try:
                RoleType(properties["role"])
            except ValueError:
                properties.pop("role", None)
        if relation_type == RelationType.PRODUCES:
            actor_subtypes = {"creator", "promoter", "affiliate"}
            org_subtypes = {"vendor", "publisher", "distributor", "sponsor"}
            raw_subtype = properties.get("relation_subtype") or properties.get("subtype") or properties.get("type")
            if isinstance(raw_subtype, str):
                raw_subtype = raw_subtype.lower().strip()
            else:
                raw_subtype = None
            if source_label == EntityType.Actor:
                properties["relation_subtype"] = raw_subtype if raw_subtype in actor_subtypes else "creator"
            elif source_label == EntityType.Organization:
                properties["relation_subtype"] = raw_subtype if raw_subtype in org_subtypes else "vendor"
            else:
                properties["relation_subtype"] = "creator"
            properties.pop("subtype", None)
            properties.pop("type", None)
        if relation_type == RelationType.COAUTHOR:
            if "platform" not in properties:
                properties["platform"] = context.platform
            if "post_id" not in properties:
                properties["post_id"] = pub_node_id
        if relation_type == RelationType.USES_TECH:
            proficiency = properties.get("proficiency")
            if not isinstance(proficiency, str) or not proficiency.strip():
                properties["proficiency"] = "user"
        if relation_type == RelationType.PARTICIPATED_IN:
            role = properties.get("role")
            if not isinstance(role, str) or not role.strip():
                properties["role"] = "participant"
        if "sentiment" not in properties and raw.get("sentiment") is not None:
            properties["sentiment"] = str(raw.get("sentiment")).strip()
        if "confidence" not in properties and raw.get("confidence") is not None:
            properties["confidence"] = raw.get("confidence")
        raw_confidence = properties.get("confidence")
        try:
            confidence_float = float(raw_confidence) if raw_confidence is not None else 0.0
        except (TypeError, ValueError):
            confidence_float = 0.0
        try:
            result.append(ExtractedRelation(
                source_id=source_id,
                source_label=source_label,
                target_id=target_id,
                target_label=target_label,
                relation_type=relation_type,
                confidence=confidence_float,
                properties=properties,
            ))
        except (ValidationError, ValueError):
            raise
        except Exception:
            logger.debug("Skipping invalid relation: source=%s target=%s type=%s", source_id, target_id, raw_rel_type)
            continue
    return result


def _parse_hashtags(raw_hashtags_llm: Any, input_raw_tags: list[str]) -> tuple[list[HashtagItem], list[ExtractedEntity], dict[str, str]]:
    hashtag_items: list[HashtagItem] = []
    entities: list[ExtractedEntity] = []
    id_map: dict[str, str] = {}
    seen_raw_lower: set[str] = set()

    raw_list = _ensure_list(raw_hashtags_llm)
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        raw = str(item.get("raw", "")).strip()
        normalized = str(item.get("normalized", "")).strip()
        if not raw or not normalized:
            continue
        raw_lower = raw.lower()
        if is_garbage_value(raw, EntityType.Hashtag):
            continue
        if raw_lower in seen_raw_lower:
            continue
        seen_raw_lower.add(raw_lower)

        hashtag_items.append(HashtagItem(raw=raw, normalized=normalized))

        entity_id = build_node_id(EntityType.Hashtag, raw_lower)
        entities.append(ExtractedEntity(
            id=entity_id,
            name=f"#{raw}",
            name_lower=raw_lower,
            label=EntityType.Hashtag,
            properties={"normalized": normalized},
        ))
        id_map[raw] = entity_id
        id_map[raw_lower] = entity_id

    for tag in input_raw_tags:
        tag_lower = tag.lower()
        if tag_lower in seen_raw_lower:
            continue
        if is_garbage_value(tag, EntityType.Hashtag):
            continue
        seen_raw_lower.add(tag_lower)

        hashtag_items.append(HashtagItem(raw=tag, normalized=tag))

        entity_id = build_node_id(EntityType.Hashtag, tag_lower)
        entities.append(ExtractedEntity(
            id=entity_id,
            name=f"#{tag}",
            name_lower=tag_lower,
            label=EntityType.Hashtag,
            properties={"normalized": tag},
        ))
        id_map[tag] = entity_id
        id_map[tag_lower] = entity_id

    return hashtag_items, entities, id_map


def _create_structural_relations(
    entities: list[ExtractedEntity],
    microconcepts: list[ExtractedEntity],
    hashtag_entities: list[ExtractedEntity],
    pub_node_id: str,
    author_node_id: str,
    context: PostBatchContext,
) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    extra_entities: list[ExtractedEntity] = []
    relations: list[ExtractedRelation] = []
    post_label = EntityType.Post

    published_at = getattr(context, 'published_at', None)
    if published_at is not None:
        try:
            published_ts = int(published_at.timestamp())
        except AttributeError:
            published_ts = int(published_at) if isinstance(published_at, (int, float)) else int(time.time())
    else:
        published_ts = int(time.time())

    relations.append(ExtractedRelation(
        source_id=author_node_id,
        source_label=EntityType.Actor,
        target_id=pub_node_id,
        target_label=post_label,
        relation_type=RelationType.PUBLISHED,
        properties={"published_at": published_ts},
    ))

    for mc in microconcepts:
        relations.append(ExtractedRelation(
            source_id=pub_node_id,
            source_label=post_label,
            target_id=mc.id or "",
            target_label=EntityType.MicroConcept,
            relation_type=RelationType.ABOUT,
            properties={"weight": 1.0},
        ))

    seen_belongs_to: set[tuple[str, str]] = set()
    for ent in entities:
        if ent.label not in (EntityType.Entity, EntityType.Product, EntityType.Organization, EntityType.Event):
            continue
        for mc in microconcepts:
            src_id = ent.id or ""
            tgt_id = mc.id or ""
            key = (src_id, tgt_id)
            if key in seen_belongs_to:
                continue
            seen_belongs_to.add(key)
            relations.append(ExtractedRelation(
                source_id=src_id,
                source_label=ent.label,
                target_id=tgt_id,
                target_label=EntityType.MicroConcept,
                relation_type=RelationType.BELONGS_TO,
                properties={},
            ))

    for ht in hashtag_entities:
        relations.append(ExtractedRelation(
            source_id=pub_node_id,
            source_label=post_label,
            target_id=ht.id or "",
            target_label=EntityType.Hashtag,
            relation_type=RelationType.TAGGED_WITH,
        ))

    return extra_entities, relations


class GraphExtractor:

    def __init__(self, settings: Settings) -> None:
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=100),
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
            verify=False,
        )
        if settings.ml_inference_key_id and settings.ml_inference_secret and settings.ml_inference_base_url:
            self._client = EvolutionAsyncOpenAI(
                key_id=settings.ml_inference_key_id,
                secret=settings.ml_inference_secret,
                base_url=settings.ml_inference_base_url,
                http_client=http_client,
                max_retries=0,
            )
            self._model = settings.ml_inference_model or settings.cloud_ru_llm_model
        else:
            self._client = AsyncOpenAI(
                base_url=settings.cloud_ru_base_url,
                api_key=settings.cloud_ru_api_key,
                http_client=http_client,
                max_retries=0,
            )
            self._model = settings.cloud_ru_llm_model

    async def close(self) -> None:
        await self._client.close()

    async def extract(
        self,
        context: PostBatchContext,
        chunk_text: str,
        retries: int = 3,
    ) -> OpenSPGExtractionResult:
        if not chunk_text:
            return OpenSPGExtractionResult(
                psychographics=ExtractedPsychographics(
                    primary_tone=None,
                    primary_hormone=None,
                ),
            )

        raw_metadata = context.raw_metadata or {}
        if isinstance(raw_metadata, str):
            try:
                raw_metadata = json.loads(raw_metadata)
            except (json.JSONDecodeError, TypeError):
                raw_metadata = {}
        if not isinstance(raw_metadata, dict):
            raw_metadata = {}
        metadata_hashtags = raw_metadata.get("hashtags", [])

        input_raw_tags = extract_raw_hashtags(
            text=chunk_text,
            raw_metadata_hashtags=metadata_hashtags,
            author_bio=context.author_biography,
            author_title=context.author_title,
        )

        messages: Any = [
            {"role": "system", "content": _build_system_prompt(context.author_title)},
            {"role": "user", "content": (
                f"Проанализируй следующий пост и извлеки граф знаний.\n\n"
                f"Текст поста: {chunk_text}\n"
                f"Автор: {context.author_title}\n"
                f"Язык профиля автора: {getattr(context, 'author_language', None) or 'Не указан'}\n"
                f"Описание профиля автора: {getattr(context, 'author_biography', '') or 'Отсутствует'}\n"
                f"Имеется только транскрипция (без подписи): {getattr(context, 'is_transcription_only', False)}\n"
                f"Платформа: {context.platform}\n"
                f"Тип поста: {context.post_type}\n"
                f"Сырые хэштеги поста: {json.dumps(input_raw_tags, ensure_ascii=False)}"
            )},
        ]

        last_error: Exception | None = None
        for attempt in range(retries):
            t0 = time.perf_counter()
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=4096,
                    timeout=60.0,
                )
            except (APIError, APITimeoutError, APIConnectionError, RateLimitError) as e:
                elapsed = (time.perf_counter() - t0) * 1000
                delay = 2 ** attempt
                logger.warning(
                    "LLM call attempt %d/%d failed after %.1fms (model=%s): %s | retrying in %ds",
                    attempt + 1, retries, elapsed, self._model, e, delay,
                )
                last_error = e
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                continue

            llm_elapsed = (time.perf_counter() - t0) * 1000
            logger.debug(
                "LLM call attempt %d/%d completed in %.1fms (model=%s)",
                attempt + 1, retries, llm_elapsed, self._model,
            )

            raw_content = response.choices[0].message.content
            if not raw_content:
                delay = 2 ** attempt
                logger.warning(
                    "Empty LLM response on attempt %d/%d | retrying in %ds",
                    attempt + 1, retries, delay,
                )
                last_error = ValueError("Empty LLM response")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                continue

            try:
                parsed = repair_and_load_json(raw_content)
            except ValueError as e:
                delay = 2 ** attempt
                logger.warning(
                    "JSON parse error on attempt %d/%d: %s | retrying in %ds",
                    attempt + 1, retries, e, delay,
                )
                last_error = e
                if attempt < retries - 1:
                    messages.append({"role": "assistant", "content": raw_content})
                    messages.append({"role": "user", "content": (
                        f"Твой предыдущий ответ вызвал ошибку: {e}. "
                        "Исправь ошибки и ответь строго валидным JSON без текста вокруг."
                    )})
                    await asyncio.sleep(delay)
                continue

            try:
                raw_entities = parsed.get("entities", [])
                raw_relations = parsed.get("relations", [])
                raw_psychographics = parsed.get("psychographics", {})
                raw_microconcepts = parsed.get("microconcepts")

                entities, entity_id_map = _parse_entities(raw_entities)
                microconcept_entities, microconcept_id_map = _parse_microconcepts(raw_microconcepts)
                entities.extend(microconcept_entities)

                raw_hashtags_llm = parsed.get("hashtags", [])
                hashtag_items, hashtag_entities, hashtag_id_map = _parse_hashtags(raw_hashtags_llm, input_raw_tags)

                global_id_map: dict[str, str] = {}
                global_id_map.update(entity_id_map)
                global_id_map.update(microconcept_id_map)
                global_id_map.update(hashtag_id_map)
                global_id_map["post"] = context.pub_node_id
                global_id_map["publication"] = context.pub_node_id
                global_id_map["this_post"] = context.pub_node_id
                global_id_map["content"] = context.pub_node_id
                global_id_map["publication_node"] = context.pub_node_id
                global_id_map["post_node"] = context.pub_node_id
                global_id_map["author"] = context.author_node_id
                global_id_map["actor"] = context.author_node_id
                global_id_map[clean_name_lower(context.author_title)] = context.author_node_id
                global_id_map[clean_name_lower(getattr(context, 'author_username', '') or '')] = context.author_node_id

                llm_relations = _parse_relations(raw_relations, global_id_map, context.pub_node_id, context.author_node_id, context)
                mentions_targets = {
                    r.target_id
                    for r in llm_relations
                    if r.relation_type == RelationType.MENTIONS and r.source_id == context.pub_node_id
                }
                for ent in entities:
                    if (
                        ent.label in (EntityType.Entity, EntityType.Organization, EntityType.Product, EntityType.Event)
                        and ent.id not in mentions_targets
                    ):
                        raise ValueError(
                            f"Extracted entity '{ent.name}' ({ent.id}) is missing a MENTIONS relation from the post"
                        )
                extra_entities, structural_relations = _create_structural_relations(
                    entities,
                    microconcept_entities,
                    hashtag_entities,
                    context.pub_node_id,
                    context.author_node_id,
                    context,
                )
                all_relations = llm_relations + structural_relations

                psychographics = _parse_psychographics(raw_psychographics)
                is_spam = bool(parsed.get("is_spam_or_gambling", False))
                thinking = str(parsed.get("thinking", ""))

                entities.extend(hashtag_entities)
                entities.extend(extra_entities)

                result = OpenSPGExtractionResult(
                    thinking=thinking,
                    entities=entities,
                    relations=all_relations,
                    psychographics=psychographics,
                    is_spam_or_gambling=is_spam,
                    hashtags=hashtag_items,
                )

                forbidden_names = {
                    clean_name_lower(context.author_title),
                    clean_name_lower(getattr(context, 'author_username', '') or ''),
                    clean_name_lower(getattr(context, 'author_handle', '') or ''),
                }
                result.sanitize_and_validate(
                    allowed_ids={context.pub_node_id, context.author_node_id},
                    forbidden_names=forbidden_names,
                    author_title=context.author_title,
                    author_handle=getattr(context, 'author_username', '') or getattr(context, 'author_handle', ''),
                )
            except Exception as e:
                delay = 2 ** attempt
                logger.warning(
                    "Validation error on attempt %d/%d: %s | retrying in %ds",
                    attempt + 1, retries, e, delay,
                )
                last_error = e
                if attempt < retries - 1:
                    messages.append({"role": "assistant", "content": raw_content})
                    messages.append({"role": "user", "content": (
                        f"Твой предыдущий ответ вызвал ошибку: {e}. "
                        "Исправь ошибки и ответь строго валидным JSON без текста вокруг."
                    )})
                    await asyncio.sleep(delay)
                continue

            logger.debug(
                "Extraction done in %.1fms: %d entities, %d relations (model=%s)",
                llm_elapsed, len(result.entities), len(result.relations), self._model,
            )
            return result

        raise last_error or RuntimeError("Extraction failed after all retries")
