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

from src.config.config import Settings
from src.graph.builder.reader import PostBatchContext
from src.graph.ontology import (
    EntityType,
    ExtractedEntity,
    ExtractedPsychographics,
    ExtractedRelation,
    HormoneType,
    OpenSPGExtractionResult,
    RelationType,
    ToneType,
    is_author_entity,
)
from src.graph.utils import build_node_id, clean_name_lower, extract_hashtags

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
        "Ты — строгий графовый экстрактор знаний, работающий по архитектуре OpenSPG и KAG. "
        "Твоя задача — извлечь структурированное представление знаний из текста поста социальной сети. "
        "Отвечай ТОЛЬКО валидным JSON-объектом без дополнительного текста. "
        "JSON должен содержать следующие ключи:\n"
        "  - thinking: короткое пошаговое рассуждение (строка)\n"
        "  - entities: массив объектов {name, label, entity_type, properties}\n"
        "    Допустимые label: Entity, Organization, Product, Event, Place\n"
        "    Для Entity entity_type: technology, method, person, term, general\n"
        "    Для Organization entity_type: company, brand, agency, media, non_profit\n"
        "    Для Product entity_type: software, gadget, course, app, physical_good, service\n"
        "    Для Event entity_type: conference, release, incident, festival, trend\n"
        "    Для Place entity_type: city, country, region, venue. "
        "Извлекай в properties дочерние детали локации, если они присутствуют в тексте: "
        "country_code (2-буквенный ISO код в нижнем регистре, например ru, us, ae), city, country, region, "
        "latitude, longitude.\n"
        "  - relations: массив объектов {source_id, source_label, target_id, target_label, relation_type, properties}\n"
        "    Допустимые relation_type: MENTIONS, ABOUT, TAGGED_AT, HAS_CONTACT, BASED_IN, WORKS_AT, "
        "PARTICIPATED_IN, USES_TECH, PRODUCES, TARGETS, LOCATED_IN\n"
        "  - microconcepts: массив из 1-3 атомарных обобщающих тематических тегов (строк). "
        "ТЕГИ ДОЛЖНЫ БЫТЬ СТРОГО НА АНГЛИЙСКОМ ЯЗЫКЕ (например, 'Fashion', 'Sports Nutrition', "
        "'Automotive Tuning', 'Self Improvement'). Независимо от языка входного текста поста, "
        "переводи и канонизируй теги microconcepts на английский язык. "
        "Каждая строка должна быть целым осмысленным существительным или устойчивым словосочетанием (1-3 слова max). "
        "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО извлекать обрезки слов, глагольные формы, верстку, хештеги и служебные знаки.\n"
        "  - psychographics: объект с полями:\n"
        "      language: двухбуквенный ISO-код основного языка поста (ru, en, kz и т.д.).\n"
        "        Правила определения языка:\n"
        "        1. Основным источником является осмысленный текст подписи к посту.\n"
        "        2. Если подписи нет и доступна только транскрипция:\n"
        "           - Если транскрипция содержит осмысленную речь автора, определяй язык СТРОГО по ней.\n"
        "           - Если транскрипция состоит из текста фоновой песни, музыки, автоматического шума или "
        "междометий, НЕ определяй язык по песне. В этом случае используй язык из названия/биографии автора.\n"
        "      sentiment: positive | negative | neutral\n"
        "      primary_tone: analytical | expert | provocative | educational | entertainment | casual | hype_train | sell_courses\n"
        "      secondary_tone: тон или null\n"
        "      primary_hormone: dopamine | oxytocin | serotonin | cortisol | adrenaline | endorphin\n"
        "      secondary_hormone: гормон или null\n"
        "      score_dopamine, score_oxytocin, score_serotonin, score_cortisol, score_adrenaline, score_endorphin: "
        "числа с плавающей точкой от 0.0 до 1.0\n"
        "  - is_spam_or_gambling: boolean (true если пост содержит спам, скам, гемблинг, ставки, легкий заработок)\n"
        f"КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО извлекать самого автора поста '{author_title}' "
        "(в любых падежах, склонениях, формах и транслитах) в массив entities. "
        "НЕ добавляй служебные свойства (role, platform, author, post) в объекты массива entities."
    )


_TONE_MAP: dict[str, ToneType] = {
    "analytical": ToneType.Analytical,
    "expert": ToneType.Expert,
    "provocative": ToneType.Provocative,
    "educational": ToneType.Educational,
    "entertainment": ToneType.Entertainment,
    "casual": ToneType.Casual,
    "hype_train": ToneType.Hype_Train,
    "sell_courses": ToneType.Sell_Courses,
}

_ENTITY_TYPE_PROP_KEY: dict[str, str] = {
    "Entity": "entity_type",
    "Organization": "org_type",
    "Product": "product_type",
    "Event": "event_type",
    "Place": "place_type",
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
    "rus": "ru",
    "eng": "en",
    "kaz": "kk",
    "zho": "zh",
    "chi": "zh",
    "deu": "de",
    "fra": "fr",
}


def _normalize_language_code(raw: Any) -> str:
    code = str(raw or "").lower().strip()
    if not code:
        return "ru"
    code = re.split(r"[-_]", code, maxsplit=1)[0].strip()
    code = _LANGUAGE_ALIASES.get(code, code)
    return code if code in _LANGUAGE_NAMES else "ru"


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
        country_code = None
        if properties:
            country_code = properties.get("country_code")
        return build_node_id(entity_type, name, country_code=country_code)
    except (ValueError, KeyError):
        return build_node_id(EntityType.Entity, name)


def _parse_psychographics(data: Any) -> ExtractedPsychographics:
    dict_data = _ensure_dict(data)
    raw_tone = str(dict_data.get("primary_tone", "casual")).lower().strip()
    primary_tone = _TONE_MAP.get(raw_tone, ToneType.Casual)
    raw_secondary = dict_data.get("secondary_tone")
    secondary_tones: list[ToneType] = []
    if raw_secondary:
        mapped = _TONE_MAP.get(str(raw_secondary).lower().strip())
        if mapped:
            secondary_tones.append(mapped)
    raw_hormone = str(dict_data.get("primary_hormone", "dopamine")).lower().strip()
    try:
        primary_hormone = HormoneType(raw_hormone)
    except ValueError:
        primary_hormone = HormoneType.dopamine
    raw_secondary_hormone = dict_data.get("secondary_hormone")
    secondary_hormone: HormoneType | None = None
    if raw_secondary_hormone:
        try:
            secondary_hormone = HormoneType(str(raw_secondary_hormone).lower().strip())
        except ValueError:
            pass
    scores: dict[str, float] = {}
    for key in ("score_dopamine", "score_oxytocin", "score_serotonin", "score_cortisol", "score_adrenaline", "score_endorphin"):
        val = dict_data.get(key)
        if val is not None:
            try:
                scores[key] = float(val)
            except (ValueError, TypeError):
                scores[key] = 0.0
    return ExtractedPsychographics(
        language=_normalize_language_code(dict_data.get("language")),
        primary_tone=primary_tone,
        secondary_tones=secondary_tones,
        primary_hormone=primary_hormone,
        secondary_hormone=secondary_hormone,
        scores=scores,
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
        if not name or len(name) < 2:
            continue
        raw_label = str(raw.get("label", "Entity")).strip()
        try:
            label = EntityType(raw_label)
        except ValueError:
            label = EntityType.Entity
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
            name = str(tag).strip().strip("\"'«»„“”‘’`").strip()
        else:
            name = str(item).strip().strip("\"'«»„“”‘’`").strip()
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


def _parse_relations(raw_relations: Any, global_id_map: dict[str, str]) -> list[ExtractedRelation]:
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
        raw_source_label = str(raw.get("source_label", "Entity")).strip()
        raw_target_label = str(raw.get("target_label", "Entity")).strip()
        try:
            source_label = EntityType(raw_source_label)
        except ValueError:
            source_label = EntityType.Entity
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
        try:
            result.append(ExtractedRelation(
                source_id=source_id,
                source_label=source_label,
                target_id=target_id,
                target_label=target_label,
                relation_type=relation_type,
                properties=properties,
            ))
        except Exception:
            logger.debug("Skipping invalid relation: source=%s target=%s type=%s", source_id, target_id, raw_rel_type)
            continue
    return result


def _create_structural_relations(
    entities: list[ExtractedEntity],
    microconcepts: list[ExtractedEntity],
    pub_node_id: str,
    author_node_id: str,
    chunk_text: str,
    raw_psychographics: dict[str, Any],
) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    extra_entities: list[ExtractedEntity] = []
    relations: list[ExtractedRelation] = []
    post_label = EntityType.Post

    relations.append(ExtractedRelation(
        source_id=author_node_id,
        source_label=EntityType.Actor,
        target_id=pub_node_id,
        target_label=post_label,
        relation_type=RelationType.PUBLISHED,
    ))

    for mc in microconcepts:
        relations.append(ExtractedRelation(
            source_id=pub_node_id,
            source_label=post_label,
            target_id=mc.id or "",
            target_label=EntityType.MicroConcept,
            relation_type=RelationType.ABOUT,
        ))

    for ent in entities:
        if ent.label == EntityType.MicroConcept:
            continue
        if ent.label == EntityType.Place:
            relations.append(ExtractedRelation(
                source_id=pub_node_id,
                source_label=post_label,
                target_id=ent.id or "",
                target_label=EntityType.Place,
                relation_type=RelationType.TAGGED_AT,
            ))
        else:
            relations.append(ExtractedRelation(
                source_id=pub_node_id,
                source_label=post_label,
                target_id=ent.id or "",
                target_label=ent.label,
                relation_type=RelationType.MENTIONS,
            ))

    tone_val = str(raw_psychographics.get("primary_tone", "casual")).lower().strip()
    tone_type = _TONE_MAP.get(tone_val, ToneType.Casual)
    tone_name = tone_type.value.lower()
    tone_entity_id = build_node_id(EntityType.Tone, tone_name)
    extra_entities.append(ExtractedEntity(
        id=tone_entity_id,
        name=tone_name,
        name_lower=clean_name_lower(tone_name),
        label=EntityType.Tone,
    ))
    relations.append(ExtractedRelation(
        source_id=pub_node_id,
        source_label=post_label,
        target_id=tone_entity_id,
        target_label=EntityType.Tone,
        relation_type=RelationType.HAS_TONE,
    ))

    lang_code = _normalize_language_code(raw_psychographics.get("language"))
    lang_entity_id = build_node_id(EntityType.Language, lang_code)
    extra_entities.append(ExtractedEntity(
        id=lang_entity_id,
        name=_LANGUAGE_NAMES[lang_code],
        name_lower=clean_name_lower(_LANGUAGE_NAMES[lang_code]),
        label=EntityType.Language,
        properties={"code": lang_code},
    ))
    relations.append(ExtractedRelation(
        source_id=pub_node_id,
        source_label=post_label,
        target_id=lang_entity_id,
        target_label=EntityType.Language,
        relation_type=RelationType.HAS_LANGUAGE,
    ))

    hashtags = extract_hashtags(chunk_text)
    for tag in hashtags:
        hashtag_entity_id = build_node_id(EntityType.Hashtag, tag)
        extra_entities.append(ExtractedEntity(
            id=hashtag_entity_id,
            name=tag,
            name_lower=clean_name_lower(tag),
            label=EntityType.Hashtag,
        ))
        relations.append(ExtractedRelation(
            source_id=pub_node_id,
            source_label=post_label,
            target_id=hashtag_entity_id,
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
                    primary_tone=ToneType.Casual,
                    primary_hormone=HormoneType.dopamine,
                ),
            )

        messages: Any = [
            {"role": "system", "content": _build_system_prompt(context.author_title)},
            {"role": "user", "content": (
                f"Проанализируй следующий пост и извлеки граф знаний.\n\n"
                f"Текст поста: {chunk_text}\n"
                f"Автор: {context.author_title}\n"
                f"Биография автора: {getattr(context, 'author_biography', '') or 'Отсутствует'}\n"
                f"Имеется только транскрипция (без подписи): {getattr(context, 'is_transcription_only', False)}\n"
                f"Платформа: {context.platform}\n"
                f"Тип поста: {context.post_type}"
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

                global_id_map: dict[str, str] = {}
                global_id_map.update(entity_id_map)
                global_id_map.update(microconcept_id_map)
                global_id_map["post"] = context.pub_node_id
                global_id_map["publication"] = context.pub_node_id
                global_id_map["this_post"] = context.pub_node_id
                global_id_map["author"] = context.author_node_id
                global_id_map["actor"] = context.author_node_id

                llm_relations = _parse_relations(raw_relations, global_id_map)
                extra_entities, structural_relations = _create_structural_relations(
                    entities,
                    microconcept_entities,
                    context.pub_node_id,
                    context.author_node_id,
                    chunk_text,
                    raw_psychographics,
                )
                all_relations = llm_relations + structural_relations

                psychographics = _parse_psychographics(raw_psychographics)
                is_spam = bool(parsed.get("is_spam_or_gambling", False))
                thinking = str(parsed.get("thinking", ""))

                entities.extend(extra_entities)

                result = OpenSPGExtractionResult(
                    thinking=thinking,
                    entities=entities,
                    relations=all_relations,
                    psychographics=psychographics,
                    is_spam_or_gambling=is_spam,
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
