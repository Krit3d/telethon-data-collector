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
from src.graph.builder.prompts import SYSTEM_PROMPT, build_user_prompt
from src.graph.builder.reader import PostBatchContext
from src.graph.ontology import (
    ENTITY_CATEGORY_VALUES,
    EntityCategory,
    EntityType,
    EVENT_TYPE_VALUES,
    EventType,
    ExtractedEntity,
    ExtractedPsychographics,
    ExtractedRelation,
    HashtagItem,
    HormoneType,
    OpenSPGExtractionResult,
    ORG_TYPE_VALUES,
    OrgType,
    PREFIX_TO_LABEL_MAP,
    PRODUCT_TYPE_VALUES,
    ProductType,
    RelationSubtype,
    RelationType,
    RoleType,
    ToneType,
    _SENTIMENT_VALUES,
)
from src.graph.utils import (
    GEO_NOISE_TYPES,
    build_node_id,
    clean_identifier,
    clean_name_lower,
    extract_raw_hashtags,
    format_display_name,
    is_garbage_value,
    sanitize_properties,
)

logger = logging.getLogger(__name__)

_TRUNCATED_TAG_RE = re.compile(r"(?:\.{2,}|…|\.$|[\-\—]$)")
_INVALID_TAG_CHARS_RE = re.compile(r"[^A-Za-z0-9\s'\-&/]")

_ALLOWED_CONTENT_LABELS: frozenset[EntityType] = frozenset({EntityType.Entity, EntityType.Organization, EntityType.Product, EntityType.Event})

_GARBAGE_HASHTAGS: frozenset[str] = frozenset({
    "fyp", "foryou", "foryoupage", "fypシ", "fy",
    "reels", "reel", "reelsinstagram",
    "viral", "virals",
    "рек", "рекомендации",
    "топ", "топчик",
    "лайк", "лайки",
    "хочувтоп", "хочу в топ",
    "follow", "followme", "followers",
    "подпишись", "подписка",
    "explore", "explorepage",
})

MAX_HASHTAGS = 15


def _is_garbage_hashtag(tag: str) -> bool:
    tag_lower = tag.lower().strip()
    if tag_lower in _GARBAGE_HASHTAGS:
        return True
    if is_garbage_value(tag, EntityType.Hashtag):
        return True
    return False

def _ensure_dict(val: Any) -> dict[str, Any]:
    if isinstance(val, dict):
        return val
    if isinstance(val, list) and val and isinstance(val[0], dict):
        return val[0]
    return {}


def _ensure_list(val: Any) -> list[Any]:
    if isinstance(val, list):
        return val
    if isinstance(val, dict):
        return list(val.values())
    if val is not None:
        return [val]
    return []


_TONE_MAP: dict[str, ToneType] = {
    "analytical": ToneType.analytical,
    "expert": ToneType.expert,
    "provocative": ToneType.provocative,
    "educational": ToneType.educational,
    "entertainment": ToneType.entertainment,
    "casual": ToneType.casual,
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
    cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response")
    end = cleaned.rfind("}")
    if end > start:
        try:
            res = json.loads(cleaned[start:end+1])
            return _ensure_dict(res)
        except json.JSONDecodeError:
            pass
    try:
        repaired = repair_json(cleaned[start:], return_objects=True)
        res = _ensure_dict(repaired)
        if isinstance(res, dict):
            return res
    except Exception:
        pass
    raise ValueError("Failed to parse LLM response as JSON after all repair attempts")


def _safe_float_score(raw: Any) -> float:
    try:
        value = float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return min(max(value, 0.0), 1.0)


def _parse_psychographics(data: Any) -> ExtractedPsychographics:
    dict_data = _ensure_dict(data)
    nested = _ensure_dict(dict_data.get("hormones") or dict_data.get("scores") or dict_data.get("neurotransmitters"))
    raw_tone = dict_data.get("tone")
    primary_tone = _TONE_MAP.get(str(raw_tone).lower().strip()) if raw_tone else None
    raw_secondary = dict_data.get("secondary_tone")
    secondary_tone = _TONE_MAP.get(str(raw_secondary).lower().strip()) if raw_secondary else None
    hormone_names = ("dopamine", "oxytocin", "serotonin", "cortisol", "adrenaline", "endorphin")
    scores: dict[str, float] = {}
    for h in hormone_names:
        raw = dict_data.get(f"score_{h}")
        if raw is None:
            raw = dict_data.get(h)
        if raw is None:
            raw = nested.get(f"score_{h}")
        if raw is None:
            raw = nested.get(h)
        scores[h] = _safe_float_score(raw)
    score_dopamine = scores["dopamine"]
    score_oxytocin = scores["oxytocin"]
    score_serotonin = scores["serotonin"]
    score_cortisol = scores["cortisol"]
    score_adrenaline = scores["adrenaline"]
    score_endorphin = scores["endorphin"]
    tone_confidence: float | None = None
    raw_tone_confidence = dict_data.get("tone_confidence")
    if raw_tone_confidence is not None:
        try:
            parsed_tone_confidence = float(raw_tone_confidence)
        except (TypeError, ValueError):
            parsed_tone_confidence = None
        if parsed_tone_confidence is not None and 0.0 <= parsed_tone_confidence <= 1.0:
            tone_confidence = parsed_tone_confidence
    max_score = max(scores.values())
    if max_score <= 0.0:
        primary_hormone = None
        secondary_hormone = None
    else:
        sorted_hormones = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_hormone = HormoneType(sorted_hormones[0][0])
        secondary_hormone = HormoneType(sorted_hormones[1][0]) if sorted_hormones[1][1] > 0.0 else None
    return ExtractedPsychographics(
        language=_normalize_language_code(dict_data.get("language")),
        primary_tone=primary_tone,
        secondary_tone=secondary_tone,
        tone_confidence=tone_confidence,
        primary_hormone=primary_hormone,
        secondary_hormone=secondary_hormone,
        score_dopamine=score_dopamine,
        score_oxytocin=score_oxytocin,
        score_serotonin=score_serotonin,
        score_cortisol=score_cortisol,
        score_adrenaline=score_adrenaline,
        score_endorphin=score_endorphin,
    )


def _parse_entities(raw_entities: Any) -> tuple[list[ExtractedEntity], dict[str, str], list[ExtractedRelation], dict[str, tuple[str, float]]]:
    result: list[ExtractedEntity] = []
    id_map: dict[str, str] = {}
    relations: list[ExtractedRelation] = []
    entity_sentiment_map: dict[str, tuple[str, float]] = {}
    microconcept_cache: dict[str, str] = {}
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
        raw_label = str(raw.get("label", "")).strip()
        if not raw_label:
            continue
        try:
            label = EntityType(raw_label)
        except ValueError:
            continue
        if label not in _ALLOWED_CONTENT_LABELS:
            continue
        if is_garbage_value(name, label):
            continue
        raw_entity_type = (
            raw.get("type")
            or raw.get("entity_type")
            or raw.get("org_type")
            or raw.get("product_type")
            or raw.get("event_type")
        )
        if raw_entity_type is not None:
            raw_entity_type = str(raw_entity_type).lower().strip()
        if raw_entity_type and raw_entity_type in GEO_NOISE_TYPES:
            continue
        if raw_entity_type:
            if label == EntityType.Entity:
                if raw_entity_type in ORG_TYPE_VALUES:
                    label = EntityType.Organization
                elif raw_entity_type in PRODUCT_TYPE_VALUES:
                    label = EntityType.Product
                elif raw_entity_type in EVENT_TYPE_VALUES:
                    label = EntityType.Event
            elif label == EntityType.Event and raw_entity_type in ENTITY_CATEGORY_VALUES:
                label = EntityType.Entity
            elif label == EntityType.Product and raw_entity_type in ORG_TYPE_VALUES:
                label = EntityType.Organization
            elif label == EntityType.Organization and raw_entity_type in PRODUCT_TYPE_VALUES:
                label = EntityType.Product
        raw_sentiment = str(raw.get("sentiment", "neutral")).strip().lower()
        if raw_sentiment not in _SENTIMENT_VALUES:
            raw_sentiment = "neutral"
        raw_confidence = raw.get("confidence")
        if raw_confidence is not None:
            try:
                confidence_val = 1.0 if float(raw_confidence) >= 0.9 else 0.8
            except (TypeError, ValueError):
                confidence_val = 0.8
        else:
            confidence_val = 0.8
        name = format_display_name(name, label, is_person=(raw_entity_type == "person"))
        entity_id = build_node_id(label, name)
        properties: dict[str, Any] = {}
        if label == EntityType.Product:
            if not raw_entity_type or raw_entity_type not in PRODUCT_TYPE_VALUES:
                logger.warning("Dropping invalid Product entity '%s': missing or invalid product_type '%s'", name, raw_entity_type)
                continue
            properties["product_type"] = ProductType(raw_entity_type).value
        elif label == EntityType.Organization:
            if not raw_entity_type or raw_entity_type not in ORG_TYPE_VALUES:
                logger.warning("Dropping invalid Organization entity '%s': missing or invalid org_type '%s'", name, raw_entity_type)
                continue
            properties["org_type"] = OrgType(raw_entity_type).value
        elif label == EntityType.Entity:
            if not raw_entity_type or raw_entity_type not in ENTITY_CATEGORY_VALUES:
                logger.warning("Dropping invalid Entity '%s': missing or invalid entity_type '%s'", name, raw_entity_type)
                continue
            properties["entity_type"] = EntityCategory(raw_entity_type).value
        elif label == EntityType.Event:
            if not raw_entity_type or raw_entity_type not in EVENT_TYPE_VALUES:
                logger.warning("Dropping invalid Event entity '%s': missing or invalid event_type '%s'", name, raw_entity_type)
                continue
            properties["event_type"] = EventType(raw_entity_type).value
        try:
            result.append(ExtractedEntity(
                id=entity_id,
                name=name,
                name_lower=clean_name_lower(name),
                label=label,
                properties=properties,
                confidence=confidence_val,
            ))
            entity_sentiment_map[entity_id] = (raw_sentiment, confidence_val)
            id_map[name] = entity_id
            id_map[name.lower()] = entity_id
        except (ValidationError, ValueError) as e:
            logger.warning("Dropping invalid %s entity '%s' due to validation error: %s", label.value, name, e)
            continue
        raw_mc_list = raw.get("micro_concepts")
        if raw_mc_list and isinstance(raw_mc_list, list):
            for mc_item in raw_mc_list:
                if isinstance(mc_item, str):
                    mc_name = mc_item.strip()
                elif isinstance(mc_item, dict):
                    mc_name = str(mc_item.get("name") or mc_item.get("tag") or "").strip()
                else:
                    continue
                if not mc_name:
                    continue
                if _TRUNCATED_TAG_RE.search(mc_name):
                    continue
                if _INVALID_TAG_CHARS_RE.search(mc_name):
                    continue
                if is_garbage_value(mc_name, EntityType.MicroConcept):
                    continue
                mc_name = format_display_name(mc_name, EntityType.MicroConcept)
                mc_name_lower = clean_name_lower(mc_name)
                if mc_name_lower not in microconcept_cache:
                    mc_id = build_node_id(EntityType.MicroConcept, mc_name_lower)
                    microconcept_cache[mc_name_lower] = mc_id
                    result.append(ExtractedEntity(
                        id=mc_id,
                        name=mc_name,
                        name_lower=mc_name_lower,
                        label=EntityType.MicroConcept,
                        properties={"is_classified": False},
                    ))
                    id_map[mc_name] = mc_id
                    id_map[mc_name_lower] = mc_id
                else:
                    mc_id = microconcept_cache[mc_name_lower]
                try:
                    relations.append(ExtractedRelation(
                        source_id=entity_id,
                        source_label=label,
                        target_id=mc_id,
                        target_label=EntityType.MicroConcept,
                        relation_type=RelationType.BELONGS_TO,
                        properties={},
                    ))
                except (ValidationError, ValueError):
                    continue
    return result, id_map, relations, entity_sentiment_map


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
        name = format_display_name(name, EntityType.MicroConcept)
        name_lower = clean_name_lower(name)
        if not name_lower or name_lower in seen_name_lower:
            continue
        if is_garbage_value(name, EntityType.MicroConcept):
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
    known_ids: set[str] = set(global_id_map.values())
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        raw_source_id = str(raw.get("source") or raw.get("source_id") or "").strip()
        raw_target_id = str(raw.get("target") or raw.get("target_id") or "").strip()
        if not raw_source_id or not raw_target_id:
            continue
        source_id = global_id_map.get(raw_source_id, global_id_map.get(raw_source_id.lower(), raw_source_id))
        target_id = global_id_map.get(raw_target_id, global_id_map.get(raw_target_id.lower(), raw_target_id))
        if source_id == target_id:
            continue
        if source_id not in known_ids or target_id not in known_ids:
            continue
        if source_id == pub_node_id:
            source_label = EntityType.Post
        elif source_id == author_node_id:
            source_label = EntityType.Actor
        else:
            raw_source_label = str(raw.get("source_label", "Entity")).strip()
            source_label = None
            for prefix, etype in PREFIX_TO_LABEL_MAP.items():
                if source_id.startswith(prefix):
                    source_label = etype
                    break
            if source_label is None:
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
            target_label = None
            for prefix, etype in PREFIX_TO_LABEL_MAP.items():
                if target_id.startswith(prefix):
                    target_label = etype
                    break
            if target_label is None:
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
        properties: dict[str, Any] = sanitize_properties(dict(raw_properties)) if isinstance(raw_properties, dict) else {}
        if relation_type == RelationType.RELATED_TO:
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
            raw_subtype = properties.get("relation_subtype") or properties.get("subtype") or properties.get("type")
            if isinstance(raw_subtype, str):
                raw_subtype = raw_subtype.lower().strip()
            else:
                raw_subtype = None
            if raw_subtype is not None:
                try:
                    RelationSubtype(raw_subtype)
                except ValueError:
                    raw_subtype = None
            if raw_subtype is None:
                raw_subtype = "creator" if source_label == EntityType.Actor else "vendor"
            properties["relation_subtype"] = raw_subtype
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
                properties["role"] = "visitor"
        raw_conf = raw.get("confidence") or properties.get("confidence")
        if raw_conf is not None:
            try:
                confidence = 1.0 if float(raw_conf) >= 0.9 else 0.8
            except (TypeError, ValueError):
                confidence = 0.8
        else:
            confidence = 0.8
        properties["confidence"] = confidence
        if not source_id or not target_id or source_label is None or target_label is None:
            logger.debug("Skipping relation with invalid source/target: source=%s target=%s source_label=%s target_label=%s type=%s", source_id, target_id, source_label, target_label, raw_rel_type)
            continue
        try:
            result.append(ExtractedRelation(
                source_id=source_id,
                source_label=source_label,
                target_id=target_id,
                target_label=target_label,
                relation_type=relation_type,
                properties=properties,
            ))
        except (ValidationError, ValueError) as e:
            logger.debug("Skipping invalid relation (%s)-[%s]->(%s): %s", source_id, raw_rel_type, target_id, e)
            continue
    return result


def _parse_hashtags(raw_hashtags_llm: Any, input_raw_tags: list[str]) -> tuple[list[HashtagItem], list[ExtractedEntity], dict[str, str]]:
    hashtag_items: list[HashtagItem] = []
    entities: list[ExtractedEntity] = []
    id_map: dict[str, str] = {}
    seen_name_lower: set[str] = set()

    raw_list = _ensure_list(raw_hashtags_llm)
    for item in raw_list:
        if len(hashtag_items) >= MAX_HASHTAGS:
            break
        if isinstance(item, str):
            raw = item.lstrip("#").strip()
            if not raw:
                continue
            normalized = clean_name_lower(raw)
        elif isinstance(item, dict):
            raw = (
                item.get("raw")
                or item.get("tag")
                or item.get("hashtag")
                or item.get("name")
            )
            if raw is None:
                continue
            raw = str(raw).lstrip("#").strip()
            if not raw:
                continue
            normalized = (
                item.get("normalized")
                or item.get("name")
                or item.get("tag")
                or clean_name_lower(raw)
            )
            normalized = str(normalized).strip()
        else:
            continue
        if _is_garbage_hashtag(raw):
            continue
        name_lower = clean_name_lower(raw)
        if name_lower in seen_name_lower:
            continue
        seen_name_lower.add(name_lower)

        hashtag_items.append(HashtagItem(raw=raw, normalized=normalized))

        entity_id = build_node_id(EntityType.Hashtag, name_lower)
        entities.append(ExtractedEntity(
            id=entity_id,
            name=f"#{raw}",
            name_lower=name_lower,
            label=EntityType.Hashtag,
            properties={"raw": raw, "normalized": normalized},
        ))
        id_map[raw] = entity_id
        id_map[name_lower] = entity_id
        id_map[f"#{raw}"] = entity_id
        id_map[f"#{name_lower}"] = entity_id

    for tag in input_raw_tags:
        if len(hashtag_items) >= MAX_HASHTAGS:
            break
        if _is_garbage_hashtag(tag):
            continue
        name_lower = clean_name_lower(tag)
        if name_lower in seen_name_lower:
            continue
        seen_name_lower.add(name_lower)

        hashtag_items.append(HashtagItem(raw=tag, normalized=tag))

        entity_id = build_node_id(EntityType.Hashtag, name_lower)
        entities.append(ExtractedEntity(
            id=entity_id,
            name=f"#{tag}",
            name_lower=name_lower,
            label=EntityType.Hashtag,
            properties={"raw": tag, "normalized": tag},
        ))
        id_map[tag] = entity_id
        id_map[name_lower] = entity_id
        id_map[f"#{tag}"] = entity_id
        id_map[f"#{name_lower}"] = entity_id

    return hashtag_items, entities, id_map


def _create_structural_relations(
    microconcepts: list[ExtractedEntity],
    hashtag_entities: list[ExtractedEntity],
    pub_node_id: str,
    author_node_id: str,
    context: PostBatchContext,
) -> list[ExtractedRelation]:
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
        if mc.id is None or mc.id == "":
            continue
        relations.append(ExtractedRelation(
            source_id=pub_node_id,
            source_label=post_label,
            target_id=mc.id,
            target_label=EntityType.MicroConcept,
            relation_type=RelationType.ABOUT,
            properties={},
        ))

    for ht in hashtag_entities:
        if ht.id is None or ht.id == "":
            continue
        relations.append(ExtractedRelation(
            source_id=pub_node_id,
            source_label=post_label,
            target_id=ht.id,
            target_label=EntityType.Hashtag,
            relation_type=RelationType.TAGGED_WITH,
        ))

    for coauthor in context.post_coauthors:
        clean_ca = clean_identifier(coauthor)
        if clean_ca:
            coauthor_id = build_node_id(EntityType.Actor, clean_ca, platform=context.platform)
            relations.append(ExtractedRelation(
                source_id=context.author_node_id,
                source_label=EntityType.Actor,
                target_id=coauthor_id,
                target_label=EntityType.Actor,
                relation_type=RelationType.COAUTHOR,
                properties={"platform": context.platform, "post_id": pub_node_id},
            ))

    return relations


class LLMInfrastructureError(Exception):
    ...


class GraphExtractor:

    def __init__(self, settings: Settings) -> None:
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=100, keepalive_expiry=30.0),
            timeout=httpx.Timeout(connect=10.0, read=45.0, write=10.0, pool=15.0),
            verify=False,
            http2=False,
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

        self._semaphore = asyncio.Semaphore(settings.graph_concurrency)

    async def close(self) -> None:
        await self._client.close()

    async def extract(
        self,
        context: PostBatchContext,
        caption_text: str = "",
        transcription_text: str = "",
        retries: int = 3,
    ) -> OpenSPGExtractionResult:
        if not caption_text and not transcription_text:
            caption_text = context.content or ""
            transcription_text = context.transcription or ""
        if not caption_text and not transcription_text:
            return OpenSPGExtractionResult(
                psychographics=ExtractedPsychographics(
                    primary_tone=None,
                    primary_hormone=None,
                ),
            )

        input_raw_tags = extract_raw_hashtags(
            text=f"{caption_text} {transcription_text}".strip(),
            raw_metadata_hashtags=context.post_hashtags,
            author_bio=context.author_biography,
            author_title=context.author_title,
        )

        author_handle = (getattr(context, 'author_handle', '') or getattr(context, 'author_username', '') or '').strip()
        messages: Any = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(
                caption_text=caption_text,
                transcription_text=transcription_text,
                author_title=context.author_title,
                author_handle=author_handle,
                platform=context.platform,
                post_type=context.post_type,
                author_biography=context.author_biography,
                coauthors=context.post_coauthors,
                raw_hashtags=input_raw_tags,
            )},
        ]

        last_error: Exception | None = None
        for attempt in range(retries):
            t0 = time.perf_counter()
            try:
                async with self._semaphore:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        response_format={"type": "json_object"},
                        temperature=0.1,
                        max_tokens=3072,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                            "enable_thinking": False,
                            "thinking": {"type": "disabled"},
                        },
                    )
                llm_elapsed = (time.perf_counter() - t0) * 1000
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                total_tokens = usage.total_tokens if usage else 0
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "length":
                    logger.warning(
                        "JSON response truncated due to token limit for content_id=%d (model=%s)",
                        context.content_id, self._model,
                    )
            except (APIError, APITimeoutError, APIConnectionError, RateLimitError, httpx.HTTPError) as e:
                elapsed = (time.perf_counter() - t0) * 1000
                base_delay = 1.5
                delay = min(15.0, base_delay * (2 ** attempt)) * (0.75 + (0.5 * (time.perf_counter() % 1.0)))
                logger.warning(
                    "LLM call attempt %d/%d failed after %.1fms (model=%s, error=%s): %s | retrying in %.1fs",
                    attempt + 1, retries, elapsed, self._model, type(e).__name__, e, delay,
                )
                last_error = e
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                continue

            logger.debug(
                "LLM call attempt %d/%d completed in %.1fms (model=%s, prompt=%d, completion=%d, total=%d)",
                attempt + 1, retries, llm_elapsed, self._model,
                prompt_tokens, completion_tokens, total_tokens,
            )

            choice = response.choices[0]
            raw_content = choice.message.content or getattr(choice.message, "reasoning_content", None)
            if raw_content:
                if " response" in raw_content:
                    raw_content = raw_content.split(" response")[-1]
                elif " thinking" in raw_content:
                    raw_content = raw_content.split(" thinking")[0]
                raw_content = raw_content.strip()
            if not raw_content:
                base_delay = 1.5
                delay = min(15.0, base_delay * (2 ** attempt)) * (0.75 + (0.5 * (time.perf_counter() % 1.0)))
                logger.warning(
                    "Empty LLM response on attempt %d/%d | retrying in %.1fs",
                    attempt + 1, retries, delay,
                )
                last_error = ValueError("Empty LLM response")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                continue

            try:
                parsed = repair_and_load_json(raw_content)
            except ValueError as e:
                base_delay = 1.5
                delay = min(15.0, base_delay * (2 ** attempt)) * (0.75 + (0.5 * (time.perf_counter() % 1.0)))
                logger.warning(
                    "JSON parse error on attempt %d/%d: %s | retrying in %.1fs",
                    attempt + 1, retries, e, delay,
                )
                last_error = e
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                continue

            try:
                raw_entities = parsed.get("entities", [])
                raw_relations = parsed.get("relations", [])
                raw_psychographics = parsed.get("psychographics", {})
                raw_microconcepts = parsed.get("microconcepts")

                entities, entity_id_map, entity_belongs_to_relations, entity_sentiment_map = _parse_entities(raw_entities)
                microconcept_entities, microconcept_id_map = _parse_microconcepts(raw_microconcepts)
                entities.extend(microconcept_entities)

                raw_hashtags_llm = parsed.get("hashtags", [])
                hashtag_items, hashtag_entities, hashtag_id_map = _parse_hashtags(raw_hashtags_llm, input_raw_tags)

                global_id_map: dict[str, str] = {}
                global_id_map.update(entity_id_map)
                global_id_map.update(microconcept_id_map)
                global_id_map.update(hashtag_id_map)
                global_id_map["$author"] = context.author_node_id
                global_id_map["$post"] = context.pub_node_id
                global_id_map["author"] = context.author_node_id
                global_id_map["actor"] = context.author_node_id
                global_id_map["post"] = context.pub_node_id
                global_id_map["publication"] = context.pub_node_id
                if author_handle:
                    clean_handle = clean_name_lower(author_handle)
                    global_id_map[clean_handle] = context.author_node_id
                    global_id_map[f"@{clean_handle}"] = context.author_node_id
                    global_id_map[author_handle.lower()] = context.author_node_id
                    global_id_map[f"@{author_handle.lower()}"] = context.author_node_id

                llm_relations = _parse_relations(raw_relations, global_id_map, context.pub_node_id, context.author_node_id, context)
                for ent in entities:
                    if (
                        ent.label in (EntityType.Entity, EntityType.Organization, EntityType.Product, EntityType.Event)
                        and ent.id is not None
                    ):
                        ent_sentiment, ent_confidence = entity_sentiment_map.get(ent.id, ("neutral", 0.8))
                        try:
                            llm_relations.append(ExtractedRelation(
                                source_id=context.pub_node_id,
                                source_label=EntityType.Post,
                                target_id=ent.id,
                                target_label=ent.label,
                                relation_type=RelationType.MENTIONS,
                                confidence=ent_confidence,
                                properties={"confidence": ent_confidence, "sentiment": ent_sentiment},
                            ))
                        except (ValidationError, ValueError):
                            continue
                structural_relations = _create_structural_relations(
                    microconcept_entities,
                    hashtag_entities,
                    context.pub_node_id,
                    context.author_node_id,
                    context,
                )
                all_relations = llm_relations + entity_belongs_to_relations + structural_relations

                psychographics = _parse_psychographics(raw_psychographics)
                is_spam = bool(parsed.get("is_spam_or_gambling", False))
                thinking = str(parsed.get("thinking", ""))

                entities.extend(hashtag_entities)

                seen_entity_ids: set[str] = set()
                deduped_entities: list[ExtractedEntity] = []
                for ent in entities:
                    eid = ent.id
                    if not eid or eid in seen_entity_ids:
                        continue
                    seen_entity_ids.add(eid)
                    deduped_entities.append(ent)
                entities = deduped_entities

                result = OpenSPGExtractionResult(
                    thinking=thinking,
                    entities=entities,
                    relations=all_relations,
                    psychographics=psychographics,
                    is_spam_or_gambling=is_spam,
                    hashtags=hashtag_items,
                )

                forbidden_names = {clean_name_lower(n) for n in (context.author_title, author_handle) if n and clean_name_lower(n)}
                coauthor_ids = {
                    build_node_id(EntityType.Actor, clean_identifier(ca), platform=context.platform)
                    for ca in context.post_coauthors
                    if clean_identifier(ca)
                }
                result.sanitize_and_validate(
                    allowed_ids={context.pub_node_id, context.author_node_id} | coauthor_ids,
                    forbidden_names=forbidden_names,
                    author_title=context.author_title,
                    author_handle=author_handle,
                )
            except Exception as e:
                base_delay = 1.5
                delay = min(15.0, base_delay * (2 ** attempt)) * (0.75 + (0.5 * (time.perf_counter() % 1.0)))
                logger.warning(
                    "Validation error on attempt %d/%d: %s | retrying in %.1fs",
                    attempt + 1, retries, e, delay,
                )
                last_error = e
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                continue

            logger.info(
                "Extraction done in %.1fms: %d entities, %d relations (model=%s, prompt=%d, completion=%d, total=%d)",
                llm_elapsed, len(result.entities), len(result.relations), self._model,
                prompt_tokens, completion_tokens, total_tokens,
            )
            return result

        if isinstance(last_error, (APIError, APITimeoutError, APIConnectionError, RateLimitError, httpx.HTTPError, httpx.HTTPStatusError)):
            raise LLMInfrastructureError(str(last_error)) from last_error
        raise last_error or RuntimeError("Extraction failed after all retries")
