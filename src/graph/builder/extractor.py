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
from src.graph.builder.prompts import build_system_prompt, build_user_prompt
from src.graph.builder.reader import PostBatchContext
from src.graph.ontology import (
    EntityType,
    ExtractedEntity,
    ExtractedPsychographics,
    ExtractedRelation,
    HashtagItem,
    HormoneType,
    OpenSPGExtractionResult,
    RelationType,
    RoleType,
    SentimentType,
    ToneType,
    _SENTIMENT_VALUES,
)
from src.graph.utils import build_node_id, clean_identifier, clean_name_lower, extract_raw_hashtags, format_display_name, is_garbage_value, sanitize_properties

logger = logging.getLogger(__name__)

_TRUNCATED_TAG_RE = re.compile(r"(?:\.{2,}|…|\.$|[\-\—]$)")
_INVALID_TAG_CHARS_RE = re.compile(r"[^A-Za-z0-9\s'\-&/]")

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

_PREFIX_LABEL_MAP: dict[str, EntityType] = {
    "actor_": EntityType.Actor,
    "event_publication_": EntityType.Post,
    "microconcept_": EntityType.MicroConcept,
    "concept_": EntityType.Concept,
    "hashtag_": EntityType.Hashtag,
    "organization_": EntityType.Organization,
    "product_": EntityType.Product,
    "entity_": EntityType.Entity,
    "event_": EntityType.Event,
}

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
    scores = {"dopamine": score_dopamine, "oxytocin": score_oxytocin, "serotonin": score_serotonin, "cortisol": score_cortisol, "adrenaline": score_adrenaline, "endorphin": score_endorphin}
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


def _parse_entities(raw_entities: Any) -> tuple[list[ExtractedEntity], dict[str, str], list[ExtractedRelation]]:
    result: list[ExtractedEntity] = []
    id_map: dict[str, str] = {}
    relations: list[ExtractedRelation] = []
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
        raw_label = str(raw.get("label", "Entity")).strip()
        try:
            label = EntityType(raw_label)
        except ValueError:
            label = EntityType.Entity
        if raw_label == "Hashtag" or label == EntityType.Hashtag:
            continue
        if is_garbage_value(name, label):
            continue
        raw_properties = raw.get("properties")
        properties: dict[str, Any] = sanitize_properties(dict(raw_properties)) if isinstance(raw_properties, dict) else {}
        raw_sentiment = str(raw.get("sentiment", "neutral")).strip().lower()
        if raw_sentiment in _SENTIMENT_VALUES:
            properties["sentiment"] = raw_sentiment
        else:
            properties["sentiment"] = "neutral"
        raw_confidence = raw.get("confidence")
        if raw_confidence is not None:
            try:
                confidence_val = float(raw_confidence)
            except (TypeError, ValueError):
                confidence_val = 1.0
            if confidence_val < 0.5:
                confidence_val = 0.5
            elif confidence_val > 1.0:
                confidence_val = 1.0
        else:
            confidence_val = 1.0
        is_reclassified_regulatory = False
        if label == EntityType.Event:
            from src.graph.utils import is_regulatory_entity
            if is_regulatory_entity(name):
                is_reclassified_regulatory = True
                label = EntityType.Entity
                properties["entity_type"] = "term"
                properties.pop("event_type", None)
                raw_entity_type = raw.get("type") or raw.get("entity_type") or properties.get("type") or properties.get("entity_type")
                name = format_display_name(name, label, is_person=(raw_entity_type == "person"))
                entity_id = build_node_id(label, name)
            else:
                entity_id = _build_entity_id(label.value, name, properties)
        else:
            entity_id = _build_entity_id(label.value, name, properties)
        prop_key = _ENTITY_TYPE_PROP_KEY.get(label.value, "entity_type")
        raw_entity_type = raw.get("type") or raw.get("entity_type") or raw.get(prop_key) or properties.get("type") or properties.get("entity_type") or properties.get(prop_key)
        name = format_display_name(name, label, is_person=(raw_entity_type == "person"))
        if raw_entity_type:
            if is_reclassified_regulatory:
                properties["entity_type"] = "term"
            else:
                properties[prop_key] = str(raw_entity_type).lower().strip()
        properties.pop("type", None)
        result.append(ExtractedEntity(
            id=entity_id,
            name=name,
            name_lower=clean_name_lower(name),
            label=label,
            properties=properties,
            confidence=confidence_val,
        ))
        id_map[name] = entity_id
        id_map[name.lower()] = entity_id
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
    return result, id_map, relations


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
            for prefix, etype in _PREFIX_LABEL_MAP.items():
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
            for prefix, etype in _PREFIX_LABEL_MAP.items():
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
                properties["role"] = "visitor"
        try:
            result.append(ExtractedRelation(
                source_id=source_id,
                source_label=source_label,
                target_id=target_id,
                target_label=target_label,
                relation_type=relation_type,
                properties=properties,
            ))
        except (ValidationError, ValueError, Exception):
            logger.debug("Skipping invalid relation: source=%s target=%s type=%s", source_id, target_id, raw_rel_type)
            continue
    return result


def _parse_hashtags(raw_hashtags_llm: Any, input_raw_tags: list[str]) -> tuple[list[HashtagItem], list[ExtractedEntity], dict[str, str]]:
    hashtag_items: list[HashtagItem] = []
    entities: list[ExtractedEntity] = []
    id_map: dict[str, str] = {}
    seen_raw_lower: set[str] = set()
    MAX_HASHTAGS = 7

    def _is_garbage_hashtag(tag: str) -> bool:
        tag_lower = tag.lower().strip()
        if tag_lower in _GARBAGE_HASHTAGS:
            return True
        if is_garbage_value(tag, EntityType.Hashtag):
            return True
        return False

    raw_list = _ensure_list(raw_hashtags_llm)
    for item in raw_list:
        if len(hashtag_items) >= MAX_HASHTAGS:
            break
        if not isinstance(item, dict):
            continue
        raw = str(item.get("raw", "")).strip()
        normalized = str(item.get("normalized", "")).strip()
        if not raw or not normalized:
            continue
        raw_lower = raw.lower()
        if _is_garbage_hashtag(raw):
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
        if len(hashtag_items) >= MAX_HASHTAGS:
            break
        tag_lower = tag.lower()
        if tag_lower in seen_raw_lower:
            continue
        if _is_garbage_hashtag(tag):
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
        if mc.id is None or mc.id == "":
            continue
        relations.append(ExtractedRelation(
            source_id=pub_node_id,
            source_label=post_label,
            target_id=mc.id,
            target_label=EntityType.MicroConcept,
            relation_type=RelationType.ABOUT,
            properties={"weight": 1.0},
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
            coauthor_id = f"actor_{context.platform.lower()}_{clean_ca}"
            relations.append(ExtractedRelation(
                source_id=context.author_node_id,
                source_label=EntityType.Actor,
                target_id=coauthor_id,
                target_label=EntityType.Actor,
                relation_type=RelationType.COAUTHOR,
                properties={"platform": context.platform, "post_id": pub_node_id},
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
            {"role": "system", "content": build_system_prompt(context.author_title, author_handle)},
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

                entities, entity_id_map, entity_belongs_to_relations = _parse_entities(raw_entities)
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
                        ent_sentiment = ent.properties.pop("sentiment", "neutral")
                        ent_confidence = ent.confidence if ent.confidence > 0.0 else 1.0
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
                extra_entities, structural_relations = _create_structural_relations(
                    entities,
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
                entities.extend(extra_entities)

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

                forbidden_names = {clean_name_lower(context.author_title)}
                if author_handle:
                    forbidden_names.add(clean_name_lower(author_handle))
                coauthor_ids = {
                    f"actor_{context.platform.lower()}_{clean_identifier(ca)}"
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
