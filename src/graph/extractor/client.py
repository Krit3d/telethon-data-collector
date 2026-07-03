import asyncio
import json
import logging
import random
import re
from typing import Any

import httpx
from openai import APIStatusError, APIConnectionError, APITimeoutError, RateLimitError
from evolution_openai import EvolutionAsyncOpenAI
from pydantic import ValidationError

from src.config.config import Settings
from src.graph.schema import OpenSPGExtractionResult
from src.graph.utils import repair_and_load_json, sanitize_id, is_garbage_value

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 15.0
_STAGGER_MAX = 5.0

_CANONICAL_PREFIXES: tuple[str, ...] = (
    "actor_", "place_", "event_", "topic_", "event_publication_",
)

_DOMAIN_ENTITY_MAPPING = (
    "DOMAIN SCHEMA REFERENCE\n"
    "SOURCE_NODE: PUB_NODE_ID (always the source for all relations)\n"
    "TARGET_NODES:\n"
    "  Topic Entity -> relation: ABOUT (source=PUB_NODE_ID, target=Topic)\n"
    "  Person|Brand|Organization|Place -> relation: MENTIONS (source=PUB_NODE_ID, target=Entity)\n"
    "SENTIMENT: Only 'MENTIONS' has: {\"key\": \"sentiment\", \"value\": \"positive|negative|neutral\", \"type\": \"text\"}. Other relations: no properties.\n"
    "STRICT NODE LABELS: Actor | Place | Entity | Event\n"
    "PROHIBITED VALUES: 'unknown', 'null', 'none', 'undefined', '' (empty string), [null, null] for coordinates\n"
    "NAMES: must be verbatim from SOURCE_TEXT, no paraphrasing"
)

_SYSTEM_PREFIXES: tuple[str, ...] = (
    "actor_", "place_", "event_", "topic_", "event_publication_",
)


def _sanitize_property(prop: Any) -> dict[str, Any] | None:
    if not isinstance(prop, dict) or "key" not in prop:
        return None
    if not isinstance(prop.get("type"), str):
        prop["type"] = "text"
    if prop.get("value") is None:
        prop["value"] = ""
    if prop.get("type") == "geo" and prop.get("key") == "coordinates":
        val = prop.get("value")
        if not isinstance(val, list) or len(val) != 2:
            return None
        for coord in val:
            if coord is None or not isinstance(coord, (int, float)):
                return None
    return prop


def _force_label_from_prefix(new_id: str) -> str:
    if new_id.startswith("actor_"):
        return "Actor"
    if new_id.startswith("place_"):
        return "Place"
    if new_id.startswith("event_") or new_id.startswith("event_publication_"):
        return "Event"
    if new_id.startswith("topic_"):
        return "Entity"
    return "Entity"


def _resolve_preliminary_label(raw_label: str | None, entity_id: str | None) -> str:
    if raw_label is not None:
        cleaned = raw_label.strip().lower()
        if cleaned in ("actor", "person", "brand", "organization", "company", "creator", "user", "coauthor"):
            return "Actor"
        if cleaned in ("place", "location", "city", "country", "region", "address"):
            return "Place"
        if cleaned in ("event", "publication", "post", "show", "incident", "video", "photo"):
            return "Event"
        return "Entity"
    if entity_id is not None:
        eid_low = entity_id.lower()
        if eid_low.startswith("actor_"):
            return "Actor"
        if eid_low.startswith("place_"):
            return "Place"
        if eid_low.startswith("event_") or eid_low.startswith("event_publication_"):
            return "Event"
    return "Entity"


def _canonicalize_id(raw_id: str | None) -> str:
    if raw_id is None:
        return ""
    s = raw_id.strip().lower()
    for prefix in _CANONICAL_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return re.sub(r'[^a-z0-9]', '', s)


def _sanitize_parsed_payload(parsed: Any, pub_node_id: str | None = None) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {"entities": [], "relations": []}
    ENTITY_ID_KEYS = ("id", "_id", "id_", "entity_id", "node_id")
    ENTITY_NAME_KEYS = ("name", "display_name", "title")
    ENTITY_LABEL_KEYS = ("label", "entity_type")
    SOURCE_ID_KEYS = ("source_id", "source", "_source", "from_id", "start_node_id")
    TARGET_ID_KEYS = ("target_id", "target", "_target", "to_id", "end_node_id")
    RELATION_TYPE_KEYS = ("relation_type", "type", "relation", "edge_label")
    final_entities: list[dict[str, Any]] = []
    final_relations: list[dict[str, Any]] = []
    exact_map: dict[str, str] = {}
    canonical_map: dict[str, str] = {}
    for item in parsed.get("entities", []):
        if not isinstance(item, dict):
            continue
        entity_id: str | None = None
        for key in ENTITY_ID_KEYS:
            val = item.get(key)
            if val is not None:
                entity_id = str(val) if not isinstance(val, str) else val
                break
        name: str | None = None
        for key in ENTITY_NAME_KEYS:
            val = item.get(key)
            if val is not None:
                name = str(val) if not isinstance(val, str) else val
                break
        label: str | None = None
        for key in ENTITY_LABEL_KEYS:
            val = item.get(key)
            if val is not None:
                label = str(val) if not isinstance(val, str) else val
                break
        if entity_id is None and name is not None:
            entity_id = name.lower().replace(" ", "_")
        if entity_id is None and name is None:
            continue
        if name is None and entity_id is not None:
            name = entity_id.lstrip("_")
            for prefix in _SYSTEM_PREFIXES:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            name = name.replace("_", " ").title().strip()
        preliminary_label = _resolve_preliminary_label(label, entity_id)
        assert name is not None
        assert entity_id is not None
        name = name.strip()
        if is_garbage_value(name) or is_garbage_value(entity_id):
            continue
        eid_low = entity_id.lower()
        if any(eid_low.startswith(p) for p in _SYSTEM_PREFIXES):
            new_id = eid_low
        elif name.startswith("#"):
            name = name.lstrip("#").strip()
            new_id = f"topic_hashtag_{sanitize_id(name)}"
        elif preliminary_label == "Place":
            new_id = f"place_{sanitize_id(name)}"
        elif preliminary_label == "Actor":
            new_id = f"actor_{sanitize_id(name)}"
        elif preliminary_label == "Event":
            new_id = f"event_{sanitize_id(name)}"
        else:
            new_id = f"topic_{sanitize_id(name)}"
        label = _force_label_from_prefix(new_id)
        exact_map[entity_id] = new_id
        canonical_map[_canonicalize_id(entity_id)] = new_id
        canonical_map[_canonicalize_id(name)] = new_id
        properties = item.get("properties", [])
        if not isinstance(properties, list):
            properties = []
        properties = [p for p in [_sanitize_property(prop) for prop in properties] if p is not None]
        if not any(p.get("key") == "type" for p in properties):
            fallback_map = {"Actor": "person", "Place": "location", "Event": "event"}
            fallback_value = fallback_map.get(label, "topic")
            properties.append({"key": "type", "value": fallback_value, "type": "text"})
        final_entities.append({
            "id": new_id,
            "name": name,
            "label": label,
            "properties": properties,
        })
    valid_new_ids = {e["id"] for e in final_entities}
    if pub_node_id:
        p_id_low = pub_node_id.lower()
        exact_map[p_id_low] = p_id_low
        canonical_map[_canonicalize_id(pub_node_id)] = p_id_low
        valid_new_ids.add(p_id_low)
    for item in parsed.get("relations", []):
        if not isinstance(item, dict):
            continue
        source_id: str | None = None
        for key in SOURCE_ID_KEYS:
            val = item.get(key)
            if val is not None:
                source_id = str(val) if not isinstance(val, str) else val
                break
        target_id: str | None = None
        for key in TARGET_ID_KEYS:
            val = item.get(key)
            if val is not None:
                target_id = str(val) if not isinstance(val, str) else val
                break
        if source_id is None or target_id is None:
            continue
        relation_type: str | None = None
        for key in RELATION_TYPE_KEYS:
            val = item.get(key)
            if val is not None:
                relation_type = str(val) if not isinstance(val, str) else val
                break
        if relation_type is None:
            relation_type = "RELATED_TO"
        resolved_source = exact_map.get(source_id)
        if resolved_source is None:
            resolved_source = canonical_map.get(_canonicalize_id(source_id), source_id)
        resolved_target = exact_map.get(target_id)
        if resolved_target is None:
            resolved_target = canonical_map.get(_canonicalize_id(target_id), target_id)
        if resolved_source not in valid_new_ids:
            for _prefix in ("topic_", "place_", "actor_", "event_"):
                _candidate = f"{_prefix}{resolved_source}"
                if _candidate in valid_new_ids:
                    resolved_source = _candidate
                    break
        if resolved_target not in valid_new_ids:
            for _prefix in ("topic_", "place_", "actor_", "event_"):
                _candidate = f"{_prefix}{resolved_target}"
                if _candidate in valid_new_ids:
                    resolved_target = _candidate
                    break
        if resolved_source not in valid_new_ids or resolved_target not in valid_new_ids:
            continue
        properties = item.get("properties", [])
        if not isinstance(properties, list):
            properties = []
        properties = [p for p in [_sanitize_property(prop) for prop in properties] if p is not None]
        if relation_type.upper() == "MENTIONS" and not any(p.get("key") == "sentiment" for p in properties):
            properties.append({"key": "sentiment", "value": "neutral", "type": "text"})
        final_relations.append({
            "source_id": resolved_source,
            "target_id": resolved_target,
            "relation_type": relation_type,
            "properties": properties,
        })
    return {"entities": final_entities, "relations": final_relations}


class LLMClient:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._semaphore = asyncio.Semaphore(getattr(settings, "llm_max_concurrency", settings.graph_concurrency))
        http_limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
        http_timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
        http_client = httpx.AsyncClient(limits=http_limits, timeout=http_timeout, verify=False)
        key_id = settings.ml_inference_key_id
        secret = settings.ml_inference_secret
        base_url = settings.ml_inference_base_url
        if not key_id or not secret:
            raise RuntimeError("ML Inference credentials (key_id and secret) are not configured")
        if not base_url:
            raise RuntimeError("ML Inference base_url is not configured")
        self._client = EvolutionAsyncOpenAI(
            key_id=key_id,
            secret=secret,
            base_url=base_url,
            http_client=http_client,
            max_retries=0,
        )

    async def close(self) -> None:
        await self._client.close()

    @staticmethod
    def _backoff_delay(attempt: int, is_rate_limit: bool = False) -> float:
        exponential = min(_RETRY_BASE_DELAY * (2 ** attempt), _RETRY_MAX_DELAY)
        jitter = random.uniform(1.0, 2.0) if is_rate_limit else random.uniform(0.5, 1.5)
        return exponential * jitter

    @staticmethod
    def _sanitize_llm_content(raw: str) -> str:
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1 and end > start:
            return raw[start:end + 1]
        return raw.strip()

    def _build_prompt(
        self,
        text: str,
        pub_node_id: str,
        author_id: int,
        platform: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        from src.graph.schema import get_open_spg_llm_prompt

        base = get_open_spg_llm_prompt(text, pub_node_id, author_id, platform, metadata)
        return f"{base}{_DOMAIN_ENTITY_MAPPING}"

    async def call_llm(
        self,
        text: str,
        author_id: int,
        post_id: int,
        pub_node_id: str,
        metadata: dict[str, Any] | None = None,
        platform: str | None = None,
        stagger: bool = True,
    ) -> OpenSPGExtractionResult:
        if not self.settings.ml_inference_key_id or not self.settings.ml_inference_secret:
            raise RuntimeError("ML Inference credentials (key_id and secret) are not configured")
        if not self.settings.ml_inference_model:
            raise RuntimeError("ML Inference model is not configured")

        if stagger:
            await asyncio.sleep(random.random() * _STAGGER_MAX)

        prompt = self._build_prompt(text, pub_node_id, author_id, platform, metadata)

        last_error: BaseException | None = None

        for attempt in range(_MAX_RETRIES):
            messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": (
                        "Deterministic JSON extraction. "
                        "Keys: thinking, entities, relations. "
                        "thinking: 1-2 sentence reasoning. "
                        "entities: [{id: lowercase_snake, name: verbatim_from_text, label: one_of(Actor|Place|Entity|Event), properties: [{key, value, type}]}] "
                        "relations: [{source_id, target_id, relation_type: ABOUT|MENTIONS|LOCATED_IN, properties: [{key, value, type}]}] "
                        "CRITICAL: If a real entity name is NOT present in the source text, do NOT extract that entity. "
                        "FORBIDDEN name/id values: <name>, name>, unknown, null, none, undefined, n_a, other, id. "
                        "Names must match source text exactly. "
                        "source_id must be PUB_NODE_ID, target_id must exist in entities. "
                        "Output ONLY valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            try:
                async with self._semaphore:
                    response = await self._client.chat.completions.create(
                        model=self.settings.ml_inference_model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=2048,
                        response_format={"type": "json_object"},
                    )

                content = (
                    response.choices[0].message.content
                    if response.choices
                    else ""
                )

                if not content:
                    logger.warning(
                        "Empty LLM response on attempt %d/%d for post_id=%d",
                        attempt + 1,
                        _MAX_RETRIES,
                        post_id,
                    )
                    last_error = RuntimeError("LLM returned empty content")
                    if attempt < _MAX_RETRIES - 1:
                        delay = self._backoff_delay(attempt)
                        await asyncio.sleep(delay)
                        continue
                    break

                content = self._sanitize_llm_content(content)
                parsed = repair_and_load_json(content)

                logger.info(
                    "RAW LLM output for post_id=%d: %d entities, %d relations",
                    post_id,
                    len(parsed.get("entities", [])),
                    len(parsed.get("relations", [])),
                )

                parsed = _sanitize_parsed_payload(parsed, pub_node_id=pub_node_id)
                result = OpenSPGExtractionResult.model_validate(parsed)

                logger.info(
                    "LLM extraction succeeded for post_id=%d: "
                    "%d entities, %d relations",
                    post_id,
                    len(result.entities),
                    len(result.relations),
                )
                return result

            except RateLimitError as exc:
                last_error = exc
                delay = self._backoff_delay(attempt, is_rate_limit=True)
                logger.warning(
                    "Rate limit on attempt %d/%d for post_id=%d, "
                    "retrying in %.1fs",
                    attempt + 1,
                    _MAX_RETRIES,
                    post_id,
                    delay,
                )
                await asyncio.sleep(delay)
                continue

            except (APIConnectionError, APIStatusError, APITimeoutError) as exc:
                last_error = exc
                logger.warning(
                    "API error on attempt %d/%d for post_id=%d: %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    post_id,
                    exc,
                )
                if attempt < _MAX_RETRIES - 1:
                    delay = self._backoff_delay(attempt)
                    await asyncio.sleep(delay)
                    continue

            except (json.JSONDecodeError, ValidationError, TypeError, AttributeError) as exc:
                last_error = exc
                logger.warning(
                    "Validation/decode error on attempt %d/%d for post_id=%d: %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    post_id,
                    exc,
                )
                if attempt < _MAX_RETRIES - 1:
                    delay = self._backoff_delay(attempt)
                    await asyncio.sleep(delay)
                    continue

        if last_error is not None:
            raise last_error
        raise RuntimeError(
            f"LLM extraction failed after {_MAX_RETRIES} attempts "
            f"for post_id={post_id}"
        )
