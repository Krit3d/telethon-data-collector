import asyncio
import json
import logging
import random
from typing import Any

import httpx
from openai import APIStatusError, APIConnectionError, APITimeoutError, RateLimitError
from evolution_openai import EvolutionAsyncOpenAI
from pydantic import ValidationError

from src.config.config import Settings
from src.graph.extractor.extraction_helpers import sanitize_id
from src.graph.schema import OpenSPGExtractionResult

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 15.0
_STAGGER_MAX = 5.0

_DOMAIN_ENTITY_MAPPING = (
    "DOMAIN SCHEMA REFERENCE\n"
    "SOURCE_NODE: PUB_NODE_ID (always the source for all relations)\n"
    "TARGET_NODES:\n"
    "  Topic Entity -> relation: ABOUT (source=PUB_NODE_ID, target=Topic)\n"
    "  Person|Brand|Organization|Place -> relation: MENTIONS (source=PUB_NODE_ID, target=Entity)\n"
    "SENTIMENT RULE: MENTIONS relation MUST include property: {\"key\": \"sentiment\", \"value\": \"positive|negative|neutral\", \"type\": \"text\"}\n"
    "STRICT NODE LABELS: Actor | Place | Entity | Event\n"
    "PROHIBITED VALUES: 'unknown', 'null', 'none', 'undefined', '' (empty string), [null, null] for coordinates\n"
    "NAMES: must be verbatim from SOURCE_TEXT, no paraphrasing"
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


def _sanitize_parsed_payload(parsed: Any) -> dict[str, Any]:
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
    old_id_to_new_id: dict[str, str] = {}
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
            for prefix in ("actor_", "place_", "event_"):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            name = name.replace("_", " ").title().strip()
        if label is None:
            if entity_id is None:
                continue
            if entity_id.startswith("actor_"):
                label = "Actor"
            elif entity_id.startswith("place_"):
                label = "Place"
            elif entity_id.startswith("event_"):
                label = "Event"
            else:
                label = "Entity"
        assert name is not None
        assert entity_id is not None
        new_id = f"{label.lower()}_{sanitize_id(name)}"
        old_id_to_new_id[entity_id] = new_id
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
        source_id = old_id_to_new_id.get(source_id, source_id)
        target_id = old_id_to_new_id.get(target_id, target_id)
        properties = item.get("properties", [])
        if not isinstance(properties, list):
            properties = []
        properties = [p for p in [_sanitize_property(prop) for prop in properties] if p is not None]
        final_relations.append({
            "source_id": source_id,
            "target_id": target_id,
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
                        "You are a ROBOTIC JSON API, NOT a chatbot. You perform ONE function: structured knowledge graph extraction.\n"
                        "ABSOLUTE REQUIREMENTS:\n"
                        "- You must respond with a single JSON object containing exactly three keys: \"thinking\", \"entities\", and \"relations\".\n"
                        "- Write your step-by-step reasoning inside the \"thinking\" string property first, before outputting the arrays.\n"
                        "- Keep your reasoning extremely brief — no more than 3 bullet points.\n"
                        "- Never translate, paraphrase, or summarize the input text.\n"
                        "- Never use conversational language, greetings, or sign-offs.\n"
                        "- Never output anything outside the JSON object.\n"
                        "VIOLATION OF ANY REQUIREMENT CAUSES IMMEDIATE SYSTEM FAILURE.\n\n"
                        "You must output a JSON object with exactly three keys: 'thinking', 'entities', and 'relations'. "
                        "Strict maximum limit: 8-10 key entities per response to fit into the token budget.\n"
                        "Extract key Topics, Persons, Brands, Places, Organizations, and Events mentioned in the text.\n"
                        "Use this exact structure (fill in your own values):\n"
                        '{\n'
                        '  "thinking": "Brief reasoning here",\n'
                        '  "entities": [\n'
                        '    {\n'
                        '      "id": "actor_unique_id",\n'
                        '      "label": "Actor",\n'
                        '      "name": "Display Name",\n'
                        '      "properties": [\n'
                        '        {"key": "platform", "value": "INSTAGRAM", "type": "text"}\n'
                        '      ]\n'
                        '    }\n'
                        '  ],\n'
                        '  "relations": [\n'
                        '    {\n'
                        '      "source_id": "PUB_NODE_ID",\n'
                        '      "relation_type": "ABOUT",\n'
                        '      "target_id": "topic_unique_id",\n'
                        '      "properties": [\n'
                        '        {"key": "sentiment", "value": "positive", "type": "text"}\n'
                        '      ]\n'
                        '    }\n'
                        '  ]\n'
                        '}\n'
                        "FAILURE MODE WARNING: If you output anything other than the expected format, the parser will crash and all extracted data will be permanently lost."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            try:
                async with self._semaphore:
                    response = await self._client.chat.completions.create(
                        model=self.settings.ml_inference_model,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=4096,
                        frequency_penalty=0.0,
                        top_p=0.85,
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
                parsed = json.loads(content)

                parsed = _sanitize_parsed_payload(parsed)
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
