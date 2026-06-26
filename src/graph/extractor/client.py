import asyncio
import json
import logging
import random
import re
from typing import Any

import httpx
from json_repair import repair_json
from openai import APIStatusError, APIConnectionError, APITimeoutError, RateLimitError
from evolution_openai import EvolutionAsyncOpenAI
from pydantic import ValidationError

from src.config.config import Settings
from src.graph.schema import OpenSPGExtractionResult

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 15.0
_STAGGER_MAX = 5.0

_DOMAIN_ENTITY_MAPPING = (
    "\n\nMANDATORY PUBLICATION ANCHOR & RELATIONSHIP EXTRACTION RULES:\n"
    "The text you are analyzing is ALWAYS a social media publication (post).\n"
    "A main publication node already exists in the graph with a dynamic ID provided in the prompt.\n"
    "You MUST NOT create a new Publication entity for the post itself.\n"
    "Instead, you MUST use the provided dynamic publication node ID as the source for all outgoing relations.\n\n"
    "You MUST connect ALL other extracted entities to the provided publication node using its actual dynamic ID:\n"
    "  - For every extracted Topic (Entity): create an ABOUT relation:\n"
    "    source_id=<pub_node_id>, relation_type=\"ABOUT\", target_id=<topic_entity_id>\n"
    "  - For every extracted Person, Brand, Organization (Actors) or Place: create a MENTIONS relation:\n"
    "    source_id=<pub_node_id>, relation_type=\"MENTIONS\", target_id=<entity_id>,\n"
    "    and include the \"sentiment\" property: {\"key\": \"sentiment\", \"value\": \"<positive|negative|neutral>\", \"type\": \"text\"}\n"
    "  - You MAY also create direct relations between other entities when explicitly mentioned in the text,\n"
    "    e.g., Actor -[LOCATED_IN]-> Place, Actor -[COLLABORATES_WITH]-> Actor.\n"
    "  - Every relation originating from the publication node MUST reference valid target entity ids from the same extraction.\n\n"
    "ADDITIONAL DOMAIN ENTITY CLASSIFICATION RULES:\n"
    "You MUST classify each extracted entity using a 'type' marker property in its properties list.\n\n"
    "Classification mapping (domain concept -> OpenSPG label + type marker property):\n"
    "  - Topic (subject, theme, discussion point, trend, concept) -> label: Entity, "
    'add property: {"key": "type", "value": "topic", "type": "text"}\n'
    "  - Person (individual, public figure, influencer, expert) -> label: Actor, "
    'add property: {"key": "type", "value": "person", "type": "text"}\n'
    "  - Brand (company, product line, trademark, startup) -> label: Actor, "
    'add property: {"key": "type", "value": "brand", "type": "text"}\n'
    "  - Organization (institution, agency, team, group, department) -> label: Actor, "
    'add property: {"key": "type", "value": "organization", "type": "text"}\n'
    "  - Author (blogger, content creator, journalist) -> label: Actor, "
    'add property: {"key": "type", "value": "author", "type": "text"}\n'
    "  - Region (geographic area, country, city, district) -> label: Place, "
    'add property: {"key": "type", "value": "region", "type": "text"}\n\n'
    "STRICT ENTITY NAME RULE:\n"
    "The 'name' property of EVERY entity MUST be a clean, literal name extracted directly from the source text. "
    "It MUST contain ONLY the actual text as written by the author \u2014 nothing else.\n"
    "You are ABSOLUTELY PROHIBITED from placing any of the following inside the 'name' property:\n"
    "  - Placeholder labels or tags: [Null], [N/A], [Unknown], [Untitled]\n"
    "  - Bracketed role tags: [Author], [Brand], [Topic], [Person], [Organization]\n"
    "  - Entity classification suffixes or prefixes (e.g. 'Brand: Nike', 'Author John')\n"
    "  - Platform names or generic descriptors instead of the real name (e.g. 'YouTube Channel', 'Telegram Post')\n"
    "  - Any synthetic, inferred, or invented text that does not appear verbatim in the original text\n"
    "If the real name cannot be determined, set 'name' to an empty string and add a 'reason' property explaining why.\n\n"
    "STRICT RELATIONSHIP RULES:\n"
    "  1. You MUST output ABOUT relations for every Topic that is discussed by the publication.\n"
    "     Format: source_id=<pub_node_id>, relation_type=ABOUT, target_id=<topic_entity_id>\n"
    "  2. You MUST output MENTIONS relations when the publication references a Brand, Person, or Organization.\n"
    "     Format: source_id=<pub_node_id>, relation_type=MENTIONS, target_id=<entity_id>\n"
    "  3. Every MENTIONS relation MUST include a sentiment property:\n"
    '     {"key": "sentiment", "value": "<positive|negative|neutral>", "type": "text"}\n'
    "     Determine sentiment based on the context and tone of the mention in the text.\n\n"
    "Every extracted entity MUST include the 'type' marker property. "
    "If an entity does not fit any of the above categories, use the most appropriate "
    "OpenSPG label (Actor/Entity/Place) and set type to 'other'."
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
    processing: list[dict[str, Any]] = []
    if isinstance(parsed, list):
        processing = list(parsed)
    elif isinstance(parsed, dict):
        processing = list(parsed.get("entities", [])) + list(parsed.get("relations", []))
    else:
        return {"entities": [], "relations": []}
    flat: list[dict[str, Any]] = []
    while processing:
        item = processing.pop(0)
        if not isinstance(item, dict):
            continue
        for nested_field in ("entities", "relations"):
            nested = item.get(nested_field)
            if isinstance(nested, list):
                processing.extend(nested)
        flat.append(item)
    ENTITY_ID_KEYS = ("id", "_id", "id_", "entity_id", "node_id")
    ENTITY_NAME_KEYS = ("name", "display_name", "title")
    ENTITY_LABEL_KEYS = ("label", "entity_type")
    SOURCE_ID_KEYS = ("source_id", "source", "_source", "from_id", "start_node_id")
    TARGET_ID_KEYS = ("target_id", "target", "_target", "to_id", "end_node_id")
    RELATION_TYPE_KEYS = ("relation_type", "type", "relation", "edge_label")
    final_entities: list[dict[str, Any]] = []
    final_relations: list[dict[str, Any]] = []
    for item in flat:
        if not isinstance(item, dict):
            continue
        source_id: str | None = None
        target_id: str | None = None
        for key in SOURCE_ID_KEYS:
            val = item.get(key)
            if val is not None:
                source_id = str(val) if not isinstance(val, str) else val
                break
        for key in TARGET_ID_KEYS:
            val = item.get(key)
            if val is not None:
                target_id = str(val) if not isinstance(val, str) else val
                break
        if source_id is not None or target_id is not None:
            relation_type: str | None = None
            for key in RELATION_TYPE_KEYS:
                val = item.get(key)
                if val is not None:
                    relation_type = str(val) if not isinstance(val, str) else val
                    break
            if relation_type is None:
                relation_type = "RELATED_TO"
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
        else:
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
            assert entity_id is not None
            if label is None:
                if entity_id.startswith("actor_"):
                    label = "Actor"
                elif entity_id.startswith("place_"):
                    label = "Place"
                elif entity_id.startswith("event_"):
                    label = "Event"
                else:
                    label = "Entity"
            properties = item.get("properties", [])
            if not isinstance(properties, list):
                properties = []
            properties = [p for p in [_sanitize_property(prop) for prop in properties] if p is not None]
            final_entities.append({
                "id": entity_id,
                "name": name,
                "label": label,
                "properties": properties,
            })
    return {"entities": final_entities, "relations": final_relations}


class LLMClient:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._semaphore = asyncio.Semaphore(getattr(settings, "llm_max_concurrency", settings.graph_concurrency))
        http_limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
        http_timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
        http_client = httpx.AsyncClient(limits=http_limits, timeout=http_timeout)
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
        cleaned = re.sub(r"<(think|thinking)>.*?(?:</\1>|$)", "", raw, flags=re.DOTALL | re.IGNORECASE)
        match = re.search(r"```(?:json)?\s*\n?(.*?)(?:\n?\s*```|$)", cleaned, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return cleaned[first_brace:last_brace + 1].strip()
        return cleaned.strip()

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
        return (
            f"{base}{_DOMAIN_ENTITY_MAPPING}"
            "\n\nCRITICAL: Your final output MUST be a single valid JSON object wrapped inside a ```json ... ``` code block. "
            "NEVER use null for 'type' properties. NEVER use [null, null] for coordinates. "
            "Ensure strict schema adherence."
        )

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
        schema = OpenSPGExtractionResult.model_json_schema()

        if "$defs" in schema:
            if "ExtractedEntity" in schema["$defs"]:
                entity_schema = schema["$defs"]["ExtractedEntity"]
                if "required" in entity_schema and "properties" not in entity_schema["required"]:
                    entity_schema["required"].append("properties")
            if "ExtractedRelation" in schema["$defs"]:
                relation_schema = schema["$defs"]["ExtractedRelation"]
                if "required" in relation_schema and "properties" not in relation_schema["required"]:
                    relation_schema["required"].append("properties")

        last_error: BaseException | None = None

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a highly meticulous OpenSPG knowledge extraction engine. "
                    "You MUST FIRST write your thought process and reasoning inside "
                    "<think>...</think> tags. ONLY AFTER thinking, output your response "
                    "as a valid JSON object wrapped in ```json ... ``` blocks. "
                    "Extract entities and relations strictly following the schema. "
                    "Every entity MUST include a 'type' marker property indicating "
                    "its domain classification. "
                    "Never output null for 'type'. Never output [null, null] for coordinates. "
                    "Ensure strict adherence to the schema.\n\n"
                    "Here is the strict JSON schema you MUST follow:\n"
                    + json.dumps(schema)
                ),
            },
            {"role": "user", "content": prompt},
        ]

        for attempt in range(_MAX_RETRIES):
            try:
                async with self._semaphore:
                    response = await self._client.chat.completions.create(
                        model=self.settings.ml_inference_model,
                        messages=messages,
                        temperature=0.6,
                        max_tokens=4096,
                        frequency_penalty=0.2,
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
                raw_preview = content[:500]

                try:
                    parsed = json.loads(content)
                    parsed = _sanitize_parsed_payload(parsed)
                except json.JSONDecodeError as original_err:
                    logger.info(
                        "Raw JSON parsing failed for post_id=%d, "
                        "attempting repair",
                        post_id,
                    )
                    repaired = repair_json(content)
                    try:
                        parsed = json.loads(repaired)
                        parsed = _sanitize_parsed_payload(parsed)
                    except json.JSONDecodeError as repair_err:
                        logger.error(
                            "JSON decode failed after repair for "
                            "post_id=%d: %s",
                            post_id,
                            repair_err,
                            exc_info=True,
                        )
                        raise original_err from repair_err

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
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Your output failed validation:\n{exc}\nPlease fix the schema errors, do not lose any data, and return ONLY the corrected valid JSON."})
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
