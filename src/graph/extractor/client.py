import asyncio
import json
import logging
import random
import re
from typing import Any

from json_repair import repair_json
from openai import AsyncOpenAI, APIStatusError, APIConnectionError, RateLimitError
from pydantic import ValidationError

from src.config.config import Settings
from src.graph.schema import OpenSPGExtractionResult

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 20.0
_REQUEST_TIMEOUT = 240.0
_STAGGER_MAX = 0.3

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


class LLMClient:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.cloud_ru_api_key,
            base_url=settings.cloud_ru_base_url,
            timeout=_REQUEST_TIMEOUT,
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
        cleaned = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL)
        cleaned = re.sub(r"```json\s*", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned)
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
        if not self.settings.cloud_ru_api_key:
            raise RuntimeError("Cloud.ru API key is not configured")

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

        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.chat.completions.create(
                        model=self.settings.cloud_ru_llm_model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a highly meticulous OpenSPG knowledge extraction engine. "
                                    "Extract entities and relations from the provided text following the "
                                    "schema and classification rules strictly. "
                                    "Every entity MUST include a 'type' marker property indicating its "
                                    "domain classification (topic, person, brand, organization, author, "
                                    "publication, region, or other)."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                        max_tokens=4096,
                        frequency_penalty=0.2,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "OpenSPGExtractionResult",
                                "strict": True,
                                "schema": schema,
                            },
                        },
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
                except json.JSONDecodeError as original_err:
                    logger.info(
                        "Raw JSON parsing failed for post_id=%d, "
                        "attempting repair",
                        post_id,
                    )
                    repaired = repair_json(content)
                    try:
                        parsed = json.loads(repaired)
                    except json.JSONDecodeError as repair_err:
                        logger.error(
                            "JSON decode failed after repair for "
                            "post_id=%d: %s",
                            post_id,
                            repair_err,
                            exc_info=True,
                        )
                        raise original_err from repair_err

                try:
                    result = OpenSPGExtractionResult.model_validate(parsed)
                except ValidationError as val_err:
                    sanitized = self._sanitize_validation_payload(parsed, val_err)
                    if sanitized is not None:
                        logger.warning(
                            "Partial validation recovery for post_id=%d "
                            "after sanitizing %d entities / %d relations",
                            post_id,
                            len(sanitized.entities),
                            len(sanitized.relations),
                        )
                        return sanitized
                    logger.error(
                        "Pydantic validation failed for post_id=%d: %s "
                        "raw[:500]=%s | repaired[:500]=%s",
                        post_id,
                        val_err,
                        raw_preview,
                        content[:500],
                        exc_info=True,
                    )
                    raise

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

            except (APIConnectionError, APIStatusError) as exc:
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

            except json.JSONDecodeError as exc:
                last_error = exc
                logger.warning(
                    "JSON decode error on attempt %d/%d for post_id=%d: %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    post_id,
                    exc,
                )
                if attempt < _MAX_RETRIES - 1:
                    delay = self._backoff_delay(attempt)
                    await asyncio.sleep(delay)
                    continue

            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "Validation error on attempt %d/%d for post_id=%d: %s",
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

    @staticmethod
    def _sanitize_validation_payload(
        parsed: dict[str, Any], original_error: ValidationError
    ) -> OpenSPGExtractionResult | None:
        try:
            entities_raw = parsed.get("entities", [])
            relations_raw = parsed.get("relations", [])

            sanitized_entities: list[Any] = []
            for idx, ent in enumerate(entities_raw):
                if not isinstance(ent, dict):
                    continue
                ent_copy = dict(ent)
                if not isinstance(ent_copy.get("name"), str) or not ent_copy["name"].strip():
                    ent_copy["name"] = f"Entity_{idx}"
                if not isinstance(ent_copy.get("id"), str) or not ent_copy["id"].strip():
                    ent_copy["id"] = f"entity_{idx}"
                if not isinstance(ent_copy.get("label"), str) or not ent_copy["label"].strip():
                    ent_copy["label"] = "Entity"
                props = ent_copy.get("properties", [])
                if not isinstance(props, list):
                    ent_copy["properties"] = []
                else:
                    cleaned_props = []
                    for prop in props:
                        if not isinstance(prop, dict):
                            continue
                        p = dict(prop)
                        if not isinstance(p.get("key"), str) or not p["key"].strip():
                            continue
                        if p.get("value") is None:
                            p["value"] = ""
                        if not isinstance(p.get("type"), str):
                            p["type"] = "text"
                        cleaned_props.append(p)
                    ent_copy["properties"] = cleaned_props
                sanitized_entities.append(ent_copy)

            sanitized_relations: list[Any] = []
            valid_entity_ids = {e.get("id") for e in sanitized_entities if isinstance(e, dict)}
            for rel in relations_raw:
                if not isinstance(rel, dict):
                    continue
                rel_copy = dict(rel)
                src = rel_copy.get("source_id", "")
                tgt = rel_copy.get("target_id", "")
                if src not in valid_entity_ids or tgt not in valid_entity_ids:
                    continue
                if not isinstance(rel_copy.get("relation_type"), str) or not rel_copy["relation_type"].strip():
                    rel_copy["relation_type"] = "MENTIONS"
                props = rel_copy.get("properties", [])
                if not isinstance(props, list):
                    rel_copy["properties"] = []
                else:
                    cleaned_props = []
                    for prop in props:
                        if not isinstance(prop, dict):
                            continue
                        p = dict(prop)
                        if not isinstance(p.get("key"), str) or not p["key"].strip():
                            continue
                        if p.get("value") is None:
                            p["value"] = ""
                        if not isinstance(p.get("type"), str):
                            p["type"] = "text"
                        cleaned_props.append(p)
                    rel_copy["properties"] = cleaned_props
                sanitized_relations.append(rel_copy)

            return OpenSPGExtractionResult(
                entities=sanitized_entities,
                relations=sanitized_relations,
            )
        except Exception:
            return None
