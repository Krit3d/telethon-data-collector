import asyncio
import json
import logging
from typing import Any

from openai import AsyncOpenAI, APIStatusError, APIConnectionError, RateLimitError
from pydantic import ValidationError

from src.config.config import Settings
from src.graph.schema import OpenSPGExtractionResult
from src.graph.utils import _repair_json

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 2.0
_RATE_LIMIT_COOLDOWN = 60.0

_DOMAIN_ENTITY_MAPPING = (
    "\n\nADDITIONAL DOMAIN ENTITY CLASSIFICATION RULES:\n"
    "You MUST classify each extracted entity using a 'type' marker property in its properties list.\n\n"
    "Classification mapping (domain concept -> OpenSPG label + type marker property):\n"
    "  - Topic (subject, theme, discussion point, trend, concept) -> label: Concept, "
    'add property: {"key": "type", "value": "topic", "type": "text"}\n'
    "  - Person (individual, public figure, influencer, expert) -> label: Actor, "
    'add property: {"key": "type", "value": "person", "type": "text"}\n'
    "  - Brand (company, product line, trademark, startup) -> label: Actor, "
    'add property: {"key": "type", "value": "brand", "type": "text"}\n'
    "  - Organization (institution, agency, team, group, department) -> label: Actor, "
    'add property: {"key": "type", "value": "organization", "type": "text"}\n'
    "  - Author (blogger, content creator, journalist) -> label: Actor, "
    'add property: {"key": "type", "value": "author", "type": "text"}\n'
    "  - Publication (article, post, news piece, announcement) -> label: Event, "
    'add property: {"key": "type", "value": "publication", "type": "text"}\n'
    "  - Region (geographic area, country, city, district) -> label: Place, "
    'add property: {"key": "type", "value": "region", "type": "text"}\n\n'
    "STRICT RELATIONSHIP RULES:\n"
    "  1. You MUST output ABOUT relations for every Publication entity that discusses a Topic.\n"
    "     Format: source_id=<publication_id>, relation_type=ABOUT, target_id=<topic_entity_id>\n"
    "  2. You MUST output MENTIONS relations when a Publication references a Brand, Person, or Organization.\n"
    "     Format: source_id=<publication_id>, relation_type=MENTIONS, target_id=<entity_id>\n"
    "  3. Every MENTIONS relation MUST include a sentiment property:\n"
    '     {{"key": "sentiment", "value": "<positive|negative|neutral>", "type": "text"}}\n'
    "     Determine sentiment based on the context and tone of the mention in the text.\n\n"
    "Every extracted entity MUST include the 'type' marker property. "
    "If an entity does not fit any of the above categories, use the most appropriate "
    "OpenSPG label (Actor/Entity/Event/Place) and set type to 'other'."
)


class LLMClient:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.cloud_ru_api_key,
            base_url=settings.cloud_ru_base_url,
        )

    async def close(self) -> None:
        await self._client.close()
        logger.debug("LLMClient: AsyncOpenAI client closed")

    def _build_prompt(
        self,
        text: str,
        author_id: int,
        platform: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        from src.graph.schema import get_open_spg_llm_prompt

        base = get_open_spg_llm_prompt(text, author_id, platform, metadata)
        return f"{base}{_DOMAIN_ENTITY_MAPPING}"

    async def call_llm(
        self,
        text: str,
        author_id: int,
        post_id: int,
        metadata: dict[str, Any] | None = None,
        platform: str | None = None,
    ) -> OpenSPGExtractionResult:
        if not self.settings.cloud_ru_api_key:
            raise RuntimeError("Cloud.ru API key is not configured")

        prompt = self._build_prompt(text, author_id, platform, metadata)
        schema = OpenSPGExtractionResult.model_json_schema()

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
                        continue
                    break

                repaired = _repair_json(content)
                if repaired != content:
                    logger.info(
                        "JSON repair applied for post_id=%d", post_id
                    )
                    content = repaired

                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as json_err:
                    logger.error(
                        "JSON decode failed for post_id=%d: %s "
                        "(content[:300]=%s)",
                        post_id,
                        json_err,
                        content[:300],
                    )
                    last_error = RuntimeError(
                        f"JSON decode failed: {json_err}"
                    )
                    last_error.__cause__ = json_err
                    if attempt < _MAX_RETRIES - 1:
                        continue
                    break

                try:
                    result = OpenSPGExtractionResult.model_validate(parsed)
                except ValidationError as val_err:
                    logger.error(
                        "Pydantic validation failed for post_id=%d: %s",
                        post_id,
                        val_err,
                    )
                    last_error = RuntimeError(
                        f"Validation failed: {val_err}"
                    )
                    last_error.__cause__ = val_err
                    if attempt < _MAX_RETRIES - 1:
                        continue
                    break

                logger.info(
                    "LLM extraction succeeded for post_id=%d: "
                    "%d entities, %d relations",
                    post_id,
                    len(result.entities),
                    len(result.relations),
                )
                return result

            except RateLimitError:
                delay = (
                    _RATE_LIMIT_COOLDOWN
                    if attempt < _MAX_RETRIES - 1
                    else _RETRY_BASE_DELAY * (2 ** attempt)
                )
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
                    await asyncio.sleep(
                        _RETRY_BASE_DELAY * (2 ** attempt)
                    )
                    continue

        raise RuntimeError(
            f"LLM extraction failed after {_MAX_RETRIES} attempts "
            f"for post_id={post_id}"
        ) from last_error
