import json
import logging

from openai import AsyncOpenAI

from src.api.schemas import ReformulatedQuery, SearchRequest
from src.config.config import Settings


class QueryParser:

    def __init__(self, settings: Settings) -> None:
        self._llm_client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=30.0,
        )
        self._llm_model = settings.deepseek_llm_model

    async def parse(self, request: SearchRequest) -> ReformulatedQuery | None:
        cleaned_query = request.query.strip()

        if not cleaned_query:
            return None

        system_prompt = (
            "You are an expert search query reformulator. Analyze the user query and extract key parameters. "
            "Return a valid JSON object matching this schema: "
            '{"dense_query": "string", "graph_entities": ["string"], "target_iab_ids": [123], "profile_type_intent": "expert|business"}. '
            "Instructions: "
            "1. dense_query: remove conversational noise, expand acronyms, keep Russian/English domain context and core intent. "
            "2. graph_entities: extract 1-5 key entity/topic terms strictly in lowercase, without special characters or quotes. "
            "3. target_iab_ids: return 3-5 integer IAB category IDs most relevant to the intent. "
            "4. profile_type_intent: \"expert\" or \"business\"."
        )

        try:
            response = await self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User query: {cleaned_query}\nRequest author_type filter: {request.author_type}"},
                ],
                extra_body={"thinking": {"type": "disabled"}},
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=1000,
            )

            content = response.choices[0].message.content

            if content is None:
                msg = "LLM returned empty content"
                raise ValueError(msg)

            parsed = ReformulatedQuery.model_validate_json(content)
            parsed.graph_entities = [e.lower() for e in parsed.graph_entities]
            return parsed

        except Exception:
            logging.warning(
                "QueryParser LLM call failed for query=%r, returning fallback",
                cleaned_query,
            )
            return ReformulatedQuery(
                dense_query=cleaned_query,
                graph_entities=[],
                target_iab_ids=[],
                profile_type_intent="expert",
            )