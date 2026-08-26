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

    async def parse(self, request: SearchRequest) -> ReformulatedQuery:
        cleaned_query = request.query.strip()

        if not cleaned_query:
            return ReformulatedQuery(
                dense_query="",
                graph_entities=[],
                semantic_topics=[],
                profile_type_intent="expert",
            )

        system_prompt = """<system-instructions>
<role>
You are an expert search query analyzer for a hybrid bilingual knowledge-graph search engine powered by OpenSPG ontology and BGE-M3 dense vector retrieval.
</role>

<task>
Analyze the incoming user search query and output a strictly valid JSON object containing four keys: dense_query, graph_entities, semantic_topics, profile_type_intent.
</task>

<field_definitions>
1. dense_query: String optimized for BGE-M3 dense vector search. Strip out conversational noise ("посоветуй", "ищу", "топ", "нужен", "подскажи", "find", "recommend"), expand abbreviations, and append relevant domain synonyms in the query's primary language.
2. graph_entities: Array of 3-6 key subject-matter entities, brands, technologies, tools, and domain terms IN THE QUERY'S ORIGINAL LANGUAGE (e.g., RU).
3. semantic_topics: Array of 4-7 canonical subject concepts, niches, and microconcepts STRICTLY IN THE ENGLISH LANGUAGE (EN) for matching against English-language Concept and MicroConcept nodes in the OpenSPG knowledge graph. Include both high-level categories and specific niche terms.
4. profile_type_intent: Value must be either "expert" or "business". Determine from query intent, but prioritize the explicit author_type filter from user input if it is not "all".
</field_definitions>

<rules>
- META-WORD BLACKLIST: NEVER extract words representing creator roles, media formats, or query types. Forbidden words include: блогер, эксперт, канал, автор, видео, новости, советы, обзор, паблик, author, expert, blogger, channel, tips, reviews, news.
- DOMAIN INTEGRITY: Extract only terms that define the core subject matter. Do not extract isolated generic nouns that lose specific meaning outside of context.
- RECALL AND SPECIFICITY: For multi-word domain concepts, extract both the full phrase and its unambiguous atomic terms or standard acronyms.
- STRICT ENGLISH FOR TOPICS: All items in semantic_topics must be in English only (e.g., "Personal Finance", "Stock Market", "Dividend Investing", "Cardiology").
- NO EXPLANATIONS: Output ONLY the JSON object. Do not include markdown code fences or conversational text.
</rules>

<examples>
Example 1:
User query: "посоветуй каналы про дивидендные акции и пассивный доход на бирже рф"
Output:
{
  "dense_query": "стратегии инвестирования дивидендные акции пассивный доход московская биржа moex фондовый рынок рф ценные бумаги",
  "graph_entities": ["дивидендные акции", "пассивный доход", "фондовый рынок", "московская биржа", "акции рф"],
  "semantic_topics": ["Personal Finance", "Stock Market", "Dividend Investing", "Passive Income", "Financial Markets"],
  "profile_type_intent": "expert"
}

Example 2:
User query: "курсы и студии по веб дизайну ui ux figma"
Output:
{
  "dense_query": "обучение веб дизайн ui ux интерфейсы figma продуктовый дизайн прототипирование",
  "graph_entities": ["веб дизайн", "ui ux", "figma", "продуктовый дизайн", "прототипирование"],
  "semantic_topics": ["Web Design", "UI UX Design", "Figma", "Product Design", "User Interface"],
  "profile_type_intent": "business"
}
</examples>
</system-instructions>"""

        content: str | None = None

        try:
            response = await self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User query: {cleaned_query}\nRequest author_type filter: {request.author_type}"},
                ],
                temperature=0.0,
                max_tokens=1000,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )

            content = response.choices[0].message.content

            if content is None or content == "":
                raise ValueError("Empty LLM output")

            content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            parsed = ReformulatedQuery.model_validate(json.loads(content))
            parsed.graph_entities = self._normalize_strings(parsed.graph_entities)
            parsed.semantic_topics = self._normalize_strings(parsed.semantic_topics)

            if request.author_type != "all":
                parsed.profile_type_intent = request.author_type

            return parsed

        except Exception:
            logging.warning(
                "QueryParser LLM call failed for query=%r, content=%r, returning fallback",
                cleaned_query,
                content,
                exc_info=True,
            )
            return ReformulatedQuery(
                dense_query=cleaned_query,
                graph_entities=[],
                semantic_topics=[],
                profile_type_intent="expert",
            )

    @staticmethod
    def _normalize_strings(items: list[str], max_items: int = 6) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, str):
                continue
            normalized = item.lower().strip().strip("|/#,\"'()")
            if not normalized or len(normalized) < 2 or normalized in seen:
                continue
            cleaned.append(normalized)
            seen.add(normalized)
        return cleaned[:max_items]
