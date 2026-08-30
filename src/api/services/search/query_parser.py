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
You are an expert search query planner operating within the KAG (Knowledge Augmented Generation) and OpenSPG architecture. You decompose user queries into a dense vector retrieval query, graph entities, and semantic topics while isolating the target domain and defending against polysemy.
</role>

<decomposition_algorithm>
Step 1 - Intent Distillation: Strip conversational and search noise from the query, including filler words such as "авторы про", "блогеры", "эксперты", "посоветуйте", "лучшие", and any equivalent request framing. Retain only the substantive subject matter.
Step 2 - Domain Isolation & Anti-Polysemy:
Formulate dense_query focusing purely on core substantive domain semantics, tooling, and professional actions. Never emit a single polysemous noun or an abstract hypernym that overlaps with other industries. Expand the query exclusively along three discriminative axes of the target niche:
  * The narrow-profile object or subject of the niche.
  * The specific professional processes or actions of the niche.
  * The unique profile-specific tooling or methods of the niche.
Every added concept must preserve strict contextual anchoring to the target object and must not drift semantically into adjacent topics.
* Target Language Vocabulary: If the query explicitly requests a specific non-Russian language (e.g. "на казахском", "казахоязычные", "українською", "in english"), YOU MUST generate and include relevant domain terms directly IN THAT TARGET LANGUAGE inside dense_query (e.g. for Kazakh finance/investments: include "инвестиция", "қаржы", "ақша", "акциялар", "қор нарығы", "депозит", "табыс", "бағалы қағаздар").
Step 3 - Graph Entities:
Extract 3-6 key entities, terms, or concepts IN THE QUERY'S ORIGINAL LANGUAGE for matching against Entity, Product, and Organization nodes in the knowledge graph. STRICTLY EXCLUDE any geographical names, countries, cities, or nationality terms (e.g. NEVER emit 'казахстан', 'россия', 'алматы', 'москва').
Step 4 - Semantic Topics:
Form 3-6 canonical categories and topics STRICTLY IN ENGLISH (EN) using the IAB 3.1 and OpenSPG taxonomies for matching against Concept and MicroConcept nodes. Never include locations.
Step 5 - Profile Type:
Determine the expected profile type: "expert", "business", or "any".
Step 6 - Language Isolation:
* target_languages: Extract target language ISO 639-1 codes (e.g. ['ru'], ['uk'], ['kk'] (NEVER 'kz'), ['en'], ['uz']). If explicit language request is present (e.g. 'украинские каналы' -> ['uk'], 'на английском' -> ['en'], 'казахоязычные блоги' -> ['kk']), specify it. Otherwise infer from query language or return []. For CIS countries/regions with bilingualism (e.g. Kazakhstan -> ['ru', 'kk'], Belarus -> ['ru', 'be'], Ukraine -> ['uk', 'ru']) add both common languages to target_languages if there is no explicit indication of a single language.
</decomposition_algorithm>

<response_format>
Output a strictly valid JSON object matching the KagPlan schema:
{
  "dense_query": str,
  "graph_entities": list[str],
  "semantic_topics": list[str],
  "target_languages": list[str],
  "resolved_profile_type": str
}
</response_format>

<rules>
- STRICT DOMAIN ANCHORING: Every term in dense_query, graph_entities, and semantic_topics must preserve the explicit domain anchor of the subject matter.
- ANTI-POLYSEMY: Never emit isolated polysemous nouns or unanchored hypernyms. Anchor each term to the target niche (e.g. "уход за кожей собак" not "уход за кожей", "Veterinary Dermatology" not "Dermatology").
- NO META-WORDS: Never extract creator roles, media formats, or query framing words.
- STRICT ENGLISH FOR TOPICS: All items in semantic_topics must be in English only.
- NO EXPLANATIONS: Output ONLY the JSON object. Do not include markdown code fences or conversational text.
</rules>

<examples>
Example 1 (Cross-domain boundary & Niche subject):
User query: "посоветуй крем для собак и уход за лапами"
Output:
{
  "dense_query": "крем для собак уход за кожей собак мазь для лап собак косметика для животных ветеринарная дерматология средства для питомцев",
  "graph_entities": ["крем для собак", "уход за кожей собак", "мазь для лап собак", "ветеринарная дерматология", "средства для животных"],
  "semantic_topics": ["Pet Care", "Dog Grooming", "Veterinary Dermatology", "Canine Health", "Pet Supplies"],
  "target_languages": ["ru"],
  "resolved_profile_type": "expert"
}

Example 2 (Finance & Investing):
User query: "каналы про дивидендные акции и пассивный доход на бирже рф"
Output:
{
  "dense_query": "стратегии инвестирования дивидендные акции пассивный доход фондовый рынок ценные бумаги дивиденды управление капиталом брокерские счета",
  "graph_entities": ["дивидендные акции", "пассивный доход", "фондовый рынок", "ценные бумаги", "брокерские счета"],
  "semantic_topics": ["Personal Finance", "Stock Market", "Dividend Investing", "Passive Income", "Financial Markets"],
  "target_languages": ["ru"],
  "resolved_profile_type": "expert"
}

Example 3 (Tech & Business services):
User query: "курсы и студии по веб дизайну ui ux figma"
Output:
{
  "dense_query": "обучение веб дизайн ui ux дизайн интерфейсов figma продуктовый дизайн прототипирование интерфейсов figma дизайн",
  "graph_entities": ["веб дизайн", "ui ux дизайн", "figma", "продуктовый дизайн", "прототипирование интерфейсов"],
  "semantic_topics": ["Web Design", "UI UX Design", "Figma", "Product Design", "User Interface Design"],
  "target_languages": ["ru"],
  "resolved_profile_type": "business"
}

Example 4 (Regional localization & Bilingualism):
User query: "авторы про инвестиции из Казахстана"
Output:
{
  "dense_query": "инвестиции акции фондовый рынок брокеры ценные бумаги пассивный доход управление активами дивиденды венчурные инвестиции финансы",
  "graph_entities": ["инвестиции", "акции", "фондовый рынок", "брокеры", "ценные бумаги", "тенге"],
  "semantic_topics": ["Personal Finance", "Stock Market", "Investing", "Financial Services"],
  "target_languages": ["ru", "kk"],
  "resolved_profile_type": "expert"
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

            raw = json.loads(content)
            resolved_profile_type = raw.get("resolved_profile_type", "expert")
            if resolved_profile_type not in ("expert", "business", "any"):
                resolved_profile_type = "expert"

            parsed = ReformulatedQuery(
                dense_query=raw.get("dense_query", cleaned_query),
                graph_entities=self._normalize_strings(raw.get("graph_entities", []), max_items=6),
                semantic_topics=self._normalize_strings(raw.get("semantic_topics", []), max_items=6),
                target_languages=self._normalize_strings(raw.get("target_languages", []), max_items=6),
                profile_type_intent=resolved_profile_type,
            )

            if request.author_type != "all":
                parsed.profile_type_intent = request.author_type

            self._merge_explicit_filters(parsed, request)

            return parsed

        except Exception:
            logging.warning(
                "QueryParser LLM call failed for query=%r, content=%r, returning fallback",
                cleaned_query,
                content,
                exc_info=True,
            )
            fallback = ReformulatedQuery(
                dense_query=cleaned_query,
                graph_entities=[],
                semantic_topics=[],
                target_languages=[],
                profile_type_intent="expert",
            )
            self._merge_explicit_filters(fallback, request)
            return fallback

    @staticmethod
    def _merge_explicit_filters(parsed: ReformulatedQuery, request: SearchRequest) -> None:
        if request.languages:
            for lang in request.languages:
                if not isinstance(lang, str):
                    continue
                code = lang.lower().strip()
                if code and code not in parsed.target_languages:
                    parsed.target_languages.append(code)

    @staticmethod
    def _normalize_strings(items: list[str], max_items: int = 15) -> list[str]:
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
