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
Build dense_query for dense vector retrieval (BGE-M3). Never emit a single polysemous noun or an abstract hypernym that overlaps with other industries. Expand the query exclusively along three discriminative axes of the target niche:
  * The narrow-profile object or subject of the niche.
  * The specific professional processes or actions of the niche.
  * The unique profile-specific tooling or methods of the niche.
Every added concept must preserve strict contextual anchoring to the target object and must not drift semantically into adjacent topics.
Step 3 - Graph Entities:
Extract 3-6 key entities, terms, or concepts IN THE QUERY'S ORIGINAL LANGUAGE for matching against Entity, Product, and Organization nodes in the knowledge graph.
Step 4 - Semantic Topics:
Form 3-6 canonical categories and topics STRICTLY IN ENGLISH (EN) using the IAB 3.1 and OpenSPG taxonomies for matching against Concept and MicroConcept nodes.
Step 5 - Profile Type:
Determine the expected profile type: "expert", "business", or "any".
</decomposition_algorithm>

<response_format>
Output a strictly valid JSON object matching the KagPlan schema:
{
  "dense_query": str,
  "graph_entities": list[str],
  "semantic_topics": list[str],
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
  "resolved_profile_type": "expert"
}

Example 2 (Finance & Investing):
User query: "каналы про дивидендные акции и пассивный доход на бирже рф"
Output:
{
  "dense_query": "стратегии инвестирования дивидендные акции рф пассивный доход московская биржа moex фондовый рынок рф ценные бумаги дивиденды",
  "graph_entities": ["дивидендные акции", "пассивный доход", "фондовый рынок рф", "московская биржа", "акции рф"],
  "semantic_topics": ["Personal Finance", "Stock Market", "Dividend Investing", "Passive Income", "Financial Markets"],
  "resolved_profile_type": "expert"
}

Example 3 (Tech & Business services):
User query: "курсы и студии по веб дизайну ui ux figma"
Output:
{
  "dense_query": "обучение веб дизайн ui ux дизайн интерфейсов figma продуктовый дизайн прототипирование интерфейсов figma дизайн",
  "graph_entities": ["веб дизайн", "ui ux дизайн", "figma", "продуктовый дизайн", "прототипирование интерфейсов"],
  "semantic_topics": ["Web Design", "UI UX Design", "Figma", "Product Design", "User Interface Design"],
  "resolved_profile_type": "business"
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
                graph_entities=self._normalize_strings(raw.get("graph_entities", [])),
                semantic_topics=self._normalize_strings(raw.get("semantic_topics", [])),
                profile_type_intent=resolved_profile_type,
            )

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
