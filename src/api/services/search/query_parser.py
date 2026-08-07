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
            "You are an expert search query analyzer for a hybrid semantic knowledge-graph search engine. "
            "Analyze the user query and output a valid JSON object matching: "
            '{"dense_query": "string", "graph_entities": ["string"], "target_topics": ["string"], "profile_type_intent": "expert|business"}.\n\n'
            "FIELD INSTRUCTIONS:\n\n"
            "1. dense_query:\n"
            "- Enrich and reformulate the query for BGE-M3 vector search.\n"
            '- Remove conversational noise ("посоветуй", "ищу", "нужен").\n'
            "- Expand acronyms and add relevant domain synonyms in the query's native language.\n\n"
            "2. graph_entities: Extract 3-7 core subject-matter entities representing Knowledge Graph nodes.\n"
            "- CRITICAL META-WORD BLACKLIST: NEVER extract words denoting creator roles, search request types, or media formats (e.g., DO NOT extract \"автор\", \"авторы\", \"эксперт\", \"эксперты\", \"блог\", \"блогер\", \"канал\", \"видео\", \"новости\", \"советы\", \"обзор\", \"author\", \"expert\", \"blogger\", \"channel\").\n"
            "ONTOLOGICAL EXTRACTION RULE:\n"
            "- EXTRACT: Domain entities, technologies, products, specific niche concepts, brands, platforms, proper nouns, and subject topics that define WHAT the content is about.\n"
            "- OMIT: Meta-search modifiers, action verbs, and structural phrasing that define HOW the user is searching (e.g., requests for reviews, experiences, comparisons, or general seeking phrasing) UNLESS those words form an integral part of a recognized standard, proper term, or subject domain itself.\n"
            "- DOMAIN INTEGRITY & ATOMIC SAFEGUARD (CRITICAL): Never extract standalone generic nouns that lose their domain meaning or cause semantic shift when isolated (e.g., for 'крем для собак', NEVER extract standalone 'крем' or 'уход' because isolated 'крем' shifts domain to human cosmetics). Only extract standalone atomic words if they remain strictly unambiguous within the target domain context (e.g., 'собаки', 'ветеринария'). Compound concepts MUST retain their qualifying domain modifiers.\n"
            "- RECALL & SPECIFICITY RULE: For multi-word domain concepts, extract BOTH the full multi-word phrase AND its primary unambiguous atomic domain terms or acronyms (e.g., for 'когнитивно-поведенческая терапия' include BOTH 'когнитивно-поведенческая терапия' and 'кпт').\n"
            "- ONTOLOGICAL BOUNDARY RULE: Never extract standalone words that denote abstract workflow stages, execution phases, deployment environments (e.g., production/prod, staging), software lifecycles, media formats, or container types. Such terms MUST ONLY exist as part of compound entities.\n"
            "3. target_topics: Extract 2-4 standard IAB 3.1 category string names in English matching query intent.\n"
            "SUBJECT DOMAIN ANCHORING RULE (CRITICAL):\n"
            "- First identify the target subject entity of the query (e.g., animal/pet, automotive, fintech, human health, industrial equipment).\n"
            "- If the query specifies non-human subjects (e.g., pets, animals, dogs, cats, machinery), ALL target_topics MUST strictly belong to that target subject's industry (e.g., 'Pets', 'Pet Supplies', 'Veterinary Medicine').\n"
            "- NEVER assign human personal care, human cosmetics, human skin care, or human lifestyle IAB categories to products, care, or services intended for animals or non-human entities.\n"
            "4. profile_type_intent: Output 'expert' or 'business' based on query context."

            "FEW-SHOT EXAMPLES:\n\n"
            'Query: "Авторы про охоту"\n'
            "JSON:\n"
            "{\n"
            '"dense_query": "экспертные блоги охота охотничье хозяйство снаряжение для охоты промысел wildlife hunting",\n'
            '"graph_entities": ["охота", "охотничье хозяйство"],\n'
            '"target_iab_ids": [464, 132],\n'
            '"profile_type_intent": "expert"\n'
            "}\n\n"
            'Query: "Ремонт авто в Москве сервис"\n'
            "JSON:\n"
            "{\n"
            '"dense_query": "автосервис ремонт автомобилей техническое обслуживание автотехцентр москва",\n'
            '"graph_entities": ["ремонт авто", "автосервис", "техническое обслуживание"],\n'
            '"target_iab_ids": [58, 61],\n'
            '"profile_type_intent": "business"\n'
            "}\n\n"
            'Query: "собачий корма уход за шпицем"\n'
            "JSON:\n"
            "{\n"
            '"dense_query": "кормление собак уход за шпицем зоотовары ветеринария гигиена собак",\n'
            '"graph_entities": ["уход за шпицем", "корм для собак", "шпиц"],\n'
            '"target_iab_ids": [382, 383],\n'
            '"profile_type_intent": "expert"\n'
            "}"
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
            parsed.graph_entities = self._sanitize_entities(parsed.graph_entities)
            return parsed

        except Exception:
            logging.warning(
                "QueryParser LLM call failed for query=%r, returning fallback",
                cleaned_query,
            )
            return ReformulatedQuery(
                dense_query=cleaned_query,
                graph_entities=[],
                target_topics=[],
                target_iab_ids=[],
                profile_type_intent="expert",
            )

    @staticmethod
    def _sanitize_entities(entities: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        strip_chars = "|/#,\"'()"
        for entity in entities:
            if not isinstance(entity, str):
                continue
            normalized = entity.lower().strip().translate(str.maketrans("", "", strip_chars))
            if not normalized or len(normalized) < 2 or normalized in seen:
                continue
            cleaned.append(normalized)
            seen.add(normalized)
        return cleaned[:10]
