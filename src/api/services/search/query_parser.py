import logging

from openai import AsyncOpenAI

from src.api.schemas import ReformulatedQuery, SearchRequest
from src.config.config import Settings


_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "are", "not", "but", "was", "had", "has",
    "its", "you", "all", "any", "can", "may", "use", "way", "new",
    "old", "big", "too", "yet", "she", "his", "her", "our", "who",
    "which", "what", "how", "why", "this", "that", "these", "those",
    "from", "they", "them", "their", "with", "have", "been", "being",
    "some", "such", "than", "then", "also", "just", "about", "into",
    "over", "after", "very", "will", "would", "could", "should",
    "there", "where", "when", "here", "each", "both", "more", "most",
    "other", "only", "same", "like", "because", "between", "before",
    "through", "during", "without", "under", "again", "while",
    "и", "в", "на", "с", "к", "у", "о", "об", "от", "по",
    "за", "для", "из", "без", "над", "под", "при", "про", "через",
    "а", "но", "да", "или", "что", "как", "это", "тот", "так",
    "все", "его", "ее", "их", "нам", "вас", "кто", "где", "тут",
    "там", "уже", "еще", "был", "была", "было", "были", "будет",
    "будут", "меня", "тебя", "себя", "него", "нее", "них",
    "чтобы", "когда", "потом", "тогда", "теперь", "также", "можно",
    "нужно", "надо", "очень", "совсем", "опять", "снова", "пока",
    "ли", "же", "бы", "не", "ни", "этот", "эта", "эти", "весь",
    "вся", "один", "одна", "одно", "два", "три", "четыре", "пять",
    "шесть", "семь", "восемь", "девять", "десять",
    "который", "которая", "которые", "которого", "которому",
    "которым", "которых", "которыми",
    "el", "la", "le", "les", "des", "une", "du", "en", "au",
    "aux", "ce", "ces", "son", "sa", "ses", "leur", "leurs",
    "der", "die", "das", "den", "dem", "ein", "eine", "einen",
    "einer", "einem", "eines", "sie", "es", "ihr", "ihre", "sein",
    "seine", "uns", "euch", "sich", "diese", "dieser", "dieses",
    "diesen", "diesem", "nicht", "auch", "nach", "durch", "ohne",
})


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
            "Field Instructions:\n"
            "1. dense_query: Reformulate and enrich the query for vector semantic search. Expand acronyms and clarify technical domain context in the query's native language.\n"
            "2. graph_entities: Extract 3-7 core subject-matter entities representing Knowledge Graph nodes.\n"
            "ONTOLOGICAL EXTRACTION RULE:\n"
            "- EXTRACT: Domain entities, technologies, products, specific niche concepts, brands, platforms, proper nouns, and subject topics that define WHAT the content is about.\n"
            "- OMIT: Meta-search modifiers, action verbs, and structural phrasing that define HOW the user is searching (e.g., requests for reviews, experiences, comparisons, or general seeking phrasing) UNLESS those words form an integral part of a recognized standard, proper term, or subject domain itself.\n"
            "- RECALL & SPECIFICITY RULE: For multi-word domain concepts, extract BOTH the full multi-word phrase AND its primary unambiguous atomic domain word or acronym (e.g., for 'когнитивно-поведенческая терапия' include BOTH 'когнитивно-поведенческая терапия' and 'кпт'; for 'профессиональное выгорание' include BOTH 'профессиональное выгорание' and 'выгорание').\n"
            "- PLATFORM & CONTAINER RULE (CRITICAL): Never extract standalone single words that represent media platforms (e.g., 'telegram', 'instagram'), content formats/containers (e.g., 'channels', 'posts', 'videos', 'blogs'), or ultra-generic process words (e.g., 'ways', 'methods', 'results', 'cases'). These words MUST ONLY exist as part of compound entities (e.g., 'telegram-каналы', 'экспертные каналы', 'контент-воронки'). Standalone single words are allowed ONLY if they denote a highly specific, narrow domain topic (e.g., 'тревожность', 'имплантация', 'криптография')."
            "3. target_topics: Extract 2-4 standard IAB 3.1 category string names in English matching query intent.\n"
            "4. profile_type_intent: Output 'expert' or 'business' based on query context."
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
