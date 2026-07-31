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
            "You are an expert search query reformulator. Analyze the user query and extract key parameters. "
            "Return a valid JSON object matching this schema: "
            '{"dense_query": "string", "graph_entities": ["string"], "target_topics": ["string"], "profile_type_intent": "expert|business"}. '
            "Instructions: "
            "1. dense_query: Expand query terms, resolve technical abbreviations (e.g. RAG -> retrieval augmented generation, LLM -> large language models), and enrich semantic context strictly in the native language of the query. "
            "Do not introduce forced translations to other languages unless the user query explicitly uses them. "
            "2. graph_entities: MANDATORY KEYRULE: Extract 5-10 lowercase entities. You MUST include both ATOMIC SINGLE WORDS (e.g., 'rag', 'llm', 'финтех', 'продакшен') AND short 2-word composite key terms (e.g., 'rag системы', 'локальные llm'). Single atomic words are mandatory to match isolated graph nodes extracted by LLMs. "
            "Strip special punctuation characters | / # , \" ' ( ) from extracted entities. "
            "3. target_topics: MANDATORY. Extract 2-4 standard IAB 3.1 category string names in English matching query intent (e.g., ['Personal Finance', 'Personal Investing', 'Stocks and Bonds', 'Financial Planning']). NEVER return an empty array for target_topics if query has identifiable intent. "
            "4. profile_type_intent: 'expert' or 'business'."
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
            normalized = entity.lower().strip().translate(str.maketrans("", "", strip_chars))
            if not normalized or normalized in seen:
                continue
            cleaned.append(normalized)
            seen.add(normalized)
            if " " in normalized:
                for token in normalized.split():
                    token_stripped = token.strip().translate(str.maketrans("", "", strip_chars))
                    if (
                        token_stripped
                        and len(token_stripped) > 2
                        and token_stripped not in _STOPWORDS
                        and token_stripped not in seen
                    ):
                        cleaned.append(token_stripped)
                        seen.add(token_stripped)
        return cleaned[:10]
