import json
import logging
from string import Template

from openai import AsyncOpenAI

from src.api.schemas import AudienceCluster, BrandAnalysisRequest, BrandAnalysisResponse, InferredFilters, ReformulatedQuery, SearchPlanRequest, SearchPlanResponse, SearchRequest
from src.config.config import Settings
from src.graph.ontology import HormoneType, ToneType

_ALLOWED_PROFILE_TYPES = frozenset({"expert", "business", "any"})

_SYSTEM_PROMPT_TEMPLATE = Template("""<system-instructions>
<role>
You are an expert search query planner operating within the KAG (Knowledge Augmented Generation) and OpenSPG architecture. You decompose user queries and campaign briefs into a dense vector retrieval query, graph entities, semantic topics, affinity audience signals, negative constraints, psychographic targets, and inferred UI filters while isolating the target domain and defending against polysemy.
</role>

<decomposition_algorithm>
Step 1 - Direct Intent:
Formulate dense_query focusing purely on core substantive domain semantics, tooling, and professional actions. Never emit a single polysemous noun or an abstract hypernym that overlaps with other industries. Expand the query exclusively along three discriminative axes of the target niche:
  * The narrow-profile object or subject of the niche.
  * The specific professional processes or actions of the niche.
  * The unique profile-specific tooling or methods of the niche.
Every added concept must preserve strict contextual anchoring to the target object and must not drift semantically into adjacent topics.
* Target Language Vocabulary: If the query explicitly requests a specific non-Russian language (e.g. "на казахском", "казахоязычные", "українською", "in english"), YOU MUST generate and include relevant domain terms directly IN THAT TARGET LANGUAGE inside dense_query (e.g. for Kazakh finance/investments: include "инвестиция", "қаржы", "ақша", "акциялар", "қор нарығы", "депозит", "табыс", "бағалы қағаздар").
Extract 3-6 graph_entities: key entities, terms, or concepts IN THE QUERY'S ORIGINAL LANGUAGE for matching against Entity, Product, and Organization nodes in the knowledge graph. STRICTLY EXCLUDE any geographical names, countries, cities, or nationality terms (e.g. NEVER emit 'казахстан', 'россия', 'алматы', 'москва').
Form 3-6 semantic_topics: canonical categories and topics STRICTLY IN ENGLISH (EN) using the IAB 3.1 and OpenSPG taxonomies for matching against Concept and MicroConcept nodes. Never include locations.

Step 2 - Affinity Intent:
Formulate affinity_dense_query as a vector search query over adjacent spheres, interests, and lifestyle areas of the target audience (e.g. related hobbies, adjacent professional fields, complementary consumption contexts). Form 3-6 affinity_topics STRICTLY IN ENGLISH (EN) using IAB/OpenSPG categories for concept expansion. Provide affinity_reason as a short Russian-language niche name of the adjacent sphere: STRICTLY 2 to 4 words, no more than 35 characters, without introductory sentences or reasoning (e.g. "IT и разработка ПО", "Мужской спорт", "Автомобили и тюнинг").
CRITICAL AFFINITY RULE: affinity_dense_query MUST NOT contain core keywords or synonyms of the direct niche. It must target strictly adjacent lifestyle domains where the same audience spends time (e.g. for skincare/cosmetics -> wellness, nutrition, women fitness, pilates, fashion styling; for barbershops -> martial arts, crossfit, mens fashion, auto tuning).

Step 3 - Negative Constraints:
Collect negative_topics and negative_entities from the query and the brief stop_topics. These are topics and entities that must be excluded from results. Include any explicitly forbidden subjects, competitor brands, or off-brand themes.

Step 4 - Psychographics:
Determine target_tone from one of: $tones.
Determine target_hormones as an array of up to 2 hormones from: $hormones.

Step 5 - Profile Type:
Determine the expected profile type: "expert", "business", or "any".

Step 6 - Language Isolation:
* target_languages: Extract target language ISO 639-1 codes (e.g. ['ru'], ['uk'], ['kk'] (NEVER 'kz'), ['en'], ['uz']). If explicit language request is present (e.g. 'украинские каналы' -> ['uk'], 'на английском' -> ['en'], 'казахоязычные блоги' -> ['kk']), specify it. Otherwise infer from query language or return []. For CIS countries/regions with bilingualism (e.g. Kazakhstan -> ['ru', 'kk'], Belarus -> ['ru', 'be'], Ukraine -> ['uk', 'ru']) add both common languages to target_languages if there is no explicit indication of a single language.

Step 7 - Inferred Filters:
Form the inferred_filters object with fields country, languages, min_followers, max_followers, target_tone, target_hormones. Infer these values from the query and brief context. country is a country or region name, languages are ISO 639-1 codes, min_followers and max_followers are positive integers. Use null for unknown values.
</decomposition_algorithm>

<response_format>
Output a strictly valid JSON object matching the KagPlan schema:
{
  "dense_query": str,
  "graph_entities": list[str],
  "semantic_topics": list[str],
  "affinity_dense_query": str | None,
  "affinity_topics": list[str],
  "affinity_reason": str | None,
  "audience_clusters": [
    {
      "name": str,
      "dense_query": str,
      "semantic_topics": list[str]
    }
  ],
  "negative_topics": list[str],
  "negative_entities": list[str],
  "target_tone": str | None,
  "target_hormones": list[str],
  "target_languages": list[str],
  "resolved_profile_type": str,
  "inferred_filters": {
    "country": str | None,
    "languages": list[str] | None,
    "min_followers": int | None,
    "max_followers": int | None,
    "target_tone": str | None,
    "target_hormones": list[str] | None
  }
}
</response_format>

<rules>
- STRICT DOMAIN ANCHORING: Every term in dense_query, graph_entities, and semantic_topics must preserve the explicit domain anchor of the subject matter.
- ANTI-POLYSEMY: Never emit isolated polysemous nouns or unanchored hypernyms. Anchor each term to the target niche (e.g. "уход за кожей собак" not "уход за кожей", "Veterinary Dermatology" not "Dermatology").
- NO META-WORDS: Never extract creator roles, media formats, or query framing words.
- STRICT ENGLISH FOR TOPICS: All items in semantic_topics and affinity_topics must be in English only.
- NO EXPLANATIONS: Output ONLY the JSON object. Do not include markdown code fences or conversational text.
- AFFINITY_REASON FORMAT: affinity_reason must be a short Russian-language niche name of the adjacent sphere, strictly 2 to 4 words and no more than 35 characters, without introductory sentences or reasoning.
</rules>

<examples>
Example 1 (Cross-domain boundary & Niche subject):
User query: "посоветуй крем для собак и уход за лапами"
Output:
{
  "dense_query": "крем для собак уход за кожей собак мазь для лап собак косметика для животных ветеринарная дерматология средства для питомцев",
  "graph_entities": ["крем для собак", "уход за кожей собак", "мазь для лап собак", "ветеринарная дерматология", "средства для животных"],
  "semantic_topics": ["Pet Care", "Dog Grooming", "Veterinary Dermatology", "Canine Health", "Pet Supplies"],
  "affinity_dense_query": "зоомагазины груминг салоны ветеринарные клиники корма для животных дрессировка собак",
  "affinity_topics": ["Pet Food", "Pet Services", "Animal Welfare", "Dog Training"],
  "affinity_reason": "Зоотовары и груминг",
  "negative_topics": [],
  "negative_entities": [],
  "target_tone": "educational",
  "target_hormones": ["oxytocin"],
  "target_languages": ["ru"],
  "resolved_profile_type": "expert",
  "inferred_filters": {
    "country": null,
    "languages": ["ru"],
    "min_followers": null,
    "max_followers": null,
    "target_tone": "educational",
    "target_hormones": ["oxytocin"]
  }
}

Example 2 (Finance & Investing):
User query: "каналы про дивидендные акции и пассивный доход на бирже рф"
Output:
{
  "dense_query": "стратегии инвестирования дивидендные акции пассивный доход фондовый рынок ценные бумаги дивиденды управление капиталом брокерские счета",
  "graph_entities": ["дивидендные акции", "пассивный доход", "фондовый рынок", "ценные бумаги", "брокерские счета"],
  "semantic_topics": ["Personal Finance", "Stock Market", "Dividend Investing", "Passive Income", "Financial Markets"],
  "affinity_dense_query": "личные финансы накопления пенсионные планы недвижимость как инвестиция налоговая оптимизация",
  "affinity_topics": ["Real Estate", "Retirement Planning", "Tax Planning", "Wealth Management"],
  "affinity_reason": "Недвижимость и накопления",
  "negative_topics": ["криптовалюты", "форекс"],
  "negative_entities": [],
  "target_tone": "expert",
  "target_hormones": ["dopamine"],
  "target_languages": ["ru"],
  "resolved_profile_type": "expert",
  "inferred_filters": {
    "country": "Россия",
    "languages": ["ru"],
    "min_followers": null,
    "max_followers": null,
    "target_tone": "expert",
    "target_hormones": ["dopamine"]
  }
}

Example 3 (Tech & Business services):
User query: "курсы и студии по веб дизайну ui ux figma"
Output:
{
  "dense_query": "обучение веб дизайн ui ux дизайн интерфейсов figma продуктовый дизайн прототипирование интерфейсов figma дизайн",
  "graph_entities": ["веб дизайн", "ui ux дизайн", "figma", "продуктовый дизайн", "прототипирование интерфейсов"],
  "semantic_topics": ["Web Design", "UI UX Design", "Figma", "Product Design", "User Interface Design"],
  "affinity_dense_query": "графический дизайн брендинг типографика моушн дизайн разработка сайтов",
  "affinity_topics": ["Graphic Design", "Branding", "Motion Design", "Web Development"],
  "affinity_reason": "Брендинг и разработка",
  "negative_topics": [],
  "negative_entities": [],
  "target_tone": "educational",
  "target_hormones": ["dopamine"],
  "target_languages": ["ru"],
  "resolved_profile_type": "business",
  "inferred_filters": {
    "country": null,
    "languages": ["ru"],
    "min_followers": null,
    "max_followers": null,
    "target_tone": "educational",
    "target_hormones": ["dopamine"]
  }
}

Example 4 (Regional localization & Bilingualism):
User query: "авторы про инвестиции из Казахстана"
Output:
{
  "dense_query": "инвестиции акции фондовый рынок брокеры ценные бумаги пассивный доход управление активами дивиденды венчурные инвестиции финансы",
  "graph_entities": ["инвестиции", "акции", "фондовый рынок", "брокеры", "ценные бумаги", "тенге"],
  "semantic_topics": ["Personal Finance", "Stock Market", "Investing", "Financial Services"],
  "affinity_dense_query": "личные финансы накопления предпринимательство малый бизнес недвижимость",
  "affinity_topics": ["Small Business", "Real Estate", "Entrepreneurship", "Wealth Management"],
  "affinity_reason": "Бизнес и недвижимость",
  "negative_topics": [],
  "negative_entities": [],
  "target_tone": "expert",
  "target_hormones": ["dopamine"],
  "target_languages": ["ru", "kk"],
  "resolved_profile_type": "expert",
  "inferred_filters": {
    "country": "Казахстан",
    "languages": ["ru", "kk"],
    "min_followers": null,
    "max_followers": null,
    "target_tone": "expert",
    "target_hormones": ["dopamine"]
  }
}
</examples>
</system-instructions>""")

_SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.substitute(
    tones=" | ".join(member.value for member in ToneType),
    hormones=", ".join(member.value for member in HormoneType),
)

_PLAN_SYSTEM_PROMPT_TEMPLATE = Template("""<system-instructions>
<role>
You are an expert campaign brief planner. You decompose a campaign brief into a concise natural search query for the UI search bar, a full precomputed query plan for direct execution, adjacent audience rationale, and a complete set of UI filters.
</role>

<decomposition_algorithm>
Step 1 - Search Query:
Formulate search_query as a concise natural search query for the search bar of 2 to 5 words in the language of the brief. It must capture the core product, brand, or niche without meta-words, creator roles, or media formats.

Step 2 - Direct Intent:
Formulate dense_query focusing purely on core substantive domain semantics, tooling, and professional actions. Never emit a single polysemous noun or an abstract hypernym that overlaps with other industries. Expand the query exclusively along three discriminative axes of the target niche: the narrow-profile object or subject of the niche, the specific professional processes or actions of the niche, and the unique profile-specific tooling or methods of the niche.
Extract 3-6 graph_entities: key entities, terms, or concepts IN THE QUERY'S ORIGINAL LANGUAGE for matching against Entity, Product, and Organization nodes in the knowledge graph. STRICTLY EXCLUDE any geographical names, countries, cities, or nationality terms.
Form 3-6 semantic_topics: canonical categories and topics STRICTLY IN ENGLISH (EN) using the IAB 3.1 and OpenSPG taxonomies for matching against Concept and MicroConcept nodes. Never include locations.

Step 3 - Affinity Audience:
Formulate affinity_dense_query as a vector search query over adjacent spheres, interests, and lifestyle areas of the target audience. Form 3-6 affinity_topics STRICTLY IN ENGLISH (EN) using IAB/OpenSPG categories for concept expansion. Provide affinity_reason as a short Russian-language niche name of the adjacent sphere: strictly 2 to 4 words, no more than 35 characters, without introductory sentences or reasoning.
CRITICAL AFFINITY RULE: affinity_dense_query MUST NOT contain core keywords or synonyms of the direct niche. It must target strictly adjacent lifestyle domains where the same audience spends time.

Step 3b - Audience Clusters:
Formulate 3-4 audience_clusters representing DIFFERENT, INDEPENDENT spheres of interest and lifestyle of the same target audience. Each cluster must cover a distinct domain (e.g. fitness, finance, travel, parenting, tech, fashion, food, education) so that the final search results are diverse and not overlapping. For each cluster provide name (short Russian-language cluster name of 2-4 words), dense_query (expanded dense vector query in Russian for finding authors of this cluster), and semantic_topics (3-6 IAB categories STRICTLY IN ENGLISH). CRITICAL DIVERSITY RULE: clusters must not duplicate each other's core topics and must not duplicate the direct niche.

Step 4 - Negative Constraints:
Collect negative_topics and negative_entities from the brief stop_topics. These are topics and entities that must be excluded from results. Include any explicitly forbidden subjects, competitor brands, or off-brand themes.

Step 5 - Psychographics:
Determine target_tone from one of: $tones.
Determine target_hormones as an array of up to 2 hormones from: $hormones.

Step 6 - Language Isolation:
Extract target_languages as ISO 639-1 codes (e.g. ['ru'], ['uk'], ['kk'] (NEVER 'kz'), ['en'], ['uz']). If explicit language request is present, specify it. Otherwise infer from brief language or return [].

Step 7 - Inferred Filters:
Form the inferred_filters object with fields country, languages, min_followers, max_followers, target_tone, target_hormones, search_query, stop_topics. Infer these values from the brief. country is a country or region name, languages are ISO 639-1 codes, min_followers and max_followers are positive integers. Use null for unknown values.
</decomposition_algorithm>

<response_format>
Output a strictly valid JSON object matching the Plan schema:
{
  "dense_query": str,
  "graph_entities": list[str],
  "semantic_topics": list[str],
  "affinity_dense_query": str | None,
  "affinity_topics": list[str],
  "affinity_reason": str | None,
  "audience_clusters": [
    {
      "name": str,
      "dense_query": str,
      "semantic_topics": list[str]
    }
  ],
  "negative_topics": list[str],
  "negative_entities": list[str],
  "target_tone": str | None,
  "target_hormones": list[str],
  "target_languages": list[str],
  "search_query": str,
  "inferred_filters": {
    "country": str | None,
    "languages": list[str] | None,
    "min_followers": int | None,
    "max_followers": int | None,
    "target_tone": str | None,
    "target_hormones": list[str] | None,
    "search_query": str | None,
    "stop_topics": list[str]
  }
}
</response_format>

<rules>
- NO EXPLANATIONS: Output ONLY the JSON object. Do not include markdown code fences or conversational text.
- SEARCH_QUERY LENGTH: search_query must be 2 to 5 words.
- AFFINITY_REASON FORMAT: affinity_reason must be a short Russian-language niche name of the adjacent sphere, strictly 2 to 4 words and no more than 35 characters.
- STOP TOPICS: Preserve all stop topics from the brief in negative_topics, negative_entities, and inferred_filters.stop_topics.
- STRICT ENGLISH FOR TOPICS: All items in semantic_topics and affinity_topics must be in English only.
</rules>
</system-instructions>""")

_PLAN_SYSTEM_PROMPT = _PLAN_SYSTEM_PROMPT_TEMPLATE.substitute(
    tones=" | ".join(member.value for member in ToneType),
    hormones=", ".join(member.value for member in HormoneType),
)

_BRAND_ANALYSIS_SYSTEM_PROMPT_TEMPLATE = Template("""<system-instructions>
<role>
You are an expert brand audience analyst operating within the KAG (Knowledge Augmented Generation) and OpenSPG architecture. You analyze a brand description and produce a structured audience portrait: a coherent target audience description, a direct product niche cluster, 3-4 diverse lifestyle and interest clusters, psychographic targets, inferred UI filters, and a concise search query.
</role>

<ontology_rule>
STRICT TARGET AUDIENCE ONTOLOGY: target_audience_description MUST describe exclusively the END CLIENTS, BUYERS, AND CONSUMERS of the brand (the people who purchase and use the product or service). It MUST NEVER describe bloggers, influencers, or experts of the same profession as the brand. For example:
- For an auto service / STO: the target audience is car owners and drivers, men aged 25-50.
- For a children's brand: the target audience is parents and mothers.
- For a fitness studio: the target audience is office workers and people who want to lose weight.
The audience is defined by who BUYS the product, not by who creates content about it.
CRITICAL ONTOLOGICAL DISTINCTION: audience_clusters are NOT the target audience itself. They are the SPHERES OF INTEREST AND LIFESTYLE of the END CLIENTS, BUYERS, AND CONSUMERS of the brand (the people who buy the product or service). For example, for an auto service / STO the target audience is car owners and drivers (men aged 25-50), and their audience_clusters are their interest spheres: sports and martial arts, auto tourism and travel, fishing and active recreation, gadgets and technologies. For a children's brand the target audience is parents and mothers, and their audience_clusters are their interest spheres: parenting and family, home and interior, children's education, family travel. NEVER treat audience_clusters as the target audience itself and NEVER describe bloggers or experts of the same profession as the brand in either target_audience_description or audience_clusters.
</ontology_rule>

<decomposition_algorithm>
Step 1 - Audience Portrait:
Analyze the brand description and determine the target audience demographics: gender, age range, income level, lifestyle, and psychographics. Formulate target_audience_description as a detailed, coherent, and precise Russian-language portrait of the END CLIENTS, BUYERS, AND CONSUMERS of the brand (2-4 sentences) suitable for user approval. It must describe who the audience is, what they value, how they spend time and money, and which content they consume. NEVER describe bloggers or experts of the same profession as the brand.

Step 2 - Direct Cluster:
Formulate direct_cluster as the narrow product niche of the brand. name must be a short Russian-language cluster name of 2-4 words. dense_query must be an expanded dense vector query in Russian for finding authors of this niche, anchored to the core product semantics, professional processes, and niche-specific tooling. Form 3-6 semantic_topics STRICTLY IN ENGLISH (EN) using the IAB 3.1 and OpenSPG taxonomies.

Step 3 - Audience Clusters:
Formulate 3-4 audience_clusters representing DIFFERENT, INDEPENDENT spheres of interest and lifestyle of the END CLIENTS, BUYERS, AND CONSUMERS of the brand (the people who buy the product or service). Each cluster must cover a distinct domain where the brand can buy advertising to attract buyers (e.g. for car owners: men's sports and martial arts, auto tourism and travel, fishing and active recreation, gadgets and technologies; for parents: parenting and family, home and interior, children's education, family travel; for office workers: fitness and healthy lifestyle, personal finance, self-development, travel). For each cluster provide name (short Russian-language cluster name of 2-4 words), dense_query (expanded dense vector query in Russian for finding authors of this cluster), and semantic_topics (3-6 IAB categories STRICTLY IN ENGLISH). CRITICAL DIVERSITY RULE: clusters must not duplicate each other's core topics and must not duplicate the direct niche. CRITICAL ONTOLOGICAL RULE: audience_clusters are the interest spheres of the END CLIENTS, BUYERS, AND CONSUMERS of the brand, NOT the target audience itself and NOT bloggers or experts of the same profession as the brand.

Step 4 - Psychographics:
Determine target_tone from one of: $tones.
Determine target_hormones as an array of up to 2 hormones from: $hormones.

Step 5 - Inferred Filters:
Form the inferred_filters object with fields country, languages, min_followers, max_followers, target_tone, target_hormones, stop_topics. country MUST be a strict 2-letter ISO 3166-1 alpha-2 code in lowercase (e.g. "kz", "ru", "by", "uz", "ae", "us") or null if unknown. languages are ISO 639-1 codes (e.g. ['ru'], ['uk'], ['kk'] (NEVER 'kz'), ['en']). min_followers and max_followers are positive integers or null. stop_topics must preserve all stop topics from the request.

Step 6 - Suggested Query:
Formulate suggested_query as a concise natural search query of 2-4 words in Russian capturing the core product or brand niche without meta-words, creator roles, or media formats.
</decomposition_algorithm>

<response_format>
Output a strictly valid JSON object matching the BrandAnalysis schema:
{
  "target_audience_description": str,
  "direct_cluster": {
    "name": str,
    "dense_query": str,
    "semantic_topics": list[str]
  },
  "audience_clusters": [
    {
      "name": str,
      "dense_query": str,
      "semantic_topics": list[str]
    }
  ],
  "target_tone": str | null,
  "target_hormones": list[str],
  "inferred_filters": {
    "country": str | null,
    "languages": list[str] | null,
    "min_followers": int | null,
    "max_followers": int | null,
    "target_tone": str | null,
    "target_hormones": list[str] | null,
    "stop_topics": list[str]
  },
  "suggested_query": str
}
</response_format>

<rules>
- NO EXPLANATIONS: Output ONLY the JSON object. Do not include markdown code fences or conversational text.
- CLUSTER NAME FORMAT: Every cluster name must be a short Russian-language phrase of 2-4 words.
- STRICT ENGLISH FOR TOPICS: All items in semantic_topics must be in English only.
- CLUSTER DIVERSITY: audience_clusters must contain 3-4 distinct, non-overlapping lifestyle and interest spheres of the brand's end clients.
- COUNTRY FORMAT: inferred_filters.country must be a strict 2-letter lowercase ISO 3166-1 alpha-2 code or null.
- STOP TOPICS: Preserve all stop topics from the request in inferred_filters.stop_topics.
- SUGGESTED_QUERY LENGTH: suggested_query must be 2-4 words.
</rules>
</system-instructions>""")

_BRAND_ANALYSIS_SYSTEM_PROMPT = _BRAND_ANALYSIS_SYSTEM_PROMPT_TEMPLATE.substitute(
    tones=" | ".join(member.value for member in ToneType),
    hormones=", ".join(member.value for member in HormoneType),
)


class QueryParser:

    def __init__(self, settings: Settings) -> None:
        self._llm_client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=30.0,
        )
        self._llm_model = settings.deepseek_llm_model

    async def parse(self, request: SearchRequest) -> ReformulatedQuery:
        if request.precomputed_plan is not None:
            plan = request.precomputed_plan.model_copy(deep=True)
            if request.direct_cluster is not None:
                plan.direct_cluster = request.direct_cluster
            if request.audience_clusters:
                plan.audience_clusters = list(request.audience_clusters)
            if request.target_tone is not None:
                plan.target_tone = request.target_tone
                if plan.inferred_filters is not None:
                    plan.inferred_filters.target_tone = request.target_tone
            if request.target_hormones:
                for hormone in request.target_hormones:
                    if hormone not in plan.target_hormones:
                        plan.target_hormones.append(hormone)
                if plan.inferred_filters is not None:
                    plan.inferred_filters.target_hormones = list(plan.target_hormones)
            if request.stop_topics:
                for topic in request.stop_topics:
                    normalized = topic.strip()
                    if normalized and normalized not in plan.negative_topics:
                        plan.negative_topics.append(normalized)
                if plan.inferred_filters is not None:
                    for topic in request.stop_topics:
                        normalized = topic.strip()
                        if normalized and normalized not in plan.inferred_filters.stop_topics:
                            plan.inferred_filters.stop_topics.append(normalized)
            self._merge_explicit_filters(plan, request)
            return plan

        input_text = self._build_input_text(request)

        if not input_text:
            return ReformulatedQuery(dense_query="", profile_type_intent="expert")

        content: str | None = None

        try:
            response = await self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": input_text},
                ],
                temperature=0.0,
                max_tokens=1500,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )

            content = response.choices[0].message.content

            if content is None or content == "":
                raise ValueError("Empty LLM output")

            content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            raw = json.loads(content)
            if not isinstance(raw, dict):
                raise ValueError("LLM output is not a JSON object")

            return self._build_reformulated_query(raw, request)

        except Exception:
            logging.warning(
                "QueryParser LLM call failed for query=%r, content=%r, returning fallback",
                request.query,
                content,
                exc_info=True,
            )
            return self._build_fallback(request)

    async def plan(self, brief: SearchPlanRequest) -> SearchPlanResponse:
        input_text = self._build_plan_input_text(brief)

        content: str | None = None

        try:
            response = await self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": _PLAN_SYSTEM_PROMPT},
                    {"role": "user", "content": input_text},
                ],
                temperature=0.0,
                max_tokens=1200,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )

            content = response.choices[0].message.content

            if content is None or content == "":
                raise ValueError("Empty LLM output")

            content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            raw = json.loads(content)
            if not isinstance(raw, dict):
                raise ValueError("LLM output is not a JSON object")

            return self._build_plan_response(raw, brief)

        except Exception:
            logging.warning(
                "QueryParser plan LLM call failed for brief=%r, content=%r, returning fallback",
                brief.campaign_description,
                content,
                exc_info=True,
            )
            return self._build_plan_fallback(brief)

    async def analyze_brand(self, request: BrandAnalysisRequest) -> BrandAnalysisResponse:
        input_text = self._build_brand_analysis_input_text(request)

        content: str | None = None

        try:
            response = await self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": _BRAND_ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": input_text},
                ],
                temperature=0.0,
                max_tokens=2000,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )

            content = response.choices[0].message.content

            if content is None or content == "":
                raise ValueError("Empty LLM output")

            content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            raw = json.loads(content)
            if not isinstance(raw, dict):
                raise ValueError("LLM output is not a JSON object")

            return self._build_brand_analysis_response(raw, request)

        except Exception:
            logging.warning(
                "QueryParser brand analysis LLM call failed for brand=%r, content=%r, returning fallback",
                request.brand_description,
                content,
                exc_info=True,
            )
            return self._build_brand_analysis_fallback(request)

    @staticmethod
    def _build_brand_analysis_input_text(request: BrandAnalysisRequest) -> str:
        lines: list[str] = []
        description = request.brand_description.strip()
        if description:
            lines.append(f"Brand description: {description}")
        stop_topics = QueryParser._brief_stop_topics(request)
        if stop_topics:
            lines.append(f"Stop topics: {', '.join(stop_topics)}")
        return "\n".join(lines)

    def _build_brand_analysis_response(self, raw: dict[str, object], request: BrandAnalysisRequest) -> BrandAnalysisResponse:
        target_audience_description = _normalize_text(raw.get("target_audience_description"))
        if target_audience_description is None:
            target_audience_description = request.brand_description.strip()

        direct_cluster = self._build_audience_cluster(
            raw.get("direct_cluster"),
            fallback_name="Прямая ниша бренда",
            fallback_query=request.brand_description.strip(),
        )

        audience_clusters = self._build_audience_clusters(raw.get("audience_clusters"), request)

        target_tone = _normalize_tone(raw.get("target_tone"))
        target_hormones = _normalize_hormones(raw.get("target_hormones"))

        filters_block = raw.get("inferred_filters")
        if not isinstance(filters_block, dict):
            filters_block = {}

        stop_topics = _normalize_strings(filters_block.get("stop_topics", []), max_items=15)
        for topic in self._brief_stop_topics(request):
            normalized = topic.strip()
            if normalized and normalized not in stop_topics:
                stop_topics.append(normalized)

        country = _normalize_country(filters_block.get("country"))
        languages = _normalize_strings(filters_block.get("languages", []), max_items=6) or None

        min_followers = filters_block.get("min_followers")
        if not isinstance(min_followers, int) or min_followers <= 0:
            min_followers = None

        max_followers = filters_block.get("max_followers")
        if not isinstance(max_followers, int) or max_followers <= 0:
            max_followers = None

        inferred_tone = _normalize_tone(filters_block.get("target_tone"))
        if target_tone is not None:
            inferred_tone = target_tone

        inferred_hormones = _normalize_hormones(filters_block.get("target_hormones"))
        if target_hormones:
            inferred_hormones = list(target_hormones)

        inferred_filters = InferredFilters(
            country=country,
            languages=languages,
            min_followers=min_followers,
            max_followers=max_followers,
            target_tone=inferred_tone,
            target_hormones=inferred_hormones,
            stop_topics=stop_topics,
        )

        suggested_query = _normalize_text(raw.get("suggested_query"))
        if suggested_query is None:
            suggested_query = direct_cluster.name

        return BrandAnalysisResponse(
            target_audience_description=target_audience_description,
            direct_cluster=direct_cluster,
            audience_clusters=audience_clusters,
            inferred_filters=inferred_filters,
            suggested_query=suggested_query,
        )

    @staticmethod
    def _build_audience_cluster(raw_block: object, fallback_name: str, fallback_query: str) -> AudienceCluster:
        block = raw_block if isinstance(raw_block, dict) else {}
        name = _normalize_text(block.get("name"))
        if name is None:
            name = fallback_name
        dense_query = _normalize_text(block.get("dense_query"))
        if dense_query is None:
            dense_query = fallback_query
        semantic_topics = _normalize_strings(block.get("semantic_topics", []), max_items=6)
        return AudienceCluster(
            name=name,
            dense_query=dense_query,
            semantic_topics=semantic_topics,
        )

    def _build_audience_clusters(self, raw_clusters: object, request: BrandAnalysisRequest) -> list[AudienceCluster]:
        if not isinstance(raw_clusters, list):
            return []
        clusters: list[AudienceCluster] = []
        seen_names: set[str] = set()
        for item in raw_clusters:
            if len(clusters) >= 4:
                break
            cluster = self._build_audience_cluster(
                item,
                fallback_name="Кластер аудитории",
                fallback_query=request.brand_description.strip(),
            )
            if cluster.name in seen_names:
                continue
            seen_names.add(cluster.name)
            clusters.append(cluster)
        return clusters

    @staticmethod
    def _build_audience_clusters_from_plan(raw_clusters: object) -> list[AudienceCluster]:
        if not isinstance(raw_clusters, list):
            return []
        clusters: list[AudienceCluster] = []
        seen_names: set[str] = set()
        for item in raw_clusters:
            if len(clusters) >= 4:
                break
            block = item if isinstance(item, dict) else {}
            name = _normalize_text(block.get("name"))
            if name is None or name in seen_names:
                continue
            dense_query = _normalize_text(block.get("dense_query"))
            if dense_query is None:
                continue
            semantic_topics = _normalize_strings(block.get("semantic_topics", []), max_items=6)
            seen_names.add(name)
            clusters.append(
                AudienceCluster(
                    name=name,
                    dense_query=dense_query,
                    semantic_topics=semantic_topics,
                )
            )
        return clusters

    @staticmethod
    def _build_brand_analysis_fallback(request: BrandAnalysisRequest) -> BrandAnalysisResponse:
        description = request.brand_description.strip()
        stop_topics = QueryParser._brief_stop_topics(request)
        direct_cluster = AudienceCluster(
            name="Прямая ниша бренда",
            dense_query=description,
            semantic_topics=[],
        )
        audience_clusters = [
            AudienceCluster(
                name="Стиль жизни аудитории",
                dense_query=description,
                semantic_topics=[],
            ),
            AudienceCluster(
                name="Интересы аудитории",
                dense_query=description,
                semantic_topics=[],
            ),
            AudienceCluster(
                name="Потребление контента",
                dense_query=description,
                semantic_topics=[],
            ),
        ]
        inferred_filters = InferredFilters(
            stop_topics=stop_topics,
        )
        return BrandAnalysisResponse(
            target_audience_description=description,
            direct_cluster=direct_cluster,
            audience_clusters=audience_clusters,
            inferred_filters=inferred_filters,
            suggested_query=description[:50],
        )

    @staticmethod
    def _brief_stop_topics(brief: SearchPlanRequest | BrandAnalysisRequest) -> list[str]:
        stop_topics = brief.stop_topics
        if isinstance(stop_topics, str):
            return [part.strip() for part in stop_topics.replace("\n", ",").split(",") if part.strip()]
        if isinstance(stop_topics, list):
            return [item.strip() for item in stop_topics if isinstance(item, str) and item.strip()]
        return []

    @staticmethod
    def _build_plan_input_text(brief: SearchPlanRequest) -> str:
        lines: list[str] = []
        description = brief.campaign_description.strip()
        if description:
            lines.append(f"Campaign description: {description}")
        stop_topics = QueryParser._brief_stop_topics(brief)
        if stop_topics:
            lines.append(f"Stop topics: {', '.join(stop_topics)}")
        return "\n".join(lines)

    def _build_plan_response(self, raw: dict[str, object], brief: SearchPlanRequest) -> SearchPlanResponse:
        search_query = _normalize_text(raw.get("search_query"))
        if search_query is None:
            search_query = brief.campaign_description.strip()[:50]

        target_tone = _normalize_tone(raw.get("target_tone"))
        target_hormones = _normalize_hormones(raw.get("target_hormones"))

        filters_block = raw.get("inferred_filters")
        if not isinstance(filters_block, dict):
            filters_block = {}

        stop_topics = _normalize_strings(raw.get("stop_topics", []), max_items=15)
        for topic in _normalize_strings(filters_block.get("stop_topics", []), max_items=15):
            if topic not in stop_topics:
                stop_topics.append(topic)
        for topic in self._brief_stop_topics(brief):
            normalized = topic.strip()
            if normalized and normalized not in stop_topics:
                stop_topics.append(normalized)

        country = _normalize_text(filters_block.get("country"))
        languages = _normalize_strings(filters_block.get("languages", []), max_items=6) or None

        min_followers = filters_block.get("min_followers")
        if not isinstance(min_followers, int) or min_followers <= 0:
            min_followers = None

        max_followers = filters_block.get("max_followers")
        if not isinstance(max_followers, int) or max_followers <= 0:
            max_followers = None

        inferred_filters = InferredFilters(
            country=country,
            languages=languages,
            min_followers=min_followers,
            max_followers=max_followers,
            target_tone=target_tone,
            target_hormones=target_hormones,
            search_query=search_query,
            stop_topics=stop_topics,
        )

        negative_topics = _normalize_strings(raw.get("negative_topics", []), max_items=15)
        for topic in self._brief_stop_topics(brief):
            normalized = topic.strip()
            if normalized and normalized not in negative_topics:
                negative_topics.append(normalized)

        precomputed_plan = ReformulatedQuery(
            dense_query=_normalize_text(raw.get("dense_query")) or search_query,
            graph_entities=_normalize_strings(raw.get("graph_entities", []), max_items=6),
            semantic_topics=_normalize_strings(raw.get("semantic_topics", []), max_items=6),
            target_languages=_normalize_strings(raw.get("target_languages", []), max_items=6),
            profile_type_intent="expert",
            affinity_dense_query=_normalize_text(raw.get("affinity_dense_query")),
            affinity_topics=_normalize_strings(raw.get("affinity_topics", []), max_items=6),
            affinity_reason=_normalize_text(raw.get("affinity_reason")),
            audience_clusters=self._build_audience_clusters_from_plan(raw.get("audience_clusters")),
            negative_topics=negative_topics,
            negative_entities=_normalize_strings(raw.get("negative_entities", []), max_items=15),
            target_tone=target_tone,
            target_hormones=target_hormones,
            inferred_filters=inferred_filters,
        )

        return SearchPlanResponse(
            search_query=search_query,
            inferred_filters=inferred_filters,
            affinity_reason=_normalize_text(raw.get("affinity_reason")),
            precomputed_plan=precomputed_plan,
        )

    @staticmethod
    def _build_plan_fallback(brief: SearchPlanRequest) -> SearchPlanResponse:
        search_query = brief.campaign_description.strip()[:50]
        stop_topics = QueryParser._brief_stop_topics(brief)
        inferred_filters = InferredFilters(
            search_query=search_query,
            stop_topics=stop_topics,
        )
        precomputed_plan = ReformulatedQuery(
            dense_query=search_query,
            graph_entities=[],
            semantic_topics=[],
            target_languages=[],
            profile_type_intent="expert",
            affinity_dense_query=None,
            affinity_topics=[],
            affinity_reason=None,
            negative_topics=list(stop_topics),
            negative_entities=[],
            target_tone=None,
            target_hormones=[],
            inferred_filters=inferred_filters,
        )
        return SearchPlanResponse(
            search_query=search_query,
            inferred_filters=inferred_filters,
            affinity_reason=None,
            precomputed_plan=precomputed_plan,
        )

    @staticmethod
    def _build_input_text(request: SearchRequest) -> str:
        lines: list[str] = []
        query = request.query.strip()
        if query:
            lines.append(f"User query: {query}")
        if request.brief is not None:
            if request.brief.brand_product_description:
                lines.append(f"Brand product description: {request.brief.brand_product_description}")
            if request.brief.target_audience:
                lines.append(f"Target audience: {request.brief.target_audience}")
            if request.brief.stop_topics:
                lines.append(f"Stop topics: {', '.join(request.brief.stop_topics)}")
        if lines:
            lines.append(f"Request author_type filter: {request.author_type}")
        return "\n".join(lines)

    def _build_reformulated_query(self, raw: dict[str, object], request: SearchRequest) -> ReformulatedQuery:
        resolved_profile_type = raw.get("resolved_profile_type", "expert")
        if not isinstance(resolved_profile_type, str) or resolved_profile_type not in _ALLOWED_PROFILE_TYPES:
            resolved_profile_type = "expert"

        target_tone = _normalize_tone(raw.get("target_tone"))
        target_hormones = _normalize_hormones(raw.get("target_hormones"))

        parsed = ReformulatedQuery(
            dense_query=_normalize_text(raw.get("dense_query")) or request.query.strip(),
            graph_entities=_normalize_strings(raw.get("graph_entities", []), max_items=6),
            semantic_topics=_normalize_strings(raw.get("semantic_topics", []), max_items=6),
            target_languages=_normalize_strings(raw.get("target_languages", []), max_items=6),
            profile_type_intent=resolved_profile_type,
            affinity_dense_query=_normalize_text(raw.get("affinity_dense_query")),
            affinity_topics=_normalize_strings(raw.get("affinity_topics", []), max_items=6),
            affinity_reason=_normalize_text(raw.get("affinity_reason")),
            direct_cluster=request.direct_cluster,
            audience_clusters=request.audience_clusters if request.audience_clusters else self._build_audience_clusters_from_plan(raw.get("audience_clusters")),
            negative_topics=_normalize_strings(raw.get("negative_topics", []), max_items=15),
            negative_entities=_normalize_strings(raw.get("negative_entities", []), max_items=15),
            target_tone=target_tone,
            target_hormones=target_hormones,
            inferred_filters=self._build_inferred_filters(raw.get("inferred_filters"), request, target_tone, target_hormones),
        )

        if request.author_type != "all":
            parsed.profile_type_intent = request.author_type

        self._merge_explicit_filters(parsed, request)

        return parsed

    def _build_fallback(self, request: SearchRequest) -> ReformulatedQuery:
        fallback = ReformulatedQuery(
            dense_query=request.query.strip(),
            graph_entities=[],
            semantic_topics=[],
            target_languages=[],
            profile_type_intent="expert",
            affinity_dense_query=None,
            affinity_topics=[],
            affinity_reason=None,
            negative_topics=[],
            negative_entities=[],
            target_tone=None,
            target_hormones=[],
            inferred_filters=self._build_inferred_filters(None, request, None, []),
        )

        if request.author_type != "all":
            fallback.profile_type_intent = request.author_type

        self._merge_explicit_filters(fallback, request)

        return fallback

    @staticmethod
    def _build_inferred_filters(raw_block: object, request: SearchRequest, target_tone: ToneType | None, target_hormones: list[HormoneType]) -> InferredFilters:
        block = raw_block if isinstance(raw_block, dict) else {}

        country = _normalize_text(block.get("country"))
        if request.location:
            country = request.location

        languages = _normalize_strings(block.get("languages", []), max_items=6) or None
        if request.languages:
            languages = list(request.languages)

        min_followers = block.get("min_followers")
        if not isinstance(min_followers, int) or min_followers <= 0:
            min_followers = None
        if request.min_followers is not None:
            min_followers = request.min_followers

        max_followers = block.get("max_followers")
        if not isinstance(max_followers, int) or max_followers <= 0:
            max_followers = None
        if request.max_followers is not None:
            max_followers = request.max_followers

        inferred_tone = _normalize_tone(block.get("target_tone"))
        if target_tone is not None:
            inferred_tone = target_tone

        inferred_hormones = _normalize_hormones(block.get("target_hormones"))
        if target_hormones:
            inferred_hormones = list(target_hormones)

        search_query = _normalize_text(block.get("search_query"))
        if search_query is None:
            search_query = request.query.strip()

        stop_topics = _normalize_strings(block.get("stop_topics", []), max_items=15)
        for topic in request.stop_topics:
            normalized = topic.strip()
            if normalized and normalized not in stop_topics:
                stop_topics.append(normalized)

        return InferredFilters(
            country=country,
            languages=languages,
            min_followers=min_followers,
            max_followers=max_followers,
            target_tone=inferred_tone,
            target_hormones=inferred_hormones,
            search_query=search_query,
            stop_topics=stop_topics,
        )

    @staticmethod
    def _merge_explicit_filters(parsed: ReformulatedQuery, request: SearchRequest) -> None:
        if request.languages:
            for lang in request.languages:
                if not isinstance(lang, str):
                    continue
                code = lang.lower().strip()
                if code and code not in parsed.target_languages:
                    parsed.target_languages.append(code)

        if request.target_tone is not None:
            parsed.target_tone = request.target_tone
            if parsed.inferred_filters is not None:
                parsed.inferred_filters.target_tone = request.target_tone

        if request.target_hormones:
            for hormone in request.target_hormones:
                if hormone not in parsed.target_hormones:
                    parsed.target_hormones.append(hormone)
            if parsed.inferred_filters is not None:
                parsed.inferred_filters.target_hormones = list(parsed.target_hormones)

        if request.stop_topics:
            for topic in request.stop_topics:
                normalized = topic.strip()
                if normalized and normalized not in parsed.negative_topics:
                    parsed.negative_topics.append(normalized)
            if parsed.inferred_filters is not None:
                for topic in request.stop_topics:
                    normalized = topic.strip()
                    if normalized and normalized not in parsed.inferred_filters.stop_topics:
                        parsed.inferred_filters.stop_topics.append(normalized)

def _normalize_strings(items: object, max_items: int = 15) -> list[str]:
    if not isinstance(items, list):
        return []
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


def _normalize_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


_COUNTRY_ALIASES: dict[str, str] = {
    "казахстан": "kz",
    "kazakhstan": "kz",
    "kz": "kz",
    "россия": "ru",
    "russia": "ru",
    "рф": "ru",
    "ru": "ru",
    "беларусь": "by",
    "belarus": "by",
    "by": "by",
    "узбекистан": "uz",
    "uzbekistan": "uz",
    "uz": "uz",
    "оаэ": "ae",
    "uae": "ae",
    "эмираты": "ae",
    "ae": "ae",
    "сша": "us",
    "usa": "us",
    "америка": "us",
    "us": "us",
}


def _normalize_country(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    return _COUNTRY_ALIASES.get(normalized)


def _normalize_tone(value: object) -> ToneType | None:
    if isinstance(value, ToneType):
        return value
    if not isinstance(value, str):
        return None
    try:
        return ToneType(value.lower().strip())
    except ValueError:
        return None


def _normalize_hormones(items: object) -> list[HormoneType]:
    if not isinstance(items, list):
        return []
    normalized = _normalize_strings(items, max_items=2)
    hormones: list[HormoneType] = []
    for hormone in normalized:
        try:
            hormones.append(HormoneType(hormone))
        except ValueError:
            continue
    return hormones
