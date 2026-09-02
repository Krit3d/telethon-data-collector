from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.graph.ontology import HormoneType, PlatformType, ToneType


def coerce_platform(value: object) -> PlatformType:
    if isinstance(value, PlatformType):
        return value
    if isinstance(value, str):
        try:
            return PlatformType(value.strip().lower())
        except ValueError:
            pass
    raise ValueError(f"Unsupported platform value: {value!r}")


def coerce_tone(value: object) -> ToneType | None:
    if value is None or isinstance(value, ToneType):
        return value
    if isinstance(value, str):
        try:
            return ToneType(value.strip().lower())
        except ValueError:
            return None
    return None


def coerce_hormone(value: object) -> HormoneType | None:
    if value is None or isinstance(value, HormoneType):
        return value
    if isinstance(value, str):
        try:
            return HormoneType(value.strip().lower())
        except ValueError:
            return None
    return None


class InferredFilters(BaseModel):
    country: str | None = None
    languages: list[str] | None = None
    min_followers: int | None = None
    max_followers: int | None = None
    target_tone: ToneType | None = None
    target_hormones: list[HormoneType] = Field(default_factory=list)
    search_query: str | None = Field(default=None, description="Generated concise natural search query for UI search bar")
    stop_topics: list[str] = Field(default_factory=list, description="Extracted stop topics")


class SearchPlanRequest(BaseModel):
    campaign_description: str = Field(description="Answer to Question 1: Product/brand description, target audience demographics, segment, creator style")
    stop_topics: str | list[str] | None = Field(default=None, description="Answer to Question 2: Stop topics and exclusions")

    @model_validator(mode="before")
    @classmethod
    def normalize_stop_topics(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        stop_topics = data.get("stop_topics")
        if isinstance(stop_topics, str):
            normalized: list[str] = []
            for part in stop_topics.replace("\n", ",").split(","):
                stripped = part.strip()
                if stripped:
                    normalized.append(stripped)
            data["stop_topics"] = normalized
        elif isinstance(stop_topics, list):
            normalized = []
            for item in stop_topics:
                if isinstance(item, str):
                    stripped = item.strip()
                    if stripped:
                        normalized.append(stripped)
            data["stop_topics"] = normalized
        return data


class AudienceCluster(BaseModel):
    name: str = Field(description="Short Russian-language cluster name of 2-4 words")
    dense_query: str = Field(description="Expanded dense vector query for finding authors of this cluster")
    semantic_topics: list[str] = Field(default_factory=list, description="IAB topic categories in English")


class ReformulatedQuery(BaseModel):
    dense_query: str = Field(description="Dense semantic text query for embedding generation")
    graph_entities: list[str] = Field(default_factory=list, description="Target entity names for graph traversal")
    semantic_topics: list[str] = Field(default_factory=list, description="Extracted natural topic areas matching query intent")
    profile_type_intent: str = Field(default="expert", description="Inferred target profile type: expert or business")
    target_languages: list[str] = Field(default_factory=list, description="Target language codes (e.g. ru, uk, kz, en) extracted or inferred from query")
    affinity_dense_query: str | None = Field(default=None, description="Dense vector query for adjacent audience topics")
    affinity_topics: list[str] = Field(default_factory=list, description="Adjacent topic areas for concept expansion")
    affinity_reason: str | None = Field(default=None, description="Explanation of affinity audience rationale")
    direct_cluster: AudienceCluster | None = Field(default=None, description="Direct product niche cluster of the brand")
    audience_clusters: list[AudienceCluster] = Field(default_factory=list, description="Diverse lifestyle and interest clusters of the target audience")
    negative_topics: list[str] = Field(default_factory=list, description="Stop topics excluded from results")
    negative_entities: list[str] = Field(default_factory=list, description="Stop entities excluded from results")
    target_tone: ToneType | None = Field(default=None, description="Dominant creator tone identified by planner")
    target_hormones: list[HormoneType] = Field(default_factory=list, description="Target psychographic hormones")
    inferred_filters: InferredFilters | None = Field(default=None, description="AI-inferred UI filters")

    @field_validator("graph_entities", "semantic_topics", "target_languages", "affinity_topics", "audience_clusters", "negative_topics", "negative_entities", "target_hormones", mode="before")
    @classmethod
    def list_null_to_empty(cls, v: Any) -> Any:
        if v is None:
            return []
        return v

    @field_validator("profile_type_intent", mode="before")
    @classmethod
    def validate_profile_type(cls, v: Any) -> str:
        if v is None or v not in ("expert", "business"):
            return "expert"
        return v

    @field_validator("affinity_reason", mode="before")
    @classmethod
    def truncate_affinity_reason(cls, v: Any) -> Any:
        if not isinstance(v, str):
            return v
        stripped = v.strip()
        if len(stripped) <= 40:
            return stripped
        truncated = stripped[:40]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]
        return truncated.rstrip()


class SearchPlanResponse(BaseModel):
    search_query: str = Field(description="Generated clean search query for the search bar")
    inferred_filters: InferredFilters = Field(description="Complete set of UI filters inferred by AI")
    affinity_reason: str | None = Field(default=None, description="Inferred adjacent audience rationale")
    precomputed_plan: ReformulatedQuery = Field(description="Full precomputed query plan for bypass on execution")


class BrandAnalysisRequest(BaseModel):
    brand_description: str = Field(description="Brand description for audience analysis")
    stop_topics: str | list[str] | None = Field(default=None, description="Stop topics and exclusions")

    @model_validator(mode="before")
    @classmethod
    def normalize_stop_topics(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        stop_topics = data.get("stop_topics")
        if isinstance(stop_topics, str):
            normalized: list[str] = []
            for part in stop_topics.replace("\n", ",").split(","):
                stripped = part.strip()
                if stripped:
                    normalized.append(stripped)
            data["stop_topics"] = normalized
        elif isinstance(stop_topics, list):
            normalized = []
            for item in stop_topics:
                if isinstance(item, str):
                    stripped = item.strip()
                    if stripped:
                        normalized.append(stripped)
            data["stop_topics"] = normalized
        return data


class BrandAnalysisResponse(BaseModel):
    target_audience_description: str = Field(description="Detailed Russian-language portrait of the target audience for user approval")
    direct_cluster: AudienceCluster = Field(description="Direct product niche cluster of the brand")
    audience_clusters: list[AudienceCluster] = Field(default_factory=list, description="3-4 diverse lifestyle and interest clusters of the target audience")
    inferred_filters: InferredFilters = Field(description="Auto-inferred UI filters: country, languages, min_followers, max_followers, target_tone, target_hormones, stop_topics")
    suggested_query: str = Field(description="Concise search query of 2-4 words")


class BriefContext(BaseModel):
    brand_product_description: str | None = None
    target_audience: str | None = None
    stop_topics: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(description="User search query text")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results to return")
    location: str | None = Field(default="", description="Filter results by author location")
    min_followers: int | None = Field(default=None, description="Minimum follower count filter")
    max_followers: int | None = Field(default=None, description="Maximum follower count filter")
    author_type: str = Field(default="expert", description="Author type filter: expert, business, or all")
    platform: str = Field(default="all", description="Platform filter: all, telegram, instagram, youtube, tiktok, threads")
    include_contacts: bool = Field(default=False, description="Include contact details in response")
    include_analytics: bool = Field(default=True, description="Include analytics data in response")
    languages: list[str] | None = Field(default=[], description="Optional filter by ISO language codes (e.g. ['ru', 'uk', 'en'])")
    brief: BriefContext | None = Field(default=None, description="Structured campaign brief context")
    target_tone: ToneType | None = Field(default=None, description="Explicit creator tone filter")
    target_hormones: list[HormoneType] = Field(default_factory=list, description="Explicit psychographic hormone filters")
    stop_topics: list[str] = Field(default_factory=list, description="Explicit list of stop topics to exclude")
    precomputed_plan: ReformulatedQuery | None = Field(default=None, description="Precomputed plan to bypass redundant LLM calls")
    direct_cluster: AudienceCluster | None = Field(default=None, description="Direct product niche cluster of the brand")
    audience_clusters: list[AudienceCluster] = Field(default_factory=list, description="Diverse lifestyle and interest clusters of the target audience")

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        query = data.get("query")
        if isinstance(query, str):
            data["query"] = query.strip()
        location = data.get("location")
        if location is None or location == "" or (isinstance(location, str) and location.strip().lower() == "string"):
            data["location"] = None
        min_followers = data.get("min_followers")
        if min_followers is None or min_followers == "" or (isinstance(min_followers, (int, float)) and min_followers <= 0):
            data["min_followers"] = None
        max_followers = data.get("max_followers")
        if max_followers is None or max_followers == "" or (isinstance(max_followers, (int, float)) and max_followers <= 0):
            data["max_followers"] = None
        languages = data.get("languages")
        if languages is None or languages == []:
            data["languages"] = None
        elif isinstance(languages, list):
            normalized_languages = []
            for lang in languages:
                if isinstance(lang, str):
                    stripped = lang.strip().lower()
                    if stripped:
                        normalized_languages.append(stripped)
            data["languages"] = normalized_languages if normalized_languages else None
        author_type = data.get("author_type")
        if author_type is None or author_type == "" or (isinstance(author_type, str) and author_type.strip().lower() == "string"):
            data["author_type"] = "expert"
        elif isinstance(author_type, str):
            normalized = author_type.strip().lower()
            data["author_type"] = normalized if normalized in ("expert", "business", "all") else "expert"
        platform = data.get("platform")
        if platform is None or platform == "" or (isinstance(platform, str) and platform.strip().lower() == "string"):
            data["platform"] = "all"
        elif isinstance(platform, str):
            data["platform"] = platform.strip().lower()
        target_tone = data.get("target_tone")
        if target_tone is not None:
            data["target_tone"] = coerce_tone(target_tone)
        target_hormones = data.get("target_hormones")
        if target_hormones is not None:
            normalized_hormones: list[HormoneType] = []
            if isinstance(target_hormones, list):
                for hormone in target_hormones:
                    coerced = coerce_hormone(hormone)
                    if coerced is not None and coerced not in normalized_hormones:
                        normalized_hormones.append(coerced)
            data["target_hormones"] = normalized_hormones
        stop_topics = data.get("stop_topics")
        if stop_topics is not None:
            normalized_stop_topics: list[str] = []
            if isinstance(stop_topics, list):
                for topic in stop_topics:
                    if isinstance(topic, str):
                        stripped = topic.strip()
                        if stripped and stripped not in normalized_stop_topics:
                            normalized_stop_topics.append(stripped)
            data["stop_topics"] = normalized_stop_topics
        return data


class VectorPostHit(BaseModel):
    post_id: int = Field(description="Post identifier from Qdrant vector index")
    account_id: int = Field(description="Account identifier owning the post")
    score: float = Field(description="Vector similarity score from Qdrant search")
    published_at: int | None = Field(default=None, description="Unix timestamp of post publication date")


class AuthorVectorAggregate(BaseModel):
    account_id: int = Field(description="Account identifier for aggregated vector signals")
    post_scores: list[float] = Field(default_factory=list, description="List of individual post vector scores for this author")
    max_vector_score: float = Field(default=0.0, description="Maximum vector score among all matched posts")
    decay_vector_score: float = Field(default=0.0, description="Time-decayed aggregated vector score")
    matched_posts_count: int = Field(default=0, description="Number of posts matched for this author")


class GraphAuthorEvidence(BaseModel):
    account_id: int = Field(description="Account identifier for graph evidence")
    topic_coverage_weight: float = Field(default=0.0, description="Topic coverage weight computed as posts_count / 12.0")
    matched_concepts: list[str] = Field(default_factory=list, description="Matched Concept names from graph traversal")
    matched_microconcepts: list[str] = Field(default_factory=list, description="Matched MicroConcept names from graph traversal")
    total_topics_count: int = Field(default=0, description="Total number of topics associated with the author in graph")
    matched_topics_count: int = Field(default=0, description="Number of topics matched to the query for this author")
    matched_entities_count: int = Field(default=0, description="Number of graph entities matched for this author")
    direct_mentions_count: int = Field(default=0, description="Number of direct entity mentions in author posts")
    has_role_relation: bool = Field(default=False, description="Whether author has WORKS_AT role relations in graph")
    has_tech_relation: bool = Field(default=False, description="Whether author has USES_TECH technology relations in graph")
    is_creator: bool = Field(default=False, description="Whether author has PRODUCES creator relation in graph")
    is_promoter: bool = Field(default=False, description="Whether author has PRODUCES promoter relation in graph")
    is_spam_or_gambling: bool = Field(default=False, description="Whether author is flagged as spam or gambling content")
    raw_graph_score: float = Field(default=0.0, description="Raw graph traversal relevance score before normalization")
    location_name: str | None = Field(default=None, description="Raw location name from Actor node in Neo4j")
    primary_language: str | None = Field(default=None, description="Aggregated primary language from Actor node in Neo4j")
    primary_tone: str | None = Field(default=None, description="Primary tone from Neo4j graph")
    primary_hormone: str | None = Field(default=None, description="Primary hormone from Neo4j graph")
    secondary_tone: str | None = Field(default=None, description="Secondary tone from Neo4j graph")
    secondary_hormone: str | None = Field(default=None, description="Secondary hormone from Neo4j graph")
    has_negative_match: bool = Field(default=False, description="Whether author matches any negative stop-topic or stop-entity")


class DbsfScoredCandidate(BaseModel):
    account_id: int = Field(description="Account identifier for DBSF scored candidate")
    raw_vector_score: float = Field(description="Raw vector similarity score before normalization")
    raw_graph_score: float = Field(description="Raw graph traversal score before normalization")
    normalized_vector_score: float = Field(description="Normalized vector score after distribution scaling")
    normalized_graph_score: float = Field(description="Normalized graph score after distribution scaling")
    final_score: float = Field(description="Aggregated final relevance score after DBSF distributional ranking")
    match_type: str = Field(default="direct", description="Match classification: direct or affinity")
    affinity_reason: str | None = Field(default=None, description="Contextual reason for affinity match")


class HydratedAuthorRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    account_id: int = Field(description="Unique account identifier from database")
    platform: PlatformType = Field(description="Platform name (e.g. telegram, instagram)")
    username: str | None = Field(default=None, description="Account username or handle")
    title: str = Field(description="Account display title or name")
    category_path: str | None = Field(default=None, description="Full category path string")
    explanation: str | None = Field(default=None, description="Human-readable explanation of author relevance")
    static_avg_er: float | None = Field(default=None, description="Static average engagement rate for the author")
    subscribers_count: int | None = Field(default=None, description="Number of subscribers or followers")
    is_author_blog: bool = Field(description="Whether the account is flagged as an author blog")
    raw_metadata: dict[str, Any] | None = Field(default=None, description="Raw metadata dictionary from database")
    contacts: dict[str, Any] | None = Field(default=None, description="Contact information dictionary")
    has_contacts: bool = Field(default=False, description="Whether contact data is available for this author")
    profile_url: str | None = Field(default=None, description="Author profile URL on the platform")
    match_type: str = Field(default="direct", description="Match classification: direct or affinity")
    affinity_reason: str | None = Field(default=None, description="Contextual reason for affinity match")
    primary_tone: ToneType | None = Field(default=None, description="Primary tone from Neo4j graph")
    primary_hormone: HormoneType | None = Field(default=None, description="Primary hormone from Neo4j graph")

    @field_validator("platform", mode="before")
    @classmethod
    def _validate_platform(cls, v: object) -> PlatformType:
        return coerce_platform(v)

    @field_validator("primary_tone", mode="before")
    @classmethod
    def _validate_primary_tone(cls, v: object) -> ToneType | None:
        return coerce_tone(v)

    @field_validator("primary_hormone", mode="before")
    @classmethod
    def _validate_primary_hormone(cls, v: object) -> HormoneType | None:
        return coerce_hormone(v)


class QueryMetadata(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    original_query: str = Field(description="Original user search query")
    dense_query: str = Field(description="Reformulated dense query used for embedding search")
    graph_entities: list[str] = Field(default_factory=list, description="Entity names used for graph traversal")
    semantic_topics: list[str] = Field(default_factory=list, description="Natural topic areas used for concept activation")
    target_languages: list[str] = Field(default_factory=list, description="Extracted target languages")
    resolved_profile_type: str = Field(description="Final resolved profile type after normalization")
    execution_time_ms: float = Field(description="Total query execution time in milliseconds")
    timings: dict[str, float] = Field(default_factory=dict, description="Phase-level execution timing breakdown in milliseconds")
    qdrant_candidates_count: int | None = Field(default=None, description="Number of candidate posts retrieved from Qdrant vector search")
    graph_evidences_count: int | None = Field(default=None, description="Number of authors with graph evidence from Neo4j")
    total_candidates_count: int | None = Field(default=None, description="Total unique candidates after DBSF ranking")
    affinity_dense_query: str | None = None
    negative_topics: list[str] = Field(default_factory=list)


class AuthorSearchResultItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    account_id: int = Field(description="Unique account identifier")
    platform: PlatformType = Field(description="Platform name (e.g. telegram, instagram)")
    username: str | None = Field(default=None, description="Account username")
    title: str = Field(description="Account display title or name")
    url: str | None = Field(default=None, description="Account profile URL")
    final_score: float = Field(description="Aggregated final relevance score")
    vector_score: float | None = Field(default=None, description="Dense vector similarity score")
    graph_score: float | None = Field(default=None, description="Graph traversal relevance score")
    static_avg_er: float | None = Field(default=None, description="Static average engagement rate")
    category_path: str | None = Field(default=None, description="Category path string")
    explanation: str | None = Field(default=None, description="Human-readable explanation of scoring")
    contacts: dict[str, Any] | None = Field(default=None, description="Contact information dictionary")
    has_contacts: bool = Field(default=False, description="Whether contact data is available")
    subscribers_count: int | None = Field(default=None, description="Number of subscribers or followers")
    location: str | None = Field(default=None, description="Author location name from graph or profile")
    primary_language: str | None = Field(default=None, description="Author primary language code (e.g. ru, uk, en)")
    match_type: str = Field(default="direct", description="Match classification: direct or affinity")
    affinity_reason: str | None = Field(default=None, description="Contextual reason for affinity match")
    primary_tone: ToneType | None = Field(default=None, description="Primary tone from Neo4j graph")
    primary_hormone: HormoneType | None = Field(default=None, description="Primary hormone from Neo4j graph")

    @field_validator("platform", mode="before")
    @classmethod
    def _validate_platform(cls, v: object) -> PlatformType:
        return coerce_platform(v)

    @field_validator("primary_tone", mode="before")
    @classmethod
    def _validate_primary_tone(cls, v: object) -> ToneType | None:
        return coerce_tone(v)

    @field_validator("primary_hormone", mode="before")
    @classmethod
    def _validate_primary_hormone(cls, v: object) -> HormoneType | None:
        return coerce_hormone(v)


class SearchResponse(BaseModel):
    items: list[AuthorSearchResultItem] = Field(default_factory=list, description="List of search result items")
    total: int = Field(default=0, description="Total number of results found")
    query_metadata: QueryMetadata | None = Field(default=None, description="Metadata about the executed query")
    message: str | None = Field(default=None, description="Response status or informational message")
    confidence_level: str = Field(default="HIGH", description="Confidence level of returned search items: HIGH, LOW, or NONE")
    warning_message: str | None = Field(default=None, description="Human-readable notification message explaining result relevance or domain coverage")
    inferred_filters: InferredFilters | None = Field(default=None, description="AI-inferred UI filter values")
