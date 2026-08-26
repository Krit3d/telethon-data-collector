from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Any


class SearchRequest(BaseModel):
    query: str = Field(description="User search query text")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results to return")
    location: str | None = Field(default="", description="Filter results by author location")
    min_followers: int | None = Field(default=None, description="Minimum follower count filter")
    author_type: str = Field(default="expert", description="Author type filter: expert, business, or all")
    include_contacts: bool = Field(default=False, description="Include contact details in response")
    include_analytics: bool = Field(default=True, description="Include analytics data in response")

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
        if data.get("min_followers") == 0:
            data["min_followers"] = None
        author_type = data.get("author_type")
        if author_type is None or author_type == "" or (isinstance(author_type, str) and author_type.strip().lower() == "string"):
            data["author_type"] = "expert"
        elif isinstance(author_type, str):
            normalized = author_type.strip().lower()
            data["author_type"] = normalized if normalized in ("expert", "business", "all") else "expert"
        return data


class ReformulatedQuery(BaseModel):
    dense_query: str = Field(description="Dense semantic text query for embedding generation")
    graph_entities: list[str] = Field(default_factory=list, description="Target entity names for graph traversal")
    semantic_topics: list[str] = Field(default_factory=list, description="Extracted natural topic areas matching query intent")
    profile_type_intent: str = Field(default="expert", description="Inferred target profile type: expert or business")

    @field_validator("graph_entities", "semantic_topics", mode="before")
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


class DbsfScoredCandidate(BaseModel):
    account_id: int = Field(description="Account identifier for DBSF scored candidate")
    raw_vector_score: float = Field(description="Raw vector similarity score before normalization")
    raw_graph_score: float = Field(description="Raw graph traversal score before normalization")
    normalized_vector_score: float = Field(description="Normalized vector score after distribution scaling")
    normalized_graph_score: float = Field(description="Normalized graph score after distribution scaling")
    final_score: float = Field(description="Aggregated final relevance score after DBSF distributional ranking")


class HydratedAuthorRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    account_id: int = Field(description="Unique account identifier from database")
    platform: str = Field(description="Platform name (e.g. telegram, instagram)")
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


class QueryMetadata(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    original_query: str = Field(description="Original user search query")
    dense_query: str = Field(description="Reformulated dense query used for embedding search")
    graph_entities: list[str] = Field(default_factory=list, description="Entity names used for graph traversal")
    semantic_topics: list[str] = Field(default_factory=list, description="Natural topic areas used for concept activation")
    resolved_profile_type: str = Field(description="Final resolved profile type after normalization")
    execution_time_ms: float = Field(description="Total query execution time in milliseconds")
    timings: dict[str, float] = Field(default_factory=dict, description="Phase-level execution timing breakdown in milliseconds")
    qdrant_candidates_count: int | None = Field(default=None, description="Number of candidate posts retrieved from Qdrant vector search")
    graph_evidences_count: int | None = Field(default=None, description="Number of authors with graph evidence from Neo4j")
    total_candidates_count: int | None = Field(default=None, description="Total unique candidates after DBSF ranking")


class AuthorSearchResultItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    account_id: int = Field(description="Unique account identifier")
    platform: str = Field(description="Platform name (e.g. telegram, instagram)")
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


class SearchResponse(BaseModel):
    items: list[AuthorSearchResultItem] = Field(default_factory=list, description="List of search result items")
    total: int = Field(default=0, description="Total number of results found")
    query_metadata: QueryMetadata | None = Field(default=None, description="Metadata about the executed query")
    message: str | None = Field(default=None, description="Response status or informational message")
    confidence_level: str = Field(default="HIGH", description="Confidence level of returned search items: HIGH, LOW, or NONE")
    warning_message: str | None = Field(default=None, description="Human-readable notification message explaining result relevance or domain coverage")
