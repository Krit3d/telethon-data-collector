from pydantic import BaseModel, Field, model_validator
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
    def normalize_fields(cls, data: dict) -> dict:
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
    target_topics: list[str] = Field(default_factory=list, description="Extracted broad topic names matching query intent")
    target_iab_ids: list[str] = Field(default_factory=list, description="IAB category string codes (e.g., 'DSS1V', 'IAB18-1') for taxonomy matching")
    profile_type_intent: str = Field(default="expert", description="Inferred target profile type: expert or business")


class QueryMetadata(BaseModel):
    original_query: str = Field(description="Original user search query")
    dense_query: str = Field(description="Reformulated dense query used for embedding search")
    graph_entities: list[str] = Field(default_factory=list, description="Entity names used for graph traversal")
    target_iab_ids: list[str] = Field(default_factory=list, description="IAB category string codes (e.g., 'DSS1V', 'IAB18-1') for taxonomy matching")
    resolved_profile_type: str = Field(description="Final resolved profile type after normalization")
    execution_time_ms: float = Field(description="Total query execution time in milliseconds")
    timings: dict[str, float] = Field(default_factory=dict, description="Phase-level execution timing breakdown in milliseconds")
    qdrant_candidates_count: int | None = Field(default=None, description="Number of candidate posts retrieved from Qdrant vector search")
    graph_candidates_count: int | None = Field(default=None, description="Number of candidate posts retrieved from graph index search")
    total_unique_candidates_count: int | None = Field(default=None, description="Total unique candidate posts after merging Qdrant and graph results")


class AuthorSearchResultItem(BaseModel):
    account_id: int = Field(description="Unique account identifier")
    platform: str = Field(description="Platform name (e.g. telegram, instagram)")
    username: str | None = Field(default=None, description="Account username")
    title: str = Field(description="Account display title or name")
    url: str | None = Field(default=None, description="Account profile URL")
    final_score: float = Field(description="Aggregated final relevance score")
    vector_score: float | None = Field(default=None, description="Dense vector similarity score")
    graph_score: float | None = Field(default=None, description="Graph traversal relevance score")
    tms_score: float | None = Field(default=None, description="Taxonomy metadata score")
    static_avg_er: float | None = Field(default=None, description="Static average engagement rate")
    category_path: str | None = Field(default=None, description="IAB category path string")
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
