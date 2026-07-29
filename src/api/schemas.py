from pydantic import BaseModel, Field, model_validator
from typing import Any


class SearchRequest(BaseModel):
    query: str = Field(description="User search query text")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results to return")
    score_threshold: float = Field(default=0.20, description="Minimum relevance score threshold")
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
    target_iab_ids: list[int] = Field(default_factory=list, description="Matched IAB category numeric IDs")
    profile_type_intent: str = Field(default="expert", description="Inferred target profile type: expert, business, or all")


class QueryMetadata(BaseModel):
    original_query: str = Field(description="Original user search query")
    dense_query: str = Field(description="Reformulated dense query used for embedding search")
    graph_entities: list[str] = Field(default_factory=list, description="Entity names used for graph traversal")
    target_iab_ids: list[int] = Field(default_factory=list, description="IAB category IDs used for filtering")
    resolved_profile_type: str = Field(description="Final resolved profile type after normalization")
    execution_time_ms: float = Field(description="Total query execution time in milliseconds")


class AuthorSearchResultItem(BaseModel):
    account_id: int = Field(description="Unique account identifier")
    platform: str = Field(description="Platform name (e.g. telegram, instagram)")
    username: str | None = Field(default=None, description="Account username")
    title: str = Field(description="Account display title or name")
    url: str | None = Field(default=None, description="Account profile URL")
    final_score: float = Field(description="Aggregated final relevance score")
    vector_score: float = Field(description="Dense vector similarity score")
    graph_score: float = Field(description="Graph traversal relevance score")
    tms_score: float = Field(description="Temporal metadata score (recency/decay)")
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
