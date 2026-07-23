from pydantic import BaseModel, Field, model_validator
from typing import Any


class ReformulatedQuery(BaseModel):
    dense_query: str = Field(..., description="Dense semantic text query for embedding generation")
    lexical_queries: list[str] = Field(default_factory=list, description="Keywords and terms for fulltext/lexical search")
    graph_entities: list[str] = Field(default_factory=list, description="Target entity names/nodes for graph traversal")
    target_iab_ids: list[str] = Field(default_factory=list, description="List of matched IAB category numeric IDs")
    profile_type_intent: str = Field(default="expert", description="Inferred target profile type: 'expert', 'business', or 'both'")


class AuthorSearchResultItem(BaseModel):
    author_id: str
    username: str | None = None
    title: str
    description: str | None = None
    subscribers_count: int | None = None
    platform: str
    final_score: float
    vector_score: float
    graph_score: float
    avg_engagement_rate: float
    explanation: str
    category_path: str | None = None
    is_author_blog: bool | None = None
    contacts: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="User's project or brand description")
    limit: int = Field(default=10, ge=1, le=50, description="Number of results")
    score_threshold: float = Field(default=0.20, description="Minimum match threshold")
    location: str | None = Field(default="", description="Filter results by author location")
    min_followers: int | None = Field(default=None, description="Minimum subscriber count")
    author_type: str | None = Field(default="expert", description="Filter by author type: 'expert', 'business', or 'all'")

    @model_validator(mode="before")
    @classmethod
    def clean_empty_values(cls, data: dict) -> dict:
        if isinstance(data, dict):
            location = data.get("location")
            if location is None or location == "" or (isinstance(location, str) and location.lower() == "string"):
                data["location"] = None
            if data.get("min_followers") == 0:
                data["min_followers"] = None
            author_type = data.get("author_type")
            if author_type is None or author_type == "" or (isinstance(author_type, str) and author_type.lower() == "string"):
                data["author_type"] = "expert"
        return data


class SearchResponse(BaseModel):
    results: list[AuthorSearchResultItem]
    message: str | None = None
