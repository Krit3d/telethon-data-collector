from datetime import datetime
from pydantic import BaseModel, Field, model_validator


class AuthorPostSnippet(BaseModel):
    post_id: int
    text: str
    published_at: datetime
    url: str | None = None
    engagement_rate: float


class AuthorSearchResultItem(BaseModel):
    author_id: int
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
    relevant_posts: list[AuthorPostSnippet]


class SearchRequest(BaseModel):
    query: str = Field(..., description="User's project or brand description")
    limit: int = Field(default=10, ge=1, le=50, description="Number of results")
    score_threshold: float = Field(default=0.20, description="Minimum match threshold")
    location: str | None = Field(default=None, description="Filter results by author location")
    min_followers: int | None = Field(default=None, description="Minimum subscriber count")

    @model_validator(mode="before")
    @classmethod
    def clean_empty_values(cls, data: dict) -> dict:
        if isinstance(data, dict):
            if data.get("location") == "":
                data["location"] = None
            if data.get("min_followers") == 0:
                data["min_followers"] = None
        return data


class SearchResponse(BaseModel):
    results: list[AuthorSearchResultItem]
