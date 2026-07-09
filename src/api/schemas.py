from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field, model_validator


class SearchRequest(BaseModel):
    query: str = Field(..., description="Text for semantic search")
    limit: int = Field(default=10, ge=1, le=50, description="Number of results")
    score_threshold: float = Field(
        default=0.20, description="Minimum match threshold"
    )
    include_author_info: bool = Field(
        default=False,
        description="Include author (Actor) node details for found posts",
    )
    location: str | None = Field(
        default=None,
        description="Filter results by author location (country or city)",
    )
    min_followers: int | None = Field(
        default=None,
        description="Minimum subscriber count for the author account",
    )
    min_engagement_rate: float | None = Field(
        default=None,
        description="Minimum engagement rate for the post",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "AI content creators in tech",
                    "limit": 10,
                    "score_threshold": 0.20,
                    "include_author_info": False,
                    "location": "",
                    "min_followers": 0,
                    "min_engagement_rate": 0.0,
                }
            ]
        }
    }

    @model_validator(mode="before")
    @classmethod
    def clear_empty_fields(cls, data: dict[str, Any] | Any) -> dict[str, Any] | Any:
        if not isinstance(data, dict):
            return data
        if data.get("location") == "":
            data["location"] = None
        if data.get("min_followers") == 0:
            data["min_followers"] = None
        if data.get("min_engagement_rate") in (0, 0.0):
            data["min_engagement_rate"] = None
        return data


class SearchResultItem(BaseModel):
    post_id: int
    account_id: int
    text: str
    score: float
    vector_score: float
    graph_score: float
    er_score: float
    created_at: datetime
    url: str | None = None
    author_id: int | None = None
    author_name: str | None = None
    boosted: bool = False


class GraphEdge(BaseModel):
    """Represents a graph relationship (edge) between two entities."""

    source_id: str
    source_label: str
    source_name: str | None = None
    relation_type: str
    target_id: str
    target_label: str
    target_name: str | None = None


class GraphEntity(BaseModel):
    """Groups graph relationships by entity for easier frontend rendering."""

    entity_id: str
    entity_label: str
    entity_name: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    relationships: list[GraphEdge] = Field(default_factory=list)


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    graph_entities: list[GraphEntity] = Field(default_factory=list)


class IndexRequest(BaseModel):
    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of posts to index",
    )


class IndexResponse(BaseModel):
    indexed_count: int
    message: str
