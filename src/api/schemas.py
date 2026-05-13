from datetime import datetime
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description="Text for semantic search")
    limit: int = Field(default=10, ge=1, le=50, description="Number of results")
    score_threshold: float = Field(
        default=0.35, description="Minimum match threshold"
    )


class SearchResultItem(BaseModel):
    post_id: int
    channel_id: int
    text: str
    score: float
    created_at: datetime
    url: str | None = None


class GraphEdge(BaseModel):
    """Represents a graph relationship (edge) between two entities."""

    source_id: str
    source_label: str
    source_name: str | None = None
    relation_type: str
    target_id: str
    target_label: str
    target_name: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    graph_context: list[GraphEdge] = Field(default_factory=list)


class IndexRequest(BaseModel):
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of posts to index")


class IndexResponse(BaseModel):
    indexed_count: int
    message: str
