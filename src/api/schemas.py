from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


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
    url: Optional[str] = None


class SearchResponse(BaseModel):
    results: list[SearchResultItem]


class IndexRequest(BaseModel):
    limit: int = Field(
        default=100, description="How many recent posts to index"
    )


class IndexResponse(BaseModel):
    indexed_count: int
    message: str
