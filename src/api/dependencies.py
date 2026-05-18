"""Dependency injection functions for FastAPI endpoints."""

from fastapi import Request
from src.db.database import Database
from src.db.graph_repo import GraphRepository
from src.embeddings.qdrant_service import QdrantService


def get_db(request: Request) -> Database:
    """Dependency to get the database instance."""
    return request.app.state.db


def get_graph_repo(request: Request) -> GraphRepository:
    """Dependency to get the graph repository instance.

    Shares the same async session factory as the main Database so both
    ORM and graph operations use the same connection pool.
    """
    
    db = request.app.state.db
    return GraphRepository(async_session=db.async_session)


def get_qdrant(request: Request) -> QdrantService:
    """Dependency to get the Qdrant service instance."""
    return request.app.state.qdrant
