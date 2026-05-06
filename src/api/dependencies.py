"""Dependency injection functions for FastAPI endpoints."""

from fastapi import Request
from src.db.database import Database
from src.embeddings.qdrant_service import QdrantService


def get_db(request: Request) -> Database:
    """Dependency to get the database instance."""
    return request.app.state.db


def get_qdrant(request: Request) -> QdrantService:
    """Dependency to get the Qdrant service instance."""
    return request.app.state.qdrant
