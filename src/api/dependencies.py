from fastapi import Request
from src.db.database import Database
from src.db.graph_repo import GraphRepository
from src.embeddings.qdrant_service import QdrantService


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_graph_repo(request: Request) -> GraphRepository:
    db = request.app.state.db
    settings = request.app.state.settings
    return GraphRepository(async_session=db.async_session, settings=settings)


def get_qdrant(request: Request) -> QdrantService:
    """Dependency to get the Qdrant service instance."""
    return request.app.state.qdrant
