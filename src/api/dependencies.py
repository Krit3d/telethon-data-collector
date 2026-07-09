from fastapi import Request

from src.api.services.search_service import SearchService
from src.db.database import Database
from src.graph.db.search_repo import GraphSearchRepository
from src.embeddings.qdrant_service import QdrantService


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_qdrant(request: Request) -> QdrantService:
    return request.app.state.qdrant


def get_graph_search_repo(request: Request) -> GraphSearchRepository:
    db = request.app.state.db
    settings = request.app.state.settings
    return GraphSearchRepository(async_session=db.async_session, settings=settings)


def get_search_service(request: Request) -> SearchService:
    qdrant = get_qdrant(request)
    db = get_db(request)
    graph_search_repo = get_graph_search_repo(request)
    return SearchService(
        qdrant=qdrant,
        db=db,
        graph_search_repo=graph_search_repo,
    )
