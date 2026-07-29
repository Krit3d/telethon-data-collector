from functools import cache

from fastapi import Request

from src.api.services.search import SearchService
from src.api.services.search.query_parser import QueryParser
from src.api.services.search.ranker import SearchRanker, TaxonomyLoader
from src.api.services.search.retriever import SearchRetriever
from src.db.database import Database
from src.graph.db.search_repo import GraphSearchRepository
from src.embeddings.qdrant_service import QdrantService


@cache
def get_ancestors_map() -> dict[str, list[str]]:
    return TaxonomyLoader().load_ancestors_map("src/config/Content Taxonomy 3.1.tsv")


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_qdrant(request: Request) -> QdrantService:
    return request.app.state.qdrant


def get_search_service(request: Request) -> SearchService:
    settings = request.app.state.settings
    qdrant = get_qdrant(request)
    db = get_db(request)
    session = db.async_session()
    graph_search_repo = GraphSearchRepository(session=session)
    query_parser = QueryParser(settings=settings)
    retriever = SearchRetriever(session=session, qdrant_service=qdrant, graph_repo=graph_search_repo)
    ranker = SearchRanker(ancestors_map=get_ancestors_map())
    return SearchService(query_parser=query_parser, retriever=retriever, ranker=ranker)
