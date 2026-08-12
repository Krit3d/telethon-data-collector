from collections.abc import AsyncGenerator
from functools import cache

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.services.search import SearchService
from src.api.services.search.query_parser import QueryParser
from src.api.services.search.ranker import SearchRanker, TaxonomyLoader
from src.api.services.search.retriever import SearchRetriever
from src.db.database import Database
from src.embeddings.qdrant_service import QdrantService
from src.graph.client import Neo4jClient
from src.graph.search_repo import Neo4jSearchRepository


@cache
def get_ancestors_map(taxonomy_path: str) -> dict[str, list[str]]:
    return TaxonomyLoader().load_ancestors_map(taxonomy_path)


@cache
def get_name_to_id_map(taxonomy_path: str) -> dict[str, str]:
    return TaxonomyLoader().load_name_to_id_map(taxonomy_path)


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_qdrant(request: Request) -> QdrantService:
    return request.app.state.qdrant


def get_neo4j(request: Request) -> Neo4jClient:
    return request.app.state.neo4j


async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    db: Database = get_db(request)
    async with db.async_session() as session:
        yield session


async def get_search_service(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> SearchService:
    settings = request.app.state.settings
    taxonomy_path_str = str(settings.taxonomy_path)
    qdrant = get_qdrant(request)
    neo4j = get_neo4j(request)
    graph_search_repo = Neo4jSearchRepository(client=neo4j)
    query_parser = QueryParser(settings)
    retriever = SearchRetriever(session=session, qdrant_service=qdrant, graph_search_repo=graph_search_repo)
    ranker = SearchRanker(
        ancestors_map=get_ancestors_map(taxonomy_path_str),
        name_to_id_map=get_name_to_id_map(taxonomy_path_str),
    )
    return SearchService(query_parser=query_parser, retriever=retriever, ranker=ranker)
