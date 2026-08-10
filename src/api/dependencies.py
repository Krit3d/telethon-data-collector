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


TAXONOMY_PATH = "src/config/Content Taxonomy 3.1.tsv"


@cache
def get_ancestors_map() -> dict[str, list[str]]:
    return TaxonomyLoader().load_ancestors_map(TAXONOMY_PATH)


@cache
def get_name_to_id_map() -> dict[str, str]:
    return TaxonomyLoader().load_name_to_id_map(TAXONOMY_PATH)


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
    qdrant = get_qdrant(request)
    neo4j = get_neo4j(request)
    query_parser = QueryParser(settings)
    retriever = SearchRetriever(session=session, qdrant_service=qdrant, graph_search_repo=None)
    ranker = SearchRanker(ancestors_map=get_ancestors_map(), name_to_id_map=get_name_to_id_map())
    return SearchService(query_parser=query_parser, retriever=retriever, ranker=ranker)
