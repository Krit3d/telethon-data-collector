from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.services.search.dbsf_engine import DbsfRankingEngine
from src.api.services.search.graph_reasoner import GraphReasoner
from src.api.services.search.hydrator import PostgresHydrator
from src.api.services.search.query_parser import QueryParser
from src.api.services.search.retriever import VectorRetriever
from src.api.services.search.search_service import SearchService
from src.db.database import Database
from src.embeddings.qdrant_service import QdrantService
from src.graph.client import Neo4jClient
from src.graph.search_repo import Neo4jSearchRepository


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


def get_search_service(request: Request) -> SearchService:
    settings = request.app.state.settings
    db = get_db(request)
    qdrant = get_qdrant(request)
    neo4j = get_neo4j(request)
    query_parser = QueryParser(settings=settings)
    retriever = VectorRetriever(qdrant_service=qdrant)
    graph_repo = Neo4jSearchRepository(client=neo4j)
    graph_reasoner = GraphReasoner(graph_repo=graph_repo)
    dbsf_engine = DbsfRankingEngine()
    hydrator = PostgresHydrator(session_factory=db.async_session)
    return SearchService(
        query_parser=query_parser,
        retriever=retriever,
        graph_reasoner=graph_reasoner,
        dbsf_engine=dbsf_engine,
        hydrator=hydrator,
    )
