from collections.abc import AsyncGenerator

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.services.crm_client import TwentyCrmClient
from src.api.services.search.dbsf_engine import DbsfRankingEngine
from src.api.services.search.graph_reasoner import GraphReasoner
from src.api.services.search.hydrator import PostgresHydrator
from src.api.services.search.query_parser import QueryParser
from src.api.services.search.retriever import VectorRetriever
from src.api.services.search.search_service import SearchService
from src.db.database import Database
from src.db.models import User
from src.embeddings.qdrant_service import QdrantService
from src.graph.client import Neo4jClient
from src.graph.search_repo import Neo4jSearchRepository
from src.utils.security import decode_access_token


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


def get_crm_client(request: Request) -> TwentyCrmClient:
    settings = request.app.state.settings
    return TwentyCrmClient(
        base_url=settings.twenty_api_url,
        api_key=settings.twenty_api_key,
    )


async def get_current_user(
    request: Request,
    db: Database = Depends(get_db),
) -> User:
    auth_header = request.headers.get("Authorization")
    token = None
    if isinstance(auth_header, str) and auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer "):]
    if not token:
        raise HTTPException(status_code=401, detail="Недействительный токен авторизации")
    payload = decode_access_token(token, request.app.state.settings.secret_key)
    if payload is None:
        raise HTTPException(status_code=401, detail="Недействительный токен авторизации")
    user_id = payload.get("user_id")
    email = payload.get("sub") or payload.get("email")
    async with db.async_session() as session:
        if isinstance(user_id, int):
            stmt = select(User).where(User.id == user_id)
        else:
            stmt = select(User).where(func.lower(User.email) == str(email).lower())
        user = (await session.execute(stmt)).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="Пользователь не найден")
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Учетная запись деактивирована",
        )
    return user
