from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import TypedDict

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.api.dependencies import get_db, get_neo4j, get_qdrant
from src.db.database import Database
from src.embeddings.qdrant_service import QdrantService
from src.graph.client import Neo4jClient

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


class ServiceStatus(TypedDict):
    status: str
    timestamp: str
    latency_ms: float | None
    error: str | None


class HealthResponse(TypedDict):
    status: str
    timestamp: str
    services: dict[str, ServiceStatus]


async def check_postgresql(db: Database) -> ServiceStatus:
    start = time.time()
    try:
        async with db.async_session() as session:
            async with session.begin():
                result = await session.execute(text("SELECT 1"))
                result.scalar_one()

        latency = (time.time() - start) * 1000
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": round(latency, 2),
            "error": None,
        }
    except SQLAlchemyError as e:
        logger.error("PostgreSQL health check failed", exc_info=e)
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": None,
            "error": f"PostgreSQL connection error: {str(e)}",
        }


async def check_neo4j(neo4j: Neo4jClient) -> ServiceStatus:
    start = time.time()
    try:
        try:
            await neo4j.verify_connectivity()
        except AttributeError:
            await neo4j.execute_read("RETURN 1 AS result")
        latency = (time.time() - start) * 1000
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": round(latency, 2),
            "error": None,
        }
    except Exception as e:
        logger.error("Neo4j health check failed", exc_info=e)
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": None,
            "error": f"Neo4j connection error: {str(e)}",
        }


async def check_qdrant(qdrant: QdrantService) -> ServiceStatus:
    start = time.time()
    try:
        if not qdrant._initialized:
            await qdrant.initialize()

        await qdrant.client.get_collections()

        latency = (time.time() - start) * 1000
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": round(latency, 2),
            "error": None,
        }
    except Exception as e:
        logger.error("Qdrant health check failed", exc_info=e)
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": None,
            "error": f"Qdrant connection error: {str(e)}",
        }


@router.get("", response_model=HealthResponse, summary="Comprehensive health check")
@router.get("/", response_model=HealthResponse, include_in_schema=False)
async def health_check(
    request: Request,
    db: Database = Depends(get_db),
    neo4j: Neo4jClient = Depends(get_neo4j),
    qdrant: QdrantService = Depends(get_qdrant),
) -> HealthResponse:
    postgres_task = asyncio.create_task(check_postgresql(db))
    neo4j_task = asyncio.create_task(check_neo4j(neo4j))
    qdrant_task = asyncio.create_task(check_qdrant(qdrant))

    postgres_status, neo4j_status, qdrant_status = await asyncio.gather(
        postgres_task, neo4j_task, qdrant_task
    )

    services = {
        "postgresql": postgres_status,
        "neo4j": neo4j_status,
        "qdrant": qdrant_status,
    }

    all_healthy = all(s["status"] == "healthy" for s in services.values())

    response: HealthResponse = {
        "status": "healthy" if all_healthy else "unhealthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": services,
    }

    if not all_healthy:
        unhealthy_services = [
            name
            for name, status in services.items()
            if status["status"] != "healthy"
        ]
        detail = {
            "message": "One or more services are unavailable",
            "unhealthy_services": unhealthy_services,
            "services": services,
        }
        raise HTTPException(status_code=503, detail=detail)

    return response
