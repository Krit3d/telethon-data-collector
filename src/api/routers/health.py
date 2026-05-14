"""Health check router for monitoring cross-server link stability."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import TypedDict

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.api.dependencies import get_db, get_qdrant
from src.db.database import Database
from src.embeddings.qdrant_service import QdrantService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


class ServiceStatus(TypedDict):
    """Status information for a single service."""

    status: str
    timestamp: str
    latency_ms: float | None
    error: str | None


class HealthResponse(TypedDict):
    """Complete health check response."""

    status: str
    timestamp: str
    services: dict[str, ServiceStatus]


async def check_postgresql(db: Database) -> ServiceStatus:
    """Check PostgreSQL connection by executing SELECT 1.

    Args:
        db: Database instance.

    Returns:
        ServiceStatus dictionary with check results.
    """

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


async def check_apache_age(db: Database) -> ServiceStatus:
    """Check Apache AGE status by verifying telegram_graph exists.

    Args:
        db: Database instance.

    Returns:
        ServiceStatus dictionary with check results.
    """

    start = time.time()

    try:
        async with db.async_session() as session:
            async with session.begin():
                # Check if the telegram_graph exists in ag_graph
                result = await session.execute(
                    text(
                        "SELECT 1 FROM ag_graph WHERE name = 'telegram_graph' AND kind = 'normal'"
                    )
                )
                exists = result.scalar_one_or_none()

                if not exists:
                    return {
                        "status": "unhealthy",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "latency_ms": None,
                        "error": "Apache AGE graph 'telegram_graph' not found",
                    }

        latency = (time.time() - start) * 1000
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": round(latency, 2),
            "error": None,
        }
    except SQLAlchemyError as e:
        logger.error("Apache AGE health check failed", exc_info=e)
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": None,
            "error": f"Apache AGE query error: {str(e)}",
        }


async def check_qdrant(qdrant: QdrantService) -> ServiceStatus:
    """Check Qdrant connection by retrieving collections.

    Args:
        qdrant: Qdrant service instance.

    Returns:
        ServiceStatus dictionary with check results.
    """

    start = time.time()

    try:
        # Ensure Qdrant is initialized
        if not qdrant._initialized:
            await qdrant.initialize()

        # Test connection by getting collections
        collections = await qdrant.client.get_collections()

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


@router.get(
    "", response_model=HealthResponse, summary="Comprehensive health check"
)
async def health_check(
    request: Request,
    db: Database = Depends(get_db),
    qdrant: QdrantService = Depends(get_qdrant),
) -> HealthResponse:
    """Comprehensive health check endpoint for monitoring cross-server link stability.

    This endpoint performs asynchronous checks on all critical services:
    - PostgreSQL: Executes SELECT 1 to verify database connectivity
    - Apache AGE: Checks if the 'telegram_graph' exists and is accessible
    - Qdrant: Retrieves collections to verify vector database connectivity

    If any service is down, returns 503 Service Unavailable with details.
    If all services are healthy, returns 200 OK with status and timestamps.

    Args:
        request: FastAPI request object.
        db: Database instance (injected).
        qdrant: Qdrant service instance (injected).

    Returns:
        HealthResponse with status and service details.

    Raises:
        HTTPException: If any service check fails (503 status).
    """

    # Run all checks concurrently for efficiency
    postgres_task = asyncio.create_task(check_postgresql(db))
    age_task = asyncio.create_task(check_apache_age(db))
    qdrant_task = asyncio.create_task(check_qdrant(qdrant))

    # Wait for all checks to complete
    postgres_status, age_status, qdrant_status = await asyncio.gather(
        postgres_task, age_task, qdrant_task
    )

    # Build services dictionary
    services = {
        "postgresql": postgres_status,
        "apache_age": age_status,
        "qdrant": qdrant_status,
    }

    # Determine overall status
    all_healthy = all(s["status"] == "healthy" for s in services.values())

    response: HealthResponse = {
        "status": "healthy" if all_healthy else "unhealthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": services,
    }

    # Return 503 if any service is down
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
