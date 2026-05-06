"""FastAPI application entry point with production-ready configuration."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.config.config import load_settings
from src.db.database import Database
from src.embeddings.qdrant_service import QdrantService
from src.api.routers import search, index

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown operations."""
    settings = load_settings()

    db = Database(settings.db_url)
    qdrant = QdrantService(settings)
    await qdrant.initialize()

    app.state.db = db
    app.state.qdrant = qdrant

    logger.info("FastAPI application started.")
    yield

    await db.close()
    await qdrant.close()
    logger.info("FastAPI application stopped.")


app = FastAPI(
    title="Telegram Semantic Search API",
    version="1.0.0",
    lifespan=lifespan,
)

# API v1 routing
app.include_router(search.router, prefix="/api/v1")
app.include_router(index.router, prefix="/api/v1")


@app.get("/health", tags=["System"])
async def health_check() -> dict[str, str]:
    """Health check endpoint for monitoring and load balancers."""
    return {"status": "ok"}
