"""FastAPI application entry point with production-ready configuration."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.config.config import load_settings
from src.db.database import Database
from src.embeddings.qdrant_service import QdrantService
from src.graph.client import Neo4jClient
from src.api.routers import search, health

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
INDEX_FILE = WEB_DIR / "index.html"
CSS_FILE = WEB_DIR / "css" / "style.css"
JS_FILE = WEB_DIR / "js" / "app.js"
CSS_ASSET = "/css/style.css"
JS_ASSET = "/js/app.js"
NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown operations."""
    settings = load_settings()

    db = Database(settings.db_url)
    qdrant = QdrantService(settings)
    neo4j = Neo4jClient(settings)

    # Initialize database with retry logic and timeout
    try:
        await db.init_db(max_retries=5, timeout=120.0)
    except Exception as e:
        logger.error("Failed to initialize database during startup: %s", e)
        # Re-raise to prevent application from starting in a broken state
        raise

    await qdrant.initialize()
    await neo4j.connect()

    app.state.db = db
    app.state.qdrant = qdrant
    app.state.neo4j = neo4j
    app.state.settings = settings

    logger.info("FastAPI application started successfully.")
    yield

    await neo4j.close()
    await db.close()
    await qdrant.close()
    logger.info("FastAPI application stopped.")


app = FastAPI(
    title="Telegram Semantic Search API",
    version="1.0.0",
    lifespan=lifespan,
)


class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        for header, value in NO_CACHE_HEADERS.items():
            response.headers[header] = value
        return response


class NoCacheStaticFiles(StaticFiles):
    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        for header, value in NO_CACHE_HEADERS.items():
            response.headers[header] = value
        return response


app.add_middleware(NoCacheMiddleware)

# Add CORS middleware for internal production APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API v1 routing
app.include_router(search.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")


@app.get("/health", tags=["System"])
async def health_check() -> dict[str, str]:
    """Health check endpoint for monitoring and load balancers."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = INDEX_FILE.read_text(encoding="utf-8")
    css_mtime = int(os.path.getmtime(CSS_FILE))
    js_mtime = int(os.path.getmtime(JS_FILE))
    html = html.replace(CSS_ASSET, f"{CSS_ASSET}?t={css_mtime}")
    html = html.replace(JS_ASSET, f"{JS_ASSET}?t={js_mtime}")
    return HTMLResponse(content=html, headers=NO_CACHE_HEADERS)


if WEB_DIR.exists():
    app.mount("/css", NoCacheStaticFiles(directory=str(WEB_DIR / "css")), name="css")
    app.mount("/js", NoCacheStaticFiles(directory=str(WEB_DIR / "js")), name="js")
