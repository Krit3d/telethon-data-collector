"""FastAPI application entry point with production-ready configuration."""

import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote
import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.staticfiles import StaticFiles
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import Response

from src.api.routers import search, health, crm
from src.api.services.crm_client import TwentyCrmClient
from src.config.config import MEDIA_DIR, load_settings
from src.db.database import Database
from src.embeddings.qdrant_service import QdrantService
from src.graph.client import Neo4jClient

logger = logging.getLogger(__name__)

settings = load_settings()

WEB_DIR = Path(__file__).resolve().parent.parent / "web" / "search"
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
SECURITY_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), camera=(), microphone=()",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown operations."""
    db = Database(settings.db_url)
    qdrant = QdrantService(settings)
    neo4j = Neo4jClient(settings)
    crm_client = TwentyCrmClient(
        base_url=settings.twenty_api_url,
        api_key=settings.twenty_api_key,
    )

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
    app.state.crm_client = crm_client

    logger.info("FastAPI application started successfully.")
    yield

    await crm_client.aclose()
    await neo4j.close()
    await db.close()
    await qdrant.close()
    logger.info("FastAPI application stopped.")


app = FastAPI(
    title="Telegram Semantic Search API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

security = HTTPBasic()


def verify_docs_credentials(
    request: Request,
    credentials: HTTPBasicCredentials = Depends(security),
) -> None:
    settings = request.app.state.settings
    username_ok = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        settings.docs_username.encode("utf-8"),
    )
    password_ok = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        settings.docs_password.encode("utf-8"),
    )
    if not settings.docs_password or not (username_ok and password_ok):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": 'Basic realm="API Documentation"'},
        )


if settings.docs_enabled:

    @app.get(
        "/openapi.json",
        include_in_schema=False,
        dependencies=[Depends(verify_docs_credentials)],
    )
    async def openapi_schema(request: Request) -> JSONResponse:
        return JSONResponse(content=request.app.openapi())

    @app.get(
        "/docs",
        include_in_schema=False,
        dependencies=[Depends(verify_docs_credentials)],
    )
    async def swagger_ui(request: Request) -> HTMLResponse:
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=f"{request.app.title} - Swagger UI",
            swagger_ui_parameters={"persistAuthorization": True},
        )

    @app.get(
        "/redoc",
        include_in_schema=False,
        dependencies=[Depends(verify_docs_credentials)],
    )
    async def redoc_ui(request: Request) -> HTMLResponse:
        return get_redoc_html(
            openapi_url="/openapi.json",
            title=f"{request.app.title} - ReDoc",
        )


class SecurityHeadersMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for header, value in SECURITY_HEADERS.items():
                    headers[header] = value
            await send(message)

        await self.app(scope, receive, send_wrapper)


class NoCacheStaticFiles(StaticFiles):
    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        for header, value in NO_CACHE_HEADERS.items():
            response.headers[header] = value
        return response


app.add_middleware(SecurityHeadersMiddleware)

# Add CORS middleware for internal production APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "Origin",
        "X-Requested-With",
        "X-Bridge-Secret",
    ],
)

app.include_router(health.router, prefix="/api/v1/health")
app.include_router(search.router, prefix="/api/v1/search")
app.include_router(crm.router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    settings = request.app.state.settings
    html = INDEX_FILE.read_text(encoding="utf-8")
    css_mtime = int(os.path.getmtime(CSS_FILE))
    js_mtime = int(os.path.getmtime(JS_FILE))
    html = html.replace(CSS_ASSET, f"{CSS_ASSET}?t={css_mtime}")
    html = html.replace(JS_ASSET, f"{JS_ASSET}?t={js_mtime}")
    html = html.replace("__CRM_URL__", settings.crm_frontend_url)
    return HTMLResponse(content=html, headers=NO_CACHE_HEADERS)


if WEB_DIR.exists():
    app.mount(
        "/css",
        NoCacheStaticFiles(directory=str(WEB_DIR / "css"), html=True),
        name="css",
    )
    app.mount(
        "/js",
        NoCacheStaticFiles(directory=str(WEB_DIR / "js"), html=True),
        name="js",
    )

@app.get("/media/{filename}")
async def get_media(filename: str, request: Request) -> Response:
    settings = request.app.state.settings
    decoded_filename = unquote(filename).strip()
    if "\x00" in decoded_filename:
        raise HTTPException(status_code=400, detail="Некорректное имя файла")
    safe_name = Path(decoded_filename).name
    if not safe_name or safe_name != decoded_filename or safe_name.startswith("."):
        raise HTTPException(status_code=400, detail="Некорректное имя файла")
    if settings.media_bridge_url:
        headers = {}
        if settings.media_bridge_secret:
            headers["X-Bridge-Secret"] = settings.media_bridge_secret
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(
                f"{settings.media_bridge_url.rstrip('/')}/media/{safe_name}",
                headers=headers,
            )
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail="File not found")
        resp.raise_for_status()
        headers = {}
        cd = resp.headers.get("content-disposition")
        if cd:
            headers["Content-Disposition"] = cd
        return StreamingResponse(
            resp.aiter_bytes(),
            media_type=resp.headers.get("content-type", "application/octet-stream"),
            headers=headers,
        )
    base_dir = MEDIA_DIR.resolve()
    target_path = (base_dir / safe_name).resolve()
    if not target_path.is_relative_to(base_dir) or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(target_path)
