import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from src.api.schemas import FileUploadResponse
from src.api.services.crm_helpers import FORBIDDEN_EXTENSIONS, detect_media_type
from src.config.config import MEDIA_DIR
from src.utils.security import decode_access_token

router = APIRouter(tags=["CRM Media"])


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
) -> FileUploadResponse:
    original_filename = file.filename or "file"
    extension = Path(original_filename).suffix.lower()
    if extension in FORBIDDEN_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Запрещённый тип файла")
    settings = request.app.state.settings
    internal_token = request.headers.get("x-internal-token")
    is_internal = bool(settings.secret_key and internal_token == settings.secret_key)
    if not is_internal:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized upload access")
        token = auth[len("Bearer "):].strip()
        if not token:
            raise HTTPException(status_code=401, detail="Unauthorized upload access")
        payload = decode_access_token(token, settings.secret_key)
        if payload is None:
            raise HTTPException(status_code=401, detail="Unauthorized upload access")
    if settings.media_bridge_url:
        headers = {}
        if settings.media_bridge_secret:
            headers["X-Bridge-Secret"] = settings.media_bridge_secret
        content = await file.read()
        files = {"file": (original_filename, content, file.content_type)}
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"{settings.media_bridge_url.rstrip('/')}/upload",
                headers=headers,
                files=files,
            )
        resp.raise_for_status()
        data = resp.json()
        return FileUploadResponse(
            media_url=data["url"],
            media_name=data["name"],
            media_type=data["type"],
            file_size=len(content),
        )
    media_type = detect_media_type(extension)
    saved_filename = f"{uuid.uuid4().hex}{extension}"
    destination = MEDIA_DIR / saved_filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    total_size = 0
    with destination.open("wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)
            total_size += len(chunk)
    return FileUploadResponse(
        media_url=f"/media/{saved_filename}",
        media_name=original_filename,
        media_type=media_type,
        file_size=total_size,
    )