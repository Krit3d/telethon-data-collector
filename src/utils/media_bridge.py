import uuid
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from src.config.config import MEDIA_DIR, Settings

app = FastAPI(title="Media Bridge")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _detect_media_type(extension: str, content_type: str | None) -> str:
    if extension in IMAGE_EXTENSIONS or (content_type or "").startswith("image/"):
        return "image"
    if extension in VIDEO_EXTENSIONS or (content_type or "").startswith("video/"):
        return "video"
    return "document"


def _check_secret(request: Request, settings: Settings) -> None:
    if not settings.media_bridge_secret:
        return
    if request.headers.get("X-Bridge-Secret") != settings.media_bridge_secret:
        raise HTTPException(status_code=403, detail="Forbidden")


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)) -> dict:
    settings = Settings() #type: ignore[call-arg]
    _check_secret(request, settings)
    original_filename = file.filename or "file"
    unique_name = f"{uuid.uuid4().hex}_{original_filename}"
    extension = Path(original_filename).suffix.lower()
    media_type = _detect_media_type(extension, file.content_type)
    destination = MEDIA_DIR / unique_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(destination, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            await buffer.write(chunk)
    return {"url": f"/media/{unique_name}", "name": original_filename, "type": media_type}


@app.get("/media/{filename}")
async def get_media(filename: str) -> FileResponse:
    path = MEDIA_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8090)
