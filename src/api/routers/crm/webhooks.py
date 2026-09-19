import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy import func, or_, select, update

from src.api.dependencies import get_db
from src.api.services.crm_helpers import detect_media_type
from src.api.services.slidecold_client import SlideColdClient
from src.config.config import MEDIA_DIR, Settings
from src.db.database import Database
from src.db.models import Account, CreatorMessage, Deal, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["CRM Webhooks"])


async def _save_webhook_media(
    remote_url: str, filename: str, settings: Settings
) -> tuple[str | None, str | None, str | None]:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(remote_url)
    except Exception:
        return (None, None, None)
    if resp.status_code != 200:
        return (None, None, None)
    content = resp.content
    content_type = resp.headers.get("content-type")
    extension = Path(filename).suffix.lower()
    media_type = detect_media_type(extension)
    if content_type:
        if content_type.startswith("image/"):
            media_type = "image"
        elif content_type.startswith("video/"):
            media_type = "video"
    if settings.media_bridge_url:
        headers = {}
        if settings.media_bridge_secret:
            headers["X-Bridge-Secret"] = settings.media_bridge_secret
        files = {"file": (filename, content, content_type)}
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{settings.media_bridge_url.rstrip('/')}/upload",
                    headers=headers,
                    files=files,
                )
            resp.raise_for_status()
            data = resp.json()
            return (data["url"], data["name"], data["type"])
        except Exception:
            return (None, None, None)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    destination = MEDIA_DIR / unique_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return (f"/media/{unique_name}", filename, media_type)


@router.post("/slidecold")
async def slidecold_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_db),
) -> dict[str, str]:
    raw_body = await request.body()
    signature = (
        request.headers.get("x-slidecold-signature")
        or request.headers.get("X-SlideCold-Signature")
    )
    settings = request.app.state.settings
    if settings.slidecold_webhook_secret and not SlideColdClient.verify_webhook_signature(
        raw_body,
        signature,
        settings.slidecold_webhook_secret or "",
    ):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if isinstance(payload, dict):
        background_tasks.add_task(process_slidecold_event, db, payload, settings)
    return {"status": "ok"}


async def process_slidecold_event(
    db: Database, payload: dict[str, Any], settings: Settings
) -> None:
    event_type = str(payload.get("event") or payload.get("type") or "").lower()
    if event_type in ("conversation.reply", "conversation_reply", "reply", "replied"):
        raw_handle = (
            payload.get("sender")
            or payload.get("handle")
            or payload.get("username")
            or payload.get("from")
        )
        handle_clean = str(raw_handle or "").lstrip("@").strip().lower()
        reply_text = payload.get("text") or payload.get("message") or payload.get("content")
        external_message_id = (
            payload.get("id")
            or payload.get("message_id")
            or payload.get("external_id")
        )
        if not handle_clean or not external_message_id:
            return
        media_url = None
        media_name = None
        media_type = None
        remote_url = None
        filename = None
        for key in (
            "attachments",
            "attachment",
            "media_url",
            "mediaUrl",
            "image_url",
            "video_url",
            "media",
        ):
            val = payload.get(key)
            if not val:
                continue
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        cand = (
                            item.get("url")
                            or item.get("media_url")
                            or item.get("mediaUrl")
                            or item.get("src")
                        )
                        if cand:
                            remote_url = str(cand)
                            filename = str(
                                item.get("name")
                                or item.get("filename")
                                or Path(remote_url).name
                            )
                            break
                    elif isinstance(item, str):
                        remote_url = item
                        filename = Path(item).name
                        break
            elif isinstance(val, dict):
                cand = (
                    val.get("url")
                    or val.get("media_url")
                    or val.get("mediaUrl")
                    or val.get("src")
                )
                if cand:
                    remote_url = str(cand)
                    filename = str(
                        val.get("name")
                        or val.get("filename")
                        or Path(remote_url).name
                    )
            else:
                remote_url = str(val)
                filename = Path(remote_url).name
            if remote_url:
                break
        if remote_url:
            media_url, media_name, media_type = await _save_webhook_media(
                remote_url, filename or "file", settings
            )
        async with db.async_session() as session:
            account_result = await session.execute(
                select(Account)
                .where(
                    or_(
                        func.lower(Account.username) == handle_clean,
                        func.lower(Account.username) == f"@{handle_clean}",
                    )
                )
                .limit(1)
            )
            account = account_result.scalar_one_or_none()
            if account is None:
                logger.warning("Slidecold reply: account %s not found", handle_clean)
                return
            last_msg_stmt = (
                select(CreatorMessage.user_id)
                .where(CreatorMessage.account_id == account.id)
                .order_by(CreatorMessage.created_at.desc())
                .limit(1)
            )
            user_id = (await session.execute(last_msg_stmt)).scalar_one_or_none()
            if user_id is None:
                user_stmt = select(User.id).order_by(User.id.asc()).limit(1)
                user_id = (await session.execute(user_stmt)).scalar_one_or_none()
            if user_id is None:
                return
            duplicate_stmt = (
                select(CreatorMessage.id)
                .where(CreatorMessage.external_message_id == str(external_message_id))
                .limit(1)
            )
            if (await session.execute(duplicate_stmt)).scalar_one_or_none() is not None:
                return
            deal_stmt = (
                select(Deal.id)
                .where(
                    Deal.account_id == account.id,
                    Deal.user_id == user_id,
                    Deal.stage.in_([1, 2, 3, 4, 5, 6]),
                )
                .order_by(Deal.updated_at.desc())
                .limit(1)
            )
            deal_id = (await session.execute(deal_stmt)).scalar_one_or_none()
            message_kwargs: dict[str, Any] = dict(
                user_id=user_id,
                account_id=account.id,
                deal_id=deal_id,
                sender_type="creator",
                channel_type="instagram",
                channel_target=f"@{handle_clean}",
                external_message_id=str(external_message_id),
                text=reply_text,
                is_read=False,
            )
            if media_url:
                message_kwargs["media_url"] = media_url
                message_kwargs["media_name"] = media_name
                message_kwargs["media_type"] = media_type
            session.add(CreatorMessage(**message_kwargs))
            await session.commit()
    elif event_type in ("campaign.sent", "message.sent", "sent"):
        external_message_id = (
            payload.get("id")
            or payload.get("message_id")
            or payload.get("external_id")
        )
        if not external_message_id:
            return
        ext_id_str = str(external_message_id)
        async with db.async_session() as session:
            await session.execute(
                update(CreatorMessage)
                .where(
                    or_(
                        CreatorMessage.external_message_id == ext_id_str,
                        CreatorMessage.external_message_id == f"slidecold:{ext_id_str}",
                    )
                )
                .values(is_read=True)
            )
            await session.commit()
    elif event_type in ("account.flagged", "failed"):
        external_message_id = (
            payload.get("id")
            or payload.get("message_id")
            or payload.get("external_id")
        )
        if not external_message_id:
            return
        ext_id_str = str(external_message_id)
        logger.warning("Slidecold delivery failure for message %s", ext_id_str)
        async with db.async_session() as session:
            await session.execute(
                update(CreatorMessage)
                .where(
                    or_(
                        CreatorMessage.external_message_id == ext_id_str,
                        CreatorMessage.external_message_id == f"slidecold:{ext_id_str}",
                    )
                )
                .values(external_message_id=f"FAILED:{ext_id_str}")
            )
            await session.commit()