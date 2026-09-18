import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy import func, or_, select, update

from src.api.dependencies import get_db
from src.api.services.dmnode_client import DMnodeClient
from src.db.database import Database
from src.db.models import Account, CreatorMessage, Deal, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["CRM Webhooks"])


@router.post("/dmnode")
async def dmnode_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_db),
) -> dict[str, str]:
    raw_body = await request.body()
    signature = (
        request.headers.get("x-dmnode-signature-v1")
        or request.headers.get("x-dmnode-signature")
    )
    settings = request.app.state.settings
    if not DMnodeClient.verify_webhook_signature(
        raw_body,
        signature,
        settings.dmnode_webhook_secret or "",
    ):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if isinstance(payload, dict):
        background_tasks.add_task(process_dmnode_event, db, payload)
    return {"status": "ok"}


async def process_dmnode_event(db: Database, payload: dict[str, Any]) -> None:
    event_type = str(payload.get("event") or payload.get("type") or "").lower()
    if event_type in ("replied", "reply"):
        target_val = payload.get("target")
        target_obj: dict[str, Any] = target_val if isinstance(target_val, dict) else {}
        raw_handle = (
            payload.get("handle")
            or target_obj.get("handle")
            or payload.get("author")
            or payload.get("username")
            or payload.get("sender")
            or payload.get("from")
        )
        handle_clean = str(raw_handle or "").lstrip("@").strip().lower()
        reply_text = payload.get("text") or payload.get("message") or payload.get("content")
        external_message_id = (
            payload.get("external_message_id")
            or payload.get("externalId")
            or payload.get("message_id")
            or payload.get("id")
        )
        timestamp = payload.get("timestamp") or payload.get("created_at")
        if not handle_clean or not external_message_id:
            return
        created_at = None
        if timestamp:
            try:
                ts = float(timestamp)
                if ts > 1e12:
                    ts = ts / 1000.0
                created_at = datetime.fromtimestamp(ts, tz=timezone.utc)
            except (ValueError, TypeError):
                created_at = None
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
                .where(CreatorMessage.external_message_id == external_message_id)
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
                external_message_id=external_message_id,
                text=reply_text,
                is_read=False,
            )
            if created_at is not None:
                message_kwargs["created_at"] = created_at
            session.add(CreatorMessage(**message_kwargs))
            await session.commit()
    elif event_type == "sent":
        external_message_id = (
            payload.get("external_message_id")
            or payload.get("externalId")
            or payload.get("message_id")
            or payload.get("id")
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
                        CreatorMessage.external_message_id == f"dmnode:{ext_id_str}",
                    )
                )
                .values(is_read=True)
            )
            await session.commit()
    elif event_type == "failed":
        external_message_id = (
            payload.get("external_message_id")
            or payload.get("externalId")
            or payload.get("message_id")
            or payload.get("id")
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
                        CreatorMessage.external_message_id == f"dmnode:{ext_id_str}",
                    )
                )
                .values(external_message_id=f"FAILED:{ext_id_str}")
            )
            await session.commit()