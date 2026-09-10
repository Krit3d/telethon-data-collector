import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, or_, select

from src.api.dependencies import get_crm_client, get_db
from src.api.schemas import (
    CrmLoginRequest,
    CrmLoginResponse,
    CrmShortlistRequest,
    CrmShortlistResponse,
    CrmUpdateStatusRequest,
)
from src.api.services.crm_client import TwentyCrmClient
from src.db.database import Database
from src.db.models import Account

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/crm", tags=["crm"])

STATUS_UI_TO_CRM: dict[str, str] = {
    "Свободен": "SVOBODEN",
    "В сделке": "V_SDELKE",
    "На паузе": "NA_PAUZE",
    "SVOBODEN": "SVOBODEN",
    "V_SDELKE": "V_SDELKE",
    "NA_PAUZE": "NA_PAUZE",
}

STATUS_CRM_TO_UI: dict[str, str] = {
    "SVOBODEN": "Свободен",
    "V_SDELKE": "В сделке",
    "NA_PAUZE": "На паузе",
}

ALLOWED_PLATFORMS: set[str] = {"INSTAGRAM", "TELEGRAM", "YOUTUBE", "VK", "TIKTOK"}


def _normalize_platform(platform: str) -> str:
    normalized = platform.upper().strip()
    if normalized not in ALLOWED_PLATFORMS:
        return "INSTAGRAM"
    return normalized


def _build_creator_payload(account: Account) -> dict[str, Any]:
    followers = int(account.subscribers_count or 0)
    er = float(account.static_avg_er or 0.0)
    if er > 0:
        avg_reach = int(followers * (er / 100))
        cpm = int((followers * (er / 100) / 1000) * 200)
    else:
        avg_reach = int(followers * 0.1)
        cpm = 0
    return {
        "accountid": str(account.id),
        "name": account.title or account.username or "",
        "platform": _normalize_platform(account.platform),
        "handle": f"@{account.username.lstrip('@')}" if account.username else "",
        "followers": followers,
        "er": er,
        "avgreach": avg_reach,
        "cpm": cpm,
        "niche": account.category_path or "Общее",
        "status": STATUS_UI_TO_CRM.get("Свободен", "SVOBODEN"),
        "dealscount": 0,
    }


@router.post("/shortlist", response_model=CrmShortlistResponse)
async def export_to_shortlist(
    payload: CrmShortlistRequest,
    request: Request,
    db: Database = Depends(get_db),
    crm_client: TwentyCrmClient = Depends(get_crm_client),
) -> CrmShortlistResponse:
    raw_ids = [str(aid).strip() for aid in payload.account_ids if aid and str(aid).strip()]
    raw_ids = [rid for rid in raw_ids if rid.lower() not in {"undefined", "null"}]
    if not raw_ids:
        return CrmShortlistResponse(
            added_count=0,
            redirect_url=f"{request.app.state.settings.crm_frontend_url}/#/authors",
        )

    numeric_ids: list[int] = []
    string_ids: list[str] = []
    clean_usernames: list[str] = []
    for raw in raw_ids:
        string_ids.append(raw)
        unprefixed = raw.lstrip("@").strip()
        if unprefixed:
            clean_usernames.append(unprefixed.lower())
            string_ids.append(unprefixed)
        try:
            numeric_ids.append(int(raw))
        except ValueError:
            pass

    logger.info(
        "CRM export: received raw_ids=%s, numeric=%d, string=%d, usernames=%d",
        raw_ids,
        len(numeric_ids),
        len(string_ids),
        len(clean_usernames),
    )

    conditions = []
    if numeric_ids:
        conditions.append(Account.id.in_(numeric_ids))
    if string_ids:
        conditions.append(Account.platform_id.in_(string_ids))
    if clean_usernames:
        conditions.append(func.lower(Account.username).in_(clean_usernames))

    async with db.async_session() as session:
        stmt = select(Account).where(or_(*conditions))
        result = await session.execute(stmt)
        accounts = list(result.scalars().all())

    logger.info("CRM export: found %d Account records in Postgres for raw_ids=%s", len(accounts), raw_ids)

    if not accounts:
        logger.warning("No accounts found for raw_ids=%s", raw_ids)
        raise HTTPException(status_code=404, detail="Указанные авторы не найдены в базе")

    tasks = [crm_client.upsert_creator(_build_creator_payload(account)) for account in accounts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for account, res in zip(accounts, results):
        if isinstance(res, Exception):
            logger.error("Failed to export account %s: %s", account.id, res)
    success_count = sum(1 for r in results if not isinstance(r, Exception))

    return CrmShortlistResponse(
        added_count=success_count,
        redirect_url=f"{request.app.state.settings.crm_frontend_url}/#/authors",
    )


@router.post("/auth/login", response_model=CrmLoginResponse)
async def crm_login(
    payload: CrmLoginRequest,
    crm_client: TwentyCrmClient = Depends(get_crm_client),
) -> CrmLoginResponse:
    try:
        data = await crm_client.authenticate(payload.email, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except Exception as exc:
        logger.exception("Twenty login failed")
        raise HTTPException(status_code=500, detail=f"Twenty service error: {exc}")
    token = data.get("token")
    user = data.get("user")
    if not isinstance(token, str) or not token or not isinstance(user, dict):
        raise HTTPException(status_code=401, detail="Неверный email или пароль")
    return CrmLoginResponse(token=token, user=user)


@router.get("/creators")
async def crm_creators(
    limit: int = 100,
    crm_client: TwentyCrmClient = Depends(get_crm_client),
) -> dict[str, Any]:
    records = await crm_client.get_creators(limit=limit)
    translated = [dict(record, status=STATUS_CRM_TO_UI.get(record["status"], record["status"])) for record in records]
    return {"data": translated, "total": len(translated)}


@router.patch("/creators/{creator_id}")
async def crm_update_creator_status(
    creator_id: str,
    payload: CrmUpdateStatusRequest,
    crm_client: TwentyCrmClient = Depends(get_crm_client),
) -> dict[str, Any]:
    try:
        mapped_status = STATUS_UI_TO_CRM.get(payload.status, payload.status)
        return await crm_client.update_creator_status(creator_id, mapped_status)
    except Exception:
        raise HTTPException(status_code=404, detail="Автор не найден в Twenty CRM")