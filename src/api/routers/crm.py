import asyncio
import logging
from typing import Any
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import and_, func, or_, select

from src.api.dependencies import get_crm_client, get_db
from src.api.schemas import (
    CrmLoginRequest,
    CrmLoginResponse,
    CrmRegisterRequest,
    CrmShortlistRequest,
    CrmShortlistResponse,
    CrmUpdateStatusRequest,
)
from src.api.services.crm_client import TwentyCrmClient
from src.db.database import Database
from src.db.models import Account, User
from src.utils.security import hash_password, verify_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/crm", tags=["crm"])

STATUS_UI_TO_CRM: dict[str, str] = {
    "Свободен": "SVOBODEN",
    "В сделке": "V_SDELKE",
    "На паузе": "NA_PAUZE",
    "В архиве": "ARCHIVED",
    "SVOBODEN": "SVOBODEN",
    "V_SDELKE": "V_SDELKE",
    "NA_PAUZE": "NA_PAUZE",
    "ARCHIVED": "ARCHIVED",
}

STATUS_CRM_TO_UI: dict[str, str] = {
    "SVOBODEN": "Свободен",
    "V_SDELKE": "В сделке",
    "NA_PAUZE": "На паузе",
    "ARCHIVED": "В архиве",
}

ALLOWED_PLATFORMS: set[str] = {"INSTAGRAM", "TELEGRAM", "YOUTUBE", "VK", "TIKTOK"}


def _normalize_platform(platform: str) -> str:
    normalized = platform.upper().strip()
    if normalized not in ALLOWED_PLATFORMS:
        return "INSTAGRAM"
    return normalized


def _build_creator_payload(account: Account, user_email: str = "") -> dict[str, Any]:
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
        "useremail": user_email,
    }


def _deduplicate_accounts(accounts: list[Account]) -> list[Account]:
    best: dict[str, Account] = {}
    for account in accounts:
        username = (account.username or "").strip().lstrip("@").lower()
        key = username if username else str(account.id)
        current = best.get(key)
        if current is None:
            best[key] = account
            continue
        current_platform = _normalize_platform(current.platform)
        candidate_platform = _normalize_platform(account.platform)
        current_followers = int(current.subscribers_count or 0)
        candidate_followers = int(account.subscribers_count or 0)
        if candidate_platform == "INSTAGRAM" and current_platform != "INSTAGRAM":
            best[key] = account
        elif candidate_platform == "INSTAGRAM" and current_platform == "INSTAGRAM" and candidate_followers > current_followers:
            best[key] = account
        elif current_platform != "INSTAGRAM" and candidate_platform != "INSTAGRAM" and candidate_followers > current_followers:
            best[key] = account
    return list(best.values())


def get_current_user_email(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[len("Bearer "):].strip()
        if token and "@" in token:
            return token
    query_email = request.query_params.get("user_email")
    if query_email:
        return query_email
    header_email = request.headers.get("X-User-Email")
    if header_email:
        return header_email
    return None


@router.post("/shortlist", response_model=CrmShortlistResponse)
async def export_to_shortlist(
    payload: CrmShortlistRequest,
    request: Request,
    db: Database = Depends(get_db),
    crm_client: TwentyCrmClient = Depends(get_crm_client),
    current_user_email: str | None = Depends(get_current_user_email),
) -> CrmShortlistResponse:
    target_email = payload.user_email or current_user_email or request.headers.get("X-User-Email") or ""
    if not target_email:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            bearer = auth[len("Bearer "):].strip()
            if bearer and "@" in bearer:
                target_email = bearer
    raw_ids: list[str] = []
    for aid in payload.account_ids:
        if not aid:
            continue
        val = unquote(str(aid).strip())
        for part in val.split(","):
            cleaned = part.strip().lstrip("@")
            if cleaned and cleaned.lower() not in {"undefined", "null"}:
                raw_ids.append(cleaned)
    raw_ids = list(dict.fromkeys(raw_ids))
    if not raw_ids:
        return CrmShortlistResponse(
            added_count=0,
            redirect_url=f"{request.app.state.settings.crm_frontend_url}/#/authors?import_ids={','.join(raw_ids)}",
        )

    numeric_ids: list[int] = []
    string_ids: list[str] = []
    clean_usernames: list[str] = []
    for raw in raw_ids:
        clean_u = raw.lstrip("@").strip().lower()
        if clean_u:
            clean_usernames.append(clean_u)
        try:
            numeric_ids.append(int(raw))
        except ValueError:
            pass
        string_ids.append(raw)

    logger.info(
        "Shortlist search: raw_ids=%s, clean_usernames=%s, numeric_ids=%s",
        raw_ids,
        clean_usernames,
        numeric_ids,
    )

    conditions = []
    if clean_usernames:
        conditions.append(func.lower(Account.username).in_(clean_usernames))
        conditions.append(func.lower(Account.username).in_([f"@{u}" for u in clean_usernames]))
    if numeric_ids:
        conditions.append(Account.id.in_(numeric_ids))
    if string_ids:
        conditions.append(Account.platform_id.in_(string_ids))

    validity = or_(Account.status == "verified", Account.subscribers_count > 0)
    async with db.async_session() as session:
        stmt = select(Account).where(and_(or_(*conditions), validity))
        result = await session.execute(stmt)
        accounts = list(result.scalars().all())

    accounts = _deduplicate_accounts(accounts)

    logger.info("CRM export: found %d Account records in Postgres for raw_ids=%s", len(accounts), raw_ids)

    if not accounts:
        logger.warning("No accounts found in DB for ids: %s", raw_ids)
        raise HTTPException(status_code=404, detail="Указанные авторы не найдены в базе")

    tasks = [crm_client.upsert_creator(_build_creator_payload(account, target_email)) for account in accounts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for account, res in zip(accounts, results):
        if isinstance(res, Exception):
            logger.error("Error upserting %s: %s", account.id, repr(res))
    success_count = sum(1 for r in results if not isinstance(r, Exception))

    return CrmShortlistResponse(
        added_count=success_count,
        redirect_url=f"{request.app.state.settings.crm_frontend_url}/#/authors?import_ids={','.join(raw_ids)}",
    )


@router.post("/auth/login", response_model=CrmLoginResponse)
async def crm_login(
    payload: CrmLoginRequest,
    db: Database = Depends(get_db),
    crm_client: TwentyCrmClient = Depends(get_crm_client),
) -> CrmLoginResponse:
    email = payload.email.strip()
    async with db.async_session() as session:
        stmt = select(User).where(func.lower(User.email) == email.lower())
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()

    if user is not None:
        if not verify_password(payload.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Неверный email или пароль")
        return CrmLoginResponse(
            token=user.email,
            user={"email": user.email, "name": user.name or user.email.split("@")[0]},
        )

    try:
        data = await crm_client.authenticate(payload.email, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except Exception as exc:
        logger.exception("Twenty login failed")
        raise HTTPException(status_code=500, detail=f"Twenty service error: {exc}")
    token = data.get("token")
    user_data = data.get("user")
    if not isinstance(token, str) or not token or not isinstance(user_data, dict):
        raise HTTPException(status_code=401, detail="Неверный email или пароль")
    return CrmLoginResponse(token=token, user=user_data)


@router.post("/auth/register", response_model=CrmLoginResponse)
async def crm_register(
    payload: CrmRegisterRequest,
    db: Database = Depends(get_db),
) -> CrmLoginResponse:
    email = payload.email.strip()
    async with db.async_session() as session:
        stmt = select(User).where(func.lower(User.email) == email.lower())
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing is not None:
            raise HTTPException(status_code=400, detail="Пользователь с таким email уже зарегистрирован")

        user = User(
            email=email,
            name=payload.name,
            password_hash=hash_password(payload.password),
        )
        session.add(user)
        await session.commit()

    return CrmLoginResponse(
        token=user.email,
        user={"email": user.email, "name": user.name or user.email.split("@")[0]},
    )


@router.get("/creators")
async def crm_creators(
    limit: int = 100,
    crm_client: TwentyCrmClient = Depends(get_crm_client),
    current_user_email: str | None = Depends(get_current_user_email),
) -> dict[str, Any]:
    records = await crm_client.get_creators(limit=limit, user_email=current_user_email)
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


@router.delete("/creators/{creator_id}")
async def crm_delete_creator(
    creator_id: str,
    crm_client: TwentyCrmClient = Depends(get_crm_client),
) -> dict[str, str]:
    existing = await crm_client.find_creator_by_account_id(creator_id)
    if existing is not None:
        dealscount = int(existing.get("dealscount") or existing.get("dealsCount") or 0)
        if dealscount > 0:
            raise HTTPException(
                status_code=400,
                detail="Нельзя удалить автора с историей сделок или перепиской. Переместите его в архив.",
            )
    deleted = await crm_client.delete_creator(creator_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Автор не найден")
    return {"status": "deleted", "creator_id": creator_id}