import asyncio
import base64
import json
import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.orm import joinedload

from src.api.dependencies import get_crm_client, get_db
from src.api.schemas import (
    CommunicationChannelItem,
    CrmLoginRequest,
    CrmLoginResponse,
    CrmRegisterRequest,
    CrmShortlistRequest,
    CrmShortlistResponse,
    CrmUpdateStatusRequest,
    DealAuthorSummary,
    DealCreateRequest,
    DealItem,
    DealMessageItem,
    DealSendMessageRequest,
    DealUpdateRequest,
)
from src.api.services.crm_client import TwentyCrmClient
from src.db.database import Database
from src.db.models import Account, Deal, DealMessage, User
from src.utils.security import create_access_token, decode_access_token, hash_password, verify_password

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


def _decode_jwt_email(token: str) -> str | None:
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload + padding)
        data = json.loads(decoded)
    except (ValueError, TypeError):
        return None
    email = data.get("email")
    if isinstance(email, str) and email:
        return email
    sub = data.get("sub")
    if isinstance(sub, str) and "@" in sub:
        return sub
    return None


def get_current_user_email(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[len("Bearer "):].strip()
        if token:
            if "@" in token:
                return token
            if "." in token:
                email = _decode_jwt_email(token)
                if email:
                    return email
    query_email = request.query_params.get("user_email")
    if query_email:
        return query_email
    header_email = request.headers.get("X-User-Email")
    if header_email:
        return header_email
    return None


async def get_current_user(
    request: Request,
    db: Database = Depends(get_db),
) -> User:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Недействительный токен авторизации")
    token = auth[len("Bearer "):].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Недействительный токен авторизации")
    payload = decode_access_token(token, request.app.state.settings.secret_key)
    if payload is None:
        raise HTTPException(status_code=401, detail="Недействительный токен авторизации")
    email = payload.get("sub")
    user_id = payload.get("user_id")
    async with db.async_session() as session:
        if isinstance(user_id, int):
            stmt = select(User).where(User.id == user_id)
        elif isinstance(email, str) and email:
            stmt = select(User).where(func.lower(User.email) == email.lower())
        else:
            raise HTTPException(status_code=401, detail="Недействительный токен авторизации")
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="Пользователь не найден")
    return user


def _author_summary(account: Account | None) -> DealAuthorSummary | None:
    if account is None:
        return None
    return DealAuthorSummary(
        id=account.id,
        platform=account.platform,
        username=account.username,
        title=account.title,
        subscribers_count=account.subscribers_count,
        static_avg_er=account.static_avg_er,
        category_path=account.category_path,
    )


def _message_item(message: DealMessage) -> DealMessageItem:
    return DealMessageItem(
        id=message.id,
        deal_id=message.deal_id,
        sender_type=message.sender_type,
        text=message.text,
        is_read=message.is_read,
        created_at=message.created_at,
    )


def _deal_item(
    deal: Deal,
    unread_count: int = 0,
    last_message: DealMessageItem | None = None,
    account: Account | None = None,
) -> DealItem:
    return DealItem(
        id=deal.id,
        user_id=deal.user_id,
        account_id=deal.account_id,
        title=deal.title,
        stage=deal.stage,
        budget=deal.budget,
        type=deal.type,
        brand_name=deal.brand_name,
        pub_date=deal.pub_date,
        terms=deal.terms,
        created_at=deal.created_at,
        updated_at=deal.updated_at,
        author=_author_summary(account if account is not None else deal.account),
        last_message=last_message,
        unread_count=unread_count,
    )


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
    request: Request,
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
        access_token = create_access_token(
            {"sub": user.email, "user_id": user.id},
            request.app.state.settings.secret_key,
        )
        return CrmLoginResponse(
            token=access_token,
            user={"id": user.id, "email": user.email, "name": user.name or user.email.split("@")[0]},
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
    user_email = user_data.get("email")
    if not isinstance(user_email, str) or not user_email:
        user_email = email
    user_name = user_data.get("name")
    if not isinstance(user_name, str) or not user_name:
        user_name = user_email.split("@")[0]
    async with db.async_session() as session:
        stmt = select(User).where(func.lower(User.email) == user_email.lower())
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        if user is None:
            user = User(email=user_email, name=user_name, password_hash="")
            session.add(user)
            await session.commit()
        user_id = user.id
        user_email = user.email
        user_name = user.name or user_email.split("@")[0]
    access_token = create_access_token(
        {"sub": user_email, "user_id": user_id},
        request.app.state.settings.secret_key,
    )
    return CrmLoginResponse(
        token=access_token,
        user={"id": user_id, "email": user_email, "name": user_name},
    )


@router.post("/auth/register", response_model=CrmLoginResponse)
async def crm_register(
    payload: CrmRegisterRequest,
    request: Request,
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
        user_id = user.id
        user_email = user.email
        user_name = user.name or user_email.split("@")[0]

    access_token = create_access_token(
        {"sub": user_email, "user_id": user_id},
        request.app.state.settings.secret_key,
    )
    return CrmLoginResponse(
        token=access_token,
        user={"id": user_id, "email": user_email, "name": user_name},
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
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
    crm_client: TwentyCrmClient = Depends(get_crm_client),
) -> dict[str, str]:
    numeric_account_id: int | None = None
    try:
        numeric_account_id = int(creator_id)
    except ValueError:
        creator_record = await crm_client.find_creator_by_account_id(creator_id)
        if creator_record is not None:
            account_id = creator_record.get("accountid") or creator_record.get("accountId")
            if account_id is not None:
                try:
                    numeric_account_id = int(account_id)
                except (ValueError, TypeError):
                    numeric_account_id = None
    if numeric_account_id is not None:
        async with db.async_session() as session:
            await session.execute(
                delete(Deal).where(
                    Deal.account_id == numeric_account_id,
                    Deal.user_id == current_user.id,
                )
            )
            await session.commit()
    deleted = await crm_client.delete_creator(creator_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Автор не найден")
    return {"status": "deleted", "creator_id": creator_id}


@router.get("/deals", response_model=list[DealItem])
async def list_deals(
    stage: int | None = None,
    search: str | None = None,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
) -> list[DealItem]:
    async with db.async_session() as session:
        if search:
            pattern = f"%{search}%"
            stmt = (
                select(Deal)
                .outerjoin(Deal.account)
                .options(joinedload(Deal.account))
                .where(Deal.user_id == current_user.id)
                .where(
                    or_(
                        Deal.title.ilike(pattern),
                        Account.title.ilike(pattern),
                        Account.username.ilike(pattern),
                    )
                )
            )
            if stage is not None:
                stmt = stmt.where(Deal.stage == stage)
        else:
            stmt = select(Deal).options(joinedload(Deal.account)).where(Deal.user_id == current_user.id)
            if stage is not None:
                stmt = stmt.where(Deal.stage == stage)
        stmt = stmt.order_by(Deal.updated_at.desc())
        result = await session.execute(stmt)
        deals = list(result.scalars().unique().all())
        deal_ids = [d.id for d in deals]
        unread_map: dict[int, int] = {}
        last_map: dict[int, DealMessageItem] = {}
        if deal_ids:
            unread_stmt = (
                select(DealMessage.deal_id, func.count(DealMessage.id))
                .where(DealMessage.deal_id.in_(deal_ids))
                .where(DealMessage.is_read.is_(False))
                .where(DealMessage.sender_type != "user")
                .group_by(DealMessage.deal_id)
            )
            unread_result = await session.execute(unread_stmt)
            for deal_id, count in unread_result.all():
                unread_map[deal_id] = count
            last_subq = (
                select(
                    DealMessage.id,
                    DealMessage.deal_id,
                    DealMessage.sender_type,
                    DealMessage.text,
                    DealMessage.is_read,
                    DealMessage.created_at,
                    func.row_number()
                    .over(
                        partition_by=DealMessage.deal_id,
                        order_by=DealMessage.created_at.desc(),
                    )
                    .label("rn"),
                )
                .where(DealMessage.deal_id.in_(deal_ids))
                .subquery()
            )
            last_stmt = select(last_subq).where(last_subq.c.rn == 1)
            last_result = await session.execute(last_stmt)
            for row in last_result.all():
                last_map[row.deal_id] = DealMessageItem(
                    id=row.id,
                    deal_id=row.deal_id,
                    sender_type=row.sender_type,
                    text=row.text,
                    is_read=row.is_read,
                    created_at=row.created_at,
                )
        return [
            _deal_item(deal, unread_map.get(deal.id, 0), last_map.get(deal.id))
            for deal in deals
        ]


@router.post("/deals", response_model=DealItem)
async def create_deal(
    payload: DealCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
    crm_client: TwentyCrmClient = Depends(get_crm_client),
) -> DealItem:
    numeric_account_id = int(str(payload.account_id).strip())
    async with db.async_session() as session:
        account_stmt = select(Account).where(
            or_(
                Account.id == numeric_account_id,
                Account.platform_id == str(payload.account_id).strip(),
            )
        )
        account_result = await session.execute(account_stmt)
        account = account_result.scalar_one_or_none()
        if account is None:
            raise HTTPException(status_code=404, detail="Автор не найден")
        deal = Deal(
            user_id=current_user.id,
            account_id=account.id,
            title=payload.title,
            stage=1,
            budget=payload.budget,
            type=payload.type,
            brand_name=payload.brand_name,
            pub_date=payload.pub_date,
            terms=payload.terms,
        )
        session.add(deal)
        await session.flush()
        if payload.initial_message:
            msg = DealMessage(deal_id=deal.id, sender_type="user", text=payload.initial_message)
            session.add(msg)
            await session.flush()
            last_message_item = _message_item(msg)
        else:
            last_message_item = None
        await session.commit()
        await session.refresh(deal)
        item = _deal_item(deal, unread_count=0, last_message=last_message_item, account=account)
    try:
        creator_record = await crm_client.find_creator_by_account_id(str(payload.account_id))
        if creator_record is not None and "id" in creator_record:
            await crm_client.update_creator_status(creator_record["id"], "V_SDELKE")
    except Exception:
        logger.warning("Failed to update creator status in Twenty CRM", exc_info=True)
    return item


@router.get("/deals/{deal_id}", response_model=DealItem)
async def get_deal(
    deal_id: int,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
) -> DealItem:
    async with db.async_session() as session:
        stmt = select(Deal).options(joinedload(Deal.account)).where(Deal.id == deal_id, Deal.user_id == current_user.id)
        result = await session.execute(stmt)
        deal = result.scalar_one_or_none()
        if deal is None:
            raise HTTPException(status_code=404, detail="Сделка не найдена")
        unread_stmt = (
            select(func.count(DealMessage.id))
            .where(DealMessage.deal_id == deal.id)
            .where(DealMessage.is_read.is_(False))
            .where(DealMessage.sender_type != "user")
        )
        unread_result = await session.execute(unread_stmt)
        unread_count = unread_result.scalar() or 0
        last_stmt = (
            select(DealMessage)
            .where(DealMessage.deal_id == deal.id)
            .order_by(DealMessage.created_at.desc())
            .limit(1)
        )
        last_result = await session.execute(last_stmt)
        last_message = last_result.scalar_one_or_none()
        return _deal_item(
            deal,
            unread_count,
            _message_item(last_message) if last_message is not None else None,
        )


@router.patch("/deals/{deal_id}", response_model=DealItem)
async def update_deal(
    deal_id: int,
    payload: DealUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
) -> DealItem:
    async with db.async_session() as session:
        stmt = select(Deal).options(joinedload(Deal.account)).where(Deal.id == deal_id, Deal.user_id == current_user.id)
        result = await session.execute(stmt)
        deal = result.scalar_one_or_none()
        if deal is None:
            raise HTTPException(status_code=404, detail="Сделка не найдена")
        account = deal.account
        updates = payload.model_dump(exclude_unset=True)
        for field, value in updates.items():
            if value is not None:
                setattr(deal, field, value)
        deal.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(deal)
        unread_stmt = (
            select(func.count(DealMessage.id))
            .where(DealMessage.deal_id == deal.id)
            .where(DealMessage.is_read.is_(False))
            .where(DealMessage.sender_type != "user")
        )
        unread_result = await session.execute(unread_stmt)
        unread_count = unread_result.scalar() or 0
        last_stmt = (
            select(DealMessage)
            .where(DealMessage.deal_id == deal.id)
            .order_by(DealMessage.created_at.desc())
            .limit(1)
        )
        last_result = await session.execute(last_stmt)
        last_message = last_result.scalar_one_or_none()
        item = _deal_item(
            deal,
            unread_count,
            _message_item(last_message) if last_message is not None else None,
            account=account,
        )
    return item


@router.delete("/deals/{deal_id}")
async def delete_deal(
    deal_id: int,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    async with db.async_session() as session:
        stmt = select(Deal).where(Deal.id == deal_id, Deal.user_id == current_user.id)
        result = await session.execute(stmt)
        deal = result.scalar_one_or_none()
        if deal is None:
            raise HTTPException(status_code=404, detail="Сделка не найдена")
        await session.delete(deal)
        await session.commit()
    return {"status": "deleted", "deal_id": deal_id}


@router.get("/deals/{deal_id}/messages", response_model=list[DealMessageItem])
async def get_deal_messages(
    deal_id: int,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
) -> list[DealMessageItem]:
    async with db.async_session() as session:
        deal_stmt = select(Deal).where(Deal.id == deal_id, Deal.user_id == current_user.id)
        deal_result = await session.execute(deal_stmt)
        deal = deal_result.scalar_one_or_none()
        if deal is None:
            raise HTTPException(status_code=404, detail="Сделка не найдена")
        update_stmt = (
            update(DealMessage)
            .where(
                DealMessage.deal_id == deal.id,
                DealMessage.sender_type != "user",
                DealMessage.is_read.is_(False),
            )
            .values(is_read=True)
        )
        await session.execute(update_stmt)
        await session.commit()
        msg_stmt = (
            select(DealMessage)
            .where(DealMessage.deal_id == deal.id)
            .order_by(DealMessage.created_at.asc())
        )
        msg_result = await session.execute(msg_stmt)
        messages = list(msg_result.scalars().all())
        return [_message_item(message) for message in messages]


@router.post("/deals/{deal_id}/messages", response_model=DealMessageItem)
async def send_deal_message(
    deal_id: int,
    payload: DealSendMessageRequest,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
) -> DealMessageItem:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Текст сообщения не может быть пустым")
    async with db.async_session() as session:
        deal_stmt = select(Deal).where(Deal.id == deal_id, Deal.user_id == current_user.id)
        deal_result = await session.execute(deal_stmt)
        deal = deal_result.scalar_one_or_none()
        if deal is None:
            raise HTTPException(status_code=404, detail="Сделка не найдена")
        message = DealMessage(
            deal_id=deal.id,
            sender_type=payload.sender_type,
            text=payload.text.strip(),
        )
        session.add(message)
        deal.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(message)
        return _message_item(message)


@router.get("/communications", response_model=list[CommunicationChannelItem])
async def list_communications(
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
) -> list[CommunicationChannelItem]:
    async with db.async_session() as session:
        stmt = select(Deal).options(joinedload(Deal.account)).where(Deal.user_id == current_user.id)
        result = await session.execute(stmt)
        deals = list(result.scalars().unique().all())
        deal_ids = [d.id for d in deals]
        unread_map: dict[int, int] = {}
        last_map: dict[int, tuple[str, datetime]] = {}
        if deal_ids:
            unread_stmt = (
                select(DealMessage.deal_id, func.count(DealMessage.id))
                .where(DealMessage.deal_id.in_(deal_ids))
                .where(DealMessage.is_read.is_(False))
                .where(DealMessage.sender_type != "user")
                .group_by(DealMessage.deal_id)
            )
            unread_result = await session.execute(unread_stmt)
            for deal_id, count in unread_result.all():
                unread_map[deal_id] = count
            last_subq = (
                select(
                    DealMessage.deal_id,
                    DealMessage.text,
                    DealMessage.created_at,
                    func.row_number()
                    .over(
                        partition_by=DealMessage.deal_id,
                        order_by=DealMessage.created_at.desc(),
                    )
                    .label("rn"),
                )
                .where(DealMessage.deal_id.in_(deal_ids))
                .subquery()
            )
            last_stmt = select(last_subq).where(last_subq.c.rn == 1)
            last_result = await session.execute(last_stmt)
            for row in last_result.all():
                last_map[row.deal_id] = (row.text, row.created_at)
        channels: list[CommunicationChannelItem] = []
        for deal in deals:
            account = deal.account
            last_text, last_time = last_map.get(deal.id, (None, None))
            channels.append(
                CommunicationChannelItem(
                    deal_id=deal.id,
                    author_id=account.id if account is not None else deal.account_id,
                    author_name=account.title if account is not None else "",
                    author_handle=(
                        f"@{account.username.lstrip('@')}"
                        if (account is not None and account.username)
                        else (f"@{account.title}" if (account is not None and account.title) else "")
                    ),
                    platform=account.platform if account is not None else "",
                    deal_title=deal.title,
                    stage=deal.stage,
                    last_message=last_text or "",
                    last_message_time=last_time or deal.updated_at or deal.created_at,
                    unread_count=unread_map.get(deal.id, 0),
                )
            )
    channels.sort(key=lambda channel: channel.last_message_time, reverse=True)
    return channels