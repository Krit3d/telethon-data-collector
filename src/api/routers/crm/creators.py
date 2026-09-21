import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.api.dependencies import get_crm_client, get_current_user, get_db
from src.api.schemas import (
    CreatorMessageItem,
    CreatorProfileDetail,
    CreatorSendMessageRequest,
    CrmManualCreatorRequest,
    CrmShortlistRequest,
    CrmShortlistResponse,
    CrmUpdateStatusRequest,
)
from src.api.services.contact_resolver import ContactResolver
from src.api.services.crm_client import TwentyCrmClient
from src.api.services.crm_helpers import (
    STATUS_CRM_TO_UI,
    STATUS_UI_TO_CRM,
    build_creator_payload,
    deduplicate_accounts,
    message_item,
    normalize_platform,
    post_item,
    profile_url,
    resolve_account,
    resolve_target_for_channel,
)
from src.db.database import Database
from src.db.models import Account, Content, Deal, CreatorMessage, User, UserShortlist
from src.parser.creators.core.contacts import (
    is_valid_email,
    is_valid_telegram_handle,
    normalize_phone,
    normalize_telegram_handle,
)
from src.parser.creators.sc_client import ScrapeCreatorsClient

logger = logging.getLogger(__name__)

router = APIRouter(tags=["CRM Creators"])


@router.post("/shortlist", response_model=CrmShortlistResponse)
async def export_to_shortlist(
    payload: CrmShortlistRequest,
    request: Request,
    db: Database = Depends(get_db),
    crm_client: TwentyCrmClient = Depends(get_crm_client),
    current_user: User = Depends(get_current_user),
) -> CrmShortlistResponse:
    target_email = current_user.email
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

        accounts = deduplicate_accounts(accounts)

        logger.info("CRM export: found %d Account records in Postgres for raw_ids=%s", len(accounts), raw_ids)

        if accounts:
            now = datetime.now(timezone.utc)
            rows = [
                {
                    "user_id": current_user.id,
                    "account_id": account.id,
                    "status": "Свободен",
                    "source": "search",
                    "created_at": now,
                    "updated_at": now,
                }
                for account in accounts
            ]
            await session.execute(
                pg_insert(UserShortlist)
                .values(rows)
                .on_conflict_do_nothing(index_elements=["user_id", "account_id"])
            )
            await session.commit()

    if not accounts:
        logger.warning("No accounts found in DB for ids: %s", raw_ids)
        raise HTTPException(status_code=404, detail="Указанные авторы не найдены в базе")

    tasks = [crm_client.upsert_creator(build_creator_payload(account, target_email)) for account in accounts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for account, res in zip(accounts, results):
        if isinstance(res, Exception):
            logger.error("Error upserting %s: %s", account.id, repr(res))

    return CrmShortlistResponse(
        added_count=len(accounts),
        redirect_url=f"{request.app.state.settings.crm_frontend_url}/#/authors?import_ids={','.join(raw_ids)}",
    )


def _deterministic_creator_id(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") & 0x7FFFFFFFFFFFFFFF


def _build_contacts_metadata(payload: CrmManualCreatorRequest) -> dict[str, list[str]]:
    contacts: dict[str, list[str]] = {}
    if payload.telegram_commercial:
        handle = normalize_telegram_handle(payload.telegram_commercial)
        if is_valid_telegram_handle(handle):
            contacts["advertising_telegrams"] = [handle]
    if payload.telegram_personal:
        handle = normalize_telegram_handle(payload.telegram_personal)
        if is_valid_telegram_handle(handle):
            contacts["telegram_personal"] = [handle]
    if payload.email:
        email = payload.email.strip().lower()
        if is_valid_email(email):
            contacts["emails"] = [email]
    if payload.phone:
        phone = normalize_phone(payload.phone)
        if phone:
            contacts["phones"] = [phone]
    return contacts


@router.post("/creators/manual")
async def add_creator_manual(
    payload: CrmManualCreatorRequest,
    request: Request,
    db: Database = Depends(get_db),
    crm_client: TwentyCrmClient = Depends(get_crm_client),
    current_user: User = Depends(get_current_user),
) -> dict[str, object]:
    platform = payload.platform.upper().strip()
    clean_username = payload.username.lstrip("@").strip().lower()
    if not clean_username:
        raise HTTPException(status_code=400, detail="Username не может быть пустым")

    platform_id: str
    account_id: int
    title: str = clean_username
    scraped_title: str | None = None
    subscribers_count: int | None = payload.subscribers_count

    if platform == "INSTAGRAM":
        try:
            async with ScrapeCreatorsClient(request.app.state.settings) as client:
                response = await client.get(
                    endpoint="/v1/instagram/profile",
                    params={"handle": clean_username},
                )
            data = response.get("data") or response
            user = data.get("user") or data if isinstance(data, dict) else {}
            raw_pid = user.get("id") or user.get("pk") or user.get("media_id")
            if raw_pid is not None:
                try:
                    account_id = int(raw_pid)
                except (ValueError, TypeError):
                    account_id = _deterministic_creator_id(f"instagram:{clean_username}")
                platform_id = str(account_id)
            else:
                account_id = _deterministic_creator_id(f"instagram:{clean_username}")
                platform_id = str(account_id)
            scraped_title = user.get("full_name") or user.get("title") or user.get("name")
            scraped_subs = user.get("edge_followed_by")
            if isinstance(scraped_subs, dict):
                scraped_subs = scraped_subs.get("count")
            if scraped_subs is None:
                scraped_subs = user.get("followers") or user.get("followers_count") or user.get("follower_count")
            if scraped_subs is not None:
                try:
                    subscribers_count = int(scraped_subs)
                except (ValueError, TypeError):
                    pass
        except Exception:
            logger.warning(
                "Scrape Creators profile fetch failed for %s, falling back to deterministic id",
                clean_username,
                exc_info=True,
            )
            account_id = _deterministic_creator_id(f"instagram:{clean_username}")
            platform_id = str(account_id)
    elif platform == "TELEGRAM":
        account_id = _deterministic_creator_id(f"telegram:{clean_username}")
        platform_id = str(account_id)
    else:
        raise HTTPException(status_code=400, detail="Поддерживаются только INSTAGRAM и TELEGRAM")

    if payload.title:
        title = payload.title
    elif scraped_title:
        title = str(scraped_title)
    else:
        title = clean_username

    contacts = _build_contacts_metadata(payload)
    raw_metadata: dict[str, object] = {"contacts": contacts}

    now = datetime.now(timezone.utc)
    async with db.async_session() as session:
        existing_stmt = select(Account).where(
            or_(
                and_(Account.platform == platform, Account.platform_id == platform_id),
                and_(Account.platform == platform, func.lower(Account.username) == clean_username),
            )
        )
        existing_result = await session.execute(existing_stmt)
        account = existing_result.scalars().first()

        if account is None:
            account = Account(
                id=account_id,
                platform=platform,
                platform_id=platform_id,
                username=clean_username,
                title=title,
                subscribers_count=subscribers_count,
                status="manual",
                raw_metadata=raw_metadata,
            )
            session.add(account)
            await session.flush()
        else:
            existing_meta = account.raw_metadata if isinstance(account.raw_metadata, dict) else {}
            existing_contacts = existing_meta.get("contacts")
            if not isinstance(existing_contacts, dict):
                existing_contacts = {}
            merged_contacts: dict[str, list[str]] = {}
            for key in ("advertising_telegrams", "telegram_personal", "emails", "phones"):
                merged = list(existing_contacts.get(key, []))
                for item in contacts.get(key, []):
                    if item not in merged:
                        merged.append(item)
                if merged:
                    merged_contacts[key] = merged
            existing_meta["contacts"] = merged_contacts
            account.raw_metadata = existing_meta
            if account.status != "verified":
                account.status = "manual"
            if title:
                account.title = title
            if subscribers_count is not None:
                account.subscribers_count = subscribers_count

        await session.execute(
            pg_insert(UserShortlist)
            .values(
                user_id=current_user.id,
                account_id=account.id,
                status="Свободен",
                source="manual",
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=["user_id", "account_id"],
                set_={
                    "source": "manual",
                    "updated_at": now,
                },
            )
        )
        await session.commit()
        await session.refresh(account)

    try:
        await crm_client.upsert_creator(build_creator_payload(account, current_user.email))
    except Exception:
        logger.warning("Failed to upsert manually added creator in Twenty CRM", exc_info=True)

    return {
        "id": str(account.id),
        "accountid": str(account.id),
        "accountId": str(account.id),
        "handle": f"@{account.username.lstrip('@')}" if account.username else "",
        "username": account.username,
        "name": account.title or account.username or "",
        "platform": normalize_platform(account.platform),
        "followers": int(account.subscribers_count or 0),
        "subscribers_count": int(account.subscribers_count or 0),
        "status": "Свободен",
        "source": "manual",
        "account_status": account.status,
        "title": account.title or account.username or "",
    }


@router.get("/creators")
async def crm_creators(
    limit: int = 100,
    offset: int = 0,
    status: str | None = None,
    db: Database = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, object]:
    async with db.async_session() as session:
        stmt = (
            select(UserShortlist, Account)
            .join(Account, UserShortlist.account_id == Account.id)
            .where(UserShortlist.user_id == current_user.id)
        )
        if status:
            stmt = stmt.where(UserShortlist.status == STATUS_CRM_TO_UI.get(status, status))
        stmt = stmt.order_by(UserShortlist.created_at.desc()).limit(limit).offset(offset)
        result = await session.execute(stmt)
        rows = result.all()
        if not rows:
            return {"data": [], "total": 0}

        account_ids = [row[1].id for row in rows]

        active_stmt = (
            select(Deal.account_id, func.count(Deal.id))
            .where(
                Deal.user_id == current_user.id,
                Deal.account_id.in_(account_ids),
                Deal.stage.between(1, 6),
            )
            .group_by(Deal.account_id)
        )
        active_result = await session.execute(active_stmt)
        active_map = {account_id: count for account_id, count in active_result.all()}

        total_stmt = (
            select(Deal.account_id, func.count(Deal.id))
            .where(Deal.user_id == current_user.id, Deal.account_id.in_(account_ids))
            .group_by(Deal.account_id)
        )
        total_result = await session.execute(total_stmt)
        total_map = {account_id: count for account_id, count in total_result.all()}

        total_count_stmt = select(func.count(UserShortlist.user_id)).where(UserShortlist.user_id == current_user.id)
        if status:
            total_count_stmt = total_count_stmt.where(UserShortlist.status == STATUS_CRM_TO_UI.get(status, status))
        total_count_result = await session.execute(total_count_stmt)
        total_count = total_count_result.scalar() or 0

        creators = []
        for shortlist, account in rows:
            followers = int(account.subscribers_count or 0)
            er = round(float(account.static_avg_er or 0.0), 2)
            if er > 0:
                avg_reach = int(followers * (er / 100))
                calc_cpm = int((followers * (er / 100) / 1000) * 200)
            else:
                avg_reach = int(followers * 0.1)
                calc_cpm = 0
            cpm = shortlist.custom_cpm if shortlist.custom_cpm is not None else calc_cpm
            active_count = active_map.get(account.id, 0)
            if shortlist.status == "В архиве":
                creator_status = "В архиве"
            elif active_count > 0:
                creator_status = "В сделке"
            else:
                creator_status = shortlist.status
            deals_count = total_map.get(account.id, 0)
            creators.append(
                {
                    "id": str(account.id),
                    "accountid": str(account.id),
                    "accountId": str(account.id),
                    "name": account.title or account.username or "",
                    "source": shortlist.source,
                    "account_status": account.status,
                    "title": account.title or account.username or "",
                    "platform": normalize_platform(account.platform),
                    "handle": f"@{account.username.lstrip('@')}" if account.username else "",
                    "followers": followers,
                    "er": er,
                    "avgreach": avg_reach,
                    "cpm": cpm,
                    "niche": account.category_path or "Общее",
                    "status": creator_status,
                    "dealscount": deals_count,
                    "dealsCount": deals_count,
                    "notes": shortlist.notes,
                    "useremail": current_user.email,
                }
            )
        return {"data": creators, "total": total_count}


@router.patch("/creators/{creator_id}")
async def crm_update_creator_status(
    creator_id: str,
    payload: CrmUpdateStatusRequest,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
    crm_client: TwentyCrmClient = Depends(get_crm_client),
) -> dict[str, object]:
    mapped_status = STATUS_CRM_TO_UI.get(payload.status, payload.status)
    async with db.async_session() as session:
        account = await resolve_account(session, crm_client, creator_id, current_user.email)
        if account is None:
            raise HTTPException(status_code=404, detail="Автор не найден")
        now = datetime.now(timezone.utc)
        await session.execute(
            pg_insert(UserShortlist)
            .values(
                user_id=current_user.id,
                account_id=account.id,
                status=mapped_status,
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=["user_id", "account_id"],
                set_={
                    "status": mapped_status,
                    "updated_at": now,
                },
            )
        )
        if payload.status.strip().lower() in {"archived", "в архиве"} and payload.archive_active_deals:
            await session.execute(
                update(Deal)
                .where(
                    Deal.user_id == current_user.id,
                    Deal.account_id == account.id,
                    Deal.stage >= 1,
                    Deal.stage <= 6,
                )
                .values(stage=0, updated_at=datetime.now(timezone.utc))
            )
        await session.commit()
    try:
        await crm_client.update_creator_status(
            creator_id, STATUS_UI_TO_CRM.get(payload.status, payload.status), current_user.email
        )
    except Exception:
        logger.warning("Failed to update creator status in Twenty CRM", exc_info=True)
    return {
        "status": "ok",
        "creator_id": creator_id,
        "new_status": mapped_status,
    }


@router.delete("/creators/{creator_id}")
async def crm_delete_creator(
    creator_id: str,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
    crm_client: TwentyCrmClient = Depends(get_crm_client),
) -> dict[str, str]:
    async with db.async_session() as session:
        account = await resolve_account(session, crm_client, creator_id, current_user.email)
        if account is not None:
            await session.execute(
                delete(UserShortlist).where(
                    UserShortlist.user_id == current_user.id,
                    UserShortlist.account_id == account.id,
                )
            )
            await session.commit()
    try:
        await crm_client.delete_creator(creator_id, current_user.email)
    except Exception:
        logger.warning("Failed to delete creator in Twenty CRM", exc_info=True)
    return {"status": "deleted", "creator_id": creator_id}


@router.get("/creators/{creator_id}", response_model=CreatorProfileDetail)
async def get_creator_detail(
    creator_id: str,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
    crm_client: TwentyCrmClient = Depends(get_crm_client),
) -> CreatorProfileDetail:
    async with db.async_session() as session:
        account = await resolve_account(session, crm_client, creator_id)
        if account is None:
            raise HTTPException(status_code=404, detail="Автор не найден")

        content_stmt = (
            select(Content)
            .where(Content.account_id == account.id)
            .order_by(Content.published_at.desc())
            .limit(20)
        )
        content_result = await session.execute(content_stmt)
        posts = list(content_result.scalars().all())

        active_deals_stmt = (
            select(func.count(Deal.id))
            .where(
                Deal.account_id == account.id,
                Deal.user_id == current_user.id,
                Deal.stage >= 1,
                Deal.stage <= 6,
            )
        )
        active_deals_result = await session.execute(active_deals_stmt)
        active_deals_count = active_deals_result.scalar() or 0

        total_deals_stmt = (
            select(func.count(Deal.id))
            .where(Deal.account_id == account.id, Deal.user_id == current_user.id)
        )
        total_deals_result = await session.execute(total_deals_stmt)
        total_deals_count = total_deals_result.scalar() or 0

        shortlist_stmt = (
            select(UserShortlist)
            .where(
                UserShortlist.user_id == current_user.id,
                UserShortlist.account_id == account.id,
            )
        )
        shortlist_result = await session.execute(shortlist_stmt)
        shortlist = shortlist_result.scalar_one_or_none()

    followers = int(account.subscribers_count or 0)
    er = round(float(account.static_avg_er or 0.0), 2)
    if er > 0:
        avg_reach = int(followers * (er / 100))
        calc_cpm = int((followers * (er / 100) / 1000) * 200)
    else:
        avg_reach = int(followers * 0.1)
        calc_cpm = 0
    cpm = shortlist.custom_cpm if (shortlist is not None and shortlist.custom_cpm is not None) else calc_cpm

    platform = normalize_platform(account.platform)
    if shortlist is not None:
        if shortlist.status == "В архиве":
            status = "В архиве"
        elif active_deals_count > 0:
            status = "В сделке"
        else:
            status = shortlist.status
    else:
        normalized_status = (account.status or "").strip().lower()
        if normalized_status in {"archived", "в архиве"}:
            status = "В архиве"
        elif active_deals_count > 0:
            status = "В сделке"
        else:
            status = "Свободен"
    return CreatorProfileDetail(
        id=str(account.id),
        platform=platform,
        username=account.username,
        title=account.title,
        account_status=account.status,
        description=account.description,
        subscribers_count=followers,
        static_avg_er=er,
        category_path=account.category_path,
        country=account.country,
        city=account.city,
        gender=account.gender,
        status=status,
        profile_url=profile_url(platform, account.username),
        cpm=cpm,
        avg_reach=avg_reach,
        deals_count=total_deals_count,
        posts=[post_item(post, platform) for post in posts],
    )


@router.get("/creators/{creator_id}/messages", response_model=list[CreatorMessageItem])
async def get_creator_messages(
    creator_id: str,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
    crm_client: TwentyCrmClient = Depends(get_crm_client),
) -> list[CreatorMessageItem]:
    async with db.async_session() as session:
        account = await resolve_account(session, crm_client, creator_id)
        if account is None:
            raise HTTPException(status_code=404, detail="Автор не найден")
        update_stmt = (
            update(CreatorMessage)
            .where(
                CreatorMessage.account_id == account.id,
                CreatorMessage.user_id == current_user.id,
                CreatorMessage.sender_type != "user",
                CreatorMessage.is_read.is_(False),
            )
            .values(is_read=True)
        )
        await session.execute(update_stmt)
        await session.commit()
        msg_stmt = (
            select(CreatorMessage)
            .where(
                CreatorMessage.account_id == account.id,
                CreatorMessage.user_id == current_user.id,
            )
            .order_by(CreatorMessage.created_at.asc())
        )
        msg_result = await session.execute(msg_stmt)
        messages = list(msg_result.scalars().all())
        return [message_item(message) for message in messages]


@router.post("/creators/{creator_id}/messages", response_model=CreatorMessageItem)
async def send_creator_message(
    creator_id: str,
    payload: CreatorSendMessageRequest,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
    crm_client: TwentyCrmClient = Depends(get_crm_client),
) -> CreatorMessageItem:
    if not (payload.text and payload.text.strip()) and not payload.media_url:
        raise HTTPException(status_code=400, detail="Сообщение не может быть пустым")
    async with db.async_session() as session:
        account = await resolve_account(session, crm_client, creator_id)
        if account is None:
            raise HTTPException(status_code=404, detail="Автор не найден")
        deal_stmt = (
            select(Deal)
            .where(
                Deal.account_id == account.id,
                Deal.user_id == current_user.id,
                Deal.stage.in_([1, 2, 3, 4, 5, 6]),
            )
            .order_by(Deal.updated_at.desc())
            .limit(1)
        )
        deal_result = await session.execute(deal_stmt)
        deal = deal_result.scalar_one_or_none()
        resolution = ContactResolver.resolve(account.raw_metadata, account.username)
        if payload.preferred_channel:
            target_channel = payload.preferred_channel
            target_handle = resolve_target_for_channel(account, target_channel) or resolution.channel_target
        else:
            target_channel = resolution.channel_type
            target_handle = resolution.channel_target
        message = CreatorMessage(
            user_id=current_user.id,
            account_id=account.id,
            deal_id=deal.id if deal else None,
            sender_type=payload.sender_type,
            text=payload.text.strip() if payload.text else None,
            channel_type=target_channel,
            channel_target=target_handle,
            external_message_id=None,
            media_url=payload.media_url,
            media_name=payload.media_name,
            media_type=payload.media_type,
        )
        session.add(message)
        await session.commit()
        await session.refresh(message)
        return message_item(message)