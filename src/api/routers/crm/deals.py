import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import joinedload

from src.api.dependencies import get_crm_client, get_current_user, get_db
from src.api.schemas import (
    CreatorMessageItem,
    CreatorSendMessageRequest,
    DealCreateRequest,
    DealItem,
    DealUpdateRequest,
)
from src.api.services.contact_resolver import ContactResolver
from src.api.services.crm_client import TwentyCrmClient
from src.api.services.crm_helpers import (
    author_summary,
    message_item,
    resolve_account,
    resolve_target_for_channel,
)
from src.db.database import Database
from src.db.models import Account, CreatorMessage, Deal, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/deals", tags=["CRM Deals"])

FORBIDDEN_DEAL_FIELDS = {"id", "user_id", "account_id", "created_at"}


def _deal_item(
    deal: Deal,
    unread_count: int = 0,
    last_message: CreatorMessageItem | None = None,
    account: Account | None = None,
) -> DealItem:
    return DealItem(
        id=deal.id,
        user_id=deal.user_id,
        account_id=str(deal.account_id),
        title=deal.title,
        stage=deal.stage,
        budget=deal.budget,
        type=deal.type,
        brand_name=deal.brand_name,
        pub_date=deal.pub_date,
        terms=deal.terms,
        created_at=deal.created_at,
        updated_at=deal.updated_at,
        author=author_summary(account if account is not None else deal.account),
        last_message=last_message,
        unread_count=unread_count,
    )


@router.get("", response_model=list[DealItem])
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
        last_map: dict[int, CreatorMessageItem] = {}
        if deal_ids:
            unread_stmt = (
                select(CreatorMessage.deal_id, func.count(CreatorMessage.id))
                .where(CreatorMessage.deal_id.in_(deal_ids))
                .where(CreatorMessage.is_read.is_(False))
                .where(CreatorMessage.sender_type != "user")
                .group_by(CreatorMessage.deal_id)
            )
            unread_result = await session.execute(unread_stmt)
            for deal_id, count in unread_result.all():
                unread_map[deal_id] = count
            last_subq = (
                select(
                    CreatorMessage.id,
                    CreatorMessage.user_id,
                    CreatorMessage.account_id,
                    CreatorMessage.deal_id,
                    CreatorMessage.sender_type,
                    CreatorMessage.text,
                    CreatorMessage.is_read,
                    CreatorMessage.created_at,
                    CreatorMessage.media_url,
                    CreatorMessage.media_name,
                    CreatorMessage.media_type,
                    func.row_number()
                    .over(
                        partition_by=CreatorMessage.deal_id,
                        order_by=CreatorMessage.created_at.desc(),
                    )
                    .label("rn"),
                )
                .where(CreatorMessage.deal_id.in_(deal_ids))
                .subquery()
            )
            last_stmt = select(last_subq).where(last_subq.c.rn == 1)
            last_result = await session.execute(last_stmt)
            for row in last_result.all():
                last_map[row.deal_id] = CreatorMessageItem(
                    id=row.id,
                    user_id=row.user_id,
                    account_id=str(row.account_id),
                    deal_id=row.deal_id,
                    sender_type=row.sender_type,
                    text=row.text,
                    is_read=row.is_read,
                    created_at=row.created_at,
                    media_url=row.media_url,
                    media_name=row.media_name,
                    media_type=row.media_type,
                )
        return [
            _deal_item(deal, unread_map.get(deal.id, 0), last_map.get(deal.id))
            for deal in deals
        ]


@router.post("", response_model=DealItem)
async def create_deal(
    payload: DealCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
    crm_client: TwentyCrmClient = Depends(get_crm_client),
) -> DealItem:
    async with db.async_session() as session:
        account = await resolve_account(session, crm_client, str(payload.account_id).strip())
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
            resolution = ContactResolver.resolve(account.raw_metadata, account.username)
            msg = CreatorMessage(
                user_id=current_user.id,
                account_id=account.id,
                deal_id=deal.id,
                sender_type="user",
                text=payload.initial_message.strip(),
                channel_type=resolution.channel_type,
                channel_target=resolution.channel_target,
                external_message_id=None,
            )
            session.add(msg)
            await session.flush()
            last_message_item = message_item(msg)
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


@router.get("/{deal_id}", response_model=DealItem)
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
            select(func.count(CreatorMessage.id))
            .where(CreatorMessage.deal_id == deal.id)
            .where(CreatorMessage.is_read.is_(False))
            .where(CreatorMessage.sender_type != "user")
        )
        unread_result = await session.execute(unread_stmt)
        unread_count = unread_result.scalar() or 0
        last_stmt = (
            select(CreatorMessage)
            .where(CreatorMessage.deal_id == deal.id)
            .order_by(CreatorMessage.created_at.desc())
            .limit(1)
        )
        last_result = await session.execute(last_stmt)
        last_message = last_result.scalar_one_or_none()
        return _deal_item(
            deal,
            unread_count,
            message_item(last_message) if last_message is not None else None,
        )


@router.patch("/{deal_id}", response_model=DealItem)
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
            if field in FORBIDDEN_DEAL_FIELDS:
                continue
            if value is not None:
                setattr(deal, field, value)
        deal.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(deal)
        unread_stmt = (
            select(func.count(CreatorMessage.id))
            .where(CreatorMessage.deal_id == deal.id)
            .where(CreatorMessage.is_read.is_(False))
            .where(CreatorMessage.sender_type != "user")
        )
        unread_result = await session.execute(unread_stmt)
        unread_count = unread_result.scalar() or 0
        last_stmt = (
            select(CreatorMessage)
            .where(CreatorMessage.deal_id == deal.id)
            .order_by(CreatorMessage.created_at.desc())
            .limit(1)
        )
        last_result = await session.execute(last_stmt)
        last_message = last_result.scalar_one_or_none()
        item = _deal_item(
            deal,
            unread_count,
            message_item(last_message) if last_message is not None else None,
            account=account,
        )
    return item


@router.delete("/{deal_id}")
async def delete_deal(
    deal_id: int,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
) -> dict[str, object]:
    async with db.async_session() as session:
        stmt = select(Deal).where(Deal.id == deal_id, Deal.user_id == current_user.id)
        result = await session.execute(stmt)
        deal = result.scalar_one_or_none()
        if deal is None:
            raise HTTPException(status_code=404, detail="Сделка не найдена")
        await session.delete(deal)
        await session.commit()
    return {"status": "deleted", "deal_id": deal_id}


@router.get("/{deal_id}/messages", response_model=list[CreatorMessageItem])
async def get_deal_messages(
    deal_id: int,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
) -> list[CreatorMessageItem]:
    async with db.async_session() as session:
        deal_stmt = select(Deal).where(Deal.id == deal_id, Deal.user_id == current_user.id)
        deal_result = await session.execute(deal_stmt)
        deal = deal_result.scalar_one_or_none()
        if deal is None:
            raise HTTPException(status_code=404, detail="Сделка не найдена")
        update_stmt = (
            update(CreatorMessage)
            .where(
                CreatorMessage.deal_id == deal.id,
                CreatorMessage.sender_type != "user",
                CreatorMessage.is_read.is_(False),
            )
            .values(is_read=True)
        )
        await session.execute(update_stmt)
        await session.commit()
        msg_stmt = (
            select(CreatorMessage)
            .where(CreatorMessage.deal_id == deal.id)
            .order_by(CreatorMessage.created_at.asc())
        )
        msg_result = await session.execute(msg_stmt)
        messages = list(msg_result.scalars().all())
        return [message_item(message) for message in messages]


@router.post("/{deal_id}/messages", response_model=CreatorMessageItem)
async def send_deal_message(
    deal_id: int,
    payload: CreatorSendMessageRequest,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
) -> CreatorMessageItem:
    if not (payload.text and payload.text.strip()) and not payload.media_url:
        raise HTTPException(status_code=400, detail="Сообщение не может быть пустым")
    async with db.async_session() as session:
        deal_stmt = select(Deal).where(Deal.id == deal_id, Deal.user_id == current_user.id)
        deal_result = await session.execute(deal_stmt)
        deal = deal_result.scalar_one_or_none()
        if deal is None:
            raise HTTPException(status_code=404, detail="Сделка не найдена")
        account_stmt = select(Account).where(Account.id == deal.account_id)
        account_result = await session.execute(account_stmt)
        account = account_result.scalar_one_or_none()
        if account is None:
            raise HTTPException(status_code=404, detail="Автор не найден")
        resolution = ContactResolver.resolve(account.raw_metadata, account.username)
        if payload.preferred_channel:
            target_channel = payload.preferred_channel
            target_handle = resolve_target_for_channel(account, target_channel) or resolution.channel_target
        else:
            target_channel = resolution.channel_type
            target_handle = resolution.channel_target
        message = CreatorMessage(
            user_id=deal.user_id,
            account_id=deal.account_id,
            deal_id=deal.id,
            sender_type="user",
            text=payload.text.strip() if payload.text else None,
            channel_type=target_channel,
            channel_target=target_handle,
            external_message_id=None,
            media_url=payload.media_url,
            media_name=payload.media_name,
            media_type=payload.media_type,
        )
        session.add(message)
        deal.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(message)
        return message_item(message)