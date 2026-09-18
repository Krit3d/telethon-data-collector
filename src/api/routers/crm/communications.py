import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.api.dependencies import get_crm_client, get_current_user, get_db
from src.api.schemas import CommunicationChannelItem
from src.api.services.contact_resolver import ContactResolver
from src.api.services.crm_client import TwentyCrmClient
from src.api.services.crm_helpers import message_snippet, resolve_account
from src.db.database import Database
from src.db.models import Account, CreatorMessage, Deal, User, UserShortlist

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/communications", tags=["CRM Communications"])


class CommunicationInitRequest(BaseModel):
    creator_id: str


@router.post("/init", response_model=CommunicationChannelItem)
async def init_communication(
    payload: CommunicationInitRequest | None = Body(default=None),
    creator_id: str | None = None,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
    crm_client: TwentyCrmClient = Depends(get_crm_client),
) -> CommunicationChannelItem:
    identifier = creator_id or (payload.creator_id if payload is not None else None)
    if not identifier:
        raise HTTPException(status_code=400, detail="creator_id обязателен")
    async with db.async_session() as session:
        account = await resolve_account(session, crm_client, identifier)
        if account is None:
            raise HTTPException(status_code=404, detail="Автор не найден")
        resolution = ContactResolver.resolve(account.raw_metadata, account.username)
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
        existing_stmt = (
            select(CreatorMessage.id)
            .where(
                CreatorMessage.user_id == current_user.id,
                CreatorMessage.account_id == account.id,
            )
            .limit(1)
        )
        existing_result = await session.execute(existing_stmt)
        existing_message = existing_result.scalar_one_or_none()
        last_existing: CreatorMessage | None = None
        if existing_message is None:
            system_message = CreatorMessage(
                user_id=current_user.id,
                account_id=account.id,
                deal_id=deal.id if deal else None,
                sender_type="system",
                text="Диалог начат",
                is_read=True,
                channel_type=resolution.channel_type,
                channel_target=resolution.channel_target,
            )
            session.add(system_message)
            await session.commit()
            await session.refresh(system_message)
            last_message = "Диалог начат"
            last_message_time = system_message.created_at
        else:
            last_stmt = (
                select(CreatorMessage)
                .where(
                    CreatorMessage.user_id == current_user.id,
                    CreatorMessage.account_id == account.id,
                )
                .order_by(CreatorMessage.created_at.desc())
                .limit(1)
            )
            last_result = await session.execute(last_stmt)
            last_existing = last_result.scalar_one_or_none()
            last_message = message_snippet(last_existing)
            last_message_time = (
                last_existing.created_at
                if last_existing
                else (deal.updated_at if deal else account.created_at)
            )
        now = datetime.now(timezone.utc)
        await session.execute(
            pg_insert(UserShortlist)
            .values(
                user_id=current_user.id,
                account_id=account.id,
                status="Свободен",
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_nothing(index_elements=["user_id", "account_id"])
        )
        shortlist_stmt = (
            select(UserShortlist)
            .where(
                UserShortlist.user_id == current_user.id,
                UserShortlist.account_id == account.id,
            )
        )
        shortlist_result = await session.execute(shortlist_stmt)
        shortlist_entry = shortlist_result.scalar_one_or_none()
        is_archived = bool(shortlist_entry and shortlist_entry.status == "В архиве")
        await session.commit()
        return CommunicationChannelItem(
            deal_id=deal.id if deal else None,
            author_id=str(account.id),
            author_name=account.title or "",
            author_handle=(
                f"@{account.username.lstrip('@')}"
                if account.username
                else (f"@{account.title}" if account.title else "")
            ),
            platform=account.platform or "",
            deal_title=deal.title if deal else None,
            stage=deal.stage if deal else None,
            last_message=last_message,
            last_message_time=last_message_time,
            unread_count=0,
            is_archived=is_archived,
            channel_type=(
                last_existing.channel_type
                if (last_existing and last_existing.channel_type)
                else resolution.channel_type
            ),
        )


@router.get("", response_model=list[CommunicationChannelItem])
async def list_communications(
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
) -> list[CommunicationChannelItem]:
    async with db.async_session() as session:
        msg_ids_stmt = select(CreatorMessage.account_id).where(CreatorMessage.user_id == current_user.id)
        deal_ids_stmt = select(Deal.account_id).where(Deal.user_id == current_user.id)
        union_stmt = msg_ids_stmt.union(deal_ids_stmt)
        union_result = await session.execute(union_stmt)
        account_ids = [row[0] for row in union_result.all()]
        if not account_ids:
            return []
        accounts_stmt = select(Account).where(Account.id.in_(account_ids))
        accounts_result = await session.execute(accounts_stmt)
        accounts = list(accounts_result.scalars().all())
        account_map = {account.id: account for account in accounts}

        deals_stmt = select(Deal).where(
            Deal.user_id == current_user.id,
            Deal.account_id.in_(account_ids),
        )
        deals_result = await session.execute(deals_stmt)
        deals = list(deals_result.scalars().all())

        unread_stmt = (
            select(CreatorMessage.account_id, func.count(CreatorMessage.id))
            .where(
                CreatorMessage.user_id == current_user.id,
                CreatorMessage.account_id.in_(account_ids),
                CreatorMessage.is_read.is_(False),
                CreatorMessage.sender_type == "creator",
            )
            .group_by(CreatorMessage.account_id)
        )
        unread_result = await session.execute(unread_stmt)
        unread_map = {account_id: count for account_id, count in unread_result.all()}

        last_stmt = (
            select(CreatorMessage)
            .distinct(CreatorMessage.account_id)
            .where(
                CreatorMessage.user_id == current_user.id,
                CreatorMessage.account_id.in_(account_ids),
            )
            .order_by(CreatorMessage.account_id, CreatorMessage.created_at.desc())
        )
        last_result = await session.execute(last_stmt)
        last_messages = list(last_result.scalars().all())
        last_map: dict[int, CreatorMessage] = {msg.account_id: msg for msg in last_messages}

        shortlist_stmt = (
            select(UserShortlist)
            .where(
                UserShortlist.user_id == current_user.id,
                UserShortlist.account_id.in_(account_ids),
            )
        )
        shortlist_result = await session.execute(shortlist_stmt)
        shortlist_rows = list(shortlist_result.scalars().all())
        shortlist_map = {row.account_id: row for row in shortlist_rows}

        channels: list[CommunicationChannelItem] = []
        for account_id in account_ids:
            account = account_map.get(account_id)
            account_deals = [d for d in deals if d.account_id == account_id]
            active_deals = [d for d in account_deals if d.stage in [1, 2, 3, 4, 5, 6]]
            if active_deals:
                best_deal = max(active_deals, key=lambda d: d.updated_at or d.created_at)
            elif account_deals:
                best_deal = max(account_deals, key=lambda d: d.updated_at or d.created_at)
            else:
                best_deal = None
            last_message = last_map.get(account_id)
            last_time = (
                last_message.created_at
                if last_message
                else (best_deal.updated_at if best_deal else (account.created_at if account else datetime.now(timezone.utc)))
            )
            shortlist_item = shortlist_map.get(account_id)
            is_archived = bool(shortlist_item and shortlist_item.status == "В архиве")
            if last_message is not None and last_message.channel_type:
                channel_type = last_message.channel_type
            elif account is not None:
                channel_type = ContactResolver.resolve(account.raw_metadata, account.username).channel_type
            else:
                channel_type = "internal"
            channels.append(
                CommunicationChannelItem(
                    deal_id=best_deal.id if best_deal else None,
                    author_id=str(account_id),
                    author_name=account.title if account is not None else "",
                    author_handle=(
                        f"@{account.username.lstrip('@')}"
                        if (account is not None and account.username)
                        else (f"@{account.title}" if (account is not None and account.title) else "")
                    ),
                    platform=account.platform if account is not None else "",
                    deal_title=best_deal.title if best_deal else None,
                    stage=best_deal.stage if best_deal else None,
                    last_message=message_snippet(last_message),
                    last_message_time=last_time,
                    unread_count=unread_map.get(account_id, 0),
                    is_archived=is_archived,
                    channel_type=channel_type,
                )
            )
    channels.sort(key=lambda channel: channel.last_message_time, reverse=True)
    return channels