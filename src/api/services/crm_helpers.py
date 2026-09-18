from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    CreatorMessageItem,
    CreatorPostItem,
    DealAuthorSummary,
)
from src.api.services.crm_client import TwentyCrmClient
from src.db.models import Account, Content, CreatorMessage
from src.parser.creators.core.contacts import (
    is_valid_email,
    is_valid_telegram_handle,
    normalize_telegram_handle,
)

STATUS_UI_TO_CRM: dict[str, str] = {
    "Свободен": "SVOBODEN",
    "В сделке": "V_SDELKE",
    "В архиве": "ARCHIVED",
    "SVOBODEN": "SVOBODEN",
    "V_SDELKE": "V_SDELKE",
    "ARCHIVED": "ARCHIVED",
}

STATUS_CRM_TO_UI: dict[str, str] = {
    "SVOBODEN": "Свободен",
    "V_SDELKE": "В сделке",
    "ARCHIVED": "В архиве",
}

ALLOWED_PLATFORMS: set[str] = {"INSTAGRAM", "TELEGRAM", "YOUTUBE", "VK", "TIKTOK"}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
FORBIDDEN_EXTENSIONS = {".exe", ".bat", ".sh", ".py", ".php", ".js"}


def normalize_platform(platform: str) -> str:
    normalized = platform.upper().strip()
    if normalized not in ALLOWED_PLATFORMS:
        return "INSTAGRAM"
    return normalized


def build_creator_payload(account: Account, user_email: str = "") -> dict[str, Any]:
    followers = int(account.subscribers_count or 0)
    er = round(float(account.static_avg_er or 0.0), 2)
    if er > 0:
        avg_reach = int(followers * (er / 100))
        cpm = int((followers * (er / 100) / 1000) * 200)
    else:
        avg_reach = int(followers * 0.1)
        cpm = 0
    return {
        "accountid": str(account.id),
        "name": account.title or account.username or "",
        "platform": normalize_platform(account.platform),
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


def deduplicate_accounts(accounts: list[Account]) -> list[Account]:
    best: dict[str, Account] = {}
    for account in accounts:
        username = (account.username or "").strip().lstrip("@").lower()
        key = username if username else str(account.id)
        current = best.get(key)
        if current is None:
            best[key] = account
            continue
        current_platform = normalize_platform(current.platform)
        candidate_platform = normalize_platform(account.platform)
        current_followers = int(current.subscribers_count or 0)
        candidate_followers = int(account.subscribers_count or 0)
        if candidate_platform == "INSTAGRAM" and current_platform != "INSTAGRAM":
            best[key] = account
        elif candidate_platform == "INSTAGRAM" and current_platform == "INSTAGRAM" and candidate_followers > current_followers:
            best[key] = account
        elif current_platform != "INSTAGRAM" and candidate_platform != "INSTAGRAM" and candidate_followers > current_followers:
            best[key] = account
    return list(best.values())


def author_summary(account: Account | None) -> DealAuthorSummary | None:
    if account is None:
        return None
    return DealAuthorSummary(
        id=str(account.id),
        platform=account.platform,
        username=account.username,
        title=account.title,
        subscribers_count=account.subscribers_count,
        static_avg_er=account.static_avg_er,
        category_path=account.category_path,
    )


def message_item(message: CreatorMessage) -> CreatorMessageItem:
    return CreatorMessageItem(
        id=message.id,
        user_id=message.user_id,
        account_id=str(message.account_id),
        deal_id=message.deal_id,
        sender_type=message.sender_type,
        text=message.text,
        is_read=message.is_read,
        created_at=message.created_at,
        channel_type=message.channel_type,
        channel_target=message.channel_target,
        external_message_id=message.external_message_id,
        media_url=message.media_url,
        media_name=message.media_name,
        media_type=message.media_type,
    )


def message_snippet(message: CreatorMessage | None) -> str:
    if message is None:
        return ""
    if message.text and message.text.strip():
        return message.text
    if message.media_url:
        if message.media_type == "image":
            return "[Фото]"
        if message.media_type == "video":
            return "[Видео]"
        return f"[Файл: {message.media_name or 'Документ'}]"
    return ""


def detect_media_type(extension: str) -> str:
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    return "document"


def resolve_target_for_channel(account: Account, channel_type: str) -> str | None:
    raw_metadata = account.raw_metadata if isinstance(account.raw_metadata, dict) else {}
    contacts = raw_metadata.get("contacts", {})
    if not isinstance(contacts, dict):
        contacts = {}
    if channel_type == "email":
        items = contacts.get("emails")
        if isinstance(items, list):
            for cand in items:
                if isinstance(cand, str) and is_valid_email(cand):
                    return cand.strip().lower()
        return None
    if channel_type == "telegram":
        items = contacts.get("telegrams")
        if isinstance(items, list):
            for cand in items:
                if isinstance(cand, str) and is_valid_telegram_handle(cand):
                    return f"@{normalize_telegram_handle(cand)}"
        if account.platform == "TELEGRAM" and account.username:
            return account.username
        return None
    if channel_type == "whatsapp":
        items = contacts.get("phones")
        if isinstance(items, list):
            for cand in items:
                if isinstance(cand, str):
                    return cand.strip()
        return None
    if channel_type == "instagram":
        if isinstance(account.username, str):
            cleaned = account.username.lstrip("@").strip()
            if cleaned:
                return f"@{cleaned}"
        return None
    return None


def post_type(content: Content, platform: str) -> str:
    if platform == "INSTAGRAM":
        return "Reels" if content.has_media else "Post"
    if platform == "TELEGRAM":
        return "Видео" if content.has_media else "Пост"
    return "Пост"


def post_er(content: Content) -> float:
    views = content.views or 0
    if views <= 0:
        return 0.0
    reactions = content.reactions_count or 0
    comments = content.comments_count or 0
    return round(((reactions + comments) / views) * 100, 2)


def post_item(content: Content, platform: str) -> CreatorPostItem:
    return CreatorPostItem(
        id=content.id,
        platform_content_id=content.platform_content_id,
        text=content.content,
        published_at=content.published_at,
        views=content.views or 0,
        likes=content.reactions_count or 0,
        comments=content.comments_count or 0,
        shares=content.shares_count or 0,
        er=post_er(content),
        post_type=post_type(content, platform),
        url=None,
    )


def profile_url(platform: str, username: str | None) -> str:
    if not username:
        return ""
    if platform == "INSTAGRAM":
        return f"https://instagram.com/{username}"
    if platform == "TELEGRAM":
        return f"https://t.me/{username}"
    if platform == "YOUTUBE":
        return f"https://youtube.com/@{username}"
    return ""


async def resolve_account(
    session: AsyncSession,
    crm_client: TwentyCrmClient,
    identifier: str,
    user_email: str | None = None,
) -> Account | None:
    raw_identifier = identifier.strip().lstrip("@")
    if not raw_identifier:
        return None
    conditions = [
        Account.platform_id == raw_identifier,
        func.lower(Account.username) == raw_identifier.lower(),
        func.lower(Account.username) == f"@{raw_identifier}".lower(),
        func.lower(Account.title) == raw_identifier.lower(),
    ]
    try:
        int_val = int(raw_identifier)
    except (ValueError, TypeError):
        int_val = None
    if int_val is not None:
        conditions.append(Account.id == int_val)
    result = await session.execute(select(Account).where(or_(*conditions)).limit(1))
    account = result.scalar_one_or_none()
    if account is not None:
        return account
    try:
        creator_record = await crm_client.get_creator_by_id(raw_identifier, user_email)
        if creator_record is None:
            creator_record = await crm_client.find_creator_by_account_id(raw_identifier, user_email)
        if creator_record is not None:
            account_id = creator_record.get("accountid") or creator_record.get("accountId")
            if account_id is not None:
                try:
                    resolved_id = int(account_id)
                except (ValueError, TypeError):
                    resolved_id = None
                if resolved_id is not None:
                    result = await session.execute(select(Account).where(Account.id == resolved_id))
                    account = result.scalar_one_or_none()
                    if account is not None:
                        return account
    except Exception:
        return None
    return None