from typing import Any

from pydantic import BaseModel

from src.db.models import Account
from src.parser.creators.core.contacts import (
    is_valid_email,
    is_valid_telegram_handle,
    normalize_telegram_handle,
)


class ContactResolutionResult(BaseModel):
    channel_type: str
    channel_target: str | None = None
    is_available: bool = False


class ContactResolver:
    @staticmethod
    def resolve(account: Account | dict[str, Any]) -> ContactResolutionResult:
        if isinstance(account, Account):
            raw_metadata = account.raw_metadata
            platform = account.platform
            username = account.username
        else:
            raw_metadata = account.get("raw_metadata")
            platform = account.get("platform")
            username = account.get("username")

        contacts = raw_metadata.get("contacts") if isinstance(raw_metadata, dict) else {}
        contacts = contacts if isinstance(contacts, dict) else {}

        for key in ("telegram_personal", "advertising_telegrams", "telegram_handles"):
            items = contacts.get(key, [])
            if not isinstance(items, list):
                continue
            for cand in items:
                if not isinstance(cand, str):
                    continue
                if "+" in cand or "joinchat" in cand:
                    continue
                if is_valid_telegram_handle(cand):
                    return ContactResolutionResult(
                        channel_type="telegram",
                        channel_target=f"@{normalize_telegram_handle(cand)}",
                        is_available=True,
                    )

        if (
            isinstance(platform, str)
            and platform.upper() == "TELEGRAM"
            and isinstance(username, str)
            and username
        ):
            if is_valid_telegram_handle(username):
                return ContactResolutionResult(
                    channel_type="telegram",
                    channel_target=f"@{normalize_telegram_handle(username)}",
                    is_available=True,
                )

        for key in ("advertising_emails", "emails"):
            items = contacts.get(key, [])
            if not isinstance(items, list):
                continue
            for cand in items:
                if not isinstance(cand, str):
                    continue
                if is_valid_email(cand):
                    return ContactResolutionResult(
                        channel_type="email",
                        channel_target=cand.strip().lower(),
                        is_available=True,
                    )

        phones = contacts.get("phones", [])
        if isinstance(phones, list) and phones and isinstance(phones[0], str) and len(phones[0]) >= 10:
            return ContactResolutionResult(
                channel_type="whatsapp",
                channel_target=phones[0].strip(),
                is_available=True,
            )

        return ContactResolutionResult(channel_type="internal", channel_target=None, is_available=False)