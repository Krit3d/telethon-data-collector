from dataclasses import dataclass
from typing import Any

from src.parser.creators.core.contacts import (
    is_valid_email,
    is_valid_telegram_handle,
    normalize_phone,
    normalize_telegram_handle,
)


@dataclass(slots=True, frozen=True)
class ContactResolutionResult:
    channel_type: str | None = None
    channel_target: str | None = None
    is_available: bool = False
    role: str | None = None


class ContactResolver:
    @staticmethod
    def _contacts(raw_metadata: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(raw_metadata, dict):
            return {}
        contacts = raw_metadata.get("contacts")
        return contacts if isinstance(contacts, dict) else {}

    @staticmethod
    def _blacklist(contacts: dict[str, Any]) -> set[str]:
        channels = contacts.get("telegram_channels", [])
        if not isinstance(channels, list):
            return set()
        return {normalize_telegram_handle(str(item)) for item in channels if str(item).strip()}

    @staticmethod
    def _is_dm_handle(handle: str) -> bool:
        return not handle.startswith("+") and "joinchat" not in handle

    @staticmethod
    def _telegram_candidate(handle: Any, blacklist: set[str]) -> str | None:
        if not isinstance(handle, str):
            return None
        normalized = normalize_telegram_handle(handle)
        if not is_valid_telegram_handle(normalized):
            return None
        if normalized in blacklist:
            return None
        if normalized.lower().endswith("bot"):
            return None
        if not ContactResolver._is_dm_handle(normalized):
            return None
        return normalized

    @staticmethod
    def _clean_username(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        cleaned = value.strip()
        if cleaned.startswith("@"):
            cleaned = cleaned[1:]
        return cleaned.strip()

    @staticmethod
    def resolve(raw_metadata: dict[str, Any] | None, username: str | None = None) -> ContactResolutionResult:
        contacts = ContactResolver._contacts(raw_metadata)
        blacklist = ContactResolver._blacklist(contacts)

        for cand in contacts.get("advertising_telegrams", []):
            handle = ContactResolver._telegram_candidate(cand, blacklist)
            if handle is not None:
                return ContactResolutionResult("telegram", f"@{handle}", True, "commercial")

        for cand in contacts.get("telegram_personal", []):
            handle = ContactResolver._telegram_candidate(cand, blacklist)
            if handle is not None:
                return ContactResolutionResult("telegram", f"@{handle}", True, "personal")

        for cand in contacts.get("advertising_emails", []):
            if isinstance(cand, str) and is_valid_email(cand):
                return ContactResolutionResult("email", cand.strip().lower(), True, "commercial")

        for cand in contacts.get("emails", []):
            if isinstance(cand, str) and is_valid_email(cand):
                return ContactResolutionResult("email", cand.strip().lower(), True, "general")

        for cand in contacts.get("phones", []):
            if isinstance(cand, str):
                phone = normalize_phone(cand)
                if phone is not None:
                    return ContactResolutionResult("whatsapp", phone, True, "personal")

        for cand in contacts.get("telegram_handles", []):
            handle = ContactResolver._telegram_candidate(cand, blacklist)
            if handle is not None:
                return ContactResolutionResult("telegram", f"@{handle}", True, "general")

        raw_username = username if username is not None else (raw_metadata.get("username") if isinstance(raw_metadata, dict) else None)
        clean_username = ContactResolver._clean_username(raw_username)
        if clean_username:
            return ContactResolutionResult("instagram", f"@{clean_username}", True, "direct")
        return ContactResolutionResult()

    @staticmethod
    def get_all_channels(raw_metadata: dict[str, Any] | None, username: str | None = None) -> list[ContactResolutionResult]:
        contacts = ContactResolver._contacts(raw_metadata)
        blacklist = ContactResolver._blacklist(contacts)
        results: list[ContactResolutionResult] = []

        for cand in contacts.get("advertising_telegrams", []):
            handle = ContactResolver._telegram_candidate(cand, blacklist)
            if handle is not None:
                results.append(ContactResolutionResult("telegram", f"@{handle}", True, "commercial"))

        for cand in contacts.get("telegram_personal", []):
            handle = ContactResolver._telegram_candidate(cand, blacklist)
            if handle is not None:
                results.append(ContactResolutionResult("telegram", f"@{handle}", True, "personal"))

        for cand in contacts.get("advertising_emails", []):
            if isinstance(cand, str) and is_valid_email(cand):
                results.append(ContactResolutionResult("email", cand.strip().lower(), True, "commercial"))

        for cand in contacts.get("emails", []):
            if isinstance(cand, str) and is_valid_email(cand):
                results.append(ContactResolutionResult("email", cand.strip().lower(), True, "general"))

        for cand in contacts.get("phones", []):
            if isinstance(cand, str):
                phone = normalize_phone(cand)
                if phone is not None:
                    results.append(ContactResolutionResult("whatsapp", phone, True, "personal"))

        for cand in contacts.get("telegram_handles", []):
            handle = ContactResolver._telegram_candidate(cand, blacklist)
            if handle is not None:
                results.append(ContactResolutionResult("telegram", f"@{handle}", True, "general"))

        raw_username = username if username is not None else (raw_metadata.get("username") if isinstance(raw_metadata, dict) else None)
        clean_username = ContactResolver._clean_username(raw_username)
        if clean_username:
            results.append(ContactResolutionResult("instagram", f"@{clean_username}", True, "direct"))

        seen: set[tuple[str, str]] = set()
        deduplicated: list[ContactResolutionResult] = []
        for item in results:
            key = (item.channel_type or "", item.channel_target or "")
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(item)
        return deduplicated