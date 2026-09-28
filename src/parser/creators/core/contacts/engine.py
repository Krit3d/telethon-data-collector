import re
from dataclasses import dataclass
from typing import Any

from .constants import DEFAULT_CONTEXT_WINDOW
from .context_scorer import ContactRole, ROLE_PRIORITY, score_context
from ..schemas import Contacts
from .extractors import (
    extract_bio_links,
    extract_emails,
    extract_phones,
    extract_telegram_contacts,
    is_valid_email,
    is_valid_telegram_handle,
    normalize_phone,
    normalize_telegram_handle,
)
from .normalizer import deduplicate_preserve_order


@dataclass(slots=True)
class ExtractionPayload:
    biography: str | None = None
    context_text: str | None = None
    external_url: str | None = None
    raw_bio_links: list[dict[str, Any]] | None = None
    posts_content: list[str] | None = None
    transcriptions: list[str] | None = None
    author_username: str | None = None
    platform: str = "INSTAGRAM"
    trusted_emails: list[str] | None = None
    trusted_phones: list[str] | None = None
    default_region: str | None = None


class ContactEngine:
    def extract_contacts(self, payload: ExtractionPayload) -> Contacts:
        sources = self._collect_sources(payload)

        advertising_telegrams: list[str] = []
        telegram_personal: list[str] = []
        telegram_channels: list[str] = []
        other_telegrams: list[str] = []

        for source in sources:
            for extracted in extract_telegram_contacts(source, DEFAULT_CONTEXT_WINDOW):
                self._route_telegram(
                    extracted.handle,
                    extracted.role,
                    advertising_telegrams,
                    telegram_personal,
                    telegram_channels,
                    other_telegrams,
                    is_invite=extracted.is_invite,
                    is_bot=extracted.is_bot,
                    author_username=payload.author_username,
                )

        email_candidates: dict[str, ContactRole] = {}
        phone_candidates: dict[str, ContactRole] = {}

        for source in sources:
            for extracted in extract_emails(source):
                self._merge_role(email_candidates, extracted.email, extracted.role)
            for extracted in extract_phones(source, DEFAULT_CONTEXT_WINDOW, payload.default_region):
                self._merge_role(phone_candidates, extracted.phone, extracted.role)

        for email in payload.trusted_emails or []:
            if is_valid_email(email):
                self._merge_role(email_candidates, email, ContactRole.COMMERCIAL)
        for phone in payload.trusted_phones or []:
            normalized = normalize_phone(phone, payload.default_region)
            if normalized:
                self._merge_role(phone_candidates, normalized, ContactRole.COMMERCIAL)

        advertising_emails: list[str] = []
        emails: list[str] = []
        phones: list[str] = []

        for email, role in email_candidates.items():
            if role is ContactRole.COMMERCIAL:
                advertising_emails.append(email)
            else:
                emails.append(email)
        phones.extend(phone_candidates)

        for link in extract_bio_links(payload.raw_bio_links):
            url_lower = link.url.lower()
            if "t.me" in url_lower or "telegram" in url_lower:
                for extracted in extract_telegram_contacts(link.url, DEFAULT_CONTEXT_WINDOW):
                    role = link.role if link.role is not ContactRole.UNKNOWN else extracted.role
                    if link.title:
                        title_role = score_context(link.title)
                        if title_role is not ContactRole.UNKNOWN:
                            role = title_role
                        if any(word in link.title.lower() for word in ("канал", "channel", "тгк")):
                            role = ContactRole.CHANNEL
                    self._route_telegram(
                        extracted.handle,
                        role,
                        advertising_telegrams,
                        telegram_personal,
                        telegram_channels,
                        other_telegrams,
                        is_invite=extracted.is_invite,
                        is_bot=extracted.is_bot,
                        author_username=payload.author_username,
                    )
            for extracted in extract_phones(link.url):
                phones.append(extracted.phone)
            if "mailto:" in url_lower:
                email = link.url.split("mailto:", 1)[1].split("?", 1)[0].strip()
                if email and is_valid_email(email):
                    if link.role is ContactRole.COMMERCIAL:
                        advertising_emails.append(email)
                    else:
                        emails.append(email)
            if link.title:
                for extracted in extract_emails(link.title):
                    if extracted.role is ContactRole.COMMERCIAL or link.role is ContactRole.COMMERCIAL:
                        advertising_emails.append(extracted.email)
                    else:
                        emails.append(extracted.email)

        advertising_telegrams = deduplicate_preserve_order(advertising_telegrams)
        advertising_set = set(advertising_telegrams)
        telegram_channels = [
            h for h in deduplicate_preserve_order(telegram_channels)
            if h not in advertising_set
        ]
        channel_set = set(telegram_channels)
        telegram_personal = [
            h for h in deduplicate_preserve_order(telegram_personal)
            if h not in advertising_set and h not in channel_set
        ]
        personal_set = set(telegram_personal)
        other_telegrams = [
            h for h in deduplicate_preserve_order(other_telegrams)
            if h not in advertising_set and h not in personal_set and h not in channel_set
        ]
        telegram_handles = deduplicate_preserve_order(
            advertising_telegrams + telegram_channels + telegram_personal + other_telegrams
        )
        advertising_emails = deduplicate_preserve_order(advertising_emails)
        advertising_email_set = set(advertising_emails)
        emails = [e for e in deduplicate_preserve_order(emails) if e not in advertising_email_set]
        phones = deduplicate_preserve_order(phones)

        return Contacts(
            emails=emails,
            phones=phones,
            telegram_handles=telegram_handles,
            telegram_channels=telegram_channels,
            telegram_personal=telegram_personal,
            advertising_emails=advertising_emails,
            advertising_telegrams=advertising_telegrams,
        )

    def extract_from_text(self, text: str | None, author_username: str | None = None) -> Contacts:
        return self.extract_contacts(ExtractionPayload(biography=text, author_username=author_username))

    def _collect_sources(self, payload: ExtractionPayload) -> list[str]:
        sources: list[str] = []
        if payload.biography:
            sources.append(payload.biography)
        if payload.context_text:
            sources.append(payload.context_text)
        if payload.external_url and not payload.raw_bio_links:
            sources.append(payload.external_url)
        if payload.posts_content:
            sources.extend(payload.posts_content[:12])
        if payload.transcriptions:
            sources.extend(payload.transcriptions)
        return deduplicate_preserve_order(sources)

    @staticmethod
    def _merge_role(candidates: dict[str, ContactRole], value: str, role: ContactRole) -> None:
        existing = candidates.get(value)
        if existing is None or ROLE_PRIORITY[role] > ROLE_PRIORITY[existing]:
            candidates[value] = role

    def _route_telegram(
        self,
        handle: str,
        role: ContactRole,
        advertising: list[str],
        personal: list[str],
        channels: list[str],
        other: list[str],
        is_invite: bool = False,
        is_bot: bool = False,
        author_username: str | None = None,
    ) -> None:
        normalized = normalize_telegram_handle(handle)
        if not is_valid_telegram_handle(normalized):
            return
        if is_bot:
            other.append(normalized)
            return
        if is_invite or normalized.startswith("+") or "joinchat" in normalized:
            channels.append(normalized)
            return
        if role is ContactRole.CHANNEL:
            channels.append(normalized)
            return
        if role is ContactRole.COMMERCIAL:
            advertising.append(normalized)
            return
        if role is ContactRole.PERSONAL:
            personal.append(normalized)
            return
        if author_username:
            author_key = re.sub(r"[^a-z0-9]", "", author_username.lower())
            handle_key = re.sub(r"[^a-z0-9]", "", normalized.lower())
            if author_key and handle_key == author_key:
                personal.append(normalized)
                return
        other.append(normalized)