from typing import Any

from sqlalchemy import select

from src.api.schemas import DbsfScoredCandidate, HydratedAuthorRecord, SearchRequest, coerce_platform
from src.db.models import Account


class PostgresHydrator:

    def __init__(self, session_factory: Any) -> None:
        self._session_factory = session_factory

    @staticmethod
    def _extract_contacts(raw_metadata: dict[str, Any] | None) -> tuple[dict[str, Any] | None, bool]:
        if raw_metadata is None or not isinstance(raw_metadata, dict):
            return (None, False)

        contacts = raw_metadata.get("contacts")
        if contacts is not None and isinstance(contacts, dict):
            source = contacts
        else:
            source = raw_metadata

        fields = [
            "emails", "phones", "telegram_handles", "telegram_channels",
            "telegram_personal", "advertising_emails", "advertising_telegrams",
        ]
        contacts_dict: dict[str, Any] = {}
        for field in fields:
            value = source.get(field)
            if isinstance(value, list):
                contacts_dict[field] = value
            else:
                contacts_dict[field] = []

        has_contacts = any(
            isinstance(contacts_dict[field], list)
            and len(contacts_dict[field]) > 0
            and any(item for item in contacts_dict[field] if item)
            for field in fields
        )

        if not has_contacts:
            return (None, False)

        return (contacts_dict, True)

    @staticmethod
    def _extract_profile_url(platform: str, username: str | None, raw_metadata: dict[str, Any] | None) -> str | None:
        if raw_metadata is not None and isinstance(raw_metadata, dict):
            profile_url = raw_metadata.get("profile_url")
            if isinstance(profile_url, str) and profile_url:
                return profile_url

        if username:
            platform_lower = platform.lower()
            if platform_lower == "instagram":
                return f"https://instagram.com/{username}"
            if platform_lower == "telegram":
                return f"https://t.me/{username}"

        return None

    async def hydrate_and_filter_candidates(
        self,
        candidates: list[DbsfScoredCandidate],
        request: SearchRequest,
    ) -> list[tuple[DbsfScoredCandidate, HydratedAuthorRecord]]:
        if not candidates:
            return []

        candidate_ids = [c.account_id for c in candidates]

        async with self._session_factory() as session:
            stmt = select(Account).where(
                Account.id.in_(candidate_ids),
                Account.status == "verified",
            )

            if request.author_type == "expert":
                stmt = stmt.where(Account.is_author_blog.is_(True))
            elif request.author_type == "business":
                stmt = stmt.where(Account.is_author_blog.is_(False))

            if request.platform and request.platform != "all":
                stmt = stmt.where(Account.platform == request.platform.upper())

            if request.min_followers is not None:
                stmt = stmt.where(
                    Account.subscribers_count.is_not(None),
                    Account.subscribers_count >= request.min_followers,
                )

            if request.max_followers is not None:
                stmt = stmt.where(
                    Account.subscribers_count.is_not(None),
                    Account.subscribers_count <= request.max_followers,
                )

            result = await session.execute(stmt)
            accounts = result.scalars().all()

        account_map: dict[int, Account] = {acc.id: acc for acc in accounts}

        stop_words = [w.lower().strip() for w in request.stop_topics if w and len(w.strip()) > 1] if request.stop_topics else []

        result_list: list[tuple[DbsfScoredCandidate, HydratedAuthorRecord]] = []
        for candidate in candidates:
            acc = account_map.get(candidate.account_id)
            if acc is None:
                continue

            if stop_words:
                searchable_text = f"{acc.title or ''} {acc.category_path or ''} {acc.explanation or ''}".lower()
                if any(sw in searchable_text for sw in stop_words):
                    continue

            contacts, has_contacts = self._extract_contacts(acc.raw_metadata)

            profile_url = self._extract_profile_url(acc.platform, acc.username, acc.raw_metadata)

            hydrated = HydratedAuthorRecord(
                account_id=acc.id,
                platform=coerce_platform(acc.platform),
                username=acc.username,
                title=acc.title,
                category_path=acc.category_path,
                explanation=acc.explanation,
                static_avg_er=acc.static_avg_er,
                subscribers_count=acc.subscribers_count,
                is_author_blog=acc.is_author_blog if acc.is_author_blog is not None else False,
                raw_metadata=acc.raw_metadata,
                contacts=contacts if request.include_contacts else None,
                has_contacts=has_contacts,
                profile_url=profile_url,
            )

            result_list.append((candidate, hydrated))

        return result_list[:request.limit]