import logging
from datetime import datetime, timezone
from typing import Any, cast

from sqlalchemy import and_, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import joinedload

from src.config.config import Settings
from src.db.database import with_retry_on_deadlock
from src.db.models import Account, Content

logger = logging.getLogger(__name__)


class ExtractorRepository:

    def __init__(
        self,
        async_session: async_sessionmaker[AsyncSession],
        settings: Settings | None = None,
    ) -> None:
        self.async_session = async_session
        self.settings = settings

    async def get_unembedded_content(
        self, limit: int, priority_mode: bool
    ) -> list[Content]:
        async with self.async_session() as session:
            stmt = (
                select(Content)
                .join(Account, Content.account_id == Account.id)
                .options(joinedload(Content.account))
                .where(
                    Content.is_embedded == False,
                    Content.content.isnot(None),
                    Content.content != "",
                    Account.status == "parsed",
                )
            )

            if priority_mode:
                stmt = stmt.order_by(Content.published_at.desc())
            else:
                stmt = stmt.order_by(Content.published_at.asc())

            stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @with_retry_on_deadlock()
    async def mark_content_embedded(self, content_ids: list[int]) -> None:
        if not content_ids:
            return

        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    update(Content)
                    .where(Content.id.in_(content_ids))
                    .values(
                        is_embedded=True,
                        updated_at=datetime.now(timezone.utc),
                    )
                )
                result = cast(CursorResult, await session.execute(stmt))

                if result.rowcount > 0:
                    logger.debug(
                        "Marked %d content items as embedded",
                        result.rowcount,
                    )
                else:
                    logger.warning(
                        "No content items found to mark as embedded for IDs: %s",
                        content_ids[:10],
                    )

    async def _build_ungraphed_query(
        self,
        require_content: bool,
        priority_mode: bool,
    ) -> Any:
        conditions: list[Any] = [Content.is_graph_extracted == False]

        if require_content:
            conditions.append(
                or_(
                    and_(Content.content.isnot(None), Content.content != ""),
                    and_(Content.transcription.isnot(None), Content.transcription != ""),
                    and_(Content.raw_metadata.isnot(None), Content.raw_metadata != ""),
                )
            )

        stmt = (
            select(Content)
            .join(Account, Content.account_id == Account.id)
            .options(joinedload(Content.account))
            .where(*conditions, Account.status == "parsed")
        )

        if priority_mode:
            stmt = stmt.order_by(Content.published_at.desc())
        elif require_content:
            stmt = stmt.order_by(Content.published_at.asc())
        else:
            stmt = stmt.order_by(Content.id.asc())

        return stmt

    async def get_unextracted_content(
        self, limit: int = 50, priority_mode: bool = False
    ) -> list[Content]:
        async with self.async_session() as session:
            stmt = await self._build_ungraphed_query(
                require_content=False,
                priority_mode=priority_mode,
            )
            stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_ungraphed_content(
        self, limit: int, priority_mode: bool
    ) -> list[Content]:
        async with self.async_session() as session:
            stmt = await self._build_ungraphed_query(
                require_content=True,
                priority_mode=priority_mode,
            )
            stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @with_retry_on_deadlock()
    async def _mark_content_graph_extracted(
        self, content_id: int, update_updated_at: bool
    ) -> None:
        values: dict[str, Any] = {"is_graph_extracted": True}
        if update_updated_at:
            values["updated_at"] = datetime.now(timezone.utc)

        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    update(Content)
                    .where(Content.id == content_id)
                    .values(**values)
                )
                result = cast(CursorResult, await session.execute(stmt))

                if result.rowcount > 0:
                    logger.debug(
                        "Marked content id=%d as graph extracted", content_id
                    )
                else:
                    logger.warning(
                        "Content id=%d not found when marking as graph extracted",
                        content_id,
                    )

    @with_retry_on_deadlock()
    async def mark_content_extracted(self, content_id: int) -> None:
        await self._mark_content_graph_extracted(content_id, update_updated_at=False)

    @with_retry_on_deadlock()
    async def mark_content_graphed(self, content_id: int) -> None:
        await self._mark_content_graph_extracted(content_id, update_updated_at=True)
