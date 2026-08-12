from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel
from sqlalchemy import func, select, update
from sqlalchemy.engine import CursorResult
from typing import cast

from src.db.database import Database
from src.db.models import Account, Content
from src.graph.utils import build_node_id


class PostBatchContext(BaseModel):
    content_id: int
    account_id: int
    pub_node_id: str
    author_node_id: str
    platform: str
    content: str | None
    transcription: str | None
    published_at: datetime
    account_category_id: str | None
    author_title: str
    author_username: str | None
    post_type: str
    is_video: bool
    raw_metadata: dict[str, Any] | None


class Reader:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def fetch_pending_batch(
        self,
        batch_size: int = 50,
        worker_id: int = 0,
        total_workers: int = 1,
        priority_mode: bool = True,
    ) -> list[PostBatchContext]:
        now = datetime.now(timezone.utc)
        async with self._db.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Content)
                    .join(Account, Content.account_id == Account.id)
                    .where(Content.graph_status == 0)
                )
                if total_workers > 1:
                    stmt = stmt.where(func.mod(Content.id, total_workers) == worker_id)
                order = Content.published_at.desc() if priority_mode else Content.published_at.asc()
                stmt = stmt.order_by(Account.category_id.is_not(None).desc(), order)
                stmt = stmt.limit(batch_size).with_for_update(skip_locked=True)

                rows = list((await session.scalars(stmt)).all())

                if rows:
                    ids = [row.id for row in rows]
                    await session.execute(
                        update(Content)
                        .where(Content.id.in_(ids))
                        .values(graph_status=1, updated_at=now)
                    )

                contexts: list[PostBatchContext] = []
                for row in rows:
                    account = row.account
                    platform = account.platform
                    author_node_id = build_node_id(
                        "Actor", "", platform=platform, account_id=row.account_id
                    )
                    pub_node_id = build_node_id(
                        "Post", "",
                        platform=platform,
                        account_id=row.account_id,
                        content_id=row.id,
                    )
                    raw_metadata = row.raw_metadata or {}
                    post_type = raw_metadata.get("post_type", "post")
                    is_video = bool(raw_metadata.get("is_video", False))
                    contexts.append(
                        PostBatchContext(
                            content_id=row.id,
                            account_id=row.account_id,
                            pub_node_id=pub_node_id,
                            author_node_id=author_node_id,
                            platform=platform,
                            content=row.content,
                            transcription=row.transcription,
                            published_at=row.published_at,
                            account_category_id=account.category_id,
                            author_title=account.title,
                            author_username=account.username,
                            post_type=post_type,
                            is_video=is_video,
                            raw_metadata=row.raw_metadata,
                        )
                    )
                return contexts

    async def recover_stale_claims(self, timeout_minutes: int = 30) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)
        async with self._db.async_session() as session:
            async with session.begin():
                result = await session.execute(
                    update(Content)
                    .where(Content.graph_status == 1, Content.updated_at < cutoff)
                    .values(graph_status=0, updated_at=datetime.now(timezone.utc))
                )
                return cast(CursorResult, result).rowcount or 0