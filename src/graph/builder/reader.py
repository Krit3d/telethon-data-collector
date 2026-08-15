import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel
from sqlalchemy import func, select, text, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import selectinload
from typing import cast

from src.db.database import Database
from src.db.models import Account, Content
from src.graph.utils import build_node_id

logger = logging.getLogger(__name__)


class PostBatchContext(BaseModel):
    content_id: int
    account_id: int
    pub_node_id: str
    author_node_id: str
    platform: str
    platform_id: str
    content: str | None
    transcription: str | None
    published_at: datetime
    account_category_id: str | None
    account_category_path: str | None
    author_title: str
    author_username: str | None
    author_biography: str | None = None
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
        t0 = time.perf_counter()
        now = datetime.now(timezone.utc)
        async with self._db.async_session() as session:
            async with session.begin():
                candidate_authors_stmt = (
                    select(Content.account_id)
                    .join(Content.account)
                    .where(
                        Content.graph_status == 0,
                        Account.status == "verified",
                    )
                )
                if total_workers > 1:
                    candidate_authors_stmt = candidate_authors_stmt.where(
                        func.mod(Content.account_id, total_workers) == worker_id
                    )
                candidate_authors_subq = candidate_authors_stmt.distinct().limit(10).subquery()

                order = Content.published_at.desc() if priority_mode else Content.published_at.asc()
                inner = (
                    select(Content.id)
                    .where(Content.graph_status == 0)
                    .where(Content.account_id.in_(select(candidate_authors_subq.c.account_id)))
                    .order_by(order)
                    .limit(batch_size)
                    .with_for_update(skip_locked=True)
                )

                result = await session.execute(
                    update(Content)
                    .where(Content.id.in_(inner))
                    .values(graph_status=1, updated_at=now)
                    .returning(Content.id)
                )
                claimed_ids = list(result.scalars().all())

                contexts: list[PostBatchContext] = []
                if claimed_ids:
                    stmt = (
                        select(Content)
                        .where(Content.id.in_(claimed_ids))
                        .order_by(order)
                        .options(selectinload(Content.account))
                    )
                    rows = list((await session.scalars(stmt)).all())

                    for row in rows:
                        account = row.account
                        platform = account.platform
                        platform_id = account.platform_id
                        content_raw = row.raw_metadata if isinstance(row.raw_metadata, dict) else {}
                        account_raw = account.raw_metadata if isinstance(account.raw_metadata, dict) else {}
                        author_node_id = build_node_id(
                            "Actor", "", platform=platform, account_id=row.account_id
                        )
                        pub_node_id = build_node_id(
                            "Post", "",
                            platform=platform,
                            account_id=row.account_id,
                            content_id=row.id,
                        )
                        post_type = str(content_raw.get("post_type") or "post").lower()
                        is_video = post_type in ("reel", "video", "short", "tiktok") or bool(content_raw.get("video_url"))
                        contexts.append(
                            PostBatchContext(
                                content_id=row.id,
                                account_id=row.account_id,
                                pub_node_id=pub_node_id,
                                author_node_id=author_node_id,
                                platform=platform,
                                platform_id=platform_id,
                                content=row.content,
                                transcription=row.transcription,
                                published_at=row.published_at,
                                account_category_id=account.category_id,
                                account_category_path=account.category_path,
                                author_title=account.title or account.username or f"Account_{account.id}",
                                author_username=account.username,
                                author_biography=account.description,
                                post_type=post_type,
                                is_video=is_video,
                                raw_metadata=row.raw_metadata,
                            )
                        )

        elapsed = (time.perf_counter() - t0) * 1000
        if not contexts:
            logger.debug("fetch_pending_batch returned empty batch in %.1fms", elapsed)
        else:
            ids = [c.content_id for c in contexts]
            logger.debug(
                "fetch_pending_batch got %d posts in %.1fms: %s",
                len(contexts), elapsed, ids,
            )
        return contexts

    async def recover_stale_claims(self, timeout_minutes: int = 30) -> int:
        t0 = time.perf_counter()
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)
        async with self._db.async_session() as session:
            async with session.begin():
                lock_res = await session.execute(
                    text("SELECT pg_catalog.pg_try_advisory_xact_lock(hashtext('graph_stale_claims_recovery')::bigint)")
                )
                if not lock_res.scalar():
                    return 0
                result = await session.execute(
                    update(Content)
                    .where(Content.graph_status == 1, Content.updated_at < cutoff)
                    .values(graph_status=0, updated_at=datetime.now(timezone.utc))
                )
                recovered = cast(CursorResult, result).rowcount or 0
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("recover_stale_claims recovered %d records in %.1fms", recovered, elapsed)
        return recovered