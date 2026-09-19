import asyncio
import logging
import signal
from typing import Any

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import selectinload

from src.api.services.slidecold_client import SlideColdClient
from src.config.config import Settings, load_settings
from src.db.database import Database
from src.db.models import Account, CreatorMessage, Deal, User

logger = logging.getLogger(__name__)

ACTIVE_STAGES = [1, 2, 3, 4, 5, 6]


class SlideColdOutreachWorker:

    def __init__(self, settings: Settings, db: Database, slidecold: SlideColdClient) -> None:
        self.settings = settings
        self._db = db
        self._slidecold = slidecold
        self._shutdown_event = asyncio.Event()

    def handle_shutdown(self, *args: object) -> None:
        logger.info("Shutdown signal received, stopping SlideCold outreach worker...")
        self._shutdown_event.set()

    async def _process_outbox(self) -> bool:
        async with self._db.async_session() as session:
            stmt = (
                select(CreatorMessage)
                .options(selectinload(CreatorMessage.account))
                .where(CreatorMessage.channel_type == "instagram")
                .where(CreatorMessage.sender_type == "user")
                .where(CreatorMessage.external_message_id.is_(None))
                .order_by(CreatorMessage.created_at.asc())
                .limit(10)
                .with_for_update(of=CreatorMessage, skip_locked=True)
            )
            result = await session.execute(stmt)
            messages = list(result.scalars().all())
            if not messages:
                return False

            tasks: list[dict[str, Any]] = []
            for message in messages:
                tasks.append(
                    {
                        "id": message.id,
                        "channel_target": message.channel_target,
                        "text": message.text,
                        "media_url": message.media_url,
                        "username": message.account.username if message.account else None,
                    }
                )
                message.external_message_id = "SENDING"
            await session.commit()

        for task in tasks:
            try:
                message_id = await self._send_message(task)
                async with self._db.async_session() as session:
                    await session.execute(
                        update(CreatorMessage)
                        .where(CreatorMessage.id == task["id"])
                        .values(external_message_id=f"slidecold:{message_id}")
                    )
                    await session.commit()
                logger.info(
                    "SlideCold message sent: msg_id=%s, message_id=%s",
                    task["id"],
                    message_id,
                )
            except Exception as e:
                async with self._db.async_session() as session:
                    await session.execute(
                        update(CreatorMessage)
                        .where(CreatorMessage.id == task["id"])
                        .values(external_message_id=f"FAILED:{str(e)[:50]}")
                    )
                    await session.commit()
                logger.error(
                    "Failed to send SlideCold message id=%s: %s",
                    task["id"],
                    e,
                    exc_info=True,
                )

        return True

    async def _send_message(self, task: dict[str, Any]) -> str:
        recipient = (task.get("channel_target") or "").lstrip("@").strip()
        if not recipient:
            recipient = (task.get("username") or "").lstrip("@").strip()
        if not recipient:
            raise ValueError(f"No valid Instagram recipient for task {task.get('id')}")

        text = task.get("text") or ""
        media_url = task.get("media_url")
        public_media_url = f"{self.settings.api_base_url.rstrip('/')}{media_url}" if media_url else None
        if public_media_url:
            text = f"{text}\n{public_media_url}" if text else public_media_url

        message_id = await self._slidecold.send_message(
            recipient=recipient,
            text=text,
            media_url=public_media_url,
            account_id=self.settings.slidecold_account_id,
        )
        return str(message_id)

    async def _outbox_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                processed = await self._process_outbox()
            except Exception as e:
                logger.error("Outbox loop error: %s", e, exc_info=True)
                processed = False

            if self._shutdown_event.is_set():
                break

            if not processed:
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=3)
                except asyncio.TimeoutError:
                    pass

    def _reply_handle(self, reply: dict[str, Any]) -> str:
        raw = (
            reply.get("sender")
            or reply.get("handle")
            or reply.get("username")
            or reply.get("from")
        )
        return str(raw or "").lstrip("@").strip().lower()

    def _reply_text(self, reply: dict[str, Any]) -> str | None:
        return reply.get("text") or reply.get("message") or reply.get("content")

    def _reply_external_id(self, reply: dict[str, Any]) -> str | None:
        return (
            reply.get("id")
            or reply.get("message_id")
            or reply.get("external_id")
            or reply.get("external_message_id")
        )

    async def _process_reply(self, reply: dict[str, Any]) -> None:
        handle = self._reply_handle(reply)
        external_message_id = self._reply_external_id(reply)
        if not handle or not external_message_id:
            return

        async with self._db.async_session() as session:
            duplicate_stmt = (
                select(CreatorMessage.id)
                .where(CreatorMessage.external_message_id == str(external_message_id))
                .limit(1)
            )
            if (await session.execute(duplicate_stmt)).scalar_one_or_none() is not None:
                return

            account_result = await session.execute(
                select(Account)
                .where(
                    or_(
                        func.lower(Account.username) == handle,
                        func.lower(Account.username) == f"@{handle}",
                    )
                )
                .limit(1)
            )
            account = account_result.scalar_one_or_none()
            if account is None:
                logger.warning(
                    "SlideCold reply from %s ignored: creator account not found",
                    handle,
                )
                return

            last_msg_stmt = (
                select(CreatorMessage.user_id)
                .where(CreatorMessage.account_id == account.id)
                .order_by(CreatorMessage.created_at.desc())
                .limit(1)
            )
            user_id = (await session.execute(last_msg_stmt)).scalar_one_or_none()
            if user_id is None:
                user_stmt = select(User.id).order_by(User.id.asc()).limit(1)
                user_id = (await session.execute(user_stmt)).scalar_one_or_none()
            if user_id is None:
                return

            deal_stmt = (
                select(Deal.id)
                .where(
                    Deal.account_id == account.id,
                    Deal.user_id == user_id,
                    Deal.stage.in_(ACTIVE_STAGES),
                )
                .order_by(Deal.updated_at.desc())
                .limit(1)
            )
            deal_id = (await session.execute(deal_stmt)).scalar_one_or_none()

            new_message = CreatorMessage(
                user_id=user_id,
                account_id=account.id,
                deal_id=deal_id,
                sender_type="creator",
                channel_type="instagram",
                channel_target=f"@{handle}",
                external_message_id=str(external_message_id),
                text=self._reply_text(reply),
                is_read=False,
            )
            session.add(new_message)
            await session.commit()
            logger.info(
                "SlideCold inbound reply processed: account_id=%s, user_id=%s, from=%s",
                account.id,
                user_id,
                handle,
            )

    async def _poll_inbound(self) -> None:
        replies = await self._slidecold.get_replies(account_id=self.settings.slidecold_account_id)
        if not isinstance(replies, list):
            return
        for reply in replies:
            if not isinstance(reply, dict):
                continue
            try:
                await self._process_reply(reply)
            except Exception as e:
                logger.error("Failed to process SlideCold inbound reply: %s", e, exc_info=True)

    async def _inbound_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                await self._poll_inbound()
            except Exception as e:
                logger.error("Inbound loop error: %s", e, exc_info=True)

            if self._shutdown_event.is_set():
                break

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.settings.slidecold_poll_interval_s,
                )
            except asyncio.TimeoutError:
                pass

    async def start(self) -> None:
        logger.info("SlideCold outreach worker starting")
        await self._db.init_db()

        async with self._db.async_session() as session:
            await session.execute(
                update(CreatorMessage)
                .where(
                    CreatorMessage.channel_type == "instagram",
                    CreatorMessage.external_message_id == "SENDING",
                )
                .values(external_message_id=None)
            )
            await session.commit()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.handle_shutdown)
            except (NotImplementedError, RuntimeError):
                pass

        tasks = [
            asyncio.create_task(self._outbox_loop()),
            asyncio.create_task(self._inbound_loop()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Worker tasks cancelled, shutting down...")
        finally:
            self._shutdown_event.set()
            await self._slidecold.aclose()
            await self._db.close()
            logger.info("SlideCold outreach worker stopped")


async def main() -> None:
    settings = load_settings()
    if not settings.slidecold_api_key:
        raise RuntimeError("SlideCold API key is not configured")
    db = Database(settings.db_url)
    slidecold = SlideColdClient(
        api_key=settings.slidecold_api_key,
        account_id=settings.slidecold_account_id,
        base_url=settings.slidecold_base_url,
    )
    worker = SlideColdOutreachWorker(settings=settings, db=db, slidecold=slidecold)
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())