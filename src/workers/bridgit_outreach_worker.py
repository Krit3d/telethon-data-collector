import asyncio
import logging
import signal
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from src.api.services.bridgit_client import BridgitAPIError, BridgitClient
from src.config.config import Settings, load_settings
from src.db.database import Database
from src.db.models import Account, CreatorMessage

logger = logging.getLogger(__name__)

SENDING_MARKER = "SENDING"
QUEUED_PREFIX = "bridgit:queued:"
SENT_PREFIX = "bridgit:sent:"
FAILED_PREFIX = "FAILED:"
NO_RECIPIENT_MARKER = "FAILED:no_recipient"
STATS_WINDOW_HOURS = 24
MIN_STATS_INTERVAL_S = 30.0
DEFAULT_POLL_INTERVAL_S = 15.0
USERNAME_KEYS = ("username", "login", "handle", "recipient", "target", "user")
STATUS_KEYS = ("status", "error", "reason", "message", "detail")
ACTIONTIME_KEYS = ("actiontime", "action_time", "time", "timestamp", "date", "sent_at")


class BridgitOutreachWorker:

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        poll_interval: float | None = None,
        batch_size: int = 10,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        if poll_interval is not None:
            self._poll_interval = float(poll_interval)
        else:
            self._poll_interval = float(
                settings.bridgit_poll_interval_s or DEFAULT_POLL_INTERVAL_S
            )
        self._batch_size = batch_size
        self._stop_event = asyncio.Event()
        self._client = BridgitClient(
            api_key=settings.bridgit_api_key or "",
            account_login=settings.bridgit_account_login,
            base_url=settings.bridgit_base_url,
        )

    async def start(self) -> None:
        if not self._settings.bridgit_api_key or not self._settings.bridgit_account_login:
            logger.error(
                "Bridgit API key or account login is not configured; worker will not start"
            )
            return

        async with self._session_factory() as session:
            async with session.begin():
                await session.execute(
                    update(CreatorMessage)
                    .where(CreatorMessage.external_message_id == SENDING_MARKER)
                    .where(CreatorMessage.channel_type == "instagram")
                    .values(external_message_id=None)
                )

        logger.info("Bridgit outreach worker starting")
        tasks = [
            asyncio.create_task(self._outbox_loop()),
            asyncio.create_task(self._stats_loop()),
        ]
        try:
            await self._stop_event.wait()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.stop()

    async def stop(self) -> None:
        self._stop_event.set()
        await self._client.aclose()

    async def _outbox_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                batch = await self._fetch_outbox_batch()
            except Exception as exc:
                logger.error("Outbox fetch error: %s", exc, exc_info=True)
                batch = []

            if not batch:
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._poll_interval
                    )
                except asyncio.TimeoutError:
                    pass
                continue

            try:
                status = await self._client.get_account_status()
            except Exception as exc:
                logger.warning("Account status check error: %s", exc)
                await self._requeue_batch(batch)
                await asyncio.sleep(self._poll_interval)
                continue

            if status.get("result") != "ok":
                logger.warning(
                    "Bridgit account status is not ok: %s", status.get("result")
                )
                await self._requeue_batch(batch)
                await asyncio.sleep(self._poll_interval)
                continue

            for task in batch:
                if self._stop_event.is_set():
                    break
                try:
                    await self._send_outbox_message(task)
                except Exception as exc:
                    logger.error(
                        "Outbox send error for id=%s: %s",
                        task.get("id"),
                        exc,
                        exc_info=True,
                    )
                await asyncio.sleep(0.5)

    async def _requeue_batch(self, batch: list[dict[str, Any]]) -> None:
        ids = [task["id"] for task in batch]
        if not ids:
            return
        async with self._session_factory() as session:
            async with session.begin():
                await session.execute(
                    update(CreatorMessage)
                    .where(CreatorMessage.id.in_(ids))
                    .where(CreatorMessage.external_message_id == SENDING_MARKER)
                    .values(external_message_id=None)
                )

    async def _fetch_outbox_batch(self) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            async with session.begin():
                stmt = (
                    select(CreatorMessage)
                    .where(CreatorMessage.channel_type == "instagram")
                    .where(CreatorMessage.sender_type == "user")
                    .where(CreatorMessage.external_message_id.is_(None))
                    .order_by(CreatorMessage.created_at.asc())
                    .limit(self._batch_size)
                    .with_for_update(skip_locked=True)
                )
                result = await session.execute(stmt)
                messages = list(result.scalars().all())
                if not messages:
                    return []

                tasks: list[dict[str, Any]] = []
                for message in messages:
                    tasks.append(
                        {
                            "id": message.id,
                            "text": message.text,
                            "media_url": message.media_url,
                            "account_id": message.account_id,
                            "channel_target": message.channel_target,
                            "user_id": message.user_id,
                        }
                    )
                    message.external_message_id = SENDING_MARKER
            return tasks

    async def _send_outbox_message(self, task: dict[str, Any]) -> None:
        recipient = (task.get("channel_target") or "").replace("@", "").strip()
        if not recipient:
            async with self._session_factory() as session:
                account = await session.get(Account, task["account_id"])
                if account is not None and account.username:
                    recipient = account.username.replace("@", "").strip()

        if not recipient:
            async with self._session_factory() as session:
                async with session.begin():
                    await session.execute(
                        update(CreatorMessage)
                        .where(CreatorMessage.id == task["id"])
                        .values(external_message_id=NO_RECIPIENT_MARKER)
                    )
            return

        media_url = task.get("media_url")
        if media_url and media_url.startswith("/"):
            base_url = getattr(self._settings, "api_base_url", None) or getattr(
                self._settings, "media_bridge_url", ""
            ).rstrip("/")
            media_url = f"{base_url}{media_url}"

        try:
            await self._client.send_direct_message(
                recipient=recipient,
                text=task.get("text") or "",
                media_url=media_url,
            )
            external_id = f"{QUEUED_PREFIX}{task['id']}:{int(time.time())}"
        except Exception as exc:
            logger.error(
                "Outbox send failed for message_id=%s, recipient=%s: %s",
                task["id"],
                recipient,
                exc,
                exc_info=True,
            )
            external_id = f"{FAILED_PREFIX}{str(exc)[:200]}"

        async with self._session_factory() as session:
            async with session.begin():
                await session.execute(
                    update(CreatorMessage)
                    .where(CreatorMessage.id == task["id"])
                    .values(external_message_id=external_id)
                )

    async def _stats_loop(self) -> None:
        interval = max(self._poll_interval * 2, MIN_STATS_INTERVAL_S)
        while not self._stop_event.is_set():
            try:
                await self._sync_delivery_stats()
            except BridgitAPIError as exc:
                logger.warning("Bridgit stats API error: %s", exc)
            except Exception as exc:
                logger.error("Stats loop error: %s", exc, exc_info=True)

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def _sync_delivery_stats(self) -> None:
        threshold = datetime.now(timezone.utc) - timedelta(hours=STATS_WINDOW_HOURS)
        async with self._session_factory() as session:
            stmt = (
                select(CreatorMessage.id)
                .where(CreatorMessage.external_message_id.like(f"{QUEUED_PREFIX}%"))
                .where(CreatorMessage.channel_type == "instagram")
                .where(CreatorMessage.created_at >= threshold)
                .limit(1)
            )
            result = await session.execute(stmt)
            if result.first() is None:
                return
        stats = await self._client.get_direct_stats(
            start=int(threshold.timestamp()),
            end=int(datetime.now(timezone.utc).timestamp()),
        )
        success_index = self._index_log(stats.get("success_log"))
        error_index = self._index_log(stats.get("error_log"))
        if not success_index and not error_index:
            return

        async with self._session_factory() as session:
            async with session.begin():
                stmt = (
                    select(CreatorMessage)
                    .options(selectinload(CreatorMessage.account))
                    .where(CreatorMessage.external_message_id.like(f"{QUEUED_PREFIX}%"))
                    .where(CreatorMessage.channel_type == "instagram")
                    .where(CreatorMessage.created_at >= threshold)
                )
                result = await session.execute(stmt)
                messages = list(result.scalars().all())

                for message in messages:
                    username = self._message_username(message)
                    if not username:
                        continue
                    queued_ts = self._queued_timestamp(message.external_message_id)
                    if queued_ts is None:
                        continue
                    min_actiontime = queued_ts - 60
                    error_event = self._match_event(
                        error_index.get(username), min_actiontime
                    )
                    if error_event is not None:
                        message.external_message_id = (
                            f"{FAILED_PREFIX}{self._log_status(error_event['entry'])}"
                        )
                        continue
                    success_event = self._match_event(
                        success_index.get(username), min_actiontime
                    )
                    if success_event is not None:
                        message.external_message_id = (
                            f"{SENT_PREFIX}{success_event['actiontime']}"
                        )

    def _message_username(self, message: CreatorMessage) -> str | None:
        channel_target = (message.channel_target or "").replace("@", "").strip().lower()
        if channel_target:
            return channel_target
        account = message.account
        if account is not None and account.username:
            return account.username.replace("@", "").strip().lower()
        return None

    def _queued_timestamp(self, external_message_id: str | None) -> int | None:
        if not external_message_id or not external_message_id.startswith(QUEUED_PREFIX):
            return None
        parts = external_message_id.split(":")
        if len(parts) < 4:
            return None
        try:
            return int(parts[-1])
        except (TypeError, ValueError):
            return None

    def _match_event(
        self, events: list[dict[str, Any]] | None, min_actiontime: int
    ) -> dict[str, Any] | None:
        if not events:
            return None
        matched = [
            event
            for event in events
            if event.get("actiontime") is not None
            and event["actiontime"] >= min_actiontime
        ]
        if not matched:
            return None
        return max(matched, key=lambda event: event["actiontime"])

    def _index_log(self, entries: Any) -> dict[str, list[dict[str, Any]]]:
        index: dict[str, list[dict[str, Any]]] = {}
        if not isinstance(entries, list):
            return index
        for entry in entries:
            username = self._log_username(entry)
            if not username:
                continue
            index.setdefault(username, []).append(
                {"actiontime": self._log_actiontime(entry), "entry": entry}
            )
        return index

    def _log_username(self, entry: Any) -> str | None:
        if isinstance(entry, dict):
            for key in USERNAME_KEYS:
                value = entry.get(key)
                if value:
                    return str(value).replace("@", "").strip().lower()
            return None
        if isinstance(entry, str):
            return entry.replace("@", "").strip().lower()
        return None

    def _log_status(self, entry: Any) -> str:
        if isinstance(entry, dict):
            for key in STATUS_KEYS:
                value = entry.get(key)
                if value:
                    return str(value)[:200]
        return str(entry)[:200]

    def _log_actiontime(self, entry: Any) -> int | None:
        if isinstance(entry, dict):
            for key in ACTIONTIME_KEYS:
                value = entry.get(key)
                if value is None:
                    continue
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    continue
        return None


async def main() -> None:
    settings = load_settings()
    db = Database(settings.db_url)
    await db.init_db()
    worker = BridgitOutreachWorker(
        session_factory=db.async_session,
        settings=settings,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(worker.stop()))
        except (NotImplementedError, RuntimeError):
            pass

    try:
        await worker.start()
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
