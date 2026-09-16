import asyncio
import json
import logging
import mimetypes
import os
import signal
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Awaitable, cast

import httpx

from sqlalchemy import func, or_, select, update

from telethon import TelegramClient, events
from telethon.errors import (
    FloodWaitError,
    RPCError,
    UsernameInvalidError,
    UsernameNotOccupiedError,
    UserPrivacyRestrictedError,
)
from telethon.hints import EntityLike
from telethon.network.connection.tcpintermediate import ConnectionTcpIntermediate
from telethon.network.connection.tcpmtproxy import ConnectionTcpMTProxyRandomizedIntermediate
from telethon.tl.types import DocumentAttributeFilename

from src.config.config import MEDIA_DIR, Settings, load_settings
from src.db.database import Database
from src.db.models import Account, CreatorMessage, User
from src.utils.proxy import build_telethon_proxy

logger = logging.getLogger("telegram_outreach")


class TelegramOutreachWorker:
    def __init__(self, db: Database, settings: Settings, shutdown_event: asyncio.Event) -> None:
        self.db = db
        self.settings = settings
        self.shutdown_event = shutdown_event
        self.poll_interval_s: int = 5
        self.batch_size: int = 10
        self.client: TelegramClient | None = None
        self.cooldown_until: float = 0.0
        self.api_base_url: str = self.settings.api_base_url.rstrip("/")
        if not self.api_base_url:
            raise ValueError("API_BASE_URL must be configured in settings/.env for TelegramOutreachWorker")

    def _find_session(self) -> tuple[Path, Path | None] | None:
        candidates: list[Path] = [
            self.settings.session_dir / "outreach",
            self.settings.session_dir / "parser",
            self.settings.session_dir,
        ]
        for directory in candidates:
            if not directory.is_dir():
                continue
            for session_file in directory.glob("*.session"):
                session_path = session_file
                json_config_path = session_path.with_suffix(".json")
                if json_config_path.exists():
                    return session_path, json_config_path
                return session_path, None
        return None

    async def init_telethon(self) -> bool:
        found = self._find_session()
        if found is None:
            logger.error("No Telegram session found in session directories")
            return False

        session_path, json_config_path = found
        logger.info("Found session file: %s (config: %s)", session_path.name, json_config_path.name if json_config_path else None)

        proxy_url: str | None = self.settings.proxy_url
        device_model: str = getattr(self.settings, "device_model", "Desktop")
        app_version: str = getattr(self.settings, "app_version", "1.0.0")
        system_version: str = getattr(self.settings, "system_version", "Windows 10")
        lang_code: str = getattr(self.settings, "lang_code", "en")

        if json_config_path is not None and json_config_path.exists():
            try:
                with json_config_path.open("r", encoding="utf-8") as f:
                    config_data: dict[str, Any] = json.loads(f.read())
                proxy_url = config_data.get("proxy_url", proxy_url)
                device_model = config_data.get("device_model", device_model)
                app_version = config_data.get("app_version", app_version)
                system_version = config_data.get("system_version", system_version)
                lang_code = config_data.get("lang_code", lang_code)
            except Exception as e:
                logger.warning("Failed to read session JSON config: %s", e)

        proxy_dict = build_telethon_proxy(proxy_url)

        if proxy_dict and proxy_dict.get("is_mtproxy"):
            connection_type = ConnectionTcpMTProxyRandomizedIntermediate
        else:
            connection_type = ConnectionTcpIntermediate

        client = TelegramClient(
            str(session_path.with_suffix("")),
            self.settings.api_id,
            self.settings.api_hash,
            proxy=cast(dict[str, Any], proxy_dict),
            connection=connection_type,
            timeout=30,
            device_model=device_model,
            app_version=app_version,
            system_version=system_version,
            lang_code=lang_code,
        )
        self.client = client

        await client.connect()

        if not await client.is_user_authorized():
            logger.error("Telegram session is not authorized")
            await cast(Awaitable[None], client.disconnect())
            self.client = None
            return False

        me = await client.get_me()
        me_id = getattr(me, "id", None)
        me_username = getattr(me, "username", None)
        logger.info("Telegram client connected and authorized as @%s (id=%s)", me_username, me_id)

        async with self.db.async_session() as session:
            await session.execute(
                update(CreatorMessage)
                .where(
                    CreatorMessage.channel_type == "telegram",
                    CreatorMessage.external_message_id == "SENDING",
                )
                .values(external_message_id=None)
            )
            await session.commit()

        client.add_event_handler(self._handle_incoming_message, events.NewMessage(incoming=True))

        return True

    async def _resolve_outbox_file(self, media_url: str) -> tuple[str | None, bool]:
        filename = Path(media_url).name
        local_path = MEDIA_DIR / filename
        if local_path.exists():
            return str(local_path), False
        if Path(media_url).exists():
            return media_url, False
        if media_url.startswith("http://") or media_url.startswith("https://"):
            return await self._download_to_temp(media_url)
        return None, False

    async def _download_to_temp(self, url: str) -> tuple[str | None, bool]:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url)
                if response.status_code != 200:
                    logger.error("Failed to download outbox media from %s: HTTP %d", url, response.status_code)
                    return None, False
                suffix = Path(url).suffix or ""
                fd, temp_path = tempfile.mkstemp(suffix=suffix)
                with os.fdopen(fd, "wb") as f:
                    f.write(response.content)
                return temp_path, True
        except Exception as e:
            logger.error("Exception while downloading outbox media from %s: %s", url, e)
            return None, False

    def _detect_media_type(self, message: Any) -> tuple[str | None, str | None]:
        if message.photo:
            return "image", "photo.jpg"
        if message.video:
            return "video", getattr(message.video, "file_name", None) or "video.mp4"
        if message.document:
            original_filename: str | None = None
            for attr in message.document.attributes:
                if isinstance(attr, DocumentAttributeFilename):
                    original_filename = attr.file_name
                    break
            mime_type: str = getattr(message.document, "mime_type", "") or ""
            if mime_type.startswith("image/"):
                media_type = "image"
            elif mime_type.startswith("video/"):
                media_type = "video"
            else:
                media_type = "document"
            return media_type, original_filename or "document.bin"
        return None, None

    async def _save_inbound_media(self, event: events.NewMessage.Event, filename: str | None) -> str | None:
        try:
            clean_name = filename or "document.bin"
            unique_name = f"{uuid.uuid4().hex}_{clean_name}"
            target_path = MEDIA_DIR / unique_name
            downloaded = await event.message.download_media(file=str(target_path))
            if downloaded:
                saved_media_url = f"/media/{unique_name}"
                logger.info("Inbound media saved successfully: %s (original: %s)", saved_media_url, clean_name)
                return saved_media_url
            logger.error("Telegram download_media returned None for message %s", event.message.id)
            return None
        except Exception as e:
            logger.error("Failed to download inbound Telegram media: %s", e, exc_info=True)
            return None

    async def _handle_incoming_message(self, event: events.NewMessage.Event) -> None:
        if not event.is_private or event.out:
            return

        text = event.raw_text or event.message.message or ""
        has_media = event.message.media is not None
        if not text.strip() and not has_media:
            return

        detected_media_type: str | None = None
        detected_filename: str | None = None
        saved_media_url: str | None = None

        if has_media:
            detected_media_type, detected_filename = self._detect_media_type(event.message)
            saved_media_url = await self._save_inbound_media(event, detected_filename)

        sender = await event.get_sender()
        sender_id = event.sender_id
        raw_username = getattr(sender, "username", None)
        username_clean = raw_username.lstrip("@").strip().lower() if raw_username else None

        async with self.db.async_session() as session:
            stmt_dup = (
                select(CreatorMessage.id)
                .where(CreatorMessage.external_message_id == str(event.message.id))
                .limit(1)
            )
            res_dup = await session.execute(stmt_dup)
            if res_dup.scalar_one_or_none() is not None:
                return

            targets = [str(sender_id)]
            if username_clean:
                targets.extend([username_clean, f"@{username_clean}"])
            stmt_prev = (
                select(CreatorMessage.account_id, CreatorMessage.user_id, CreatorMessage.deal_id)
                .where(
                    CreatorMessage.channel_type == "telegram",
                    func.lower(CreatorMessage.channel_target).in_([t.lower() for t in targets]),
                )
                .order_by(CreatorMessage.created_at.desc())
                .limit(1)
            )
            res_prev = await session.execute(stmt_prev)
            row_prev = res_prev.first()

            matched_account_id: int | None = None
            matched_user_id: int | None = None
            matched_deal_id: int | None = None

            if row_prev is not None:
                matched_account_id, matched_user_id, matched_deal_id = row_prev
            else:
                if username_clean:
                    stmt_acc = (
                        select(Account.id)
                        .where(
                            or_(
                                func.lower(Account.username) == username_clean,
                                Account.raw_metadata["contacts"]["telegram_handles"].contains([username_clean]),
                                Account.raw_metadata["contacts"]["telegram_personal"].contains([username_clean]),
                                Account.raw_metadata["contacts"]["advertising_telegrams"].contains([username_clean]),
                                Account.raw_metadata["contacts"]["telegrams"].contains([username_clean]),
                            )
                        )
                        .limit(1)
                    )
                    res_acc = await session.execute(stmt_acc)
                    matched_account_id = res_acc.scalar_one_or_none()

                if matched_account_id is None:
                    stmt_acc_pid = select(Account.id).where(Account.platform_id == str(sender_id)).limit(1)
                    res_acc_pid = await session.execute(stmt_acc_pid)
                    matched_account_id = res_acc_pid.scalar_one_or_none()

                if matched_account_id is None:
                    logger.warning("Incoming Telegram message from unknown sender: id=%s username=%s", sender_id, username_clean)
                    return

                stmt_user = select(User.id).order_by(User.id.asc()).limit(1)
                res_user = await session.execute(stmt_user)
                matched_user_id = res_user.scalar_one_or_none()
                if matched_user_id is None:
                    return

            msg_date = event.message.date
            incoming_msg = CreatorMessage(
                user_id=matched_user_id,
                account_id=matched_account_id,
                deal_id=matched_deal_id,
                sender_type="creator",
                channel_type="telegram",
                channel_target=f"@{raw_username}" if raw_username else str(sender_id),
                external_message_id=str(event.message.id),
                text=text if text.strip() else None,
                media_url=saved_media_url,
                media_name=detected_filename,
                media_type=detected_media_type,
                is_read=False,
                created_at=msg_date,
            )
            session.add(incoming_msg)
            await session.commit()
            logger.info("Saved incoming Telegram message from @%s (id=%s) for account_id=%s", raw_username, sender_id, matched_account_id)

    async def process_outbox_batch(self) -> int:
        if self.client is None or not self.client.is_connected():
            return 0

        loop_time = asyncio.get_running_loop().time()
        if loop_time < self.cooldown_until:
            return 0

        processed_count = 0

        async with self.db.async_session() as session:
            stmt = (
                select(CreatorMessage)
                .where(
                    CreatorMessage.channel_type == "telegram",
                    CreatorMessage.sender_type == "user",
                    CreatorMessage.external_message_id.is_(None),
                    CreatorMessage.channel_target.is_not(None),
                )
                .order_by(CreatorMessage.created_at.asc())
                .limit(self.batch_size)
                .with_for_update(skip_locked=True)
            )
            messages = list((await session.execute(stmt)).scalars().all())
            if not messages:
                return 0

            tasks_data = [
                {
                    "id": m.id,
                    "target": (m.channel_target or "").lstrip("@").strip(),
                    "text": m.text or "",
                    "media_url": m.media_url,
                    "media_name": m.media_name,
                }
                for m in messages
            ]

            for m in messages:
                m.external_message_id = "SENDING"
            await session.commit()

        for task in tasks_data:
            msg_id = task["id"]
            target = task["target"]
            text = task["text"]
            media_url = task["media_url"]
            media_name = task["media_name"]

            if not target:
                async with self.db.async_session() as session:
                    await session.execute(update(CreatorMessage).where(CreatorMessage.id == msg_id).values(external_message_id="FAILED:EMPTY_TARGET"))
                    await session.commit()
                processed_count += 1
                continue

            try:
                if target.isdigit() or (target.startswith("-") and target[1:].isdigit()):
                    entity_key: EntityLike = int(target)
                else:
                    entity_key = target
                entity = cast(EntityLike, await self.client.get_entity(entity_key))
                if media_url:
                    file_path, is_temp = await self._resolve_outbox_file(media_url)
                    if file_path is None:
                        async with self.db.async_session() as session:
                            await session.execute(update(CreatorMessage).where(CreatorMessage.id == msg_id).values(external_message_id="FAILED:MEDIA_DOWNLOAD_ERROR"))
                            await session.commit()
                        logger.error("Failed to resolve media for %s: %s", target, media_url)
                        processed_count += 1
                        continue
                    try:
                        file_attributes = [DocumentAttributeFilename(file_name=media_name)] if media_name else []
                        last_pct = -1

                        def _upload_progress(current: int, total: int) -> None:
                            nonlocal last_pct
                            if total <= 0:
                                return
                            pct = int((current / total) * 100)
                            if pct >= last_pct + 20 or pct == 100:
                                last_pct = pct
                                cur_mb = current / (1024 * 1024)
                                tot_mb = total / (1024 * 1024)
                                logger.info("Uploading %s to Telegram: %d%% (%.1f / %.1f MB)", media_name or "file", pct, cur_mb, tot_mb)

                        is_video = False
                        if media_name:
                            ext = Path(media_name).suffix.lower()
                            if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
                                is_video = True
                        if text and len(text) > 1024:
                            sent = await self.client.send_file(
                                entity,
                                file=file_path,
                                caption="",
                                attributes=file_attributes,
                                progress_callback=_upload_progress,
                                supports_streaming=is_video,
                            )
                            await self.client.send_message(entity, text)
                        else:
                            sent = await self.client.send_file(
                                entity,
                                file=file_path,
                                caption=text or "",
                                attributes=file_attributes,
                                progress_callback=_upload_progress,
                                supports_streaming=is_video,
                            )
                    finally:
                        if is_temp:
                            Path(file_path).unlink(missing_ok=True)
                else:
                    sent = await self.client.send_message(entity, text or "")

                sent_id = sent.id if not isinstance(sent, list) else (sent[0].id if sent else None)

                async with self.db.async_session() as session:
                    await session.execute(update(CreatorMessage).where(CreatorMessage.id == msg_id).values(external_message_id=str(sent_id)))
                    await session.commit()
                logger.info("Successfully sent message to %s and committed external_message_id=%s", target, sent_id)
                processed_count += 1
            except FloodWaitError as e:
                async with self.db.async_session() as session:
                    await session.execute(update(CreatorMessage).where(CreatorMessage.id == msg_id).values(external_message_id=None))
                    await session.commit()
                self.cooldown_until = asyncio.get_running_loop().time() + e.seconds
                break
            except (UserPrivacyRestrictedError, UsernameNotOccupiedError, UsernameInvalidError) as e:
                async with self.db.async_session() as session:
                    await session.execute(update(CreatorMessage).where(CreatorMessage.id == msg_id).values(external_message_id=f"FAILED:{type(e).__name__}"))
                    await session.commit()
                processed_count += 1
            except RPCError as e:
                async with self.db.async_session() as session:
                    await session.execute(update(CreatorMessage).where(CreatorMessage.id == msg_id).values(external_message_id=f"FAILED:RPC:{str(e)[:50]}"))
                    await session.commit()
                processed_count += 1
            except Exception as e:
                async with self.db.async_session() as session:
                    await session.execute(update(CreatorMessage).where(CreatorMessage.id == msg_id).values(external_message_id=f"FAILED:ERR:{str(e)[:50]}"))
                    await session.commit()
                processed_count += 1

        return processed_count

    async def close(self) -> None:
        client = self.client
        if client is not None and client.is_connected():
            await cast(Awaitable[None], client.disconnect())
        self.client = None


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    settings = load_settings()
    db = Database(settings.db_url, echo=False)
    shutdown_event = asyncio.Event()
    logger.info("Starting Telegram Outreach Worker daemon...")

    loop = asyncio.get_running_loop()

    def _sig_handler() -> None:
        shutdown_event.set()

    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(s, _sig_handler)
        except NotImplementedError:
            signal.signal(s, lambda *_: _sig_handler())

    worker = TelegramOutreachWorker(db, settings, shutdown_event)
    initialized = await worker.init_telethon()

    try:
        while not shutdown_event.is_set():
            if not initialized or worker.client is None or not worker.client.is_connected():
                initialized = await worker.init_telethon()
                if not initialized:
                    try:
                        await asyncio.wait_for(shutdown_event.wait(), timeout=15)
                    except asyncio.TimeoutError:
                        pass
                    continue

            try:
                processed = await worker.process_outbox_batch()
                if processed == 0:
                    try:
                        await asyncio.wait_for(shutdown_event.wait(), timeout=worker.poll_interval_s)
                    except asyncio.TimeoutError:
                        pass
            except Exception as e:
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=worker.poll_interval_s)
                except asyncio.TimeoutError:
                    pass
    finally:
        logger.info("Shutting down Telegram Outreach Worker gracefully...")
        await worker.close()
        await db.close()
        logger.info("Telegram Outreach Worker stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass