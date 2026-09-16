import asyncio
import logging
import mimetypes
import signal
import uuid
from email.message import EmailMessage
from pathlib import Path

import aiosmtplib
from imap_tools import AND, MailBox
from sqlalchemy import or_, select, update
from sqlalchemy.orm import selectinload

from src.config.config import MEDIA_DIR, Settings, load_settings
from src.db.database import Database
from src.db.models import Account, CreatorMessage, Deal, User

logger = logging.getLogger(__name__)

ACTIVE_STAGES = [1, 2, 3, 4, 5, 6]


class EmailOutreachWorker:

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._shutdown_event = asyncio.Event()
        self._db = Database(settings.db_url)

    def handle_shutdown(self, *args: object) -> None:
        logger.info("Shutdown signal received, stopping email outreach worker...")
        self._shutdown_event.set()

    def _extract_emails(self, account: Account) -> list[str]:
        raw = account.raw_metadata
        if not isinstance(raw, dict):
            return []
        contacts = raw.get("contacts")
        if not isinstance(contacts, dict):
            return []
        emails = contacts.get("emails")
        if not isinstance(emails, list):
            return []
        return [e for e in emails if isinstance(e, str) and e.strip()]

    async def _find_previous(
        self, session, user_id: int, account_id: int
    ) -> CreatorMessage | None:
        stmt = (
            select(CreatorMessage)
            .where(CreatorMessage.user_id == user_id)
            .where(CreatorMessage.account_id == account_id)
            .where(CreatorMessage.external_message_id.isnot(None))
            .order_by(CreatorMessage.created_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def _send_message(self, task: dict) -> tuple[str, str]:
        account = task["account"]
        channel_target = task["channel_target"]
        if not channel_target:
            emails = self._extract_emails(account)
            if not emails:
                raise RuntimeError("No email available for account")
            channel_target = emails[0].strip().lower()

        msg = EmailMessage()
        msg["From"] = self.settings.smtp_user
        msg["To"] = channel_target
        if task["deal_id"]:
            msg["Subject"] = "Сотрудничество с брендом"
        else:
            msg["Subject"] = "Предложение о сотрудничестве"

        logger.info(
            "Sending email: msg_id=%s, to=%s, subject=%s",
            task["id"],
            channel_target,
            msg["Subject"],
        )

        smtp_user = self.settings.smtp_user or ""
        domain = smtp_user.split("@")[-1] if "@" in smtp_user else "localhost"
        message_id = f"<{uuid.uuid4()}@{domain}>"
        msg["Message-ID"] = message_id

        async with self._db.async_session() as session:
            previous = await self._find_previous(
                session, task["user_id"], task["account_id"]
            )
            if previous is not None and previous.external_message_id:
                msg["In-Reply-To"] = previous.external_message_id
                msg["References"] = previous.external_message_id

        msg.set_content(task["text"] or "")

        media_url = task["media_url"]
        if media_url:
            media_path = MEDIA_DIR / Path(media_url).name
            if media_path.exists():
                payload = media_path.read_bytes()
                maintype, subtype = mimetypes.guess_type(media_url)
                if maintype is None:
                    maintype = "application"
                if subtype is None:
                    subtype = "octet-stream"
                msg.add_attachment(
                    payload,
                    maintype=maintype,
                    subtype=subtype,
                    filename=task["media_name"] or media_path.name,
                )

        await aiosmtplib.send(
            msg,
            hostname=self.settings.smtp_host,
            port=self.settings.smtp_port,
            username=self.settings.smtp_user,
            password=self.settings.smtp_password,
            use_tls=self.settings.smtp_use_ssl,
            start_tls=self.settings.smtp_use_tls,
        )

        logger.info(
            "Email sent successfully: msg_id=%s, to=%s, external_message_id=%s",
            task["id"],
            channel_target,
            message_id,
        )
        return message_id, channel_target

    async def _process_outbox(self) -> bool:
        async with self._db.async_session() as session:
            stmt = (
                select(CreatorMessage)
                .options(selectinload(CreatorMessage.account))
                .where(CreatorMessage.channel_type == "email")
                .where(CreatorMessage.external_message_id.is_(None))
                .where(CreatorMessage.sender_type == "user")
                .order_by(CreatorMessage.created_at.asc())
                .limit(10)
                .with_for_update(of=CreatorMessage, skip_locked=True)
            )
            result = await session.execute(stmt)
            messages = list(result.scalars().all())
            if not messages:
                return False

            tasks: list[dict] = []
            for message in messages:
                tasks.append(
                    {
                        "id": message.id,
                        "channel_target": message.channel_target,
                        "text": message.text,
                        "media_url": message.media_url,
                        "media_name": message.media_name,
                        "deal_id": message.deal_id,
                        "account": message.account,
                        "user_id": message.user_id,
                        "account_id": message.account_id,
                    }
                )
                message.external_message_id = "SENDING"
            await session.commit()

        for task in tasks:
            try:
                message_id, final_email = await self._send_message(task)
                async with self._db.async_session() as session:
                    await session.execute(
                        update(CreatorMessage)
                        .where(CreatorMessage.id == task["id"])
                        .values(
                            external_message_id=message_id,
                            channel_target=final_email,
                        )
                    )
                    await session.commit()
            except Exception as e:
                async with self._db.async_session() as session:
                    await session.execute(
                        update(CreatorMessage)
                        .where(CreatorMessage.id == task["id"])
                        .values(external_message_id=f"FAILED:{str(e)[:50]}")
                    )
                    await session.commit()
                logger.error(
                    "Failed to send email message id=%s: %s",
                    task["id"],
                    e,
                    exc_info=True,
                )

        return True

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

    async def _find_account(
        self, session, sender_email: str
    ) -> Account | None:
        stmt = (
            select(Account)
            .where(
                or_(
                    Account.raw_metadata["contacts"]["emails"].contains(
                        [sender_email]
                    ),
                    Account.id.in_(
                        select(CreatorMessage.account_id).where(
                            CreatorMessage.channel_target == sender_email
                        )
                    ),
                )
            )
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def _resolve_user_id(
        self, session, account_id: int
    ) -> int:
        stmt = (
            select(CreatorMessage.user_id)
            .where(CreatorMessage.account_id == account_id)
            .where(CreatorMessage.sender_type == "user")
            .order_by(CreatorMessage.created_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        user_id = result.scalar_one_or_none()
        if user_id is not None:
            return user_id

        stmt = (
            select(Deal.user_id)
            .where(Deal.account_id == account_id)
            .where(Deal.stage.in_(ACTIVE_STAGES))
            .order_by(Deal.created_at.asc())
            .limit(1)
        )
        result = await session.execute(stmt)
        user_id = result.scalar_one_or_none()
        if user_id is not None:
            return user_id

        result = await session.execute(
            select(User.id).order_by(User.id.asc()).limit(1)
        )
        user_id = result.scalar_one_or_none()
        if user_id is None:
            raise RuntimeError("No users available in the system")
        return user_id

    async def _find_active_deal(
        self, session, account_id: int, user_id: int
    ) -> Deal | None:
        stmt = (
            select(Deal)
            .where(Deal.account_id == account_id)
            .where(Deal.user_id == user_id)
            .where(Deal.stage.in_(ACTIVE_STAGES))
            .order_by(Deal.updated_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    def _detect_media_type(self, content_type: str, filename: str) -> str:
        ct = (content_type or "").lower()
        if ct.startswith("image/"):
            return "image"
        if ct.startswith("video/"):
            return "video"
        ext = Path(filename).suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic", ".svg"}:
            return "image"
        if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg"}:
            return "video"
        return "document"

    async def _save_attachments(
        self, msg
    ) -> tuple[str | None, str | None, str | None]:
        for attachment in msg.attachments:
            payload = attachment.payload
            if not payload:
                continue
            filename = attachment.filename or f"attachment_{uuid.uuid4().hex}"
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            media_path = MEDIA_DIR / unique_name
            media_path.write_bytes(payload)
            media_type = self._detect_media_type(
                attachment.content_type, filename
            )
            return unique_name, filename, media_type
        return None, None, None

    def _clean_reply_text(self, raw_text: str | None) -> str:
        if not raw_text:
            return ""
        normalized = raw_text.replace("\r\n", "\n")
        lines = normalized.splitlines()
        collected: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(">"):
                continue
            if stripped.startswith("-----") or stripped.startswith("_____") or stripped.startswith("--- "):
                break
            if "<" in stripped and "@" in stripped and ">" in stripped and ":" in stripped:
                break
            if "написал:" in stripped.lower() or "wrote:" in stripped.lower():
                break
            collected.append(line)
        return "\n".join(collected).strip()

    async def _process_inbound(self, msg) -> None:
        sender_email = msg.from_.lower().strip()
        message_id_header = (
            msg.headers.get("message-id", (None,))[0]
            or f"inbound_{uuid.uuid4()}"
        )

        async with self._db.async_session() as session:
            existing = await session.execute(
                select(CreatorMessage.id).where(
                    CreatorMessage.external_message_id == message_id_header
                )
            )
            if existing.scalar_one_or_none() is not None:
                return

            account = await self._find_account(session, sender_email)
            if account is None:
                logger.warning(
                    "Incoming email from %s ignored: creator account not found in database",
                    sender_email,
                )
                return

            user_id = await self._resolve_user_id(session, account.id)
            deal = await self._find_active_deal(session, account.id, user_id)
            deal_id = deal.id if deal is not None else None

            unique_name, media_name, media_type = await self._save_attachments(msg)
            media_url = f"/media/{unique_name}" if unique_name is not None else None

            clean_text = self._clean_reply_text(msg.text)

            new_message = CreatorMessage(
                user_id=user_id,
                account_id=account.id,
                deal_id=deal_id,
                sender_type="creator",
                channel_type="email",
                channel_target=sender_email,
                external_message_id=message_id_header,
                text=(clean_text if clean_text else None),
                is_read=False,
                media_url=media_url,
                media_name=media_name,
                media_type=media_type,
            )
            session.add(new_message)
            await session.commit()
            logger.info(
                "Inbound email processed: author_id=%s, user_id=%s, from=%s, external_message_id=%s",
                account.id,
                user_id,
                sender_email,
                message_id_header,
            )

    async def _poll_inbound(self) -> None:
        if not self.settings.imap_host or not self.settings.imap_user or not self.settings.imap_password:
            raise RuntimeError("IMAP settings are not fully configured")

        imap_host: str = self.settings.imap_host
        imap_user: str = self.settings.imap_user
        imap_password: str = self.settings.imap_password

        def _fetch() -> list:
            with MailBox(imap_host, port=self.settings.imap_port) as mailbox:
                mailbox.login(imap_user, imap_password)
                return list(
                    mailbox.fetch(AND(seen=False), mark_seen=True)
                )

        logger.debug("Checking for unread inbound emails via IMAP")
        messages = await asyncio.to_thread(_fetch)
        if messages:
            logger.info("Found %s new inbound emails via IMAP", len(messages))
        for msg in messages:
            try:
                await self._process_inbound(msg)
            except Exception as e:
                logger.error(
                    "Failed to process inbound email from %s: %s",
                    getattr(msg, "from_", "unknown"),
                    e,
                    exc_info=True,
                )

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
                    timeout=self.settings.email_poll_interval_s,
                )
            except asyncio.TimeoutError:
                pass

    async def start(self) -> None:
        logger.info("Email outreach worker starting")
        await self._db.init_db()

        async with self._db.async_session() as session:
            await session.execute(
                update(CreatorMessage)
                .where(
                    CreatorMessage.channel_type == "email",
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
            await self._db.close()
            logger.info("Email outreach worker stopped")


async def main() -> None:
    settings = load_settings()
    worker = EmailOutreachWorker(settings=settings)
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())