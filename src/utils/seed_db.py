import asyncio
import logging
from telethon import TelegramClient
from telethon.tl.types import Channel as TlChannel

from src.config.config import load_settings
from src.db.database import Database
from src.utils.logger import setup_logging


async def main():
    settings = load_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    if not settings.channels:
        logger.error(
            "No channels to add. Check CHANNELS in .env or channels.txt"
        )
        return

    db = Database(settings.db_url)
    await db.init_db()

    # Get first available session
    session_files = list(settings.session_dir.glob("*.session"))
    if not session_files:
        logger.error(f"No sessions found in {settings.session_dir}")
        return

    session_path = str(session_files[0]).replace(".session", "")
    client = TelegramClient(session_path, settings.api_id, settings.api_hash)

    await client.connect()  # type: ignore
    if not await client.is_user_authorized():
        logger.error(f"Session {session_path} is not authorized.")
        return

    logger.info(f"Starting to add {len(settings.channels)} channels to DB...")

    for ch_name in settings.channels:
        try:
            entity = await client.get_entity(ch_name)
            if isinstance(entity, TlChannel):
                # Convert ID to -100... format
                raw_id = str(entity.id)
                formatted_id = (
                    int(f"-100{raw_id}")
                    if not raw_id.startswith("-100")
                    else entity.id
                )

                channel_data = {
                    "id": formatted_id,
                    "username": entity.username or ch_name,
                    "title": entity.title,
                    "description": None,
                    "subscribers_count": None,
                    "status": "pending",
                    "is_author_blog": True,
                }
                await db.upsert_channel(channel_data)
                logger.info(
                    f"✅ Added channel: {ch_name} (ID: {formatted_id})"
                )
            else:
                logger.warning(f"⚠️ {ch_name} is not a Telegram channel.")
        except Exception as e:
            logger.error(f"❌ Error with {ch_name}: {e}")

    await client.disconnect()  # type: ignore
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
