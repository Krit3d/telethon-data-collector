"""Interactive CLI utility for Telegram account authorization.

This script guides the user through the process of creating a new Telegram session
by phone number, then saves both the .session file and a corresponding .json
configuration file for use with crawler/parser workers.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from telethon import TelegramClient
from telethon.errors import (
    FloodWaitError,
    SessionPasswordNeededError,
)
from telethon.network.connection.tcpintermediate import (
    ConnectionTcpIntermediate,
)
from telethon.network.connection.tcpmtproxy import (
    ConnectionTcpMTProxyRandomizedIntermediate,
)

from src.config.config import load_settings
from src.utils.proxy import build_telethon_proxy

logger = logging.getLogger(__name__)

# Device parameters matching worker defaults
DEVICE_MODEL = "PC 64bit"
SYSTEM_VERSION = "Windows 10"
APP_VERSION = "4.16.8"
LANG_CODE = "en"
SYSTEM_LANG_CODE = "en-US"


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with a consistent format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def prompt_text(field_name: str, default: str | None = None) -> str:
    """Prompt user for text input with optional default value."""
    prompt = f"Enter {field_name}"
    if default:
        prompt += f" [{default}]"
    prompt += ": "

    value = input(prompt).strip()

    if not value and default:
        return default

    if not value:
        raise ValueError(f"{field_name} cannot be empty")

    return value


def prompt_api_id(default: int) -> int:
    """Prompt for API_ID with default."""
    raw = input(f"Enter API_ID [{default}]: ").strip()
    if not raw:
        return default

    try:
        return int(raw)
    except ValueError:
        raise ValueError("API_ID must be an integer")


def prompt_proxy_url() -> str | None:
    """Prompt for optional proxy URL."""
    raw = input("Enter Proxy URL (optional, press Enter to skip): ").strip()
    return raw if raw else None


async def create_session(
    session_name: str,
    phone: str,
    api_id: int,
    api_hash: str,
    proxy_url: str | None,
    target: str,
    sessions_dir: Path,
) -> Path:
    """Create Telegram session and corresponding JSON config file.

    Args:
        session_name: Name for the session (without extension)
        phone: Phone number in international format
        api_id: Telegram API ID
        api_hash: Telegram API hash
        proxy_url: Optional proxy URL
        target: Target subdirectory ("crawler" or "parser")
        sessions_dir: Base sessions directory

    Returns:
        Path to the created .session file

    Raises:
        Various Telethon exceptions on auth failure
    """
    target_dir = sessions_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)

    session_path = target_dir / f"{session_name}.session"
    json_path = target_dir / f"{session_name}.json"

    # Build client kwargs with device parameters
    client_kwargs: dict = {
        "device_model": DEVICE_MODEL,
        "system_version": SYSTEM_VERSION,
        "app_version": APP_VERSION,
        "lang_code": LANG_CODE,
        "system_lang_code": SYSTEM_LANG_CODE,
        "timeout": 60,
        "connection": ConnectionTcpIntermediate,
    }

    # Configure proxy if provided
    proxy_config = None
    if proxy_url:
        try:
            proxy_config = build_telethon_proxy(proxy_url)
            if proxy_config and proxy_config.pop("is_mtproxy", False):
                client_kwargs["connection"] = (
                    ConnectionTcpMTProxyRandomizedIntermediate
                )
                client_kwargs["proxy"] = (
                    proxy_config["addr"],
                    proxy_config["port"],
                    proxy_config["secret"],
                )
            elif proxy_config:
                client_kwargs["proxy"] = proxy_config
            logger.info(f"Using proxy configuration for {session_name}")
        except ValueError as e:
            logger.error(f"Invalid proxy URL: {e}")
            raise

    # Create client
    client = TelegramClient(
        str(session_path.with_suffix("")),  # Telethon adds .session
        api_id,
        api_hash,
        **client_kwargs,
    )

    try:
        logger.info(f"Connecting to Telegram for session '{session_name}'...")
        await client.connect()

        if not await client.is_user_authorized():
            logger.info("Starting interactive authorization...")
            try:
                await client.start(phone=phone)  # type: ignore
            except SessionPasswordNeededError:
                # Telethon will prompt for 2FA password via console
                logger.info("Two-factor authentication required")
                await client.start(phone=phone)  # type: ignore
            except FloodWaitError as e:
                wait_seconds = getattr(e, "seconds", 0)
                logger.error(
                    f"Flood wait: must wait {wait_seconds} seconds before retrying"
                )
                raise

            logger.info("Authorization successful!")
        else:
            logger.info("Session already authorized")

        # Verify authorization
        me = await client.get_me()
        # me can be User or InputPeerUser depending on session state
        user_name = getattr(me, "first_name", None) or getattr(
            me, "username", "Unknown"
        )
        user_id = getattr(me, "id", "Unknown")
        logger.info(f"Logged in as: {user_name} (ID: {user_id})")

        # Build configuration JSON matching crawler expectations
        config = {
            "api_id": api_id,
            "api_hash": api_hash,
            "proxy_url": proxy_url,
            "device_model": DEVICE_MODEL,
            "system_version": SYSTEM_VERSION,
            "app_version": APP_VERSION,
            "lang_code": LANG_CODE,
            "system_lang_code": SYSTEM_LANG_CODE,
        }

        # Write JSON config
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Session files created:\n"
            f"  • {session_path}\n"
            f"  • {json_path}"
        )

        return session_path

    finally:
        client.disconnect()
        logger.info(f"Client disconnected for session '{session_name}'")


async def main() -> None:
    """Main entry point for interactive session creation."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Interactive Telegram account authorization and config generator"
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["crawler", "parser"],
        default="crawler",
        help="Target worker type (saves to sessions/<target>/ directory)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Load global settings to get default API credentials
    try:
        settings = load_settings()
        default_api_id = settings.api_id
        default_api_hash = settings.api_hash
    except SystemExit:
        # Settings may fail if .env missing; use placeholders
        logger.warning(
            "Could not load settings from .env. You'll need to provide API_ID/API_HASH manually."
        )
        default_api_id = 0
        default_api_hash = ""

    # Determine sessions directory
    sessions_dir = Path(
        settings.session_dir if "settings" in locals() else Path("sessions")
    )
    sessions_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Target directory: sessions/{args.target}")
    logger.info(f"Sessions base directory: {sessions_dir.absolute()}")

    try:
        # Interactive prompts
        print("\n" + "=" * 60)
        print("Telegram Account Authorization")
        print("=" * 60 + "\n")

        session_name = prompt_text("session name (e.g., account_1)")
        phone = prompt_text(
            "phone number (international format, e.g., +79991234567)"
        )

        api_id = (
            prompt_api_id(default_api_id)
            if default_api_id
            else int(input("Enter API_ID (required): ").strip())
        )
        api_hash = (
            prompt_text("API_HASH", default_api_hash)
            if default_api_hash
            else (input("Enter API_HASH (required): ").strip())
        )

        # Validate required fields
        if not api_hash:
            raise ValueError("API_HASH is required")

        proxy_url = prompt_proxy_url()

        print("\n" + "-" * 60)
        print("Summary:")
        print(f"  Session name : {session_name}")
        print(f"  Phone        : {phone}")
        print(f"  API_ID       : {api_id}")
        print(
            f"  API_HASH     : {api_hash[:6]}...{api_hash[-4:] if len(api_hash) > 10 else '****'}"
        )
        print(f"  Proxy        : {proxy_url or 'None'}")
        print(f"  Target       : {args.target}")
        print("-" * 60 + "\n")

        confirm = input("Proceed with authorization? (y/N): ").strip().lower()
        if confirm not in {"y", "yes"}:
            logger.info("Cancelled by user")
            sys.exit(0)

        # Create session
        session_path = await create_session(
            session_name=session_name,
            phone=phone,
            api_id=api_id,
            api_hash=api_hash,
            proxy_url=proxy_url,
            target=args.target,
            sessions_dir=sessions_dir,
        )

        print("\n" + "=" * 60)
        print("✅ SUCCESS!")
        print("=" * 60)
        print(f"Session created: {session_path}")
        print(f"Config saved:   {session_path.with_suffix('.json')}")
        print(
            "\nYou can now use this session with your crawler/parser workers."
        )
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Failed to create session: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
