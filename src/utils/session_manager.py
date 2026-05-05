import argparse
import asyncio
import json
import logging
from pathlib import Path

from opentele.api import API, UseCurrentSession
from opentele.td import TDesktop
from opentele.exception import TDesktopUnauthorized, TDesktopHasNoAccount
from telethon import TelegramClient
from telethon.errors import (
    AuthKeyError,
    UserDeactivatedError,
    SessionRevokedError,
    FloodWaitError,
    RPCError,
)
from telethon.network.connection.tcpintermediate import ConnectionTcpIntermediate
from telethon.network.connection.tcpmtproxy import (
    ConnectionTcpMTProxyRandomizedIntermediate,
)

from src.utils.proxy import build_telethon_proxy

logger = logging.getLogger(__name__)


def load_proxies(proxies_file: Path) -> list[str]:
    """Load proxy strings from a text file, ignoring empty lines and comments."""
    if not proxies_file.exists():
        return []
    lines = proxies_file.read_text(encoding="utf-8").splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


async def check_proxy_alive(proxy_dict: dict) -> bool:
    """Fast TCP check to see if the proxy port is open before passing to Telethon."""
    addr = proxy_dict.get("addr")
    port = proxy_dict.get("port")
    if not addr or not port:
        return False

    try:
        # Simple socket connection test with 5s timeout
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(addr, port), timeout=5.0
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception as e:
        logger.error(f"Proxy {addr}:{port} is dead or unreachable: {e}")
        return False


async def process_sessions(
    raw_sessions_dir: Path, target_folder: str, proxies: list[str]
) -> None:
    target_dir = raw_sessions_dir.parent / "sessions" / target_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    tdata_folders = sorted(
        raw_sessions_dir.rglob("tdata"), key=lambda p: p.parent.name.lower()
    )

    if not tdata_folders:
        logger.warning(f"No tdata folders found in {raw_sessions_dir}")
        return

    logger.info(f"Found {len(tdata_folders)} tdata folders")

    valid_count = 0
    invalid_count = 0

    for idx, tdata_path in enumerate(tdata_folders):
        account_folder = tdata_path.parent
        account_name = account_folder.name
        proxy_url = proxies[idx] if idx < len(proxies) else None

        json_files = list(account_folder.glob("*.json"))
        if not json_files:
            logger.warning(f"Skipping {account_name}: no JSON file found")
            invalid_count += 1
            continue

        try:
            json_data = json.loads(json_files[0].read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to read JSON for {account_name}: {e}")
            invalid_count += 1
            continue

        api_id = json_data.get("app_id") or json_data.get("api_id")
        api_hash = json_data.get("app_hash") or json_data.get("api_hash")
        password = json_data.get("twoFA") or json_data.get("password")

        if not api_id or not api_hash:
            logger.warning(f"Skipping {account_name}: missing api_id/api_hash")
            invalid_count += 1
            continue

        # Validate and convert types
        try:
            api_id = int(api_id)
        except (ValueError, TypeError):
            logger.error(f"Invalid api_id for {account_name}: {api_id} (must be int)")
            invalid_count += 1
            continue

        try:
            api_hash = str(api_hash)
        except Exception:
            logger.error(f"Invalid api_hash for {account_name}")
            invalid_count += 1
            continue

        # Log credentials (masked) for debugging
        masked_hash = api_hash[:6] + "****" + api_hash[-4:] if len(api_hash) > 10 else "****"
        logger.info(
            f"Account {account_name}: api_id={api_id}, api_hash={masked_hash}, "
            f"has_2FA={'Yes' if password else 'No'}"
        )

        # Extract device fingerprint to prevent bans
        device_model = json_data.get("device_model", "PC 64bit")
        system_version = json_data.get("system_version", "Windows 10")
        app_version = json_data.get("app_version", "4.16.8")
        lang_code = json_data.get("lang_code", "en")
        system_lang_code = json_data.get("system_lang_code", "en-US")

        # These kwargs will be passed to TelegramClient via ToTelethon's **kwargs
        # Only include device/language parameters; avoid network params that opentele manages
        client_kwargs = {
            "device_model": device_model,
            "system_version": system_version,
            "app_version": app_version,
            "lang_code": lang_code,
            "system_lang_code": system_lang_code,
            "timeout": 60,
            "connection": ConnectionTcpIntermediate,
        }

        # Handle proxy separately - we'll pass it directly to ToTelethon
        proxy_kwargs = None
        if proxy_url:
            try:
                proxy_config = build_telethon_proxy(proxy_url)
                if proxy_config:
                    if proxy_config.pop("is_mtproxy", False):
                        # MTProxy: use special connection type
                        proxy_kwargs = {
                            "connection": ConnectionTcpMTProxyRandomizedIntermediate,
                            "proxy": (
                                proxy_config["addr"],
                                proxy_config["port"],
                                proxy_config["secret"],
                            ),
                        }
                    else:
                        # Pre-check SOCKS/HTTP proxy
                        is_alive = await check_proxy_alive(proxy_config)
                        if not is_alive:
                            logger.error(
                                f"Skipping {account_name} due to dead proxy."
                            )
                            invalid_count += 1
                            continue
                        proxy_kwargs = {"proxy": proxy_config}
            except Exception as e:
                logger.error(f"Invalid proxy for {account_name}: {e}")
                invalid_count += 1
                continue

        dest_session_base = target_dir / account_name
        dest_session_file = target_dir / f"{account_name}.session"
        dest_json = target_dir / f"{account_name}.json"

        client = None
        try:
            logger.info(f"Loading TDATA for {account_name} from {tdata_path}...")
            td = TDesktop(str(tdata_path))

            # Handle 2FA/password if present
            if password:
                logger.info(f"Account {account_name}: using 2FA password")
                try:
                    td.CheckPassword(password)
                except Exception as pwd_err:
                    logger.warning(
                        f"Account {account_name}: password check failed: {pwd_err}"
                    )

            # Initialize API with validated credentials
            api_instance = API.TelegramDesktop(
                api_id=api_id, api_hash=api_hash
            )

            logger.info(f"Converting TDATA to Telethon client for {account_name}...")
            totelethon_kwargs = dict(client_kwargs)
            if proxy_kwargs:
                totelethon_kwargs.update(proxy_kwargs)
            
            client = await td.ToTelethon(
                session=str(dest_session_base),
                flag=UseCurrentSession,
                api=api_instance,
                **totelethon_kwargs,
            )

            logger.info(f"Connecting client for {account_name}...")
            await client.connect()

            # Check authorization status
            is_authorized = await client.is_user_authorized()
            if not is_authorized:
                logger.warning(f"Session {account_name} is not authorized. Attempting to get detailed error...")
                try:
                    # This will trigger the actual auth check and return specific error
                    me = await client.get_me()
                    logger.info(
                        f"✅ Success: {account_name} (User ID: {me.id}). Saved to {target_folder}"
                    )
                except Exception as auth_err:
                    error_type = type(auth_err).__name__
                    logger.error(
                        f"❌ Account {account_name} NOT authorized. Error: {error_type}: {auth_err}"
                    )
                    invalid_count += 1
                    continue
            else:
                me = await client.get_me()
                logger.info(
                    f"✅ Success: {account_name} (User ID: {me.id}). Saved to {target_folder}"
                )

            json_data["proxy_url"] = proxy_url
            dest_json.write_text(
                json.dumps(json_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            valid_count += 1

        except (TDesktopUnauthorized, TDesktopHasNoAccount) as e:
            logger.error(f"Account {account_name}: opentele error: {e}")
            invalid_count += 1
            if dest_session_file.exists():
                dest_session_file.unlink()
        except (AuthKeyError, UserDeactivatedError, SessionRevokedError) as e:
            error_type = type(e).__name__
            logger.error(f"Account {account_name}: Telegram error ({error_type}): {e}")
            invalid_count += 1
            if dest_session_file.exists():
                dest_session_file.unlink()
        except FloodWaitError as e:
            delay = getattr(e, "seconds", 0) or 1
            logger.error(
                f"Account {account_name}: FloodWaitError (wait {delay}s). Skipping."
            )
            invalid_count += 1
            if dest_session_file.exists():
                dest_session_file.unlink()
        except RPCError as e:
            logger.error(f"Account {account_name}: RPCError ({type(e).__name__}): {e}")
            invalid_count += 1
            if dest_session_file.exists():
                dest_session_file.unlink()
        except Exception as e:
            logger.error(f"Failed to process {account_name}: {type(e).__name__}: {e}")
            invalid_count += 1
            if dest_session_file.exists():
                dest_session_file.unlink()
        finally:
            if client:
                await client.disconnect()

    logger.info(f"Done! Valid: {valid_count}, Invalid: {invalid_count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_folder", type=str)
    parser.add_argument(
        "target_folder", type=str, choices=["crawler", "parser"]
    )
    parser.add_argument("--proxies-file", type=str, default="proxies.txt")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    raw_dir = Path(args.raw_folder).resolve()
    proxies_file = Path(args.proxies_file).resolve()

    proxies = load_proxies(proxies_file)
    asyncio.run(process_sessions(raw_dir, args.target_folder, proxies))


if __name__ == "__main__":
    main()
