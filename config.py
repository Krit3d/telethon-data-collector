import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_id: int
    api_hash: str
    db_url: str
    session_name: str
    channels: list[str]
    posts_limit: int
    concurrency: int
    network_retries: int
    network_retry_base_delay_s: float
    proxy_url: str | None
    avatars_dir: Path


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_settings() -> Settings:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Telegram channels parser (Telethon)"
    )
    parser.add_argument(
        "--posts", type=int, default=None, help="Posts per channel"
    )
    parser.add_argument(
        "--concurrency", type=int, default=None, help="Max parallel channels"
    )
    parser.add_argument(
        "--channels-limit",
        type=int,
        default=None,
        help="Limit number of channels to parse",
    )
    parser.add_argument(
        "--channels-file",
        type=str,
        default=os.getenv("CHANNELS_FILE", "channels.txt"),
        help="Path to file with channels list",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level",
    )

    args = parser.parse_args()
    _setup_logging(args.log_level)

    api_id_raw = os.getenv("API_ID")
    api_hash = os.getenv("API_HASH")
    db_url = os.getenv("DB_URL")
    session_name = os.getenv("SESSION_NAME", "telethon")

    if not api_id_raw or not api_hash or not db_url:
        raise SystemExit(
            "Missing env vars. Required: API_ID, API_HASH, DB_URL (see .env.example)"
        )

    channels = _load_channels_from_file(Path(args.channels_file))
    channels_env = _parse_channels_env(os.getenv("CHANNELS"))

    if channels_env:
        # env has priority
        channels = channels_env

    if not channels:
        raise SystemExit(
            "No channels provided. Set CHANNELS env var or create channels.txt"
        )

    if args.channels_limit is not None:
        channels = channels[: max(0, args.channels_limit)]

    posts_limit = int(
        args.posts if args.posts is not None else os.getenv("POSTS_LIMIT", 10)
    )
    concurrency = int(
        args.concurrency
        if args.concurrency is not None
        else os.getenv("CONCURRENCY", 10)
    )
    network_retries = int(os.getenv("NETWORK_RETRIES", 5))
    network_retry_base_delay_s = float(
        os.getenv("NETWORK_RETRY_BASE_DELAY_S", 1.0)
    )
    proxy_url = os.getenv("PROXY_URL", "").strip() or None
    avatars_dir = Path(os.getenv("AVATARS_DIR", "avatars"))

    return Settings(
        api_id=int(api_id_raw),
        api_hash=api_hash,
        db_url=db_url,
        session_name=session_name,
        channels=channels,
        posts_limit=posts_limit,
        concurrency=max(1, concurrency),
        network_retries=max(0, network_retries),
        network_retry_base_delay_s=max(0.1, network_retry_base_delay_s),
        proxy_url=proxy_url,
        avatars_dir=avatars_dir,
    )


def _load_channels_from_file(path: Path) -> list[str]:
    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8")

    return _parse_channels_env(content)


def _parse_channels_env(value: str | None) -> list[str]:
    if not value:
        return []

    raw = value.replace("\r\n", "\n").replace("\r", "\n")
    parts: list[str] = []

    for token in raw.replace(",", "\n").split("\n"):
        t = token.strip()

        if not t:
            continue
        elif t.startswith("https://t.me/"):
            t = t.removeprefix("https://t.me/").strip("/")
        elif t.startswith("@"):
            t = t[1:]

        parts.append(t)

    # preserve order, remove duplicates
    seen: set[str] = set()
    out: list[str] = []

    for ch in parts:
        if ch in seen:
            continue

        seen.add(ch)
        out.append(ch)

    return out
