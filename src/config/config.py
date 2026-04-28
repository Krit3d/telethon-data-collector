import argparse
import logging
import os
from pathlib import Path

from pydantic import (
    Field,
    ValidationError,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Immutable runtime configuration for the parser.

    All values can be supplied via environment variables or an ``.env`` file.
    CLI arguments are handled by the surrounding ``load_settings()`` function
    and are passed as keyword overrides.
    """

    model_config = SettingsConfigDict(
        env_file="src/config/.env",
        env_file_encoding="utf-8",
        frozen=True,  # immutable, like the original @dataclass(frozen=True)
        extra="ignore",  # ignore extra env vars / inputs
    )

    # ---- Mandatory Telegram credentials ----
    api_id: int = Field(..., description="Telegram API ID")
    api_hash: str = Field(..., description="Telegram API hash")
    db_url: str = Field(
        ...,
        description="PostgreSQL connection URL",
    )

    # ---- Optional vector search settings ----
    # Qdrant vector database for semantic search of posts
    qdrant_url: str = Field(
        ...,
        description="Qdrant HTTP API URL",
    )
    qdrant_grpc_url: str | None = Field(
        default=None,
        description="Qdrant gRPC URL (optional, for faster operations)",
    )
    qdrant_collection_name: str = Field(
        default="telegram_posts",
        description="Qdrant collection name for post embeddings",
    )
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformer model name for generating embeddings",
    )
    qdrant_batch_size: int = Field(
        default=100,
        description="Batch size for Qdrant upsert operations",
    )
    qdrant_timeout: int = Field(
        default=30,
        description="Qdrant request timeout in seconds",
    )
    qdrant_retries: int = Field(
        default=3,
        description="Number of retries for failed Qdrant operations",
    )

    # ---- Optional general settings ----
    session_dir: Path = Field(
        default=Path("sessions"),
        description="Directory for storing sessions",
    )
    avatars_dir: Path = Field(
        default=Path("avatars"),
        description="Directory for downloaded channel avatars",
    )
    posts_limit: int = Field(
        default=10,
        description="Maximum number of posts to fetch per channel",
    )
    concurrency: int = Field(
        default=10,
        description="Maximum number of channels parsed in parallel",
    )
    network_retries: int = Field(
        default=5,
        description="Retry count for transient network failures",
    )
    network_retry_base_delay_s: float = Field(
        default=1.0,
        description="Base delay in seconds for retry backoff",
    )
    proxy_url: str | None = Field(
        default=None,
        description="Optional proxy URL for Telegram connections",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    # ---- Channel sourcing ----
    channels_file: Path = Field(
        default=Path("channels.txt"),
        description="Path to file with channel list",
    )
    channels_limit: int | None = Field(
        default=None,
        description="Limit the number of channels to parse",
    )
    # Raw string from CHANNELS env var; if set, it overrides the file.
    channels_env_raw: str | None = Field(
        default=None,
        validation_alias="CHANNELS",
        description="Comma/newline separated list of channels (env overrides file)",
    )

    @property
    def channels(self) -> list[str]:
        """Final normalised list of channels to parse."""

        if self.channels_env_raw:
            c = _parse_channels_env(self.channels_env_raw)
        else:
            c = _load_channels_from_file(self.channels_file)

        if self.channels_limit is not None:
            c = c[: max(0, self.channels_limit)]

        return c


# ----- Helper functions -----


def _setup_logging(level: str) -> None:
    """Configure global logging with a normalised log level."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_channels_from_file(path: Path) -> list[str]:
    """Read and normalise channel references from a text file."""

    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8")

    return _parse_channels_env(content)


def _parse_channels_env(value: str | None) -> list[str]:
    """Parse channels text into a deduplicated, normalised channel list."""

    if not value:
        return []

    raw = value.replace("\r\n", "\n").replace("\r", "\n")
    parts: list[str] = []

    for token in raw.replace(",", "\n").split("\n"):
        t = token.strip()

        if not t:
            continue
        if t.startswith("https://t.me/"):
            t = t.removeprefix("https://t.me/").strip("/")
        elif t.startswith("@"):
            t = t[1:]

        parts.append(t)

    # Preserve order, remove duplicates
    seen: set[str] = set()
    out: list[str] = []

    for ch in parts:
        if ch in seen:
            continue

        seen.add(ch)
        out.append(ch)

    return out


# ----- Public entry point -----


def load_settings() -> Settings:
    """Load parser settings from environment, .env file, and CLI overrides.

    Returns:
        Fully validated ``Settings`` instance.

    Raises:
        SystemExit: If required environment variables are missing or no channels
            are provided.
    """

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

    # Collect CLI overrides – only non‑None values are forwarded so that
    # environment variables (and defaults) can still act as fallbacks.
    overrides: dict = {}

    if args.posts is not None:
        overrides["posts_limit"] = args.posts
    if args.concurrency is not None:
        overrides["concurrency"] = args.concurrency
    if args.channels_limit is not None:
        overrides["channels_limit"] = args.channels_limit

    # channels_file is always provided by argparse (default or env), we pass it.
    overrides["channels_file"] = Path(args.channels_file)

    try:
        settings = Settings(**overrides)
    except ValidationError as exc:
        # Mimic the original "Missing env vars" behaviour.
        raise SystemExit(str(exc)) from exc

    if not settings.channels:
        raise SystemExit(
            "No channels provided. Set CHANNELS env var or create channels.txt"
        )

    return settings
