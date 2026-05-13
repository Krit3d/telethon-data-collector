import argparse
import json
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
        env_file=".env",
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
    qdrant_url: str | None = Field(
        default=None,
        description="Qdrant HTTP API URL (optional, for data collection node)",
    )
    qdrant_grpc_url: str | None = Field(
        default=None,
        description="Qdrant gRPC URL (optional, for faster operations)",
    )
    qdrant_collection_name: str | None = Field(
        default=None,
        description="Qdrant collection name for post embeddings",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="API key for Qdrant authentication",
    )
    embedding_model_name: str = Field(
        default="intfloat/multilingual-e5-large",
        description="Name of the fastembed model to use for local embeddings",
    )
    embedding_model: str = Field(
        default="bge-m3", description="Model name for the API payload"
    )
    embedding_threads: int = Field(
        default=3,
        description="Number of threads to use for embedding generation",
    )
    qdrant_batch_size: int | None = Field(
        default=None,
        description="Batch size for Qdrant upsert operations",
    )
    qdrant_timeout: int | None = Field(
        default=None,
        description="Qdrant request timeout in seconds",
    )
    qdrant_retries: int | None = Field(
        default=None,
        description="Number of retries for failed Qdrant operations",
    )

    # Crawler settings
    crawler_delay_min: int = Field(
        default=45,
        description="Minimum delay between Telegram API calls",
    )
    crawler_delay_max: int = Field(
        default=120,
        description="Maximum delay between Telegram API calls",
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

    # ---- Natparsing stealth settings ----
    use_natural_delays: bool = Field(
        default=False,
        description="Enable natural human-like delay distribution (Pareto/Gamma) instead of uniform random",
    )
    mobile_proxy_rotation_url: str | None = Field(
        default=None,
        description="URL to call for mobile proxy IP rotation (e.g., http://proxy-api:8080/rotate)",
    )
    organic_heatup_minutes: int = Field(
        default=30,
        description="Duration of organic session heatup in minutes (slow start period)",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    # ---- API server settings ----
    api_host: str = Field(
        default="0.0.0.0",
        description="FastAPI server host",
    )
    api_port: int = Field(
        default=8000,
        description="FastAPI server port",
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


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_object = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Include worker_id if present in the log record
        worker_id = getattr(record, "worker_id", None)
        if worker_id is not None:
            log_object["worker_id"] = str(worker_id)

        # Include exception info if present
        if record.exc_info:
            log_object["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_object, ensure_ascii=False)


# ----- Helper functions -----


def _setup_logging(level: str) -> None:
    """Configure global logging with a normalised log level and format.

    Supports JSON and TEXT formats via LOG_FORMAT environment variable.
    TEXT format includes service name and worker_id in the output.
    """

    # Clear existing handlers to prevent duplicate logs and override library defaults
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    log_format = os.getenv("LOG_FORMAT", "TEXT").upper()
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    if log_format == "JSON":
        # JSON structured logging
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S.%fZ"))
        root_logger.addHandler(handler)
    else:
        # TEXT format with worker_id support
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


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
    args, _ = parser.parse_known_args()
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
        logging.warning(
            "No channels provided. Set CHANNELS env var or create channels.txt. "
            "The API will start without channel parsing capability."
        )

    return settings
