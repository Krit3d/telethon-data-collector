"""Centralized logging module for production-ready structured logging.

This module provides JSON and TEXT logging formats with custom filters
to suppress noisy Telethon RPC warnings in production.
"""

import json
import logging
import os


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


class TelethonNoiseFilter(logging.Filter):
    """Filter to suppress noisy Telethon RPC error warnings.

    This filter inspects log records and returns False if the record
    comes from a telethon logger AND contains spammy messages like
    "RPC error" or "Invalid channel object".
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Return True to keep the record, False to suppress it."""
        # Check if the record is from telethon logger
        if record.name.startswith("telethon"):
            message = record.getMessage()
            # Suppress RPC error warnings and invalid channel object messages
            if "RPC error" in message or "Invalid channel object" in message:
                return False
        return True


def setup_logging(level: str) -> None:
    """Configure global logging with a normalized log level and format.

    Supports JSON and TEXT formats via LOG_FORMAT environment variable.
    TEXT format includes service name and worker_id in the output.
    Applies TelethonNoiseFilter to suppress noisy RPC warnings.

    Args:
        level: Logging level as a string (e.g., "INFO", "DEBUG").
    """

    # Clear existing handlers to prevent duplicate logs and override library defaults
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    log_format = os.getenv("LOG_FORMAT", "TEXT").upper()
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # Create the stream handler
    handler = logging.StreamHandler()

    if log_format == "JSON":
        # JSON structured logging
        handler.setFormatter(JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S.%fZ"))
    else:
        # TEXT format with worker_id support
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

    # Apply the Telethon noise filter to suppress RPC warnings
    handler.addFilter(TelethonNoiseFilter())

    root_logger.addHandler(handler)
