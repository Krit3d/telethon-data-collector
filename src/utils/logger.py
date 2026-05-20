"""Centralized logging module for production-ready structured logging.

This module provides TEXT logging format optimized for production use.
Supports optional colorized output while maintaining Docker log compatibility.
Suppresses noisy external library logs (Telethon, asyncio, urllib3).
"""

import logging
import os
import sys

# Color codes for terminal output (only if TTY)
if sys.stderr.isatty() and os.getenv("NO_COLOR") is None:
    _CYAN = "\033[36m"
    _GREEN = "\033[32m"
    _YELLOW = "\033[33m"
    _RED = "\033[31m"
    _BOLD_RED = "\033[1;31m"
    _RESET = "\033[0m"
else:
    _CYAN = _GREEN = _YELLOW = _RED = _BOLD_RED = _RESET = ""


class ColorFormatter(logging.Formatter):
    """Custom formatter with colorized output and clean structured format.

    Format: timestamp | LEVEL    | message
    No module paths in output for clean horizontal-space efficient logs.
    """

    LEVEL_COLORS = {
        logging.DEBUG: _CYAN,
        logging.INFO: _GREEN,
        logging.WARNING: _YELLOW,
        logging.ERROR: _RED,
        logging.CRITICAL: _BOLD_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        # Colorize levelname
        color = self.LEVEL_COLORS.get(record.levelno, _RESET)
        levelname_colored = f"{color}{record.levelname:<8s}{_RESET}"

        # Format timestamp
        asctime = self.formatTime(record, self.datefmt)

        # Build clean log line: timestamp | LEVEL | message
        log_line = f"{asctime} | {levelname_colored} | {record.getMessage()}"

        # Append exception info if present (only for errors)
        if record.exc_info and record.levelno >= logging.WARNING:
            log_line += "\n" + self.formatException(record.exc_info)

        return log_line


def _configure_external_loggers() -> None:
    """Set appropriate log levels for noisy external libraries.

    Suppresses spam from Telethon RPC warnings and other verbose libraries
    while keeping errors visible.
    """
    noisy_loggers = [
        "telethon",
        "telethon.network",
        "telethon.extensions",
        "telethon.crypto",
        "asyncio",
        "urllib3",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def setup_logging(level: str) -> None:
    """Configure global logging with clean format and suppressed noise.

    Args:
        level: Logging level as a string (e.g., "INFO", "DEBUG").
    """

    # Clear existing handlers to prevent duplicate logs
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # Create stream handler with clean formatter
    handler = logging.StreamHandler(sys.stderr)
    formatter = ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Suppress noisy external library logs
    _configure_external_loggers()

    # Disable propagation for external loggers to prevent double-logging
    logging.getLogger("telethon").propagate = False
