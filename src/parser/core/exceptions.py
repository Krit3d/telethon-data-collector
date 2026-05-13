"""Custom exceptions used throughout the parser module."""

from __future__ import annotations


class ChannelTaskRejected(Exception):
    """Exception raised when a channel task should be rejected due to invalid channel."""

    pass


class WorkerError(Exception):
    """Base exception for worker-related errors."""

    pass


class SessionExpiredError(WorkerError):
    """Exception raised when a Telegram session has expired or been revoked."""

    pass


class ShadowBanDetectedError(WorkerError):
    """Exception raised when a shadow ban is detected."""

    pass
