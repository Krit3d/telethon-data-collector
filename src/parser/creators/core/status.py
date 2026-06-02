"""
Account status Enum and state transition validation for creators parser.

Defines valid statuses and enforces strict state machine rules to prevent
Telegram-specific statuses (like 'ready_for_parsing') from leaking to
creator platforms (Instagram, TikTok, YouTube, Threads).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Final

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Account

logger = logging.getLogger(__name__)

# Telegram-specific statuses that must NEVER be set on creator platform accounts
TELEGRAM_ONLY_STATUSES: Final[frozenset[str]] = frozenset({
    "ready_for_parsing",
})

# Creator platform statuses (valid for Instagram, TikTok, YouTube, Threads)
CREATOR_STATUSES: Final[frozenset[str]] = frozenset({
    "pending",
    "processing",
    "parsed",
    "rejected",
    "failed",
})

# All valid statuses across all platforms
VALID_STATUSES: Final[frozenset[str]] = CREATOR_STATUSES | TELEGRAM_ONLY_STATUSES


class AccountStatus(str, Enum):
    """Enum for account lifecycle statuses.

    Telegram pipeline: pending -> processing -> ready_for_parsing -> processing -> parsed
    Creator pipeline: pending -> processing -> parsed (or rejected/failed)
    """

    PENDING = "pending"
    PROCESSING = "processing"
    PARSED = "parsed"
    REJECTED = "rejected"
    FAILED = "failed"
    READY_FOR_PARSING = "ready_for_parsing"

    def is_valid_for_platform(self, platform: str) -> bool:
        """Check if this status is valid for the given platform.

        Args:
            platform: Platform name (INSTAGRAM, TELEGRAM, etc.).

        Returns:
            True if the status is valid for the platform, False otherwise.
        """
        if platform == "TELEGRAM":
            return True  # All statuses valid for Telegram
        return self.value in CREATOR_STATUSES


class StatusTransitionError(ValueError):
    """Raised when an invalid status transition is attempted."""

    def __init__(self, account_id: int, platform: str, current_status: str, new_status: str) -> None:
        self.account_id = account_id
        self.platform = platform
        self.current_status = current_status
        self.new_status = new_status
        message = (
            f"Invalid status transition for account {account_id} "
            f"(platform={platform}): {current_status} -> {new_status}"
        )
        super().__init__(message)


# Valid transition map: current_status -> set of allowed new statuses
VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"processing", "rejected"}),
    "processing": frozenset({"parsed", "rejected", "failed", "pending"}),
    "parsed": frozenset({"processing"}),  # Allow re-processing
    "rejected": frozenset({"pending", "processing"}),  # Allow re-queuing
    "failed": frozenset({"pending", "processing"}),  # Allow retry
    "ready_for_parsing": frozenset({"processing", "rejected"}),  # Telegram only
}


def validate_status_transition(
    account_id: int,
    platform: str,
    current_status: str | None,
    new_status: str,
) -> None:
    """Validate that a status transition is allowed.

    Args:
        account_id: Database ID of the account.
        platform: Platform name (INSTAGRAM, TELEGRAM, etc.).
        current_status: Current status of the account (None if new account).
        new_status: Proposed new status.

    Raises:
        StatusTransitionError: If the transition is invalid.
        ValueError: If the new_status is not a valid status.
    """
    # Check that new_status is a valid status
    if new_status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{new_status}' for account {account_id}. "
            f"Valid statuses: {VALID_STATUSES}"
        )

    # Check that new_status is valid for the platform
    try:
        status_enum = AccountStatus(new_status)
        if not status_enum.is_valid_for_platform(platform):
            raise StatusTransitionError(
                account_id=account_id,
                platform=platform,
                current_status=current_status or "None",
                new_status=new_status,
            )
    except ValueError:
        # Not a valid enum value - should have been caught above
        pass

    # If new account (no current status), any valid status is allowed
    if current_status is None:
        return

    # Check valid transition
    allowed_transitions = VALID_TRANSITIONS.get(current_status, frozenset())
    if new_status not in allowed_transitions:
        raise StatusTransitionError(
            account_id=account_id,
            platform=platform,
            current_status=current_status,
            new_status=new_status,
        )


async def update_account_status_safe(
    session: AsyncSession,
    account_id: int,
    new_status: str,
    platform: str | None = None,
) -> None:
    """Update account status with validation.

    Args:
        session: SQLAlchemy async session.
        account_id: Database ID of the account.
        new_status: New status to set.
        platform: Platform name. If None, will be fetched from database.

    Raises:
        StatusTransitionError: If the transition is invalid.
        ValueError: If account not found.
    """
    # Fetch current account state with row lock
    stmt = (
        select(Account)
        .where(Account.id == account_id)
        .with_for_update()
    )
    result = await session.execute(stmt)
    account = result.scalar_one_or_none()

    if account is None:
        raise ValueError(f"Account {account_id} not found")

    # Get platform if not provided
    effective_platform: str = platform if platform is not None else account.platform

    # Validate transition
    validate_status_transition(
        account_id=account_id,
        platform=effective_platform,
        current_status=account.status,
        new_status=new_status,
    )

    # Apply update
    account.status = new_status
    account.updated_at = datetime.now(timezone.utc)
