"""
Multi-platform Ingestion Coordinator for creators scraping.

Queries the database for pending accounts, processes them concurrently across
different platforms (Instagram, Threads, TikTok, YouTube), and updates their scraping status.

This module is designed to run as a production-grade background daemon with
graceful shutdown handling for OS signals (SIGTERM, SIGINT).
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config.config import Settings, load_settings
from src.db.models import Account
from src.parser.creators.platforms import get_platform_parser
from src.parser.creators.sc_client import ScrapeCreatorsClient

logger = logging.getLogger(__name__)

# Status constants
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_PARSED = "parsed"
STATUS_REJECTED = "rejected"
STATUS_FAILED = "failed"

# Platforms that are handled by the creators coordinator (not Telegram)
CREATOR_PLATFORMS = ["INSTAGRAM", "THREADS", "TIKTOK", "YOUTUBE"]

# Default threshold for re-processing accounts (24 hours ago)
DEFAULT_STATUS_UPDATE_THRESHOLD_HOURS = 24

# Minimum subscribers/followers threshold for processing
MIN_SUBSCRIBERS_THRESHOLD = 3000

# Default poll interval for the daemon loop (seconds)
DEFAULT_CREATORS_POLL_INTERVAL_S = 300


class CreatorsCoordinator:
    """
    Multi-platform ingestion coordinator for creators scraping.

    Queries the database for accounts that need processing, processes them
    concurrently with configurable concurrency limits, and updates their
    scraping status in the database.
    """

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        settings: Settings,
        shutdown_event: asyncio.Event | None = None,
        platform: str | None = None,
    ) -> None:
        """
        Initialize the coordinator with database and settings.

        Args:
            session_maker: SQLAlchemy async session maker for database operations.
            settings: Application settings containing configuration values.
            shutdown_event: asyncio.Event | None for graceful shutdown coordination.
            platform: Optional platform filter to isolate scraping queues per container.
        """
        self.session_maker = session_maker
        self.settings = settings
        self._shutdown_event = shutdown_event
        self.platform_filter: str | None = platform
        self.min_subscribers: int = settings.creators_min_subscribers
        self.status_threshold_hours: int = (
            settings.creators_status_threshold_hours
        )
        self.poll_interval_s: int = settings.creators_poll_interval_s
        logger.info(
            f"CreatorsCoordinator initialized with min_subscribers={self.min_subscribers}, "
            f"status_threshold_hours={self.status_threshold_hours}, "
            f"poll_interval_s={self.poll_interval_s}, "
            f"platform_filter={self.platform_filter}"
        )

    def is_shutdown_requested(self) -> bool:
        """
        Check if a graceful shutdown has been requested.

        Returns:
            True if shutdown event is set, False otherwise.
        """
        if self._shutdown_event is None:
            return False
        return self._shutdown_event.is_set()

    async def run_once(
        self,
        batch_size: int = 10,
        concurrency_limit: int = 3,
    ) -> None:
        """
        Run a single ingestion cycle: query pending accounts and process them concurrently.

        Queries the accounts table for accounts where platform is in CREATOR_PLATFORMS
        AND status is "pending" or "failed" with updated_at older than the configured
        threshold, ordered by updated_at ascending. Processes these accounts concurrently
        using an asyncio.Semaphore.

        A single ScrapeCreatorsClient is created and shared across all accounts in this
        batch to optimize connection handling and track credit usage.

        Args:
            batch_size: Maximum number of accounts to process in this cycle.
            concurrency_limit: Maximum number of concurrent account processing tasks.
        """
        logger.info(
            f"Starting ingestion cycle with batch_size={batch_size}, "
            f"concurrency_limit={concurrency_limit}"
        )

        # Calculate the threshold datetime for re-processing
        threshold_time = datetime.now(timezone.utc) - timedelta(
            hours=self.status_threshold_hours
        )

        # Query for accounts to process
        async with self.session_maker() as session:
            # Determine platform filter condition
            if self.platform_filter:
                platform_condition = Account.platform == self.platform_filter
            else:
                platform_condition = Account.platform.in_(CREATOR_PLATFORMS)

            stmt = (
                select(Account)
                .where(
                    platform_condition,
                    (
                        (Account.status == STATUS_PENDING)
                        | (
                            (Account.status == STATUS_FAILED)
                            & (Account.updated_at < threshold_time)
                        )
                    ),
                )
                .order_by(Account.updated_at.asc())
                .limit(batch_size)
            )

            result = await session.execute(stmt)
            accounts = result.scalars().all()

        if not accounts:
            logger.info("No accounts to process in this cycle")
            return

        logger.info(f"Found {len(accounts)} accounts to process")

        # Use a single ScrapeCreatorsClient for all accounts in this batch
        # This optimizes connection handling and allows credit tracking
        async with ScrapeCreatorsClient(self.settings) as client:
            # Record initial credits if available
            initial_credits: int | None = client.last_credits_remaining

            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(concurrency_limit)

            # Create tasks for processing each account
            tasks = [
                asyncio.create_task(
                    self._process_single_account(
                        account_id=account.id,
                        platform=account.platform,
                        username=account.username,
                        semaphore=semaphore,
                        client=client,
                    )
                )
                for account in accounts
                if account.username  # Only process accounts with a username
            ]

            if not tasks:
                logger.info("No valid accounts with usernames to process")
                return

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log any exceptions that occurred
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Unexpected error in task {i}: {result!r}", exc_info=result
                    )

            # Calculate credit usage for this batch
            final_credits: int | None = client.last_credits_remaining

            # Log batch credit summary at INFO level
            if initial_credits is not None and final_credits is not None:
                credits_spent = initial_credits - final_credits
                logger.info(
                    f"=== BATCH CREDIT SUMMARY: Spent: {credits_spent} credits | Remaining: {final_credits} ==="
                )
            elif final_credits is not None:
                # Initial credits not available (first request may not have populated it)
                logger.info(
                    f"=== BATCH CREDIT SUMMARY: Remaining: {final_credits} ==="
                )
            else:
                # Credit data not available
                logger.info(
                    "=== BATCH CREDIT SUMMARY: Credit data not available ==="
                )

        logger.info(
            f"Ingestion cycle completed. Processed {len(tasks)} accounts."
        )

    async def _process_single_account(
        self,
        account_id: int,
        platform: str,
        username: str | None,
        semaphore: asyncio.Semaphore,
        client: ScrapeCreatorsClient,
    ) -> None:
        """
        Process a single account: parse profile and content, update status.

        Protects the execution block with the semaphore for concurrency control.
        Uses the shared ScrapeCreatorsClient passed from run_once.
        Gets the corresponding platform parser using the factory.
        Updates the account status throughout the process.

        Args:
            account_id: Database ID of the account to process.
            platform: Platform name (e.g., "INSTAGRAM", "THREADS", "TIKTOK", "YOUTUBE").
            username: Platform username/handle (without @ prefix).
            semaphore: asyncio.Semaphore for concurrency control.
            client: Shared ScrapeCreatorsClient instance for API requests.
        """
        if not username:
            logger.warning(f"Account {account_id} has no username, skipping")
            return

        async with semaphore:
            logger.info(
                f"Processing account {account_id} (platform={platform}, username={username})"
            )

            # Update status to "processing"
            await self._update_account_status(account_id, STATUS_PROCESSING)

            try:
                # Get the platform parser using the factory with shared client
                parser = get_platform_parser(
                    platform=platform,
                    session_maker=self.session_maker,
                    client=client,
                    settings=self.settings,
                )

                # Execute parse_profile to get subscriber count
                logger.debug(
                    f"Parsing profile for {username} on {platform}"
                )
                db_account_id = await parser.parse_profile(username)

                if db_account_id is None:
                    # Profile parsing returned None - likely below threshold
                    logger.info(
                        f"Profile {username} on {platform} rejected "
                        f"(below subscriber threshold or parsing failed)"
                    )
                    await self._update_account_status(
                        account_id, STATUS_REJECTED
                    )
                    return

                # Get the updated account to check status and subscriber count
                # Use db_account_id returned from parse_profile to ensure we're
                # operating on the correctly resolved/merged database record
                async with self.session_maker() as session:
                    stmt = select(Account).where(Account.id == db_account_id)
                    result = await session.execute(stmt)
                    account = result.scalar_one_or_none()

                # Check if account was rejected during profile parsing
                if account and account.status == STATUS_REJECTED:
                    logger.info(
                        f"Account {username} was rejected during profile parsing. "
                        f"Skipping content parse."
                    )
                    return

                if account and account.subscribers_count is not None:
                    if account.subscribers_count < self.min_subscribers:
                        logger.info(
                            f"Account {username} has {account.subscribers_count} "
                            f"subscribers, below threshold {self.min_subscribers}. "
                            f"Marking as rejected."
                        )
                        await self._update_account_status(
                            db_account_id, STATUS_REJECTED
                        )
                        return

                # Execute parse_content to fetch and store content
                # Use db_account_id to ensure operations run against the correctly
                # resolved/merged database record
                logger.debug(
                    f"Parsing content for {username} on {platform}"
                )
                await parser.parse_content(
                    account_id=db_account_id,
                    platform_id=username,  # Using username as platform_id for API calls
                    max_items=50,
                )

                # After content parsing, check the current status before updating to "parsed"
                # The status may have been changed to rejected or failed during content parsing
                async with self.session_maker() as session:
                    stmt = select(Account).where(Account.id == db_account_id)
                    result = await session.execute(stmt)
                    account_after_content = result.scalar_one_or_none()

                if account_after_content and account_after_content.status in (
                    STATUS_REJECTED,
                    STATUS_FAILED,
                ):
                    logger.info(
                        f"Account {username} status is '{account_after_content.status}' "
                        f"after content parsing. Not updating to '{STATUS_PARSED}'."
                    )
                    return

                # Only update status to "parsed" if account is still in "processing" state
                if account_after_content and account_after_content.status == STATUS_PROCESSING:
                    # On successful completion, update status to "parsed"
                    # Use db_account_id for status update
                    await self._update_account_status(db_account_id, STATUS_PARSED)
                    logger.info(
                        f"Successfully processed account {db_account_id} "
                        f"({username} on {platform})"
                    )

            except Exception as e:
                # On failure, catch the exception, log it, and update status to "failed"
                logger.error(
                    f"Failed to process account {account_id} "
                    f"({username} on {platform}): {e!r}",
                    exc_info=e,
                )
                await self._update_account_status(account_id, STATUS_FAILED)

    async def _update_account_status(
        self,
        account_id: int,
        status: str,
    ) -> None:
        """
        Update the account status and refresh the updated_at timestamp.

        Args:
            account_id: Database ID of the account to update.
            status: New status value (e.g., "processing", "parsed", "failed").
        """
        try:
            async with self.session_maker() as session:
                stmt = (
                    update(Account)
                    .where(Account.id == account_id)
                    .values(
                        status=status,
                        updated_at=datetime.now(timezone.utc),
                    )
                )
                await session.execute(stmt)
                await session.commit()
                logger.debug(
                    f"Updated account {account_id} status to '{status}'"
                )
        except Exception as e:
            logger.error(
                f"Failed to update account {account_id} status to '{status}': {e!r}",
                exc_info=e,
            )

    async def reset_orphaned_processing_accounts(self) -> None:
        """
        Reset orphaned creator accounts stuck in 'processing' status back to 'pending'.

        This method is called during startup to handle accounts that were left in
        'processing' status due to container restarts or crashes. It only affects
        creator platforms (INSTAGRAM, THREADS, TIKTOK, YOUTUBE) and explicitly
        excludes TELEGRAM to prevent state collisions with the Telegram crawler.
        When platform_filter is set, only resets accounts for that specific platform.
        """
        async with self.session_maker() as session:
            # Determine platform filter condition
            if self.platform_filter:
                platform_condition = Account.platform == self.platform_filter
            else:
                platform_condition = Account.platform.in_(CREATOR_PLATFORMS)

            # Create update statement for orphaned processing accounts
            stmt = (
                update(Account)
                .where(
                    platform_condition,
                    Account.status == STATUS_PROCESSING,
                )
                .values(
                    status=STATUS_PENDING,
                    updated_at=datetime.now(timezone.utc),
                )
                .returning(Account.id)
            )
            result = await session.execute(stmt)
            updated_rows = result.fetchall()
            await session.commit()

            # Log the number of reset rows
            count = len(updated_rows)
            if count > 0:
                logger.info(
                    f"Successfully restored {count} orphaned creator accounts to 'pending'"
                )
            else:
                logger.info("No orphaned creator accounts found to restore")


async def _sleep_with_shutdown_check(
    shutdown_event: asyncio.Event,
    interval_s: int,
    check_interval_s: int = 5,
) -> bool:
    """
    Sleep for a specified interval, checking for shutdown requests periodically.

    This function breaks the sleep into small chunks to allow responsive
    shutdown handling without busy-waiting.

    Args:
        shutdown_event: Event to check for shutdown requests.
        interval_s: Total sleep duration in seconds.
        check_interval_s: Interval between shutdown checks in seconds.

    Returns:
        True if shutdown was requested, False if sleep completed normally.
    """
    elapsed: float = 0.0
    while elapsed < interval_s:
        if shutdown_event.is_set():
            logger.info(
                "Shutdown requested during sleep cycle, waking up early."
            )
            return True
        sleep_chunk: float = min(check_interval_s, interval_s - elapsed)
        await asyncio.sleep(sleep_chunk)
        elapsed += sleep_chunk
    return False


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the creators coordinator.

    Returns:
        Parsed arguments namespace with 'platform' attribute.
    """
    parser = argparse.ArgumentParser(
        description="Multi-platform creators scraping coordinator daemon"
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Process only a specific creator platform (e.g., INSTAGRAM, THREADS, TIKTOK, YOUTUBE)",
    )
    return parser.parse_args()


async def main() -> None:
    """
    Main async entrypoint for the creators coordinator daemon.

    This function:
    - Parses CLI arguments (--platform)
    - Loads settings via load_settings()
    - Initializes the database engine and session maker
    - Sets up signal handlers for graceful shutdown (SIGTERM, SIGINT)
    - Runs the coordinator in an infinite polling loop
    - Ensures clean exit on shutdown signals
    """
    # Parse CLI arguments before setting up the engine
    args: argparse.Namespace = parse_args()

    # Validate platform argument if provided
    platform_filter: str | None = None
    if args.platform is not None:
        platform_upper: str = args.platform.upper()
        if platform_upper not in CREATOR_PLATFORMS:
            raise ValueError(
                f"Invalid platform '{args.platform}'. "
                f"Must be one of: {', '.join(CREATOR_PLATFORMS)}"
            )
        platform_filter = platform_upper
        logger.info(f"CLI argument --platform={platform_filter} validated")

    settings: Settings = load_settings()

    # Initialize database engine and session maker
    engine = create_async_engine(
        settings.db_url,
        echo=False,
        pool_size=10,
        max_overflow=5,
        pool_timeout=15.0,
        pool_pre_ping=True,
        pool_recycle=3600,
    )
    session_maker: async_sessionmaker[AsyncSession] = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    # Create shutdown event for signal handling
    shutdown_event: asyncio.Event = asyncio.Event()

    # Instantiate the coordinator with optional platform filter
    coordinator: CreatorsCoordinator = CreatorsCoordinator(
        session_maker=session_maker,
        settings=settings,
        shutdown_event=shutdown_event,
        platform=platform_filter,
    )

    # Run auto-heal to reset orphaned processing accounts
    try:
        logger.info(
            "Running auto-heal to reset orphaned processing accounts..."
        )
        await coordinator.reset_orphaned_processing_accounts()
    except Exception as e:
        logger.error(
            f"Failed to reset orphaned processing accounts: {e!r}",
            exc_info=e,
        )
        logger.warning("Continuing startup despite auto-heal failure...")

    def _signal_handler(sig: int, frame: Any | None = None) -> None:
        """
        Handle OS termination signals (SIGTERM, SIGINT).

        Sets the shutdown event to trigger graceful shutdown. The main loop
        will finish the current batch and exit cleanly.
        """
        sig_name: str = signal.Signals(sig).name
        logger.info(
            f"Received signal {sig_name}, initiating graceful shutdown..."
        )
        logger.info(
            "Waiting for current batch to complete before shutting down..."
        )
        shutdown_event.set()

    # Register signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda s=sig: _signal_handler(s))
        except NotImplementedError:
            # Fallback for Windows or environments without add_signal_handler
            signal.signal(sig, _signal_handler)

    logger.info("Creators coordinator daemon started. Press Ctrl+C to stop.")

    try:
        while not shutdown_event.is_set():
            # Check shutdown before starting a new batch
            if shutdown_event.is_set():
                logger.info("Shutdown requested before batch start, exiting.")
                break

            logger.info("Starting new ingestion batch...")
            try:
                # Run a single ingestion cycle
                await coordinator.run_once(
                    batch_size=settings.creators_batch_size,
                    concurrency_limit=settings.creators_concurrency,
                )
            except Exception as exc:
                logger.error(
                    f"Unexpected error in ingestion batch: {exc!r}",
                    exc_info=exc,
                )

            # Check shutdown before sleeping
            if shutdown_event.is_set():
                logger.info(
                    "Shutdown requested after batch completion, exiting."
                )
                break

            # Sleep with periodic shutdown checks
            logger.info(
                f"Sleeping for {coordinator.poll_interval_s}s until next batch..."
            )
            shutdown_during_sleep: bool = await _sleep_with_shutdown_check(
                shutdown_event=shutdown_event,
                interval_s=coordinator.poll_interval_s,
            )
            if shutdown_during_sleep:
                logger.info("Shutdown requested during sleep, exiting.")
                break

    finally:
        # Graceful shutdown: finish current processing, commit transactions, close connections
        logger.info("Shutting down gracefully...")

        # Close database engine
        logger.info("Closing database connections...")
        await engine.dispose()

        logger.info("Creators coordinator daemon stopped cleanly.")
        sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, exiting.")
        sys.exit(0)
    except SystemExit as e:
        # Allow clean sys.exit(0)
        raise
    except Exception as e:
        logger.error(f"Fatal error in coordinator daemon: {e!r}", exc_info=e)
        sys.exit(1)
