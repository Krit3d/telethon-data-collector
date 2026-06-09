"""
Multi-platform Ingestion Coordinator for creators scraping.

Queries the database for pending accounts, processes them concurrently across
different platforms (Instagram, Threads, TikTok, YouTube), and updates their scraping status.

This module is designed to run as a production-grade background daemon with
graceful shutdown handling for OS signals (SIGTERM, SIGINT).

Uses atomic database methods to eliminate race conditions and deadlocks.
"""

import argparse
import asyncio
import logging
import signal
import sys
from typing import Any

from sqlalchemy import select

from src.config.config import Settings, load_settings
from src.db.database import Database
from src.db.models import Account
from src.parser.creators.core.queries import SearchQueriesManager
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

    Uses atomic database methods to eliminate race conditions and deadlocks.
    """

    def __init__(
        self,
        db: Database,
        settings: Settings,
        shutdown_event: asyncio.Event | None = None,
        platform: str | None = None,
    ) -> None:
        """
        Initialize the coordinator with database and settings.

        Args:
            db: Database instance for atomic database operations.
            settings: Application settings containing configuration values.
            shutdown_event: asyncio.Event | None for graceful shutdown coordination.
            platform: Optional platform filter to isolate scraping queues per container.
        """
        self.db = db
        self.settings = settings
        self._shutdown_event = shutdown_event
        self.platform_filter: str | None = platform
        self.min_subscribers: int = settings.creators_min_subscribers
        self.status_threshold_hours: int = (
            settings.creators_status_threshold_hours
        )
        self.poll_interval_s: int = settings.creators_poll_interval_s
        
        # Initialize SearchQueriesManager for Round-Robin query rotation
        self._queries_manager = SearchQueriesManager(settings.search_queries_path)
        self._current_query_index: int = 0
        
        logger.info(
            f"CreatorsCoordinator initialized with min_subscribers={self.min_subscribers}, "
            f"status_threshold_hours={self.status_threshold_hours}, "
            f"poll_interval_s={self.poll_interval_s}, "
            f"platform_filter={self.platform_filter}"
        )

    def _get_next_queries(self, count: int) -> list[tuple[str, str]]:
        """
        Fetch the next 'count' queries using Round-Robin rotation from balanced queries.

        Safely retrieves a slice of queries starting from self._current_query_index,
        wrapping around if the end of the queries list is reached. Updates the
        self._current_query_index accordingly for the next rotation.

        Args:
            count: Number of queries to fetch.

        Returns:
            List of tuples (query, category) representing the next queries to process.
        """
        balanced_queries = self._queries_manager.get_balanced_queries()

        if not balanced_queries:
            return []

        total_queries = len(balanced_queries)

        # Handle case where count > total_queries by wrapping multiple times
        selected_queries: list[tuple[str, str]] = []

        for _ in range(count):
            selected_queries.append(balanced_queries[self._current_query_index])
            self._current_query_index = (self._current_query_index + 1) % total_queries

        return selected_queries

    async def _ensure_pending_queue(
        self,
        platform: str,
        client: ScrapeCreatorsClient,
    ) -> None:
        """
        Ensure the pending queue has enough accounts to process.

        Checks the current count of accounts with status "pending" for the active
        platform. If the count is less than creators_batch_size * 2, it means the
        queue is running dry. In this case, fetches the next 3 queries from
        _get_next_queries and runs Stage 1 (Discovery) to populate the database
        with fresh "pending" candidates.

        Args:
            platform: Platform name to check and discover candidates for.
            client: Shared ScrapeCreatorsClient instance for API requests.
        """
        # Determine the platform to filter by
        active_platform = self.platform_filter if self.platform_filter else platform

        # Count pending accounts for the active platform using atomic method
        pending_count = await self.db.count_pending_creator_accounts(active_platform)

        logger.debug(
            f"Pending accounts for platform '{active_platform}': {pending_count}"
        )

        # Check if queue is running dry
        threshold = self.settings.creators_batch_size * 2

        if pending_count < threshold:
            logger.info(
                f"Pending queue running low ({pending_count} < {threshold}). "
                f"Running Stage 1 (Discovery) to populate candidates..."
            )

            # Fetch next 3 queries using Round-Robin rotation
            next_queries = self._get_next_queries(count=3)

            if not next_queries:
                logger.warning("No queries available for discovery.")
                return

            # Get the platform parser
            parser = get_platform_parser(
                platform=active_platform,
                session_maker=self.db.async_session,
                client=client,
                settings=self.settings,
            )

            # Run discovery for each query
            for query, category in next_queries:
                if self.is_shutdown_requested():
                    logger.info("Shutdown requested during discovery. Aborting.")
                    return

                try:
                    logger.info(
                        f"Discovering candidates for query='{query}', "
                        f"category='{category}' on platform={active_platform}"
                    )
                    discovered_count = await parser.discover_candidates(query, category)
                    logger.info(
                        f"Discovered {discovered_count} accounts for query='{query}', "
                        f"category='{category}' on platform={active_platform}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to discover candidates for query='{query}', "
                        f"category='{category}' on platform={active_platform}: {e!r}",
                        exc_info=e,
                    )

            logger.info("Stage 1 (Discovery) completed for this cycle.")
        else:
            logger.debug(
                f"Pending queue has sufficient accounts ({pending_count} >= {threshold}). "
                f"Skipping discovery."
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
        Run a single ingestion cycle: ensure pending queue, claim accounts, and process them.

        This method implements a self-feeding 3-stage pipeline:
        - Stage 1 (Discovery): Ensures the pending queue has enough candidates by
          running discovery if the count is below threshold.
        - Stage 2 (Profile Parsing): Processes pending accounts to parse profiles.
        - Stage 3 (Content Parsing): Fetches and stores content for parsed accounts.

        A single ScrapeCreatorsClient is created and shared across all operations in
        this batch to optimize connection handling and track credit usage.

        Uses atomic claim_creator_accounts to eliminate race conditions.

        Args:
            batch_size: Maximum number of accounts to process in this cycle.
            concurrency_limit: Maximum number of concurrent account processing tasks.
        """
        logger.info(
            f"Starting ingestion cycle with batch_size={batch_size}, "
            f"concurrency_limit={concurrency_limit}"
        )

        # Use a single ScrapeCreatorsClient for all operations in this batch
        # This optimizes connection handling and allows credit tracking
        async with ScrapeCreatorsClient(self.settings) as client:
            # Record initial credits if available
            initial_credits: int | None = client.last_credits_remaining

            # Determine the platform for this cycle
            platform = self.platform_filter if self.platform_filter else "INSTAGRAM"

            # Stage 1: Ensure pending queue is populated (Discovery)
            # This runs BEFORE querying for pending accounts to ensure the queue
            # is populated if it's running low
            await self._ensure_pending_queue(platform, client)

            # Stage 2: Atomically claim accounts for processing
            # This eliminates race conditions by using CTE with FOR UPDATE SKIP LOCKED
            accounts = await self.db.claim_creator_accounts(
                platforms=[self.platform_filter] if self.platform_filter else CREATOR_PLATFORMS,
                batch_size=batch_size,
                status_threshold_hours=self.status_threshold_hours,
            )

            if not accounts:
                logger.info("No accounts to process in this cycle")
                return

            logger.info(f"Found {len(accounts)} accounts to process")

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
        Updates the account status throughout the process using atomic database methods.

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

            try:
                # Get the platform parser using the factory with shared client
                parser = get_platform_parser(
                    platform=platform,
                    session_maker=self.db.async_session,
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
                    await self.db.update_creator_account_status(
                        account_id, STATUS_REJECTED
                    )
                    return

                # Get the updated account to check status and subscriber count
                # Use db_account_id returned from parse_profile to ensure we're
                # operating on the correctly resolved/merged database record
                async with self.db.async_session() as session:
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
                        await self.db.update_creator_account_status(
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
                async with self.db.async_session() as session:
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
                    await self.db.update_creator_account_status(db_account_id, STATUS_PARSED)
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
                await self.db.update_creator_account_status(account_id, STATUS_FAILED)

    async def reset_orphaned_processing_accounts(self) -> None:
        """
        Reset orphaned creator accounts stuck in 'processing' status back to 'pending'.

        This method is called during startup to handle accounts that were left in
        'processing' status due to container restarts or crashes. It only affects
        creator platforms (INSTAGRAM, THREADS, TIKTOK, YOUTUBE) and explicitly
        excludes TELEGRAM to prevent state collisions with the Telegram crawler.
        When platform_filter is set, only resets accounts for that specific platform.

        Uses the atomic reset_orphaned_creator_accounts database method.
        """
        count = await self.db.reset_orphaned_creator_accounts(
            platforms=[self.platform_filter] if self.platform_filter else CREATOR_PLATFORMS
        )

        if count > 0:
            logger.info(
                f"Successfully restored {count} orphaned creator accounts to 'pending'"
            )
        else:
            logger.info("No orphaned creator accounts found to restore")

    async def run_discovery(self) -> None:
        """
        Run the candidate discovery phase using balanced queries from search_queries.json.

        This method:
        - Loads balanced queries via SearchQueriesManager
        - Iterates over filtered platform(s)
        - For each platform, instantiates the parser via get_platform_parser()
        - Calls await parser.discover_candidates(query, category) for each query
        - Logs the number of discovered accounts
        - Checks for shutdown requests after each iteration
        - Relies on concurrency_limit semaphore for API rate limiting (no artificial sleeps)
        """
        logger.info("Starting candidate discovery phase...")

        # Instantiate SearchQueriesManager using settings.search_queries_path
        queries_manager = SearchQueriesManager(self.settings.search_queries_path)
        balanced_queries = queries_manager.get_balanced_queries()

        if not balanced_queries:
            logger.warning("No balanced queries available. Exiting discovery phase.")
            return

        logger.info(f"Loaded {len(balanced_queries)} balanced queries for discovery.")

        # Determine which platforms to iterate over
        if self.platform_filter:
            platforms_to_process = [self.platform_filter]
        else:
            platforms_to_process = CREATOR_PLATFORMS

        logger.info(f"Discovery will process platforms: {platforms_to_process}")

        # Iterate through each platform
        for platform in platforms_to_process:
            logger.info(f"Starting discovery for platform: {platform}")

            # Instantiate the parser via get_platform_parser
            # Create a temporary client for discovery
            async with ScrapeCreatorsClient(self.settings) as client:
                parser = get_platform_parser(
                    platform=platform,
                    session_maker=self.db.async_session,
                    client=client,
                    settings=self.settings,
                )

                # Iterate through balanced queries
                for query, category in balanced_queries:
                    # Check for shutdown request
                    if self.is_shutdown_requested():
                        logger.info("Shutdown requested during discovery. Aborting early.")
                        return

                    try:
                        logger.info(
                            f"Discovering candidates for query='{query}', "
                            f"category='{category}' on platform={platform}"
                        )
                        discovered_count = await parser.discover_candidates(query, category)
                        logger.info(
                            f"Discovered {discovered_count} accounts for query='{query}', "
                            f"category='{category}' on platform={platform}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to discover candidates for query='{query}', "
                            f"category='{category}' on platform={platform}: {e!r}",
                            exc_info=e,
                        )
    
                logger.info(f"Completed discovery for platform: {platform}")

        logger.info("Candidate discovery phase completed.")


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
        Parsed arguments namespace with 'platform' and 'discover' attributes.
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
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Run candidate discovery phase using queries from search_queries.json",
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

    # Initialize database using Database class for optimized connection pooling
    db = Database(settings.db_url, echo=False)

    # Create shutdown event for signal handling
    shutdown_event: asyncio.Event = asyncio.Event()

    # Instantiate the coordinator with optional platform filter
    # Pass db instance for atomic database operations
    coordinator: CreatorsCoordinator = CreatorsCoordinator(
        db=db,
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

    # Check if discovery mode is requested
    if args.discover:
        logger.info("Discovery mode enabled. Running candidate discovery phase...")
        try:
            await coordinator.run_discovery()
            logger.info("Candidate discovery phase completed successfully.")
        except Exception as e:
            logger.error(
                f"Error during candidate discovery phase: {e!r}",
                exc_info=e,
            )
            # Close database gracefully before exiting
            await db.close()
            sys.exit(1)

        # Close database gracefully and exit cleanly without running the daemon loop
        logger.info("Closing database connections...")
        await db.close()
        logger.info("Discovery phase completed. Exiting.")
        sys.exit(0)

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

        # Close database connections gracefully using Database class
        logger.info("Closing database connections...")
        await db.close()

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
