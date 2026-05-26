"""
Multi-platform Ingestion Coordinator for creators scraping.

Queries the database for pending accounts, processes them concurrently across
different platforms (Instagram, TikTok), and updates their scraping status.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config.config import Settings
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
CREATOR_PLATFORMS = ["INSTAGRAM", "TIKTOK"]

# Default threshold for re-processing accounts (24 hours ago)
DEFAULT_STATUS_UPDATE_THRESHOLD_HOURS = 24

# Minimum subscribers/followers threshold for processing
MIN_SUBSCRIBERS_THRESHOLD = 3000


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
    ) -> None:
        """
        Initialize the coordinator with database and settings.

        Args:
            session_maker: SQLAlchemy async session maker for database operations.
            settings: Application settings containing configuration values.
        """
        self.session_maker = session_maker
        self.settings = settings
        self.min_subscribers: int = getattr(
            settings, "CREATORS_MIN_SUBSCRIBERS", MIN_SUBSCRIBERS_THRESHOLD
        )
        self.status_threshold_hours: int = getattr(
            settings,
            "CREATORS_STATUS_THRESHOLD_HOURS",
            DEFAULT_STATUS_UPDATE_THRESHOLD_HOURS,
        )
        logger.info(
            f"CreatorsCoordinator initialized with min_subscribers={self.min_subscribers}, "
            f"status_threshold_hours={self.status_threshold_hours}"
        )

    async def run_once(
        self,
        batch_size: int = 10,
        concurrency_limit: int = 3,
    ) -> None:
        """
        Run a single ingestion cycle: query pending accounts and process them concurrently.

        Queries the accounts table for accounts where platform is NOT "TELEGRAM"
        AND status is "pending" or older than a configurable threshold (e.g., 24 hours),
        ordered by updated_at ascending. Processes these accounts concurrently using
        an asyncio.Semaphore.

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
            stmt = (
                select(Account)
                .where(
                    Account.platform.in_(CREATOR_PLATFORMS),
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

        logger.info(f"Ingestion cycle completed. Processed {len(tasks)} accounts.")

    async def _process_single_account(
        self,
        account_id: int,
        platform: str,
        username: str | None,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """
        Process a single account: parse profile and content, update status.

        Protects the execution block with the semaphore for concurrency control.
        Instantiates ScrapeCreatorsClient using an async context manager.
        Gets the corresponding platform parser using the factory.
        Updates the account status throughout the process.

        Args:
            account_id: Database ID of the account to process.
            platform: Platform name (e.g., "INSTAGRAM", "TIKTOK").
            username: Platform username/handle (without @ prefix).
            semaphore: asyncio.Semaphore for concurrency control.
        """
        if not username:
            logger.warning(
                f"Account {account_id} has no username, skipping"
            )
            return

        async with semaphore:
            logger.info(
                f"Processing account {account_id} (platform={platform}, username={username})"
            )

            # Update status to "processing"
            await self._update_account_status(account_id, STATUS_PROCESSING)

            client: ScrapeCreatorsClient | None = None
            try:
                # Instantiate ScrapeCreatorsClient using async context manager
                client = ScrapeCreatorsClient(self.settings)
                async with client:
                    # Get the platform parser using the factory
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

                    # Get the updated account to check subscriber count
                    async with self.session_maker() as session:
                        stmt = select(Account).where(
                            Account.id == account_id
                        )
                        result = await session.execute(stmt)
                        account = result.scalar_one_or_none()

                        if account and account.subscribers_count is not None:
                            if account.subscribers_count < self.min_subscribers:
                                logger.info(
                                    f"Account {username} has {account.subscribers_count} "
                                    f"subscribers, below threshold {self.min_subscribers}. "
                                    f"Marking as rejected."
                                )
                                await self._update_account_status(
                                    account_id, STATUS_REJECTED
                                )
                                return

                    # Execute parse_content to fetch and store content
                    logger.debug(
                        f"Parsing content for {username} on {platform}"
                    )
                    await parser.parse_content(
                        account_id=account_id,
                        platform_id=username,  # Using username as platform_id for API calls
                        max_items=50,
                    )

                    # On successful completion, update status to "parsed"
                    await self._update_account_status(
                        account_id, STATUS_PARSED
                    )
                    logger.info(
                        f"Successfully processed account {account_id} "
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

            finally:
                # Ensure client is closed if not using context manager properly
                if client and not client._session:
                    await client.close()

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
