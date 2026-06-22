import argparse
import asyncio
import logging
import signal
import sys
from collections import defaultdict
from typing import Any

from sqlalchemy import select

from src.config.config import Settings, load_settings
from src.db.database import Database
from src.db.models import Account
from src.parser.creators.core.queries import SearchQueriesManager
from src.parser.creators.platforms import get_platform_parser
from src.parser.creators.sc_client import ScrapeCreatorsClient

logger = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_PARSED = "parsed"
STATUS_REJECTED = "rejected"
STATUS_FAILED = "failed"

CREATOR_PLATFORMS = ["INSTAGRAM", "THREADS", "TIKTOK", "YOUTUBE"]

DEFAULT_STATUS_UPDATE_THRESHOLD_HOURS = 24

MIN_SUBSCRIBERS_THRESHOLD = 5000

DEFAULT_CREATORS_POLL_INTERVAL_S = 300


class CreatorsCoordinator:

    def __init__(
        self,
        db: Database,
        settings: Settings,
        shutdown_event: asyncio.Event | None = None,
        platform: str | None = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self._shutdown_event = shutdown_event
        self.platform_filter: str | None = platform
        self.min_subscribers: int = settings.creators_min_subscribers
        self.status_threshold_hours: int = (
            settings.creators_status_threshold_hours
        )
        self.poll_interval_s: int = settings.creators_poll_interval_s

        self._queries_manager = SearchQueriesManager(settings.search_queries_path)
        self._current_category_index: int = 0
        self._category_query_indices: dict[str, int] = {}

        logger.info(
            f"CreatorsCoordinator initialized with min_subscribers={self.min_subscribers}, "
            f"status_threshold_hours={self.status_threshold_hours}, "
            f"poll_interval_s={self.poll_interval_s}, "
            f"platform_filter={self.platform_filter}"
        )

    def _get_next_queries(self, count: int) -> list[tuple[str, str]]:
        balanced_queries = self._queries_manager.get_balanced_queries()

        if not balanced_queries:
            return []

        grouped: dict[str, list[str]] = defaultdict(list)
        for query, category in balanced_queries:
            grouped[category].append(query)

        sorted_categories: list[str] = sorted(grouped.keys())

        self._current_category_index = getattr(self, "_current_category_index", 0)
        self._category_query_indices = getattr(self, "_category_query_indices", {})

        selected_queries: list[tuple[str, str]] = []

        for _ in range(count):
            category = sorted_categories[self._current_category_index % len(sorted_categories)]
            local_index = self._category_query_indices.get(category, 0)
            queries_in_cat = grouped[category]
            query = queries_in_cat[local_index % len(queries_in_cat)]

            selected_queries.append((query, category))

            self._category_query_indices[category] = local_index + 1
            self._current_category_index = (self._current_category_index + 1) % len(sorted_categories)

        return selected_queries

    async def _ensure_pending_queue(
        self,
        platform: str,
        client: ScrapeCreatorsClient,
    ) -> None:
        active_platform = self.platform_filter if self.platform_filter else platform
        threshold = self.settings.creators_batch_size * 2

        balanced_queries = self._queries_manager.get_balanced_queries()
        max_attempts = len(balanced_queries)
        attempts = 0

        parser = get_platform_parser(
            platform=active_platform,
            session_maker=self.db.async_session,
            client=client,
            settings=self.settings,
        )

        while attempts < max_attempts and not self.is_shutdown_requested():
            pending_count = await self.db.count_pending_creator_accounts(active_platform)

            if pending_count >= threshold:
                logger.info(
                    f"Pending queue sufficiently populated ({pending_count} >= {threshold}). "
                    f"Stopping discovery loop."
                )
                break

            next_queries = self._get_next_queries(count=1)
            if not next_queries:
                logger.warning("No more queries available for discovery.")
                break

            query, category = next_queries[0]

            logger.info(
                f"Pending queue running low ({pending_count} < {threshold}). "
                f"Starting discovery attempt {attempts + 1}/{max_attempts}..."
            )

            prev_pending_count = pending_count

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

            new_pending_count = await self.db.count_pending_creator_accounts(active_platform)

            if new_pending_count <= prev_pending_count:
                logger.info(
                    f"Discovery loop broken early after {attempts + 1}/{max_attempts} "
                    f"attempts: no new pending accounts added "
                    f"(pending_count={new_pending_count}). "
                    f"Saving API credits."
                )
                break

            attempts += 1

    def is_shutdown_requested(self) -> bool:
        if self._shutdown_event is None:
            return False
        return self._shutdown_event.is_set()

    async def run_once(
        self,
        batch_size: int = 10,
        concurrency_limit: int = 3,
    ) -> None:
        logger.info(
            f"Starting ingestion cycle with batch_size={batch_size}, "
            f"concurrency_limit={concurrency_limit}"
        )

        async with ScrapeCreatorsClient(self.settings) as client:
            initial_credits: int | None = client.last_credits_remaining

            platform = self.platform_filter if self.platform_filter else "INSTAGRAM"

            await self._ensure_pending_queue(platform, client)

            final_pending_count = await self.db.count_pending_creator_accounts(platform)

            if final_pending_count == 0:
                logger.info("No pending accounts available. Skipping ingestion cycle.")
                return

            if final_pending_count < batch_size:
                logger.info(
                    f"Processing partial batch of {final_pending_count} pending accounts "
                    f"(batch_size={batch_size})."
                )

            accounts = await self.db.claim_creator_accounts(
                platforms=[self.platform_filter] if self.platform_filter else CREATOR_PLATFORMS,
                batch_size=batch_size,
                status_threshold_hours=self.status_threshold_hours,
            )

            if not accounts:
                logger.info("No accounts to process in this cycle")
                return

            logger.info(f"Found {len(accounts)} accounts to process")

            semaphore = asyncio.Semaphore(concurrency_limit)

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
                if account.username
            ]

            if not tasks:
                logger.info("No valid accounts with usernames to process")
                return

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Unexpected error in task {i}: {result!r}", exc_info=result
                    )

            if client.background_tasks:
                logger.info(
                    f"Waiting for {len(client.background_tasks)} background transcript tasks to complete..."
                )
                await asyncio.gather(*client.background_tasks, return_exceptions=True)
                client.background_tasks.clear()

            final_credits: int | None = client.last_credits_remaining

            if initial_credits is not None and final_credits is not None:
                credits_spent = initial_credits - final_credits
                logger.info(
                    f"=== BATCH CREDIT SUMMARY: Spent: {credits_spent} credits | Remaining: {final_credits} ==="
                )
            elif final_credits is not None:
                logger.info(
                    f"=== BATCH CREDIT SUMMARY: Remaining: {final_credits} ==="
                )
            else:
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
        if not username:
            logger.warning(f"Account {account_id} has no username, skipping")
            return

        async with semaphore:
            logger.info(
                f"Processing account {account_id} (platform={platform}, username={username})"
            )

            try:
                parser = get_platform_parser(
                    platform=platform,
                    session_maker=self.db.async_session,
                    client=client,
                    settings=self.settings,
                )

                logger.debug(
                    f"Parsing profile for {username} on {platform}"
                )
                db_account_id = await parser.parse_profile(username)

                if db_account_id is None:
                    logger.info(
                        f"Profile {username} on {platform} rejected "
                        f"(below subscriber threshold or parsing failed)"
                    )
                    await self.db.update_creator_account_status(
                        account_id, STATUS_REJECTED
                    )
                    return

                async with self.db.async_session() as session:
                    stmt = select(Account).where(Account.id == db_account_id)
                    result = await session.execute(stmt)
                    account = result.scalar_one_or_none()

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

                logger.debug(
                    f"Parsing content for {username} on {platform}"
                )
                await parser.parse_content(
                    account_id=db_account_id,
                    platform_id=username,
                    max_items=50,
                )

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

                if account_after_content and account_after_content.status == STATUS_PROCESSING:
                    await self.db.update_creator_account_status(db_account_id, STATUS_PARSED)
                    logger.info(
                        f"Successfully processed account {db_account_id} "
                        f"({username} on {platform})"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to process account {account_id} "
                    f"({username} on {platform}): {e!r}",
                    exc_info=e,
                )
                await self.db.update_creator_account_status(account_id, STATUS_FAILED)

    async def reset_orphaned_processing_accounts(self) -> None:
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
        logger.info("Starting candidate discovery phase...")

        queries_manager = SearchQueriesManager(self.settings.search_queries_path)
        balanced_queries = queries_manager.get_balanced_queries()

        if not balanced_queries:
            logger.warning("No balanced queries available. Exiting discovery phase.")
            return

        logger.info(f"Loaded {len(balanced_queries)} balanced queries for discovery.")

        if self.platform_filter:
            platforms_to_process = [self.platform_filter]
        else:
            platforms_to_process = CREATOR_PLATFORMS

        logger.info(f"Discovery will process platforms: {platforms_to_process}")

        for platform in platforms_to_process:
            logger.info(f"Starting discovery for platform: {platform}")

            async with ScrapeCreatorsClient(self.settings) as client:
                parser = get_platform_parser(
                    platform=platform,
                    session_maker=self.db.async_session,
                    client=client,
                    settings=self.settings,
                )

                for query, category in balanced_queries:
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
    args: argparse.Namespace = parse_args()

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

    db = Database(settings.db_url, echo=False)

    shutdown_event: asyncio.Event = asyncio.Event()

    coordinator: CreatorsCoordinator = CreatorsCoordinator(
        db=db,
        settings=settings,
        shutdown_event=shutdown_event,
        platform=platform_filter,
    )

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
            await db.close()
            sys.exit(1)

        logger.info("Closing database connections...")
        await db.close()
        logger.info("Discovery phase completed. Exiting.")
        sys.exit(0)

    def _signal_handler(sig: int, frame: Any | None = None) -> None:
        sig_name: str = signal.Signals(sig).name
        logger.info(
            f"Received signal {sig_name}, initiating graceful shutdown..."
        )
        logger.info(
            "Waiting for current batch to complete before shutting down..."
        )
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda s=sig: _signal_handler(s))
        except NotImplementedError:
            signal.signal(sig, _signal_handler)

    logger.info("Creators coordinator daemon started. Press Ctrl+C to stop.")

    try:
        while not shutdown_event.is_set():
            if shutdown_event.is_set():
                logger.info("Shutdown requested before batch start, exiting.")
                break

            logger.info("Starting new ingestion batch...")
            try:
                await coordinator.run_once(
                    batch_size=settings.creators_batch_size,
                    concurrency_limit=settings.creators_concurrency,
                )
            except Exception as exc:
                logger.error(
                    f"Unexpected error in ingestion batch: {exc!r}",
                    exc_info=exc,
                )

            if shutdown_event.is_set():
                logger.info(
                    "Shutdown requested after batch completion, exiting."
                )
                break

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
        logger.info("Shutting down gracefully...")

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
        raise
    except Exception as e:
        logger.error(f"Fatal error in coordinator daemon: {e!r}", exc_info=e)
        sys.exit(1)
