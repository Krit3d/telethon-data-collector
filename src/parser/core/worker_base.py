"""Base class for Telegram workers with common lifecycle and error handling."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable

from telethon import TelegramClient
from telethon.errors import (
    AuthKeyError,
    RPCError,
    UserDeactivatedError,
)
from telethon.errors.rpcerrorlist import FloodWaitError
from telethon.network.connection.tcpintermediate import (
    ConnectionTcpIntermediate,
)
from telethon.network.connection.tcpmtproxy import (
    ConnectionTcpMTProxyRandomizedIntermediate,
)

from src.config.config import Settings
from src.db.database import Database
from src.utils.proxy import build_telethon_proxy
from .exceptions import SessionExpiredError, ShadowBanDetectedError

logger = logging.getLogger(__name__)


class BaseTelegramWorker:
    """Base class for workers that interact with Telegram via Telethon.

    Provides common functionality:
    - Session and client initialization with proxy support
    - Connection lifecycle management
    - Safe API call wrapper with FloodWait handling, retry logic, and shadowban tracking
    - Graceful degradation for transient errors
    """

    def __init__(
        self,
        worker_id: int,
        session_path: Path,
        db: Database,
        settings: Settings,
        api_id: int,
        api_hash: str,
        proxy_url: str | None = None,
        device_model: str = "PC 64bit",
        system_version: str = "Windows 10",
        app_version: str = "4.16.8",
        lang_code: str = "en",
        system_lang_code: str = "en-US",
        delay_min: float = 5.0,
        delay_max: float = 15.0,
    ):
        """Initialize the base worker.

        Args:
            worker_id: Unique identifier for this worker instance.
            session_path: Path to the .session file.
            db: Database instance for data persistence.
            settings: Global settings configuration.
            api_id: Telegram API ID.
            api_hash: Telegram API hash.
            proxy_url: Optional proxy URL (socks5, http, or mtproxy).
            device_model: Device model string for Telethon client.
            system_version: System version string.
            app_version: App version string.
            lang_code: Language code (e.g., 'en').
            system_lang_code: System language code (e.g., 'en-US').
            delay_min: Minimum random delay in seconds before API calls (default: 5.0).
            delay_max: Maximum random delay in seconds before API calls (default: 15.0).
        """

        self.worker_id = worker_id
        self.session_path = session_path
        self.db = db
        self.settings = settings
        self.api_id = api_id
        self.api_hash = api_hash
        self.proxy_url = proxy_url
        self.device_model = device_model
        self.system_version = system_version
        self.app_version = app_version
        self.lang_code = lang_code
        self.system_lang_code = system_lang_code
        self.delay_min = delay_min
        self.delay_max = delay_max

        # State tracking for graceful degradation
        self.client: TelegramClient | None = None
        self.consecutive_shadowbans = 0
        self.safe_mode = False
        self.is_alive = True  # Flag to indicate if worker should continue running
        self._flood_wait_cooldown_until = 0  # Timestamp for FloodWait cooldown
        
        # Logger with worker_id context for structured logging
        self.logger = logging.LoggerAdapter(
            logging.getLogger(self.__class__.__name__),
            {"worker_id": self.worker_id}
        )

    def _build_proxy_config(self) -> dict[str, Any] | None:
        """Build Telethon proxy configuration from proxy_url.

        Returns:
            Proxy configuration dict or None if no proxy is configured.
        """

        if not self.proxy_url:
            return None

        try:
            proxy_config = build_telethon_proxy(self.proxy_url)
            return proxy_config
        except ValueError as e:
            self.logger.error(
                "Invalid proxy URL %s: %s",
                self.proxy_url,
                e,
            )
            return None

    def _create_client(self) -> TelegramClient:
        """Create and configure a Telethon client.

        Returns:
            Configured TelegramClient instance.

        Raises:
            ValueError: If proxy configuration is invalid.
        """

        proxy_config = self._build_proxy_config()

        client_kwargs: dict[str, Any] = {
            "device_model": self.device_model,
            "system_version": self.system_version,
            "app_version": self.app_version,
            "lang_code": self.lang_code,
            "system_lang_code": self.system_lang_code,
            "use_ipv6": False,
            "timeout": 60,
            "connection": ConnectionTcpIntermediate,
            "connection_retries": self.settings.network_retries,
            "retry_delay": self.settings.network_retry_base_delay_s,
            "flood_sleep_threshold": 0,
        }

        if proxy_config:
            if proxy_config.pop("is_mtproxy", False):
                client_kwargs["connection"] = (
                    ConnectionTcpMTProxyRandomizedIntermediate
                )
                client_kwargs["proxy"] = (
                    proxy_config["addr"],
                    proxy_config["port"],
                    proxy_config["secret"],
                )
            else:
                client_kwargs["proxy"] = proxy_config

        session_abs_path = self.session_path.with_suffix("").absolute()

        self.client = TelegramClient(
            str(session_abs_path),
            self.api_id,
            self.api_hash,
            **client_kwargs,
        )

        return self.client

    async def connect(self) -> None:
        """Establish connection to Telegram and verify authorization.

        Raises:
            SessionExpiredError: If the session is not authorized.
        """

        if self.client is None:
            self._create_client()

        if self.client is None:
            raise RuntimeError("Failed to create Telegram client")

        # Log connection details
        if self.proxy_url:
            proxy_config = self._build_proxy_config()
            if proxy_config and proxy_config.get("is_mtproxy"):
                self.logger.info(
                    "Connecting to MTProxy %s:%d",
                    proxy_config["addr"],
                    proxy_config["port"],
                )
            else:
                self.logger.info(
                    "Connecting to Telegram via proxy",
                )
        else:
            self.logger.info(
                "Connecting to Telegram directly (no proxy)",
            )

        await self.client.connect()

        if not await self.client.is_user_authorized():
            self.logger.critical(
                "SESSION UNAUTHORIZED. Path: %s. "
                "Interactive login is impossible in Docker. Check your session files!",
                self.session_path,
            )
            raise SessionExpiredError(
                f"Session {self.session_path} is not authorized"
            )

        self.logger.info("Connected and authorized")

    async def disconnect(self) -> None:
        """Disconnect the Telegram client if connected."""
        client = self.client
        if client and client.is_connected():
            await client.disconnect()  # type: ignore
            self.logger.info("Disconnected")

    async def safe_api_call(
        self,
        operation_name: str,
        operation: Callable[[], Any],
        *,
        network_retries: int | None = None,
        base_delay_s: float | None = None,
        rpc_error_fatal: bool = False,
        max_flood_wait_s: int = 3600,
    ) -> Any:
        """Execute a Telethon API call with comprehensive error handling.

        This method provides robust error handling for Telegram API calls, including:
        - Automatic retry logic for FloodWaitError with configurable maximum wait time
        - Exponential backoff for network errors (OSError, TimeoutError, ConnectionError)
        - Shadowban detection and mitigation with connection lifecycle management
        - Safe mode activation after repeated shadowban indicators
        - Graceful degradation for non-fatal errors
        - Cool-down period for severe FloodWait (>1000s): disconnect and wait without retrying

        Args:
            operation_name: Human-readable identifier for the operation (used in logging).
            operation: Callable (sync or async) that performs the API call.
            network_retries: Maximum retry attempts for transient network errors.
                If None, uses settings.network_retries.
            base_delay_s: Base delay in seconds for exponential backoff calculation.
                If None, uses settings.network_retry_base_delay_s.
            rpc_error_fatal: If True, RPC errors are re-raised; if False, they're logged
                and None is returned for graceful degradation.
            max_flood_wait_s: Maximum cumulative FloodWait sleep time before giving up.
                Prevents indefinite blocking from extreme rate limits. Default: 3600 seconds.

        Returns:
            The result of the operation, or None if the operation failed gracefully
            (non-fatal RPC error, network error after retries exhausted, or other ValueError).

        Raises:
            FloodWaitError: Re-raised when cumulative FloodWait exceeds max_flood_wait_s
                or after sleeping and retrying within limits.
            RPCError: Re-raised if rpc_error_fatal=True or after retry limit.
            SessionExpiredError: Re-raised for session authentication errors.
            ShadowBanDetectedError: Raised when shadowban is confirmed (3 consecutive
                "No user has" errors). The worker will disconnect, sleep 3 hours,
                and reconnect before raising.
        """

        if network_retries is None:
            network_retries = self.settings.network_retries
        if base_delay_s is None:
            base_delay_s = self.settings.network_retry_base_delay_s

        attempt = 0
        flood_wait_total = 0

        while True:
            try:
                # Check if we're in a FloodWait cooldown period
                current_time = time.time()
                if current_time < self._flood_wait_cooldown_until:
                    remaining = self._flood_wait_cooldown_until - current_time
                    self.logger.warning(
                        "%s: In FloodWait cooldown. %.0fs remaining. Disconnected during cooldown.",
                        operation_name,
                        remaining,
                    )
                    # Ensure client is disconnected during cooldown to avoid holding TCP
                    if self.client and self.client.is_connected():
                        await self.disconnect()
                    await asyncio.sleep(min(remaining, 60))  # Wake up every 60s to recheck
                    continue  # Retry after cooldown expires

                # Apply random delay before API call to avoid rate limiting
                delay = random.uniform(self.delay_min, self.delay_max)
                self.logger.info(
                    "Sleeping for %.1f seconds before API call",
                    delay,
                )
                await asyncio.sleep(delay)

                result = operation()

                if asyncio.iscoroutine(result):
                    return await result

                return result

            except FloodWaitError as e:
                delay = int(getattr(e, "seconds", 0)) or 1
                flood_wait_total += delay

                self.logger.warning(
                    "%s: FloodWaitError, sleeping %ds (total: %ds)",
                    operation_name,
                    delay,
                    flood_wait_total,
                )

                # SEVERE FLOODWAIT COOLDOWN: if delay > 1000 seconds, disconnect and wait
                if delay > 1000:
                    self.logger.critical(
                        "%s: FloodWait > 1000s (%ds). Entering cooldown: disconnect and wait without retrying.",
                        operation_name,
                        delay,
                    )
                    # Set cooldown timestamp (current time + delay + 10s safety)
                    self._flood_wait_cooldown_until = time.time() + delay + 10
                    # Disconnect immediately to free proxy and avoid holding TCP
                    await self.disconnect()
                    # Wait for cooldown to expire
                    await asyncio.sleep(delay + 10)
                    # After cooldown, reconnect and retry
                    await self.connect()
                    self.logger.info(
                        "%s: Cooldown expired, reconnected and resuming operations",
                        operation_name,
                    )
                    # Reset flood_wait_total to allow fresh attempts after cooldown
                    flood_wait_total = 0
                    continue

                if flood_wait_total >= max_flood_wait_s:
                    self.logger.error(
                        "%s: Cumulative FloodWait %ds exceeds limit %ds. "
                        "Aborting operation.",
                        operation_name,
                        flood_wait_total,
                        max_flood_wait_s,
                    )
                    raise

                # Add 10 second buffer to FloodWait delay for extra safety
                await asyncio.sleep(delay + 10)
                # After sleeping, retry the operation (loop continues)
                continue

            except (OSError, asyncio.TimeoutError, ConnectionError) as e:
                if attempt >= network_retries:
                    self.logger.exception(
                        "%s: network error, retries exhausted",
                        operation_name,
                    )
                    raise

                delay = base_delay_s * (2**attempt)
                attempt += 1

                self.logger.warning(
                    "%s: network error (%s), retry %d/%d in %.1fs",
                    operation_name,
                    type(e).__name__,
                    attempt,
                    network_retries,
                    delay,
                )

                await asyncio.sleep(delay)

            except RPCError as e:
                # Check for fatal account errors
                error_text = str(e)
                if (
                    "method that is not available for frozen accounts" in error_text
                    or isinstance(e, (AuthKeyError, UserDeactivatedError))
                ):
                    self.logger.critical(
                        "Account is DEAD/FROZEN. "
                        "Terminating worker to protect proxy. Error: %s",
                        e,
                    )
                    self.is_alive = False
                    # Disconnect immediately to free proxy
                    await self.disconnect()
                    # Raise SessionExpiredError to signal termination
                    raise SessionExpiredError(
                        f"Account fatal error: {type(e).__name__}"
                    )

                if rpc_error_fatal:
                    self.logger.error(
                        "[%s] fatal RPC error: %s",
                        operation_name,
                        e,
                    )
                else:
                    self.logger.warning(
                        "[%s] RPC error: %s",
                        operation_name,
                        e,
                    )
                # Always raise RPCError so crawler fallback logic can handle it
                raise

            except ValueError as e:
                # Special case: Shadowban detection for "No user has" errors
                if "No user has" in str(e):
                    self.consecutive_shadowbans += 1
                    self.logger.warning(
                        "Shadowban suspected (consecutive: %d). Error: %s",
                        self.consecutive_shadowbans,
                        e,
                    )

                    if not self.safe_mode:
                        self.safe_mode = True
                        self.logger.warning(
                            "Switched to SAFE MODE due to shadowban",
                        )

                    if self.consecutive_shadowbans >= 3:
                        self.logger.error(
                            "3 consecutive shadowbans. "
                            "Disconnecting, sleeping 3 hours, then reconnecting.",
                        )
                        # Critical: Disconnect to avoid holding TCP connection during long sleep
                        await self.disconnect()
                        await asyncio.sleep(10800)  # 3 hours
                        await self.connect()
                        # Reset counter after recovery
                        self.consecutive_shadowbans = 0

                        # Raise error AFTER state update and recovery steps
                        raise ShadowBanDetectedError(
                            f"Shadowban detected and handled (count=3, "
                            f"disconnected and slept for 3 hours)"
                        )

                    # For 1-2 shadowbans, sleep briefly before retry
                    await asyncio.sleep(60)

                    # Raise error to signal shadowban (but not yet critical)
                    raise ShadowBanDetectedError(
                        f"Shadowban detected (count={self.consecutive_shadowbans})"
                    )

                self.logger.warning(
                    "%s: ValueError: %s",
                    operation_name,
                    e,
                )
                raise

            except Exception as e:
                self.logger.exception(
                    "%s: unexpected error: %s",
                    operation_name,
                    e,
                )
                raise

    async def run(self) -> None:
        """Main worker loop to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement run()")
