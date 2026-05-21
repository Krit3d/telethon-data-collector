"""Base class for Telegram workers with common lifecycle and error handling."""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

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
        delay_min: float | None = None,
        delay_max: float | None = None,
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
            delay_min: Minimum random delay in seconds before API calls.
                If None, uses settings.parser_delay_min (default: 1.0).
            delay_max: Maximum random delay in seconds before API calls.
                If None, uses settings.parser_delay_max (default: 3.0).
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

        # Fallback cascade: explicit argument -> settings config -> hardcoded safe defaults
        self.delay_min = max(
            0.1,  # Absolute structural minimum to prevent 0 delay
            (
                delay_min
                if delay_min is not None
                else getattr(settings, "parser_delay_min", 1.0)
            ),
        )
        self.delay_max = max(
            self.delay_min
            + 0.1,  # Max must always be strictly greater than min
            (
                delay_max
                if delay_max is not None
                else getattr(settings, "parser_delay_max", 3.0)
            ),
        )

        # Organic session heatup tracking
        self._start_time = time.time()
        self._heatup_duration_s = settings.organic_heatup_minutes * 60
        self._is_heatup_complete = False

        # State tracking for graceful degradation
        self.client: TelegramClient | None = None
        self.consecutive_shadowbans = 0
        self.safe_mode = False
        self.is_alive = (
            True  # Flag to indicate if worker should continue running
        )
        self.last_activity: float = (
            time.time()
        )  # Timestamp of last API call activity
        self._flood_wait_cooldown_until = (
            0  # Timestamp for standard FloodWait cooldown
        )
        self._severe_flood_wait_until = (
            0  # Timestamp for severe FloodWait (>3600s) 12h cooldown
        )
        self._entity_cache: dict[tuple[int, int | None], Any] = (
            {}
        )  # In-memory entity cache: (channel_id, access_hash) -> entity

        # Logger with worker_id context for structured logging
        self.logger = logging.LoggerAdapter(
            logging.getLogger(self.__class__.__name__),
            {"worker_id": self.worker_id},
        )
        # Track last proxy rotation time to avoid excessive rotations
        self._last_rotation_time = 0
        self._rotation_cooldown_s = 300  # 5 minutes between rotations minimum

    def _calculate_heatup_factor(self) -> float:
        """Calculate the current heatup factor for organic session warmup.

        During the first `organic_heatup_minutes` of operation, the worker
        gradually increases its activity. Factor starts at 0.1 (10% speed)
        and asymptotically approaches 1.0 over the heatup period.

        If `settings.enable_warmup` is False, returns 1.0 immediately to
        bypass the slow start period for faster parsing.

        Returns:
            Float between 0.1 and 1.0 representing current activity multiplier.
        """

        # Bypass warmup if disabled in settings
        if not self.settings.enable_warmup:
            return 1.0

        elapsed = time.time() - self._start_time
        if elapsed >= self._heatup_duration_s:
            self._is_heatup_complete = True
            return 1.0

        # Smooth exponential growth: factor = 0.1 + 0.9 * (elapsed / duration)^2
        # This gives a gentle start that accelerates over time
        ratio = elapsed / self._heatup_duration_s
        factor = 0.1 + 0.9 * (ratio**2)
        return max(0.1, min(1.0, factor))

    def natural_delay(self, base_delay: float | None = None) -> float:
        """Generate a human-like delay using Pareto/Gamma distribution.

        Instead of uniform random delays, this mimics natural human reading
        patterns with occasional longer pauses (Pareto distribution heavy tail).

        Args:
            base_delay: Base delay in seconds to scale. If None, uses settings
                or heatup-adjusted value.

        Returns:
            Delay in seconds (always >= 1.0 for safety).
        """

        # Apply heatup factor to base delay
        heatup_factor = self._calculate_heatup_factor()

        if base_delay is None:
            # Use worker-specific delays (parser or crawler) and scale by heatup factor
            base_delay = random.uniform(self.delay_min, self.delay_max)

        # Scale by heatup factor (longer delays during warmup)
        scaled_delay = (
            base_delay / heatup_factor if heatup_factor < 1.0 else base_delay
        )

        # Apply Pareto distribution for heavy-tail effect (human-like pauses)
        # Pareto shape parameter alpha=3 gives ~80% short, ~20% long delays
        if HAS_NUMPY:
            pareto_factor = np.random.pareto(3) + 1  # +1 to make mean ~1.5
        else:
            # random.paretovariate(alpha) returns Pareto type I (xm=1, alpha>=1), equivalent to np.random.pareto(alpha) + 1
            pareto_factor = random.paretovariate(3)

        # Combine: base scaled delay * Pareto factor
        natural_delay = scaled_delay * pareto_factor

        # Clamp to reasonable bounds (1-300 seconds)
        return max(1.0, min(300.0, natural_delay))

    async def _rotate_mobile_proxy(self) -> bool:
        """Rotate mobile proxy IP by calling the rotation URL.

        Uses the mobile_proxy_rotation_url from settings to request a new
        IP address from the proxy provider. Respects a cooldown period to
        avoid excessive rotations.

        Returns:
            True if rotation was successful, False otherwise.
        """

        rotation_url = self.settings.mobile_proxy_rotation_url
        if not rotation_url:
            return False

        current_time = time.time()
        if current_time - self._last_rotation_time < self._rotation_cooldown_s:
            self.logger.warning(
                "Proxy rotation skipped: cooldown period not elapsed (%.0fs remaining)",
                self._rotation_cooldown_s
                - (current_time - self._last_rotation_time),
            )
            return False

        try:
            from aiohttp import ClientSession, ClientTimeout

            timeout = ClientTimeout(total=10)
            async with ClientSession() as session:
                async with session.get(
                    rotation_url, timeout=timeout
                ) as response:
                    if response.status == 200:
                        self._last_rotation_time = current_time
                        self.logger.info(
                            "Worker %d: Mobile proxy rotated successfully via %s",
                            self.worker_id,
                            rotation_url,
                        )
                        return True
                    else:
                        self.logger.error(
                            "Worker %d: Proxy rotation failed with status %d",
                            self.worker_id,
                            response.status,
                        )
                        return False
        except Exception as e:
            self.logger.error(
                "Worker %d: Proxy rotation request failed: %s",
                self.worker_id,
                e,
            )
            return False

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
            "flood_sleep_threshold": 24,
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
        socket_timeout: float = 30.0,
    ) -> Any:
        """Execute a Telethon API call with comprehensive error handling.

        This method provides robust error handling for Telegram API calls, including:
        - Socket timeout wrapping (30s default) to prevent infinite network hangs
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
            socket_timeout: Timeout in seconds for the API call to prevent infinite hangs.
                Default: 30.0 seconds. Treated as transient error for retry logic.

        Returns:
            The result of the operation, or None if the operation failed gracefully
            (non-fatal RPC error, network error after retries exhausted, or other ValueError).

        Raises:
            FloodWaitError: Re-raised when delay > 5 seconds (caller handles cooldown),
                or when cumulative FloodWait exceeds max_flood_wait_s.
            RPCError: Re-raised if rpc_error_fatal=True or after retry limit.
            SessionExpiredError: Re-raised for session authentication errors.
            ShadowBanDetectedError: Raised when shadowban is confirmed (3 consecutive
                "No user has" errors). The worker will disconnect, sleep 3 hours,
                and reconnect before raising.
        """
        self.last_activity = (
            time.time()
        )  # Update activity at start of safe_api_call

        if network_retries is None:
            network_retries = self.settings.network_retries
        if base_delay_s is None:
            base_delay_s = self.settings.network_retry_base_delay_s

        attempt = 0
        flood_wait_total = 0

        while True:
            try:
                # Check if we're in a SEVERE FloodWait cooldown period (>3600s, 12h)
                current_time = time.time()
                if current_time < self._severe_flood_wait_until:
                    remaining = self._severe_flood_wait_until - current_time
                    self.logger.critical(
                        "%s: In SEVERE FloodWait cooldown (12h). %.0fs remaining. Worker is terminated for this session.",
                        operation_name,
                        remaining,
                    )
                    # Ensure client is disconnected during cooldown
                    if self.client and self.client.is_connected():
                        await self.disconnect()
                    # Mark worker as dead and exit immediately
                    self.is_alive = False
                    self.logger.critical(
                        "%s: Worker %d is DEAD due to severe FloodWait. Session protected. Cooldown expires in %.0f seconds.",
                        operation_name,
                        self.worker_id,
                        remaining,
                    )
                    return  # Exit the operation and worker loop

                # Check if we're in a standard FloodWait cooldown period
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
                    await asyncio.sleep(
                        min(remaining, 60)
                    )  # Wake up every 60s to recheck
                    continue  # Retry after cooldown expires

                # Apply natural or uniform delay before API call to avoid rate limiting
                if self.settings.use_natural_delays:
                    delay = self.natural_delay()
                    self.logger.debug(
                        "Natural delay: sleeping for %.1f seconds (heatup_factor=%.2f)",
                        delay,
                        self._calculate_heatup_factor(),
                    )
                else:
                    delay = random.uniform(self.delay_min, self.delay_max)
                    self.logger.debug(
                        "Sleeping for %.1f seconds before API call",
                        delay,
                    )
                await asyncio.sleep(delay)

                # Execute operation with socket timeout to prevent infinite hangs
                self.last_activity = (
                    time.time()
                )  # Update activity right before API call
                result = operation()

                if inspect.isawaitable(result):
                    # Wrap awaitable in wait_for to enforce socket timeout
                    try:
                        await_result = await asyncio.wait_for(
                            result, timeout=socket_timeout
                        )
                        self.last_activity = (
                            time.time()
                        )  # Update activity after successful await
                        return await_result
                    except (asyncio.TimeoutError, TimeoutError):
                        # Treat socket timeout as transient network error
                        self.logger.warning(
                            "%s: Socket timeout after %.1fs, treating as transient error",
                            operation_name,
                            socket_timeout,
                        )
                        raise  # Re-raise to be caught by network error handler
                else:
                    # Sync operation - log that socket timeout is not enforced
                    if socket_timeout > 0:
                        self.logger.debug(
                            "%s: Sync operation, socket timeout not enforced",
                            operation_name,
                        )
                    self.last_activity = (
                        time.time()
                    )  # Update activity after successful sync call
                    return result

            except FloodWaitError as e:
                delay = int(getattr(e, "seconds", 0)) or 1
                flood_wait_total += delay

                self.logger.warning(
                    "%s: FloodWaitError detected, delay=%ds (total: %ds)",
                    operation_name,
                    delay,
                    flood_wait_total,
                )

                # Trigger mobile proxy rotation if configured (on any FloodWait)
                if self.settings.mobile_proxy_rotation_url:
                    self.logger.info(
                        "%s: FloodWait detected - triggering mobile proxy rotation",
                        operation_name,
                    )
                    rotation_success = await self._rotate_mobile_proxy()
                    if rotation_success:
                        self.logger.info(
                            "%s: Mobile proxy rotated successfully",
                            operation_name,
                        )
                    else:
                        self.logger.warning(
                            "%s: Mobile proxy rotation failed or skipped",
                            operation_name,
                        )

                # SEVERE FLOODWAIT COOLDOWN: if delay > 3600 seconds, disconnect and wait 12h
                if delay > 3600:
                    self.logger.critical(
                        "%s: FloodWait > 3600s (%ds). CRITICAL: Entering 12h cooldown. Worker will be terminated to save session.",
                        operation_name,
                        delay,
                    )
                    # Set severe cooldown timestamp (current time + 12 hours = 43200 seconds)
                    self._severe_flood_wait_until = time.time() + 43200
                    # Disconnect immediately to free proxy and avoid holding TCP
                    await self.disconnect()
                    # Mark worker as dead for this session - it will not restart after cooldown
                    self.is_alive = False
                    self.logger.critical(
                        "%s: Worker %d is now DEAD. Session saved from permanent ban. Cooldown: 12h (%ds).",
                        operation_name,
                        self.worker_id,
                        43200,
                    )
                    return  # Exit the operation and worker loop

                # For delays > 5 seconds, raise immediately to caller for pool-friendly handling
                # This prevents blocking the worker indefinitely inside safe_api_call
                if delay > 5:
                    self.logger.warning(
                        "%s: FloodWait delay %ds > 5s, setting cooldown and raising to caller",
                        operation_name,
                        delay,
                    )
                    # Set cooldown timestamp for when this session should be available again
                    self._flood_wait_cooldown_until = time.time() + delay + 10
                    # Keep session alive - it will be returned to pool with cooldown
                    self.is_alive = True
                    # Re-raise FloodWaitError so caller can handle pool return
                    raise

                if flood_wait_total >= max_flood_wait_s:
                    self.logger.error(
                        "%s: Cumulative FloodWait %ds exceeds limit %ds. "
                        "Aborting operation.",
                        operation_name,
                        flood_wait_total,
                        max_flood_wait_s,
                    )
                    raise

                # For delays <= 5 seconds, sleep here and retry
                # Add 10 second buffer to FloodWait delay for extra safety
                self.logger.info(
                    "%s: Sleeping %.1fs for FloodWait (delay <= 5s, will retry)",
                    operation_name,
                    delay + 10,
                )
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
                    "method that is not available for frozen accounts"
                    in error_text
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
