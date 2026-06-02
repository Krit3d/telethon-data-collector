"""Thin, generic HTTP transport client for the Scrape Creators API.

This module provides *only* network I/O: session management, authentication,
retries, and raw JSON passthrough.  All platform-specific parsing logic lives
in higher-level services or parsers.
"""

import asyncio
import logging
import json
from typing import Any

import aiohttp

from src.config.config import Settings

logger = logging.getLogger(__name__)
# Dedicated logger for credit statistics - can be configured independently
# To enable credit statistics, set level to INFO for this logger:
# logging.getLogger("src.parser.creators.sc_client.credits").setLevel(logging.INFO)
credits_logger = logging.getLogger(__name__ + ".credits")


class ScrapeCreatorsClient:
    """Thin asynchronous HTTP client for the Scrape Creators API.

    Focuses exclusively on transport concerns:
    * Lazy `aiohttp.ClientSession` initialisation protected by an `asyncio.Lock`
    * Configurable base URL, header name, and authentication scheme (via `Settings`)
    * Resilient `_request` helper with configurable retries, exponential backoff,
      and strict HTTP 429 (Rate-Limit) handling
    * Generic `get` / `post` convenience methods that return raw parsed JSON

    Business logic, data schemas, and platform-specific parsing must **not** be
    added to this class.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise the client from application settings.

        Args:
            settings: Application settings containing the API key, base URL,
                header name, authentication scheme, and network retry configuration.

        Raises:
            ValueError: If the Scrape Creators API key is not configured.
        """
        self._settings = settings
        api_key = settings.scrape_creators_api_key
        if not api_key:
            raise ValueError(
                "Scrape Creators API key is not configured. "
                "Set SCRAPE_CREATORS_API_KEY in your environment or .env file."
            )
        self._api_key: str = api_key
        self._base_url = settings.scrape_creators_base_url.rstrip("/")
        self._header_name = settings.scrape_creators_header_name
        self._auth_scheme = settings.scrape_creators_auth_scheme
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()
        # Track remaining credits for statistics/monitoring
        self.last_credits_remaining: int | None = None

    async def __aenter__(self) -> "ScrapeCreatorsClient":
        """Async context manager entry - ensure the session is initialised."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - close the underlying session."""
        await self.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Lazily create the `aiohttp.ClientSession` (thread-safe via lock).

        Returns:
            The active `aiohttp.ClientSession`.
        """
        if self._session is None or self._session.closed:
            async with self._session_lock:
                # Double-checked locking pattern
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=30),
                        headers={
                            "User-Agent": "ScrapeCreatorsClient/1.0 (Production)",
                        },
                    )
        return self._session

    async def close(self) -> None:
        """Close the underlying `aiohttp.ClientSession`.

        Safe to call multiple times; only closes when the session is open.
        """
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _build_auth_header(self) -> str:
        """Build the authentication header value.

        Returns:
            Formatted header value, e.g. ``"Bearer <key>"`` or just ``"<key>"``
            when no scheme is configured.
        """
        if self._auth_scheme:
            return f"{self._auth_scheme} {self._api_key}"
        return self._api_key

    def _get_network_retries(self) -> int:
        """Get the network retries configuration from settings with fallback.

        Returns:
            The number of network retries to use. Defaults to 3 if not
            configured or set to None.
        """
        retries = getattr(self._settings, 'network_retries', None)
        if retries is None:
            return 3
        return retries

    def _get_network_retry_base_delay(self) -> float:
        """Get the network retry base delay configuration from settings with fallback.

        Returns:
            The base delay in seconds for retry backoff. Defaults to 1.0 if not
            configured or set to None.
        """
        delay = getattr(self._settings, 'network_retry_base_delay_s', None)
        if delay is None:
            return 1.0
        return delay

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        json_data: dict | None = None,
        max_retries: int | None = None,
    ) -> dict:
        """Perform an HTTP request with retries, fixed-interval delays, and rate-limit handling.

        Retries up to the specified number of times on:
        * Network errors (`aiohttp.ClientError`, `asyncio.TimeoutError`)
        * HTTP 429 Too Many Requests (with Retry-After header support)
        * Server errors (HTTP 5xx)

        Client errors (HTTP 4xx, except 429) are caught by the specific
        `aiohttp.ClientResponseError` handler and raised immediately without
        retrying, as they indicate invalid requests that will not succeed
        on subsequent attempts.

        For 429 responses the client honours the ``Retry-After`` header when
        present, falling back to a fixed delay.

        The retry configuration uses a fixed delay between retries:
        * ``network_retry_base_delay_s``: Fixed delay in seconds (default: 1.0)

        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: API endpoint path (leading ``/`` is optional).
            params: Optional query-string parameters.
            json_data: Optional JSON body for POST/PUT requests.
            max_retries: Maximum number of retries. If not provided, defaults to 5
                (which means 6 total attempts).

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            aiohttp.ClientResponseError: When a client error (4xx) occurs.
                This is raised immediately without retry.
            aiohttp.ClientError: When a server error (5xx) or network error
                fails after all retries.
            asyncio.TimeoutError: When the request times out after all retries.
            ValueError: When the response body cannot be parsed as JSON.
        """
        session = await self._ensure_session()

        # Determine retry configuration: use explicit parameter if provided,
        # otherwise default to 5 retries (6 total attempts)
        if max_retries is not None:
            actual_max_retries = max_retries
        else:
            actual_max_retries = 5

        fixed_delay = self._get_network_retry_base_delay()

        # Normalise endpoint so it always starts with a slash
        normalised_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        full_url = f"{self._base_url}{normalised_endpoint}"

        headers = {self._header_name: self._build_auth_header()}
        request_kwargs: dict[str, Any] = {"headers": headers}
        if params is not None:
            request_kwargs["params"] = params
        if json_data is not None:
            request_kwargs["json"] = json_data

        for attempt in range(actual_max_retries + 1):
            try:
                async with session.request(method, full_url, **request_kwargs) as response:
                    if response.status == 429:
                        # --- Rate-limit handling ---
                        if attempt == actual_max_retries:
                            # All retries exhausted for 429 response
                            logger.error(
                                "Max retries (%d) exceeded for 429 response on %s %s",
                                actual_max_retries,
                                method,
                                full_url,
                            )
                            response.raise_for_status()

                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = float(retry_after)
                            except ValueError:
                                logger.warning(
                                    "Invalid Retry-After header: '%s', using fixed delay",
                                    retry_after,
                                )
                                wait_time = fixed_delay
                        else:
                            wait_time = fixed_delay

                        logger.warning(
                            "Rate limited (429) on attempt %d/%d for %s %s. "
                            "Waiting %.2f seconds before retrying.",
                            attempt + 1,
                            actual_max_retries + 1,
                            method,
                            full_url,
                            wait_time,
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    # --- Client error handling (4xx except 429) ---
                    # Client errors (4xx) indicate invalid requests that won't succeed on retry
                    if 400 <= response.status < 500:
                        # Read and log the response body before raising the error
                        error_detail = None
                        # Try to parse as JSON first
                        try:
                            error_data = await response.json()
                            # Extract meaningful error message from common keys
                            if isinstance(error_data, dict):
                                for key in ("message", "error", "messageStatus", "errorStatus", "details"):
                                    if key in error_data:
                                        error_detail = str(error_data[key])
                                        break
                                if error_detail is None:
                                    error_detail = str(error_data)
                            else:
                                error_detail = str(error_data)
                        except (aiohttp.ContentTypeError, json.JSONDecodeError):
                            # JSON parsing failed, try raw text
                            try:
                                error_detail = await response.text()
                            except Exception:
                                error_detail = "(unable to read response body)"
                        # Log the detailed error
                        logger.error(
                            "API Error %d detail: %s. Request: %s %s. Raising immediately without retry.",
                            response.status,
                            error_detail,
                            method,
                            full_url,
                        )
                        # Raise ClientResponseError - will be caught by the specific
                        # except block below and raised immediately without retry
                        response.raise_for_status()

                    # --- Server error handling (5xx) ---
                    # Server errors (5xx) may be transient, so we raise ClientError
                    # (not ClientResponseError) to allow the general except block
                    # to catch it and retry with backoff
                    if 500 <= response.status < 600:
                        raise aiohttp.ClientError(f"Server error: {response.status}")

                    # Parse JSON body
                    try:
                        response_dict = await response.json()
                    except aiohttp.ContentTypeError as e:
                        logger.error(
                            "Failed to parse JSON response from %s %s: %s",
                            method,
                            full_url,
                            e,
                        )
                        raise ValueError(
                            f"Invalid JSON response from {method} {full_url}"
                        ) from e

                    # Extract remaining credits from the root of the JSON response
                    credits_remaining: int | None = None
                    if isinstance(response_dict, dict):
                        credits_remaining = response_dict.get("credits_remaining")

                    # Log successful request with remaining credits (concise format)
                    if credits_remaining is not None:
                        self.last_credits_remaining = int(credits_remaining)
                        # Log credits to dedicated statistics logger (configured independently)
                        credits_logger.debug("Credits remaining: %d", self.last_credits_remaining)
                    else:
                        # Concise debug log when no credits info available
                        logger.debug("API request success: %s %s", method, normalised_endpoint)

                    return response_dict

            except aiohttp.ClientResponseError as cre:
                # ClientResponseError (4xx, 5xx) - handle 4xx immediately without retry
                # 5xx errors should have been raised as ClientError (not ClientResponseError)
                # in the try block, so this block mainly handles 4xx errors.
                if cre.status == 429:
                    # 429 should have been handled in the try block with retries
                    # If we get here, max retries for 429 were exceeded
                    raise
                else:
                    # For other 4xx errors (400, 401, 403, 404, etc.), raise immediately
                    # without retrying, as these indicate invalid requests
                    raise cre

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                # Only retry for network errors, timeouts, and server errors (5xx)
                # Client errors (4xx) should have been caught above and raised immediately
                logger.warning(
                    "Request failed on attempt %d/%d for %s %s: %s",
                    attempt + 1,
                    actual_max_retries + 1,
                    method,
                    full_url,
                    str(e),
                )
                if attempt < actual_max_retries:
                    wait_time = fixed_delay
                    logger.warning("Retrying after %.2f seconds...", wait_time)
                    await asyncio.sleep(wait_time)
                else:
                    # All retries exhausted - log error and re-raise the exception
                    logger.error(
                        "Max retries (%d) exceeded for %s %s",
                        actual_max_retries,
                        method,
                        full_url,
                    )
                    raise

        # Should never be reached, but keeps mypy happy
        raise RuntimeError("Max retries exceeded without success")

    async def get(
        self,
        endpoint: str,
        params: dict | None = None,
        max_retries: int | None = None,
    ) -> dict:
        """Send a GET request and return the raw JSON response.

        Args:
            endpoint: API endpoint path (e.g. ``"/v1/tiktok/profile"``).
            params: Optional query-string parameters.
            max_retries: Maximum number of retries (default: None, uses client default).

        Returns:
            Parsed JSON response dictionary.
        """
        logger.debug("GET %s params=%s", endpoint, params)
        return await self._request("GET", endpoint, params=params, max_retries=max_retries)

    async def post(
        self,
        endpoint: str,
        json_data: dict | None = None,
        max_retries: int | None = None,
    ) -> dict:
        """Send a POST request and return the raw JSON response.

        Args:
            endpoint: API endpoint path (e.g. ``"/v1/instagram/posts"``).
            json_data: Optional JSON body.
            max_retries: Maximum number of retries (default: None, uses client default).

        Returns:
            Parsed JSON response dictionary.
        """
        logger.debug("POST %s json=%s", endpoint, json_data)
        return await self._request("POST", endpoint, json_data=json_data, max_retries=max_retries)
