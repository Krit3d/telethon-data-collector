from __future__ import annotations

import asyncio
import logging
import json
from typing import Any

import aiohttp

from src.config.config import Settings


logger = logging.getLogger(__name__)
credits_logger = logging.getLogger(__name__ + ".credits")


class ScrapeCreatorsClient:

    def __init__(self, settings: Settings) -> None:
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
        self.global_semaphore = asyncio.Semaphore(settings.creators_concurrency)
        self.last_credits_remaining: int | None = None
        self.background_tasks: set[asyncio.Task[Any]] = set()

    async def __aenter__(self) -> ScrapeCreatorsClient:
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=30),
                        headers={
                            "User-Agent": "ScrapeCreatorsClient/1.0 (Production)",
                        },
                    )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _build_auth_header(self) -> str:
        if self._auth_scheme:
            return f"{self._auth_scheme} {self._api_key}"
        return self._api_key

    def _get_network_retries(self) -> int:
        retries = getattr(self._settings, 'network_retries', None)
        if retries is None:
            return 3
        return retries

    def _get_network_retry_base_delay(self) -> float:
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
        session = await self._ensure_session()

        if max_retries is not None:
            actual_max_retries = max_retries
        else:
            actual_max_retries = self._get_network_retries()

        fixed_delay = self._get_network_retry_base_delay()

        normalised_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        full_url = f"{self._base_url}{normalised_endpoint}"

        headers = {self._header_name: self._build_auth_header()}
        request_kwargs: dict[str, Any] = {"headers": headers}
        if params is not None:
            request_kwargs["params"] = params
        if json_data is not None:
            request_kwargs["json"] = json_data

        if "transcript" in endpoint:
            request_kwargs["timeout"] = aiohttp.ClientTimeout(total=35.0)

        for attempt in range(actual_max_retries + 1):
            is_server_fault = False
            try:
                async with session.request(method, full_url, **request_kwargs) as response:
                    if response.status == 429:
                        if attempt == actual_max_retries:
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

                    if response.status == 404 and "search" in endpoint:
                        logger.info(
                            "Search returned 404 (no results). Returning empty dataset gracefully."
                        )
                        return {"profiles": [], "data": [], "items": []}

                    if 400 <= response.status < 500:
                        error_detail = None
                        try:
                            error_data = await response.json()
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
                            try:
                                error_detail = await response.text()
                            except Exception:
                                error_detail = "(unable to read response body)"
                        logger.error(
                            "API Error %d detail: %s. Request: %s %s. Raising immediately without retry.",
                            response.status,
                            error_detail,
                            method,
                            full_url,
                        )
                        response.raise_for_status()

                    if 500 <= response.status < 600:
                        is_server_fault = True
                        raise aiohttp.ClientError(f"Server error: {response.status}")

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

                    credits_remaining: int | None = None
                    if isinstance(response_dict, dict):
                        credits_remaining = response_dict.get("credits_remaining")

                    if credits_remaining is not None:
                        self.last_credits_remaining = int(credits_remaining)
                        credits_logger.debug("Credits remaining: %d", self.last_credits_remaining)
                    else:
                        logger.debug("API request success: %s %s", method, normalised_endpoint)

                    return response_dict

            except aiohttp.ClientResponseError as cre:
                if cre.status == 429:
                    raise
                else:
                    raise cre

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(
                    "Request failed on attempt %d/%d for %s %s: %s",
                    attempt + 1,
                    actual_max_retries + 1,
                    method,
                    full_url,
                    str(e),
                )
                if attempt < actual_max_retries:
                    if "transcript" in endpoint:
                        wait_time = 4.0 * (attempt + 1)
                    else:
                        wait_time = fixed_delay
                    logger.warning("Retrying after %.2f seconds...", wait_time)
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        "Max retries (%d) exceeded for %s %s",
                        actual_max_retries,
                        method,
                        full_url,
                    )
                    raise

        raise RuntimeError("Max retries exceeded without success")

    async def get(
        self,
        endpoint: str,
        params: dict | None = None,
        max_retries: int | None = None,
    ) -> dict:
        logger.debug("GET %s params=%s", endpoint, params)
        return await self._request("GET", endpoint, params=params, max_retries=max_retries)

    async def post(
        self,
        endpoint: str,
        json_data: dict | None = None,
        max_retries: int | None = None,
    ) -> dict:
        logger.debug("POST %s json=%s", endpoint, json_data)
        return await self._request("POST", endpoint, json_data=json_data, max_retries=max_retries)
