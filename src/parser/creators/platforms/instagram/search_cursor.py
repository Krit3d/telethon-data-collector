import json
import logging
import os
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

CURSORS_FILE = "src/config/search_cursors.json"


class InstagramSearchPaginator:

    def __init__(self, query: str, max_depth: int = 10, expiration_hours: int = 24) -> None:
        self.query: str = query
        self.max_depth: int = max_depth
        self.depth: int = 0
        self.is_finished: bool = False
        self._cursor: str = self._load_cursor(expiration_hours)

    def _load_cursor(self, expiration_hours: int = 24) -> str:
        if not os.path.exists(CURSORS_FILE):
            return "1"
        try:
            with open(CURSORS_FILE, "r", encoding="utf-8") as f:
                data: dict[str, dict[str, str]] = json.load(f)
        except (json.JSONDecodeError, OSError):
            return "1"
        entry = data.get(self.query)
        if not entry or "updated_at" not in entry or "cursor" not in entry:
            return "1"
        try:
            updated_at = datetime.fromisoformat(entry["updated_at"])
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - updated_at > timedelta(hours=expiration_hours):
                return "1"
            return str(entry["cursor"])
        except (ValueError, TypeError):
            return "1"

    def _save_cursor(self, cursor: str) -> None:
        try:
            os.makedirs(os.path.dirname(CURSORS_FILE), exist_ok=True)
            data: dict[str, dict[str, str]] = {}
            if os.path.exists(CURSORS_FILE):
                try:
                    with open(CURSORS_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    data = {}
            data[self.query] = {
                "cursor": cursor,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(CURSORS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.error(
                "Failed to save cursor for query '%s': %s",
                self.query,
                exc,
                exc_info=True,
            )

    def should_continue(self) -> bool:
        return not self.is_finished and self.depth < self.max_depth

    def get_params(self) -> dict[str, Any]:
        self.depth += 1
        params: dict[str, Any] = {"query": self.query}
        if self._cursor != "1":
            params["cursor"] = self._cursor
        return params

    def extract_profiles(self, response: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
        if isinstance(response, list):
            return [item for item in response if isinstance(item, dict)]
        if isinstance(response, dict):
            for key in ("profiles", "data", "items"):
                value = response.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        logger.warning(
            "Unexpected Instagram search API response structure for query: '%s'. "
            "Response type: %s",
            self.query,
            type(response).__name__,
        )
        return []

    def handle_empty_response(self) -> None:
        logger.info(
            "No profiles found in Instagram search results for query: '%s' at cursor '%s'",
            self.query,
            self._cursor,
        )
        self._save_cursor("1")
        self.is_finished = True

    def handle_error(self, error: Exception) -> None:
        logger.error(
            "Instagram candidate discovery failed for query: '%s' at cursor '%s': %s",
            self.query,
            self._cursor,
            error,
            exc_info=True,
        )
        self._save_cursor("1")
        self.is_finished = True

    def register_candidates(
        self,
        response: dict[str, Any] | list[Any],
        new_candidates_found: int,
        discovered_count: int,
    ) -> None:
        if new_candidates_found > 0:
            if isinstance(response, dict):
                next_cursor = response.get("cursor") or str(int(self._cursor) + 1)
            else:
                next_cursor = str(int(self._cursor) + 1)
            self._save_cursor(next_cursor)
            logger.info(
                "Instagram candidate discovery completed for query: '%s'. "
                "Discovered %d valid candidates (%d new) on page %s.",
                self.query,
                discovered_count,
                new_candidates_found,
                self._cursor,
            )
            self.is_finished = True
        else:
            if isinstance(response, dict):
                next_cursor = response.get("cursor") or str(int(self._cursor) + 1)
            else:
                next_cursor = str(int(self._cursor) + 1)
            self._cursor = next_cursor
            logger.info(
                "All %d candidates on page %s for query '%s' already exist. Advancing to next page.",
                discovered_count,
                self._cursor,
                self.query,
            )

    async def sleep(self) -> None:
        if not self.is_finished:
            await asyncio.sleep(1.0)

    def finalize_exhausted(self) -> None:
        if self.depth >= self.max_depth and not self.is_finished:
            logger.warning(
                "Instagram candidate discovery for query: '%s' exhausted max_depth=%d without finding new candidates.",
                self.query,
                self.max_depth,
            )
            self._save_cursor("1")
            self.is_finished = True
