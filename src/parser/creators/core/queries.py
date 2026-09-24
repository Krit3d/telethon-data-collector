from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_queries_cache: dict[str, SearchQueriesSchema] = {}


class CategoryQuery(BaseModel):
    name: str
    queries: list[str] = Field(default_factory=list)


class SearchQueriesSchema(BaseModel):
    queries: list[str] = Field(default_factory=list)
    categories: list[CategoryQuery] = Field(default_factory=list)


class SearchQueriesManager:

    def __init__(self, json_path: Path | str | None = None) -> None:
        if json_path is None:
            self.json_path = Path("src/config/search_queries.json")
        else:
            self.json_path = Path(json_path)

        self._schema: SearchQueriesSchema | None = None
        self._load_and_validate()

    def _load_and_validate(self) -> None:
        cache_key = str(self.json_path.resolve())

        if cache_key in _queries_cache:
            self._schema = _queries_cache[cache_key]
            return

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._schema = SearchQueriesSchema.model_validate(data)
            _queries_cache[cache_key] = self._schema
            logger.info(
                f"Successfully loaded search queries from {self.json_path}. "
                f"Found {len(self._schema.queries)} general queries, "
                f"{len(self._schema.categories)} categories."
            )
        except FileNotFoundError as e:
            logger.error(f"Search queries file not found: {self.json_path}. Error: {e}")
            self._schema = SearchQueriesSchema()
            logger.warning("Falling back to empty search queries schema.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {self.json_path}. Error: {e}")
            self._schema = SearchQueriesSchema()
            logger.warning("Falling back to empty search queries schema.")
        except Exception as e:
            logger.error(
                f"Unexpected error loading search queries from {self.json_path}. "
                f"Error: {e}",
                exc_info=True,
            )
            self._schema = SearchQueriesSchema()
            logger.warning("Falling back to empty search queries schema.")

    def get_balanced_queries(self) -> list[tuple[str, str]]:
        if self._schema is None or not self._schema.categories:
            logger.warning("No categories available for balanced query generation.")
            return []

        category_queries: list[tuple[str, list[str]]] = [
            (cat.name, cat.queries) for cat in self._schema.categories if cat.queries
        ]

        if not category_queries:
            logger.warning("No queries found in any category.")
            return []

        max_query_count = max(len(queries) for _, queries in category_queries)

        balanced: list[tuple[str, str]] = []

        for i in range(max_query_count):
            for category_name, queries in category_queries:
                if i < len(queries):
                    balanced.append((queries[i], category_name))

        logger.debug(f"Generated {len(balanced)} balanced queries across {len(category_queries)} categories.")
        return balanced
