import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CategoryQuery(BaseModel):
    name: str
    queries: list[str] = Field(default_factory=list)


class SearchQueriesSchema(BaseModel):
    queries: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    categories: list[CategoryQuery] = Field(default_factory=list)


class SearchQueriesManager:

    def __init__(self, json_path: Path | str | None = None) -> None:
        if json_path is None:
            self.json_path = Path("src/config/search_queries.json")
        else:
            self.json_path = Path(json_path)

        self._schema: SearchQueriesSchema | None = None
        self._category_patterns: dict[str, re.Pattern[str]] | None = None
        self._load_and_validate()

    def _load_and_validate(self) -> None:
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._schema = SearchQueriesSchema.model_validate(data)
            logger.info(
                f"Successfully loaded search queries from {self.json_path}. "
                f"Found {len(self._schema.queries)} general queries, "
                f"{len(self._schema.keywords)} keywords, "
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

    def get_compiled_keywords_pattern(self) -> re.Pattern[str]:
        if self._schema is None or not self._schema.keywords:
            return re.compile(r"(?!.*)")

        escaped_keywords = [re.escape(keyword) for keyword in self._schema.keywords]
        pattern = r"\b(" + "|".join(escaped_keywords) + r")\b"

        return re.compile(pattern, re.IGNORECASE)

    def get_category_patterns(self) -> dict[str, re.Pattern[str]]:
        if self._category_patterns is not None:
            return self._category_patterns

        if self._schema is None or not self._schema.categories:
            self._category_patterns = {}
            return self._category_patterns

        self._category_patterns = {}
        for category in self._schema.categories:
            if not category.queries:
                continue
            escaped = [re.escape(q) for q in category.queries]
            joined = "|".join(escaped)
            boundary_pattern = (
                rf"(?<![a-zA-Zа-яА-ЯёЁ0-9])(?:{joined})(?![a-zA-Zа-яА-ЯёЁ0-9])"
            )
            self._category_patterns[category.name] = re.compile(
                boundary_pattern, re.IGNORECASE
            )

        return self._category_patterns

    def classify_text(self, text: str) -> str | None:
        if not text:
            return None

        patterns = self.get_category_patterns()
        if not patterns:
            return None

        for category_name, pattern in patterns.items():
            if pattern.search(text):
                return category_name

        return None
