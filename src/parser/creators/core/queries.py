"""
Balanced, category-aware candidate discovery queue manager.

This module provides Pydantic V2 models for validating search_queries.json
and a SearchQueriesManager class that implements a symmetric Round-Robin
query selection algorithm to ensure balanced API usage across categories.
"""

import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CategoryQuery(BaseModel):
    """
    A category containing a list of search queries.

    Attributes:
        name: The name/identifier of the category.
        queries: List of search query strings for this category.
    """

    name: str
    queries: list[str] = Field(default_factory=list)


class SearchQueriesSchema(BaseModel):
    """
    Schema for validating the structure of search_queries.json.

    Attributes:
        queries: Top-level list of general search queries.
        keywords: List of keywords for content filtering/pattern matching.
        categories: List of categorized query groups.
    """

    queries: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    categories: list[CategoryQuery] = Field(default_factory=list)


class SearchQueriesManager:
    """
    Manages loading, validating, and balancing search queries from JSON configuration.

    This manager implements a symmetric Round-Robin algorithm to ensure that
    no single category dominates API usage when discovering candidates.

    Attributes:
        json_path: Path to the search_queries.json configuration file.
        _schema: Validated SearchQueriesSchema instance.
    """

    def __init__(self, json_path: Path | str | None = None) -> None:
        """
        Initialize the SearchQueriesManager.

        Args:
            json_path: Path to the JSON configuration file.
                Defaults to 'src/config/search_queries.json' relative to project root.
        """
        if json_path is None:
            # Default path relative to project structure
            self.json_path = Path("src/config/search_queries.json")
        else:
            self.json_path = Path(json_path)

        self._schema: SearchQueriesSchema | None = None
        self._load_and_validate()

    def _load_and_validate(self) -> None:
        """
        Load and validate the JSON configuration file.

        Handles FileNotFoundError and json.JSONDecodeError gracefully by
        logging errors and falling back to safe empty defaults.
        """
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
        """
        Compile queries across all categories into a symmetric Round-Robin sequence.

        This ensures no single category dominates API usage by rotating through
        categories one query at a time until all queries are exhausted.

        Returns:
            List of (query, category_name) tuples in balanced order.

        Example:
            If Category A has [A1, A2] and Category B has [B1, B2, B3],
            the returned sequence will be:
            [("A1", "Category A"), ("B1", "Category B"), ("A2", "Category A"),
             ("B2", "Category B"), ("B3", "Category B")]
        """
        if self._schema is None or not self._schema.categories:
            logger.warning("No categories available for balanced query generation.")
            return []

        # Collect all category queries with their category names
        category_queries: list[tuple[str, list[str]]] = [
            (cat.name, cat.queries) for cat in self._schema.categories if cat.queries
        ]

        if not category_queries:
            logger.warning("No queries found in any category.")
            return []

        # Calculate the maximum number of queries in any category
        max_query_count = max(len(queries) for _, queries in category_queries)

        balanced: list[tuple[str, str]] = []

        # Round-robin through categories
        for i in range(max_query_count):
            for category_name, queries in category_queries:
                if i < len(queries):
                    balanced.append((queries[i], category_name))

        logger.debug(f"Generated {len(balanced)} balanced queries across {len(category_queries)} categories.")
        return balanced

    def get_compiled_keywords_pattern(self) -> re.Pattern[str]:
        """
        Compile the top-level list of keywords into a case-insensitive regex pattern.

        The pattern uses word boundaries to ensure whole-word matching.
        If the keywords list is empty, returns a pattern that never matches.

        Returns:
            Compiled regex pattern for keyword matching.
        """
        if self._schema is None or not self._schema.keywords:
            # Return a pattern that never matches
            return re.compile(r"(?!.*)")

        # Escape keywords for safe regex usage and join with |
        escaped_keywords = [re.escape(keyword) for keyword in self._schema.keywords]
        pattern = r"\b(" + "|".join(escaped_keywords) + r")\b"

        return re.compile(pattern, re.IGNORECASE)
