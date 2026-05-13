"""Knowledge graph extractor for OpenSPG-based extraction pipeline."""

import json
import logging
from typing import Any

import aiohttp
from pydantic import ValidationError

from src.config.config import Settings
from src.db.database import Database
from src.embeddings.qdrant_service import QdrantService
from src.graph.schema import (
    ExtractionResult,
    OpenSPGExtractionResult,
    SPGEdge,
    SPGNode,
    get_open_spg_llm_prompt,
    open_spg_result_to_extraction_result,
)

logger = logging.getLogger(__name__)


class KnowledgeExtractor:
    """Extracts knowledge triples (nodes and edges) from text content using LLM."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the extractor with configuration settings.

        Args:
            settings: Application settings containing LLM configuration.
        """
        self.settings = settings
        self._session: aiohttp.ClientSession | None = None

    async def _call_llm(self, text: str, author_id: int) -> ExtractionResult:
        """Call the LLM API to extract knowledge triples from text.

        Args:
            text: Input text to analyze.
            author_id: Telegram user ID of the post author (used for author node).

        Returns:
            ExtractionResult containing extracted nodes and edges.
            Returns empty result on any error (timeout, validation, etc.).
        """
        if not self.settings.llm_api_key:
            logger.warning("LLM API key not configured, skipping extraction")
            return ExtractionResult(nodes=[], edges=[])

        # Lazy initialization of HTTP session
        if self._session is None:
            self._session = aiohttp.ClientSession()

        # Construct the OpenSPG prompt
        prompt = get_open_spg_llm_prompt(text, author_id)

        try:
            async with self._session.post(
                f"{self.settings.llm_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.llm_model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a knowledge graph extraction assistant. Always respond with valid JSON matching the requested schema.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    logger.error(
                        "LLM API request failed: status=%d, response=%s",
                        response.status,
                        await response.text(),
                    )
                    return ExtractionResult(nodes=[], edges=[])

                data = await response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                if not content:
                    logger.warning("LLM returned empty content")
                    return ExtractionResult(nodes=[], edges=[])

                # Parse the JSON response as OpenSPG format
                try:
                    open_spg_result = OpenSPGExtractionResult.model_validate_json(content)
                except ValidationError as e:
                    logger.error(
                        "Failed to parse LLM response as OpenSPGExtractionResult: %s\nResponse: %s",
                        e,
                        content[:500],
                    )
                    return ExtractionResult(nodes=[], edges=[])

                # Convert OpenSPG result to legacy ExtractionResult for backward compatibility
                extraction_result = open_spg_result_to_extraction_result(open_spg_result)
                nodes = extraction_result.nodes
                edges = extraction_result.edges

                logger.info(
                    "LLM extraction successful: %d nodes, %d edges",
                    len(nodes),
                    len(edges),
                )
                return ExtractionResult(nodes=nodes, edges=edges)

        except aiohttp.ClientError as e:
            logger.error(
                "LLM API request failed with network error: %s",
                e,
                exc_info=True,
            )
            return ExtractionResult(nodes=[], edges=[])
        except TimeoutError as e:
            logger.error(
                "LLM API request timed out: %s",
                e,
                exc_info=True,
            )
            return ExtractionResult(nodes=[], edges=[])
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to decode LLM API response: %s",
                e,
                exc_info=True,
            )
            return ExtractionResult(nodes=[], edges=[])
        except Exception as e:
            logger.error(
                "Unexpected error during LLM extraction: %s",
                e,
                exc_info=True,
            )
            return ExtractionResult(nodes=[], edges=[])

    async def extract_triplets(
        self, text: str, author_id: int
    ) -> ExtractionResult:
        """
        Extract knowledge triples from the given text using LLM.

        Args:
            text: Input text to analyze.
            author_id: Telegram user ID of the post author.

        Returns:
            ExtractionResult containing extracted nodes and edges.
        """
        logger.debug("Extracting triples from text: %s", text[:100])
        return await self._call_llm(text, author_id)

    async def close(self) -> None:
        """Close the aiohttp session and clean up resources."""
        if self._session is not None:
            await self._session.close()
            self._session = None
            logger.debug("KnowledgeExtractor: aiohttp session closed")

    async def process_post(
        self,
        post_id: int,
        text: str,
        author_id: int,
        db: Database,
        qdrant: QdrantService | None = None,
    ) -> None:
        """
        Process a single post: extract knowledge triples and persist to AGE graph.

        Args:
            post_id: Database ID of the post.
            text: Text content of the post.
            author_id: Telegram user ID of the post author.
            db: Database instance for persistence operations.
            qdrant: Optional QdrantService for syncing entities to vector store.
        """

        logger.info("Processing post id=%s for knowledge extraction", post_id)

        # Extract triples from text
        result = await self.extract_triplets(text, author_id)

        if not result.nodes and not result.edges:
            logger.debug("Post id=%s: no knowledge triples extracted", post_id)
            return

        # Build node ID -> label mapping for edge upserts
        node_id_to_label: dict[str, str] = {}
        for node in result.nodes:
            node_id = node.properties.get("id")
            if node_id:
                node_id_to_label[str(node_id)] = node.label

        # Upsert nodes to AGE graph
        for node in result.nodes:
            try:
                await db.upsert_graph_node(
                    label=node.label,
                    properties=node.properties,
                    merge_key="id",
                )
                logger.debug(
                    "Upserted node: label=%s, id=%s",
                    node.label,
                    node.properties.get("id"),
                )
            except Exception as e:
                logger.error(
                    "Failed to upsert node (post_id=%s, label=%s): %s",
                    post_id,
                    node.label,
                    e,
                )
                raise

        # Upsert edges to AGE graph
        for edge in result.edges:
            start_label = node_id_to_label.get(edge.start_node_id, "Entity")
            end_label = node_id_to_label.get(edge.end_node_id, "Entity")
            try:
                await db.upsert_graph_edge(
                    start_label=start_label,
                    start_merge_key="id",
                    start_merge_val=edge.start_node_id,
                    edge_label=edge.edge_label,
                    end_label=end_label,
                    end_merge_key="id",
                    end_merge_val=edge.end_node_id,
                    edge_properties=edge.properties,
                )
                logger.debug(
                    "Upserted edge: %s(%s)-%s->%s(%s)",
                    start_label,
                    edge.start_node_id,
                    edge.edge_label,
                    end_label,
                    edge.end_node_id,
                )
            except Exception as e:
                logger.error(
                    "Failed to upsert edge (post_id=%s, edge=%s): %s",
                    post_id,
                    edge.edge_label,
                    e,
                )
                raise

        # Sync entities to Qdrant (if service is available)
        if qdrant is not None:
            try:
                await qdrant.upsert_entities(result.nodes)
            except Exception as e:
                logger.error(
                    "Failed to sync entities to Qdrant (post_id=%s): %s",
                    post_id,
                    e,
                    exc_info=True,
                )
                # Do not raise - Qdrant failure should not crash the pipeline

        logger.info(
            "Completed processing post id=%s: %d nodes, %d edges",
            post_id,
            len(result.nodes),
            len(result.edges),
        )
