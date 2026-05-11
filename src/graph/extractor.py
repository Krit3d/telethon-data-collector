"""Knowledge graph extractor for OpenSPG-based extraction pipeline."""

import logging

from src.db.database import Database
from src.graph.schema import ExtractionResult, SPGEdge, SPGNode

logger = logging.getLogger(__name__)


class KnowledgeExtractor:
    """Extracts knowledge triples (nodes and edges) from text content."""

    async def extract_triplets(self, text: str, author_id: int) -> ExtractionResult:
        """
        Extract knowledge triples from the given text.

        This is a mock implementation that looks for family-related keywords
        and generates Person nodes and HAS_CHILD relationships using the
        author_id to create a unique author node.

        Args:
            text: Input text to analyze.
            author_id: Telegram user ID of the post author.

        Returns:
            ExtractionResult containing extracted nodes and edges.
        """
        logger.debug("Extracting triples from text: %s", text[:100])

        # Mock extraction logic: look for family-related keywords
        keywords = ["child", "son", "daughter", "born"]
        text_lower = text.lower()

        if any(keyword in text_lower for keyword in keywords):
            # Create a mock child node with unique ID based on author
            child_node = SPGNode(
                label="Person",
                properties={
                    "id": f"person_child_{author_id}",
                    "name": "Child",
                    "description": "Extracted from text mentioning family",
                },
            )

            # Create an author node using the actual author_id
            author_node = SPGNode(
                label="Person",
                properties={
                    "id": f"person_{author_id}",
                    "name": f"Author {author_id}",
                    "description": f"Post author with Telegram ID {author_id}",
                },
            )

            # Create HAS_CHILD edge
            edge = SPGEdge(
                start_node_id=f"person_{author_id}",
                edge_label="HAS_CHILD",
                end_node_id=f"person_child_{author_id}",
                properties={},
            )

            logger.info(
                "Mock extraction found family relationship: %s -> HAS_CHILD -> %s",
                author_node.properties.get("name"),
                child_node.properties.get("name"),
            )

            return ExtractionResult(nodes=[author_node, child_node], edges=[edge])

        # No extraction found
        logger.debug("No relevant keywords found, returning empty result")
        return ExtractionResult(nodes=[], edges=[])

    async def process_post(
        self, post_id: int, text: str, author_id: int, db: Database
    ) -> None:
        """
        Process a single post: extract knowledge triples and persist to AGE graph.

        Args:
            post_id: Database ID of the post.
            text: Text content of the post.
            author_id: Telegram user ID of the post author.
            db: Database instance for persistence operations.
        """
        logger.info("Processing post id=%s for knowledge extraction", post_id)

        # Extract triples from text
        result = await self.extract_triplets(text, author_id)

        if not result.nodes and not result.edges:
            logger.debug("Post id=%s: no knowledge triples extracted", post_id)
            return

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
            try:
                await db.upsert_graph_edge(
                    start_label="Person",
                    start_merge_key="id",
                    start_merge_val=edge.start_node_id,
                    edge_label=edge.edge_label,
                    end_label="Person",
                    end_merge_key="id",
                    end_merge_val=edge.end_node_id,
                    edge_properties=edge.properties,
                )
                logger.debug(
                    "Upserted edge: %s-%s->%s",
                    edge.start_node_id,
                    edge.edge_label,
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

        logger.info(
            "Completed processing post id=%s: %d nodes, %d edges",
            post_id,
            len(result.nodes),
            len(result.edges),
        )
