"""Pydantic models for SPG (Semantic Parsing Graph) schema elements."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class SPGNode(BaseModel):
    """Represents a graph node in the knowledge graph."""

    label: str
    """Node type/label (e.g., 'Person', 'Organization')."""

    properties: dict[str, Any]
    """Dictionary of node properties."""

    model_config = ConfigDict(extra="allow")


class SPGEdge(BaseModel):
    """Represents a graph edge/relationship in the knowledge graph."""

    start_node_id: str
    """ID of the start node."""

    edge_label: str
    """Relationship type (e.g., 'HAS_CHILD', 'WORKS_AT')."""

    end_node_id: str
    """ID of the end node."""

    properties: dict[str, Any] = {}
    """Optional dictionary of edge properties."""

    model_config = ConfigDict(extra="allow")


class ExtractionResult(BaseModel):
    """Container for the result of a knowledge extraction operation."""

    nodes: list[SPGNode]
    """List of extracted nodes."""

    edges: list[SPGEdge]
    """List of extracted edges."""

    model_config = ConfigDict(extra="allow")
