"""
Knowledge Graph repository for Apache AGE operations.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

logger = logging.getLogger(__name__)

# Mapping from ID prefix to corresponding graph label
ID_PREFIX_TO_LABEL: dict[str, str] = {
    "actor_": "Actor",
    "entity_": "Entity",
    "event_": "Event",
    "place_": "Place",
    "content_": "Content",
}


def _parse_agtype(value: Any) -> Any:
    """Robust parser for AGE agtype returns (dict, str, None).

    Handles three cases:
    1. ``None`` → returned as-is.
    2. ``dict`` → returned as-is (already parsed by the driver).
    3. ``str`` or other → stripped of surrounding quotes, then attempted
       ``json.loads``.  If that fails the stripped string is returned as-is
       so that plain string IDs (e.g. ``"entity_123"``) are never lost.

    Args:
        value: Raw value returned from an Apache AGE query.

    Returns:
        Parsed Python object: dict, str, or None.
    """

    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        # If it's already a string, check if it's a JSON string that needs parsing
        text_val = value.strip()
        # Only strip quotes if they are balanced and at both ends
        if len(text_val) >= 2 and (
            (text_val[0] == '"' and text_val[-1] == '"')
            or (text_val[0] == "'" and text_val[-1] == "'")
        ):
            text_val = text_val[1:-1]
        try:
            return json.loads(text_val)
        except (json.JSONDecodeError, TypeError):
            return text_val
    # For any other type, convert to string and process
    text_val = str(value).strip()
    if (text_val.startswith('"') and text_val.endswith('"')) or (
        text_val.startswith("'") and text_val.endswith("'")
    ):
        text_val = text_val[1:-1]
    try:
        return json.loads(text_val)
    except (json.JSONDecodeError, TypeError):
        return text_val


class GraphRepository:
    """Repository for all Apache AGE knowledge graph operations.

    Accepts an ``async_sessionmaker`` so it shares the same engine and
    connection pool as the main ``Database`` class.

    Args:
        async_session: An ``async_sessionmaker`` bound to the target engine.
    """

    # Default timeout (seconds) for individual label-level subgraph queries.
    # Over high-latency VPN links (Tailscale, 300ms+) this must be generous
    # enough to accommodate several round-trips while still failing fast
    # enough to not block the entire search request.
    _SUBGRAPH_QUERY_TIMEOUT: float = 15.0

    def __init__(self, async_session: async_sessionmaker[AsyncSession]) -> None:
        self.async_session = async_session
        self._subgraph_query_timeout = self._SUBGRAPH_QUERY_TIMEOUT

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _check_graph_has_edges(self) -> bool:
        """Lightweight pre-check: whether the graph contains any edges at all.

        Runs a simple ``MATCH ()-[r]->() RETURN count(r) LIMIT 1`` query.
        If the graph has zero edges the expensive per-label subgraph queries
        can be skipped entirely, saving multiple high-latency round-trips.

        Returns:
            ``True`` if at least one edge exists, ``False`` otherwise.
            On timeout or error returns ``True`` (fail-open) so the caller
            proceeds to per-label queries which have their own timeout guards.
        """
        
        try:
            async with self.async_session() as session:
                async with session.begin():
                    result = await asyncio.wait_for(
                        session.execute(text("""
                            SELECT * FROM ag_catalog.cypher('telegram_graph',
                                $$ MATCH ()-[r]->() RETURN count(r) LIMIT 1 $$
                            ) AS (cnt agtype)
                        """)),
                        timeout=5.0,
                    )
                    row = result.first()
                    if row is None:
                        return False
                    count_val = _parse_agtype(row[0])
                    edge_count = int(count_val) if count_val is not None else 0
                    logger.debug(
                        "Graph edge existence check: %d total edges", edge_count
                    )
                    return edge_count > 0
        except asyncio.TimeoutError:
            logger.warning(
                "Graph edge existence check timed out — assuming edges exist"
            )
            return True  # Fail open: proceed to per-label queries
        except Exception:
            logger.exception("Graph edge existence check failed")
            return True  # Fail open

    def _group_ids_by_label(self, ids: list[str]) -> dict[str, list[str]]:
        """Group IDs by their corresponding graph label based on prefix.

        IDs with unrecognized prefixes are skipped and logged.

        Args:
            ids: List of entity IDs to group.

        Returns:
            Dictionary mapping label names to lists of IDs belonging to that label.
        """
        grouped: dict[str, list[str]] = {}
        skipped: list[str] = []

        for entity_id in ids:
            matched = False
            for prefix, label in ID_PREFIX_TO_LABEL.items():
                if entity_id.startswith(prefix):
                    grouped.setdefault(label, []).append(entity_id)
                    matched = True
                    break
            if not matched:
                skipped.append(entity_id)

        if skipped:
            logger.debug(
                "Skipped IDs with unrecognized prefixes (no label mapping)",
                extra={"skipped_ids": skipped, "total_skipped": len(skipped)},
            )

        return grouped

    def _sanitize_properties(self, properties: dict) -> dict:
        """Sanitize property keys to ensure they are strictly alphanumeric snake_case.

        This utility method prevents hidden characters or invalid syntax from
        breaking the Cypher query parser. Keys are stripped, converted to
        snake_case, and validated against a strict pattern.

        Args:
            properties: Raw properties dictionary.

        Returns:
            Dictionary with sanitized keys (original values preserved).

        Raises:
            ValueError: If any key cannot be sanitized to a valid identifier.
        """

        sanitized: dict[str, Any] = {}
        for key, value in properties.items():
            clean_key = key.strip().lower()
            clean_key = re.sub(r"[^a-z0-9_]", "_", clean_key)
            clean_key = re.sub(r"_+", "_", clean_key)
            clean_key = clean_key.strip("_")

            if not re.match(r"^[a-z0-9_]+$", clean_key):
                raise ValueError(
                    f"Property key '{key}' could not be sanitized to a valid identifier "
                    f"(got '{clean_key}'). Only alphanumeric characters and underscores are allowed."
                )

            sanitized[clean_key] = value

        return sanitized

    # ------------------------------------------------------------------
    # Node / edge write operations
    # ------------------------------------------------------------------

    async def upsert_graph_node(
        self, label: str, properties: dict, merge_key: str = "id"
    ) -> None:
        """Upsert a node in the Apache AGE graph.

        Uses MERGE to create or update a node with the given label and properties.
        The node is matched on the merge_key (default: 'id').

        IMPORTANT: Labels and property keys cannot be parameterized in Cypher,
        so we validate them to prevent injection attacks. Properties are set
        using an explicit SET clause with individual bind parameters to avoid
        the "SET clause expects a map" error in Apache AGE.

        Args:
            label: Graph node label (e.g., 'Account', 'Content'). Must be alphanumeric.
            properties: Dictionary of node properties.
            merge_key: Property name to use for MERGE matching (default: 'id').

        Raises:
            ValueError: If label, merge_key, or any property key contains
                non-alphanumeric characters.
        """

        if not re.match(r"^[A-Za-z0-9_]+$", label):
            raise ValueError(
                f"Invalid label '{label}': must be alphanumeric with underscores"
            )

        if not re.match(r"^[A-Za-z0-9_]+$", merge_key):
            raise ValueError(
                f"Invalid merge_key '{merge_key}': must be alphanumeric with underscores"
            )

        props = self._sanitize_properties(properties)

        if merge_key not in props:
            props[merge_key] = properties.get(merge_key)

        set_clauses: list[str] = []
        set_params: dict[str, Any] = {}
        for key, value in props.items():
            if key == merge_key:
                continue
            set_clauses.append(f"n.`{key}` = $prop_{key}")
            set_params[f"prop_{key}"] = value

        set_clause_str = ", ".join(set_clauses) if set_clauses else ""

        query = text(f"""
            SELECT * FROM cypher('telegram_graph',
                $$ MERGE (n:{label} {{{merge_key}: $merge_val}})
                   {f"SET {set_clause_str}" if set_clause_str else ""}
                   RETURN n $$,
                CAST(:params AS agtype)
            ) AS (v agtype)
        """)

        params_dict: dict[str, Any] = {
            "merge_val": props.get(merge_key),
            **set_params,
        }
        params = json.dumps(params_dict)

        async with self.async_session() as session:
            async with session.begin():
                await session.execute(query, {"params": params})

    async def upsert_graph_edge(
        self,
        start_label: str,
        start_merge_key: str,
        start_merge_val: Any,
        edge_label: str,
        end_label: str,
        end_merge_key: str,
        end_merge_val: Any,
        edge_properties: dict | None = None,
    ) -> None:
        """Upsert an edge in the Apache AGE graph.

        Matches start and end nodes by label and merge keys, then MERGEs the edge.
        Edge properties are set using an explicit SET clause with individual bind
        parameters to avoid the "SET clause expects a map" error in Apache AGE.

        IMPORTANT: Labels and relationship types cannot be parameterized in Cypher,
        so we validate them to prevent injection attacks. All property keys are
        sanitized to valid identifiers. Properties are assigned individually.

        Args:
            start_label: Label of the start node. Must be alphanumeric.
            start_merge_key: Property name to match start node.
            start_merge_val: Value for start node matching.
            edge_label: Relationship type label. Must be alphanumeric.
            end_label: Label of the end node. Must be alphanumeric.
            end_merge_key: Property name to match end node.
            end_merge_val: Value for end node matching.
            edge_properties: Optional dictionary of edge properties.

        Raises:
            ValueError: If any label, edge_label, merge key, or property key
                contains non-alphanumeric characters.
        """

        for identifier_name, identifier_value in [
            ("start_label", start_label),
            ("edge_label", edge_label),
            ("end_label", end_label),
        ]:
            if not re.match(r"^[A-Za-z0-9_]+$", identifier_value):
                raise ValueError(
                    f"Invalid {identifier_name} '{identifier_value}': "
                    "must be alphanumeric with underscores"
                )

        if not re.match(r"^[A-Za-z0-9_]+$", start_merge_key):
            raise ValueError(
                f"Invalid start_merge_key '{start_merge_key}': "
                "must be alphanumeric with underscores"
            )
        if not re.match(r"^[A-Za-z0-9_]+$", end_merge_key):
            raise ValueError(
                f"Invalid end_merge_key '{end_merge_key}': "
                "must be alphanumeric with underscores"
            )

        edge_properties = edge_properties or {}
        props = self._sanitize_properties(edge_properties)

        set_clauses: list[str] = []
        set_params: dict[str, Any] = {}
        for key, value in props.items():
            set_clauses.append(f"r.`{key}` = $prop_{key}")
            set_params[f"prop_{key}"] = value

        set_clause_str = ", ".join(set_clauses) if set_clauses else ""

        query = text(f"""
            SELECT * FROM cypher('telegram_graph',
                $$ MATCH (a:{start_label} {{{start_merge_key}: $sid}})
                   MATCH (b:{end_label} {{{end_merge_key}: $eid}})
                   MERGE (a)-[r:{edge_label}]->(b)
                   {f"SET {set_clause_str}" if set_clause_str else ""}
                   RETURN r $$,
                CAST(:params AS agtype)
            ) AS (v agtype)
        """)

        params_dict: dict[str, Any] = {
            "sid": start_merge_val,
            "eid": end_merge_val,
            **set_params,
        }
        params = json.dumps(params_dict)

        async with self.async_session() as session:
            async with session.begin():
                await session.execute(query, {"params": params})

    # ------------------------------------------------------------------
    # Node / edge read operations
    # ------------------------------------------------------------------

    async def _query_nodes_by_label(
        self, label: str, node_ids: list[str]
    ) -> list[dict]:
        """Fetch nodes of a single label by their IDs using a single Cypher query.

        Uses ``UNWIND`` to pass each ID individually, which is more reliable in
        Apache AGE for index lookups on ``agtype`` properties compared to the
        ``ANY`` operator.

        Args:
            label: The graph label to query (e.g., 'Actor', 'Entity').
            node_ids: List of node original_id values to fetch.

        Returns:
            List of node dicts with keys: id, label, name, properties.
        """

        if not node_ids:
            return []

        limit_val = len(node_ids) * 50

        # UNWIND pattern — each ID is matched individually for reliable index hits
        # in Apache AGE. Using WHERE clause instead of inline property matcher
        # helps the optimizer pick the right index.
        query = text(f"""
            SELECT * FROM ag_catalog.cypher('telegram_graph',
                $$ UNWIND $ids_list AS hid
                   MATCH (n:{label})
                   WHERE n.id = hid
                   RETURN n.id AS n_id, label(n) AS n_label, n.name AS n_name, n
                   LIMIT {limit_val} $$,
                CAST(:params AS agtype)
            ) AS (
                n_id    agtype,
                n_label agtype,
                n_name  agtype,
                n       agtype
            )
        """)

        try:
            async with self.async_session() as session:
                async with session.begin():
                    # Python-level timeout — reliable across high-latency VPN
                    # links where server-side statement_timeout may not fire
                    # before the client-side command_timeout.
                    result = await asyncio.wait_for(
                        session.execute(
                            query,
                            {"params": json.dumps({"ids_list": node_ids})},
                        ),
                        timeout=self._subgraph_query_timeout,
                    )
                    rows = result.all()

                    nodes: list[dict] = []
                    for row in rows:
                        node_id = _parse_agtype(row[0])
                        node_label = _parse_agtype(row[1])
                        node_name = _parse_agtype(row[2])
                        node_data = _parse_agtype(row[3])

                        properties: dict[str, Any] = {}
                        if isinstance(node_data, dict):
                            for key, value in node_data.items():
                                if key not in ("id", "label", "name"):
                                    properties[key] = value

                        nodes.append(
                            {
                                "id": node_id,
                                "label": node_label,
                                "name": node_name,
                                "properties": properties,
                            }
                        )
                    return nodes
        except asyncio.TimeoutError:
            logger.warning(
                "Node query timed out after %.1fs for label '%s' with %d ids",
                self._subgraph_query_timeout,
                label,
                len(node_ids),
            )
            return []
        except Exception:
            logger.exception(
                "Database error fetching nodes for label '%s' with ids=%s",
                label,
                node_ids,
            )
            return []

    async def get_nodes_by_ids(
        self, node_ids: list[str], label: str | None = None
    ) -> list[dict]:
        """Fetch full node details from Apache AGE graph by node IDs.

        Args:
            node_ids: List of node IDs to fetch.
            label: Optional node label filter to restrict search to a specific label.

        Returns:
            List of dictionaries with node details: id, label, name, properties (as dict).

        Note: Returns empty list if node_ids is empty. Individual label queries
        that fail (e.g., missing label) are logged as warnings and their results
        are skipped; other label queries still succeed.
        """

        if not node_ids:
            return []

        valid_ids = [
            nid for nid in node_ids if re.match(r"^[a-zA-Z0-9_-]+$", nid)
        ]
        if not valid_ids:
            return []

        if label:
            return await self._query_nodes_by_label(label, valid_ids)

        ids_by_label = self._group_ids_by_label(valid_ids)
        if not ids_by_label:
            return []

        logger.debug(
            "IDs grouped by label for graph query",
            extra={"ids_by_label": ids_by_label},
        )

        start_time = time.perf_counter()
        tasks = [
            self._query_nodes_by_label(label, ids)
            for label, ids in ids_by_label.items()
        ]
        results: list[list[dict]] = await asyncio.gather(*tasks, return_exceptions=False)  # type: ignore[arg-type]
        duration = time.perf_counter() - start_time

        all_nodes: list[dict] = []
        for nodes_list in results:
            all_nodes.extend(nodes_list)

        logger.debug(
            "Fetched nodes from graph (grouped by label)",
            extra={
                "node_count": len(all_nodes),
                "requested_ids": len(node_ids),
                "duration_seconds": duration,
            },
        )
        return all_nodes

    async def get_subgraph_for_entities(
        self, entity_ids: list[str]
    ) -> list[dict]:
        """Fetch the subgraph (direct neighbors) for a list of entity IDs from Apache AGE.

        Args:
            entity_ids: List of entity original_id values to query graph relationships for.

        Returns:
            List of dictionaries representing graph edges with keys:
            - source_id: ID of the source node
            - source_label: Label of the source node
            - source_name: Name property of the source node
            - target_id: ID of the target node
            - target_label: Label of the target node
            - target_name: Name property of the target node

        Note: Returns empty list if entity_ids is empty or no valid IDs.
        Individual label queries that fail (e.g., missing label) are logged as
        warnings and their results are skipped; other label queries still succeed.
        """

        if not entity_ids:
            return []

        valid_ids = [
            uid for uid in entity_ids if re.match(r"^[a-zA-Z0-9_-]+$", uid)
        ]
        if not valid_ids:
            return []

        ids_by_label = self._group_ids_by_label(valid_ids)
        if not ids_by_label:
            return []

        # Fast pre-check: if the graph has zero edges we can skip all the
        # expensive per-label subgraph queries and return immediately.  This
        # saves multiple high-latency round-trips over the VPN tunnel.
        graph_has_edges = await self._check_graph_has_edges()
        if not graph_has_edges:
            logger.info(
                "Graph has no edges — skipping subgraph query for %d entities",
                len(valid_ids),
            )
            return []

        async def query_for_label(label: str, ids: list[str]) -> list[dict]:
            """Fetch subgraph edges for a single label group.

            Uses two directed MATCH patterns (outgoing + incoming) instead of one
            undirected ``(a)-[r]-(b)`` because Apache AGE can route directed
            traversals more efficiently through its internal adjacency structures.

            A Python-level ``asyncio.wait_for`` guard ensures the query cannot
            block the request for longer than ``_SUBGRAPH_QUERY_TIMEOUT`` seconds,
            which is essential when operating over high-latency VPN links
            (Tailscale, 300ms+) where even moderate table scans amplify into
            multi-second delays.

            Args:
                label: Graph label (e.g. 'Entity', 'Event').
                ids: List of node ``id`` values to look up.

            Returns:
                List of edge dicts (empty list on timeout, error, or no edges).
            """
            if not ids:
                return []

            limit_val = len(ids) * 50

            # Use two directed patterns instead of undirected (a)-[r]-(b).
            # This lets Apache AGE optimize each direction independently and
            # avoids the bidirectional expansion that can cause full scans.
            # We UNION the two directed traversals so the caller receives a
            # single result set identical in shape to the original query.
            query = text(f"""
                SELECT * FROM ag_catalog.cypher('telegram_graph',
                    $$ UNWIND $ids_list AS hid
                       MATCH (a:{label})
                       WHERE a.id = hid
                       MATCH (a)-[r]->(b)
                       RETURN a.id AS a_id, label(a) AS a_label, a.name AS a_name,
                              type(r) AS rel_type, b.id AS b_id, label(b) AS b_label, b.name AS b_name
                       LIMIT {limit_val}
                       UNION
                       UNWIND $ids_list AS hid
                       MATCH (a:{label})
                       WHERE a.id = hid
                       MATCH (a)<-[r]-(b)
                       RETURN a.id AS a_id, label(a) AS a_label, a.name AS a_name,
                              type(r) AS rel_type, b.id AS b_id, label(b) AS b_label, b.name AS b_name
                       LIMIT {limit_val} $$,
                    CAST(:params AS agtype)
                ) AS (
                    a_id     agtype,
                    a_label  agtype,
                    a_name   agtype,
                    rel_type agtype,
                    b_id     agtype,
                    b_label  agtype,
                    b_name   agtype
                )
            """)

            try:
                async with self.async_session() as session:
                    async with session.begin():
                        # Python-level timeout guard — essential for high-latency
                        # VPN links where server-side statement_timeout may not
                        # fire before the client-side command_timeout.
                        result = await asyncio.wait_for(
                            session.execute(
                                query,
                                {"params": json.dumps({"ids_list": ids})},
                            ),
                            timeout=self._subgraph_query_timeout,
                        )
                        rows = result.all()

                        edges: list[dict] = []
                        for row in rows:
                            edges.append(
                                {
                                    "source_id": _parse_agtype(row[0]),
                                    "source_label": _parse_agtype(row[1]),
                                    "source_name": _parse_agtype(row[2]),
                                    "relation_type": _parse_agtype(row[3]),
                                    "target_id": _parse_agtype(row[4]),
                                    "target_label": _parse_agtype(row[5]),
                                    "target_name": _parse_agtype(row[6]),
                                }
                            )

                        logger.debug(
                            "Fetched subgraph edges for label",
                            extra={
                                "label": label,
                                "ids_count": len(ids),
                                "edges_found": len(edges),
                            },
                        )

                        # If no edges found, fetch sample IDs to diagnose mismatch
                        if len(edges) == 0 and len(ids) > 0:
                            try:
                                sample_result = await asyncio.wait_for(
                                    session.execute(text(f"""
                                        SELECT * FROM ag_catalog.cypher('telegram_graph',
                                            $$ MATCH (n:{label}) RETURN n.id LIMIT 3 $$
                                        ) AS (sample_id agtype)
                                    """)),
                                    timeout=self._subgraph_query_timeout,
                                )
                                sample_rows = sample_result.all()
                                sample_ids = [
                                    str(_parse_agtype(row[0]))
                                    for row in sample_rows
                                ]
                                logger.warning(
                                    "No edges found for label - possible ID mismatch",
                                    extra={
                                        "label": label,
                                        "requested_ids": ids[:5],
                                        "sample_ids_in_db": sample_ids,
                                    },
                                )
                            except asyncio.TimeoutError:
                                logger.warning(
                                    "Sample ID fetch timed out for label '%s'",
                                    label,
                                )
                            except Exception as sample_e:
                                logger.warning(
                                    "Failed to fetch sample IDs for label",
                                    extra={
                                        "label": label,
                                        "error": str(sample_e),
                                    },
                                )

                        return edges

            except asyncio.TimeoutError:
                logger.warning(
                    "Subgraph query timed out after %.1fs for label '%s' with %d ids=%s",
                    self._subgraph_query_timeout,
                    label,
                    len(ids),
                    ids[:5],
                )
                return []
            except Exception:
                logger.exception(
                    "Database error fetching subgraph for label '%s' with ids=%s",
                    label,
                    ids,
                )
                return []

        start_time = time.perf_counter()
        tasks = [
            query_for_label(label, ids) for label, ids in ids_by_label.items()
        ]
        results: list[list[dict]] = await asyncio.gather(*tasks, return_exceptions=False)  # type: ignore[arg-type]
        duration = time.perf_counter() - start_time

        all_edges: list[dict] = []
        for edges_list in results:
            all_edges.extend(edges_list)

        logger.info(
            f"Graph subgraph query for {len(entity_ids)} entities "
            f"(grouped by label) took {duration:.4f}s, found {len(all_edges)} edges"
        )

        return all_edges

    # ------------------------------------------------------------------
    # Raw Cypher execution
    # ------------------------------------------------------------------

    async def execute_cypher(self, query: str) -> Any:
        """Execute a raw Cypher query against the Apache AGE graph.

        The query is executed in the context of the ``telegram_graph`` graph.

        Args:
            query: Cypher query string, e.g.::

                SELECT * FROM cypher('telegram_graph',
                    $$ MATCH (n) RETURN n $$
                ) AS (n agtype);

        Returns:
            Raw query result rows.
        """
        
        async with self.async_session() as session:
            async with session.begin():
                result = await session.execute(text(query))
                return result.scalars().all()

            async with session.begin():
                result = await session.execute(text(query))
                return result.scalars().all()
