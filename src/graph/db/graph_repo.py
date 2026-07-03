import asyncio
import json
import logging
import random
import re
import time
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from src.config.config import Settings
from src.db.database import with_retry_on_deadlock
from src.graph.db.helpers import ID_PREFIX_TO_LABEL, parse_agtype, connection_retry

logger = logging.getLogger(__name__)


class GraphRepository:

    _SUBGRAPH_QUERY_TIMEOUT: float = 15.0

    _JUNK_IDS_AND_NAMES = frozenset({
        "loc", "location", "place", "actor", "event", "entity", "topic",
        "unknown", "null", "none", "undefined", "n_a", "other", "id"
    })

    _verified_entities: set[str] = set()
    _cache_warmed: bool = False
    _cache_lock: asyncio.Lock | None = None

    def __init__(
        self,
        async_session: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        self.async_session = async_session
        self.settings = settings
        self.graph_name: str = settings.graph_name
        self._subgraph_query_timeout = self._SUBGRAPH_QUERY_TIMEOUT

    def _resolve_graph_name(self) -> str:
        name = self.graph_name
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(
                f"Invalid graph name '{name}': must be a valid identifier"
            )
        return name

    async def _execute_in_transaction(
        self, query: Any, params: dict[str, Any] | None = None,
        session: AsyncSession | None = None,
    ) -> Any:
        if session is not None:
            return await session.execute(query, params)
        async with self.async_session() as new_session:
            async with new_session.begin():
                return await new_session.execute(query, params)

    async def warm_up_cache(self) -> None:
        graph_name = self._resolve_graph_name()
        try:
            query = text(
                f"SELECT * FROM ag_catalog.cypher('{graph_name}', $$ MATCH (n) RETURN n.id $$) AS (id agtype)"
            )
            result = await self._execute_in_transaction(query)
            for row in result:
                raw_id = parse_agtype(row[0])
                if raw_id is not None:
                    GraphRepository._verified_entities.add(str(raw_id))
            GraphRepository._cache_warmed = True
            logger.info(
                "L1 cache warmed with %d pre-existing entities",
                len(GraphRepository._verified_entities),
            )
        except Exception:
            logger.warning("L1 cache warming failed", exc_info=True)

    async def initialize_graph(self) -> None:
        graph_name = self._resolve_graph_name()
        check_query = text(
            "SELECT 1 FROM ag_catalog.ag_graph WHERE name = :graph_name LIMIT 1"
        )
        result = await self._execute_in_transaction(
            check_query, {"graph_name": graph_name}
        )
        if result.first() is None:
            create_query = text(
                f"SELECT ag_catalog.create_graph('{graph_name}')"
            )
            await self._execute_in_transaction(create_query)
            logger.info("Created AGE graph '%s'", graph_name)
        else:
            logger.debug("AGE graph '%s' already exists", graph_name)
        await self.warm_up_cache()

    async def _check_graph_has_edges(self) -> bool:
        graph_name = self._resolve_graph_name()
        try:
            query = text(
                f"SELECT * FROM ag_catalog.cypher('{graph_name}',"
                " $$ MATCH ()-[r]->() RETURN r LIMIT 1 $$"
                ") AS (r agtype)"
            )
            result = await asyncio.wait_for(
                self._execute_in_transaction(query),
                timeout=5.0,
            )
            row = result.first()
            has_edges = row is not None
            logger.debug(
                "Graph edge existence check: edges_present=%s", has_edges
            )
            return has_edges
        except asyncio.TimeoutError:
            logger.warning(
                "Graph edge existence check timed out, assuming edges exist"
            )
            return True
        except Exception:
            logger.exception("Graph edge existence check failed")
            return True

    def _group_ids_by_label(self, ids: list[str]) -> dict[str, list[str]]:
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

    def _clean_string_for_age(self, val: str) -> str:
        val = re.sub(r"[\r\n\t]+", " ", val)
        val = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", val)
        val = val.replace("\\", "/")
        val = re.sub(r" +", " ", val)
        try:
            val = val.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")
        except (ValueError, UnicodeError):
            pass
        return val.strip()

    def _sanitize_properties(self, properties: dict) -> dict:
        sanitized: dict[str, Any] = {}
        for key, value in properties.items():
            clean_key = key.strip().lower()
            clean_key = re.sub(r"[^a-z0-9_]", "_", clean_key)
            clean_key = re.sub(r"_+", "_", clean_key)
            clean_key = clean_key.strip("_")

            if not re.match(r"^[a-z0-9_]+$", clean_key):
                logger.warning(
                    "Property key '%s' is invalid and cannot be sanitized to a valid identifier (got '%s'). Skipping.",
                    key,
                    clean_key,
                )
                continue

            if isinstance(value, (list, dict)):
                sanitized[clean_key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, str):
                sanitized[clean_key] = self._clean_string_for_age(value)
            else:
                sanitized[clean_key] = value

        return sanitized

    _NUMERIC_KEYS: set[str] = {
        "follower_count", "views", "reactions_count",
        "comments_count", "shares_count", "db_post_id",
        "last_modified_at", "engagement_rate", "confidence",
    }

    def _clean_numeric_properties(self, props: dict[str, Any]) -> dict[str, Any]:
        keys_to_remove: list[str] = []

        for key in props:
            if key not in self._NUMERIC_KEYS:
                continue

            value = props[key]

            if value is None or value == "":
                keys_to_remove.append(key)
                continue

            if isinstance(value, str):
                stripped = re.sub(r"[^0-9.\-]", "", value)

                if not stripped or stripped in ("-", ".") or stripped.count(".") > 1:
                    logger.warning(
                        "Removing non-numeric property '%s' with unparseable value '%s'",
                        key, value,
                    )
                    keys_to_remove.append(key)
                    continue

                try:
                    if "." in stripped:
                        parsed = float(stripped)
                    else:
                        parsed = int(stripped)
                except (ValueError, OverflowError):
                    logger.warning(
                        "Failed to parse numeric property '%s' from value '%s', removing it",
                        key, value,
                    )
                    keys_to_remove.append(key)
                    continue

                props[key] = parsed
                continue

            if isinstance(value, float):
                if value == int(value):
                    props[key] = int(value)
                continue

            if isinstance(value, int):
                continue

            keys_to_remove.append(key)

        for key in keys_to_remove:
            props.pop(key, None)

        return props

    async def _delete_and_create_node(
        self, label: str, properties: dict, merge_key: str,
        merge_val: Any, session: AsyncSession,
    ) -> None:
        graph_name = self._resolve_graph_name()

        delete_query = text(f"""
            SELECT * FROM cypher('{graph_name}',
                $$ MATCH (n:{label} {{{merge_key}: $merge_val}})
                   DETACH DELETE n $$,
                CAST(:params AS agtype)
            ) AS (v agtype)
        """)
        await self._execute_in_transaction(
            delete_query, {"params": json.dumps({"merge_val": merge_val})}, session=session
        )

        set_clauses: list[str] = []
        set_params: dict[str, Any] = {"merge_val": merge_val}
        for key, value in properties.items():
            if key == merge_key or value is None:
                continue
            set_clauses.append(f"n.`{key}` = $prop_{key}")
            set_params[f"prop_{key}"] = value

        set_clause_str = ", ".join(set_clauses) if set_clauses else ""
        params = json.dumps(set_params)

        if set_clause_str:
            create_query = text(f"""
                SELECT * FROM cypher('{graph_name}',
                    $$ CREATE (n:{label} {{{merge_key}: $merge_val}})
                       SET {set_clause_str}
                       RETURN n $$,
                    CAST(:params AS agtype)
                ) AS (v agtype)
            """)
        else:
            create_query = text(f"""
                SELECT * FROM cypher('{graph_name}',
                    $$ CREATE (n:{label} {{{merge_key}: $merge_val}})
                       RETURN n $$,
                    CAST(:params AS agtype)
                ) AS (v agtype)
            """)
        await self._execute_in_transaction(
            create_query, {"params": params}, session=session
        )

    @with_retry_on_deadlock(max_retries=5, base_delay=0.2)
    async def upsert_graph_node(
        self, label: str, properties: dict, merge_key: str = "id",
        session: AsyncSession | None = None,
    ) -> None:
        label = re.sub(r"[^A-Za-z0-9_]", "", label)
        if not label:
            label = "Entity"
        if not merge_key:
            merge_key = "id"

        props = self._sanitize_properties(properties)
        props = self._clean_numeric_properties(props)

        if merge_key not in props:
            props[merge_key] = properties.get(merge_key)

        set_clauses: list[str] = []
        set_params: dict[str, Any] = {}
        for key, value in props.items():
            if key == merge_key or value is None:
                continue
            set_clauses.append(f"n.`{key}` = $prop_{key}")
            set_params[f"prop_{key}"] = value

        set_clause_str = ", ".join(set_clauses) if set_clauses else ""
        graph_name = self._resolve_graph_name()

        query = text(f"""
            SELECT * FROM cypher('{graph_name}',
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

        if session is not None:
            await self._execute_in_transaction(query, {"params": params}, session=session)
            return
        async with self.async_session() as new_session:
            async with new_session.begin():
                await new_session.execute(
                    text("SELECT pg_catalog.pg_advisory_xact_lock(hashtext(:val)::bigint)"),
                    {"val": props.get(merge_key)}
                )
                await new_session.execute(query, {"params": params})

    @with_retry_on_deadlock(max_retries=5, base_delay=0.2)
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
        session: AsyncSession | None = None,
    ) -> None:
        start_label = re.sub(r"[^A-Za-z0-9_]", "", start_label)
        if not start_label:
            start_label = "Entity"
        end_label = re.sub(r"[^A-Za-z0-9_]", "", end_label)
        if not end_label:
            end_label = "Entity"
        edge_label = re.sub(r"[^A-Za-z0-9_]", "", edge_label)
        edge_label = edge_label.upper()
        if not edge_label:
            edge_label = "RELATED_TO"
        if not start_merge_key:
            start_merge_key = "id"
        if not end_merge_key:
            end_merge_key = "id"

        edge_properties = edge_properties or {}
        props = self._sanitize_properties(edge_properties)

        set_clauses: list[str] = []
        set_params: dict[str, Any] = {}
        for key, value in props.items():
            if value is None:
                continue
            set_clauses.append(f"r.`{key}` = $prop_{key}")
            set_params[f"prop_{key}"] = value

        set_clause_str = ", ".join(set_clauses) if set_clauses else ""
        graph_name = self._resolve_graph_name()

        query = text(f"""
            SELECT * FROM cypher('{graph_name}',
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

        if session is not None:
            await self._execute_in_transaction(query, {"params": params}, session=session)
            return
        sorted_vals = sorted([str(start_merge_val), str(end_merge_val)])
        async with self.async_session() as new_session:
            async with new_session.begin():
                for val in sorted_vals:
                    await new_session.execute(
                        text("SELECT pg_catalog.pg_advisory_xact_lock(hashtext(:val)::bigint)"),
                        {"val": val}
                    )
                await new_session.execute(query, {"params": params})

    async def _query_nodes_by_label(
        self, label: str, node_ids: list[str]
    ) -> list[dict]:
        if not node_ids:
            return []

        limit_val = len(node_ids) * 50
        graph_name = self._resolve_graph_name()

        query = text(f"""
            SELECT * FROM ag_catalog.cypher('{graph_name}',
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
            result = await asyncio.wait_for(
                self._execute_in_transaction(
                    query,
                    {"params": json.dumps({"ids_list": node_ids})},
                ),
                timeout=self._subgraph_query_timeout,
            )
            rows = result.all()

            nodes: list[dict] = []
            for row in rows:
                node_id = parse_agtype(row[0])
                node_label = parse_agtype(row[1])
                node_name = parse_agtype(row[2])
                node_data = parse_agtype(row[3])

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
        results: list[list[dict]] = await asyncio.gather(*tasks, return_exceptions=False)
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

        graph_has_edges = await self._check_graph_has_edges()
        if not graph_has_edges:
            logger.info(
                "Graph has no edges, skipping subgraph query for %d entities",
                len(valid_ids),
            )
            return []

        graph_name = self._resolve_graph_name()

        async def query_for_label(label: str, ids: list[str]) -> list[dict]:
            if not ids:
                return []

            limit_val = len(ids) * 50

            query = text(f"""
                SELECT * FROM ag_catalog.cypher('{graph_name}',
                    $$ UNWIND $ids_list AS hid
                       MATCH (a:{label})
                       WHERE a.id = hid
                       MATCH (a)-[r]->(b)
                       RETURN a.id AS a_id, label(a) AS a_label, a.name AS a_name,
                              type(r) AS rel_type, b.id AS b_id, label(b) AS b_label, b.name AS b_name
                       LIMIT {limit_val}
                       UNION
                       UNWIND $ids_list AS hid
                       MATCH (b:{label})
                       WHERE b.id = hid
                       MATCH (a)-[r]->(b)
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
                result = await asyncio.wait_for(
                    self._execute_in_transaction(
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
                            "source_id": parse_agtype(row[0]),
                            "source_label": parse_agtype(row[1]),
                            "source_name": parse_agtype(row[2]),
                            "relation_type": parse_agtype(row[3]),
                            "target_id": parse_agtype(row[4]),
                            "target_label": parse_agtype(row[5]),
                            "target_name": parse_agtype(row[6]),
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

                if len(edges) == 0 and len(ids) > 0:
                    try:
                        sample_result = await asyncio.wait_for(
                            self._execute_in_transaction(text(f"""
                                SELECT * FROM ag_catalog.cypher('{graph_name}',
                                    $$ MATCH (n:{label}) RETURN n.id LIMIT 3 $$
                                ) AS (sample_id agtype)
                            """)),
                            timeout=self._subgraph_query_timeout,
                        )
                        sample_rows = sample_result.all()
                        sample_ids = [
                            str(parse_agtype(row[0]))
                            for row in sample_rows
                        ]
                        logger.warning(
                            "No edges found for label, possible ID mismatch",
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
        results: list[list[dict]] = await asyncio.gather(*tasks, return_exceptions=False)
        duration = time.perf_counter() - start_time

        all_edges: list[dict] = []
        for edges_list in results:
            all_edges.extend(edges_list)

        logger.info(
            "Graph subgraph query for %d entities "
            "(grouped by label) took %.4fs, found %d edges",
            len(entity_ids),
            duration,
            len(all_edges),
        )

        return all_edges

    @connection_retry
    async def save_extraction_result(
        self, post_id: int, result: Any
    ) -> None:
        if GraphRepository._cache_lock is None:
            GraphRepository._cache_lock = asyncio.Lock()
        async with GraphRepository._cache_lock:
            if not GraphRepository._cache_warmed:
                await self.warm_up_cache()
        filtered_entities = []
        discarded_ids = set()
        for entity in result.entities:
            eid_clean = entity.id.lower()
            for prefix in ("place_", "actor_", "topic_", "event_", "entity_"):
                if eid_clean.startswith(prefix):
                    eid_clean = eid_clean[len(prefix):]
                    break
            if (
                eid_clean in self._JUNK_IDS_AND_NAMES
                or entity.name.lower().strip() in self._JUNK_IDS_AND_NAMES
                or len(entity.name.strip()) < 2
            ):
                discarded_ids.add(entity.id)
                continue
            filtered_entities.append(entity)
        result.entities = filtered_entities
        if discarded_ids:
            result.relations = [
                r for r in result.relations
                if r.source_id not in discarded_ids
                and r.target_id not in discarded_ids
            ]

        entities_to_upsert = [
            e for e in result.entities
            if e.id not in GraphRepository._verified_entities
        ]

        unverified_ids: set[str] = set()
        for entity in entities_to_upsert:
            unverified_ids.add(entity.id)
        for relation in result.relations:
            if relation.source_id not in GraphRepository._verified_entities:
                unverified_ids.add(relation.source_id)
            if relation.target_id not in GraphRepository._verified_entities:
                unverified_ids.add(relation.target_id)

        sorted_unverified_ids = sorted(unverified_ids)
        entity_id_to_label: dict[str, str] = {}

        async with self.async_session() as session:
            async with session.begin():
                if sorted_unverified_ids:
                    await session.execute(
                        text("SELECT pg_catalog.pg_advisory_xact_lock(hashtext(val)::bigint) FROM unnest(CAST(:vals AS text[])) AS val"),
                        {"vals": sorted_unverified_ids}
                    )

                for entity in entities_to_upsert:
                    entity_id_to_label[entity.id] = entity.label
                    props: dict[str, Any] = {
                        "id": entity.id,
                        "name": entity.name,
                        **entity.get_property_dict(),
                    }
                    await self.upsert_graph_node(
                        label=entity.label,
                        properties=props,
                        merge_key="id",
                        session=session,
                    )

                for relation in result.relations:
                    start_label = entity_id_to_label.get(relation.source_id, "Entity")
                    end_label = entity_id_to_label.get(relation.target_id, "Entity")
                    await self.upsert_graph_edge(
                        start_label=start_label,
                        start_merge_key="id",
                        start_merge_val=relation.source_id,
                        edge_label=relation.relation_type,
                        end_label=end_label,
                        end_merge_key="id",
                        end_merge_val=relation.target_id,
                        edge_properties=relation.get_property_dict(),
                        session=session,
                    )

        GraphRepository._verified_entities.update(sorted_unverified_ids)

    async def execute_cypher(self, query: str) -> Any:
        result = await self._execute_in_transaction(text(query))
        return result.scalars().all()
