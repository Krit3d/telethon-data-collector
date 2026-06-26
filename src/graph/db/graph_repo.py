import asyncio
import json
import logging
import random
import re
import time
from typing import Any, ClassVar

from sqlalchemy import text
from sqlalchemy.exc import InternalError, OperationalError, DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from src.config.config import Settings
from src.graph.db.helpers import ID_PREFIX_TO_LABEL, parse_agtype, connection_retry

logger = logging.getLogger(__name__)


class GraphRepository:

    _SUBGRAPH_QUERY_TIMEOUT: float = 15.0
    _write_semaphore: ClassVar[asyncio.Semaphore | None] = None

    def __init__(
        self,
        async_session: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        self.async_session = async_session
        self.settings = settings
        self.graph_name: str = settings.graph_name
        self._subgraph_query_timeout = self._SUBGRAPH_QUERY_TIMEOUT
        if GraphRepository._write_semaphore is None:
            write_concurrency = getattr(settings, 'graph_write_concurrency', 6)
            GraphRepository._write_semaphore = asyncio.Semaphore(write_concurrency)

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
        # Strip surrogate characters that are invalid in UTF-8
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

            # Serialize lists and dicts to JSON strings for Apache AGE compatibility
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
        params_dict: dict[str, Any] = {"merge_val": merge_val}
        set_params: dict[str, Any] = {}
        for key, value in properties.items():
            if key == merge_key or value is None:
                continue
            set_params[f"prop_{key}"] = value
        params_dict.update(set_params)
        params = json.dumps(params_dict)

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
        for key in set_params:
            set_clauses.append(f"n.`{key.replace('prop_', '')}` = ${key}")
        set_clause_str = ", ".join(set_clauses)

        create_query = text(f"""
            SELECT * FROM cypher('{graph_name}',
                $$ CREATE (n:{label} {{{merge_key}: $merge_val
                   {',' + set_clause_str if set_clause_str else ''}}})
                   RETURN n $$,
                CAST(:params AS agtype)
            ) AS (v agtype)
        """)
        await self._execute_in_transaction(
            create_query, {"params": params}, session=session
        )

    async def upsert_graph_node(
        self, label: str, properties: dict, merge_key: str = "id",
        session: AsyncSession | None = None,
    ) -> None:
        if not re.match(r"^[A-Za-z0-9_]+$", label):
            raise ValueError(
                f"Invalid label '{label}': must be alphanumeric with underscores"
            )

        if not re.match(r"^[A-Za-z0-9_]+$", merge_key):
            raise ValueError(
                f"Invalid merge_key '{merge_key}': must be alphanumeric with underscores"
            )

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

        for attempt in range(5):
            try:
                if session is not None:
                    async with session.begin_nested():
                        await self._execute_in_transaction(query, {"params": params}, session=session)
                else:
                    await self._execute_in_transaction(query, {"params": params}, session=session)
                break
            except (InternalError, OperationalError, DBAPIError, IntegrityError) as exc:
                err_msg = str(exc)
                if any(x in err_msg.lower() for x in ("entity failed to be updated", "concurrently updated", "lock timeout", "deadlock", "tuple concurrently updated", "duplicate key value", "unique constraint")):
                    if attempt < 4:
                        await asyncio.sleep((2 ** attempt) * 0.1 + random.uniform(0.05, 0.2))
                        continue
                    merge_val = props.get(merge_key)
                    logger.warning(
                        "AGE entity update failed for (%s, %s=%s): %s. "
                        "Attempting DETACH DELETE + CREATE fallback.",
                        label, merge_key, merge_val, exc,
                    )
                    if session is not None:
                        async with session.begin_nested():
                            await self._delete_and_create_node(
                                label, props, merge_key, merge_val, session
                            )
                    else:
                        async with self.async_session() as fallback_session:
                            async with fallback_session.begin():
                                await self._delete_and_create_node(
                                    label, props, merge_key, merge_val, fallback_session
                                )
                    logger.info(
                        "Fallback DETACH DELETE + CREATE succeeded for (%s, %s=%s)",
                        label, merge_key, merge_val,
                    )
                    break
                raise

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

        for attempt in range(5):
            try:
                if session is not None:
                    async with session.begin_nested():
                        await self._execute_in_transaction(query, {"params": params}, session=session)
                else:
                    await self._execute_in_transaction(query, {"params": params}, session=session)
                break
            except (InternalError, OperationalError, DBAPIError, IntegrityError) as exc:
                err_msg = str(exc)
                if any(x in err_msg.lower() for x in ("entity failed to be updated", "concurrently updated", "lock timeout", "deadlock", "tuple concurrently updated", "duplicate key value", "unique constraint")):
                    if attempt < 4:
                        await asyncio.sleep((2 ** attempt) * 0.1 + random.uniform(0.05, 0.2))
                        continue
                    raise
                raise

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
        assert GraphRepository._write_semaphore is not None
        async with GraphRepository._write_semaphore:
            async with self.async_session() as session:
                async with session.begin():
                    entity_id_to_label: dict[str, str] = {}
                    sorted_entities = sorted(result.entities, key=lambda e: e.id)
                    for entity in sorted_entities:
                        entity_id_to_label[entity.id] = entity.label
                        try:
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
                            logger.debug(
                                "Upserted entity: label=%s, id=%s",
                                entity.label,
                                entity.id,
                            )
                        except Exception as exc:
                            logger.error(
                                "Failed to upsert entity (post_id=%d, label=%s, id=%s): %s",
                                post_id,
                                entity.label,
                                entity.id,
                                exc,
                            )
                            raise

                    pub_node_id: str | None = None
                    for entity in sorted_entities:
                        if entity.id.startswith("event_publication_"):
                            pub_node_id = entity.id
                            break

                    if pub_node_id is not None:
                        pub_label = entity_id_to_label.get(pub_node_id, "Event")
                        sorted_mention_targets = sorted(
                            [e for e in sorted_entities if e.id != pub_node_id],
                            key=lambda e: e.id,
                        )
                        for entity in sorted_mention_targets:
                            try:
                                await self.upsert_graph_edge(
                                    start_label=pub_label,
                                    start_merge_key="id",
                                    start_merge_val=pub_node_id,
                                    edge_label="MENTIONS",
                                    end_label=entity.label,
                                    end_merge_key="id",
                                    end_merge_val=entity.id,
                                    edge_properties={},
                                    session=session,
                                )
                                logger.debug(
                                    "Created MENTIONS edge: %s -[:MENTIONS]-> %s(%s)",
                                    pub_node_id,
                                    entity.label,
                                    entity.id,
                                )
                            except Exception as exc:
                                logger.error(
                                    "Failed MENTIONS edge (post_id=%d, entity_id=%s): %s",
                                    post_id,
                                    entity.id,
                                    exc,
                                )

                    sorted_relations = sorted(result.relations, key=lambda r: (r.source_id, r.relation_type, r.target_id))
                    for relation in sorted_relations:
                        start_label = entity_id_to_label.get(relation.source_id, "Entity")
                        end_label = entity_id_to_label.get(relation.target_id, "Entity")
                        try:
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
                            logger.debug(
                                "Upserted relation: %s(%s)-[%s]->%s(%s)",
                                start_label,
                                relation.source_id,
                                relation.relation_type,
                                end_label,
                                relation.target_id,
                            )
                        except Exception as exc:
                            logger.error(
                                "Failed relation upsert (post_id=%d, type=%s): %s",
                                post_id,
                                relation.relation_type,
                                exc,
                            )
                            raise

    async def execute_cypher(self, query: str) -> Any:
        result = await self._execute_in_transaction(text(query))
        return result.scalars().all()
