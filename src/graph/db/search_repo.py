import json
import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from src.config.config import Settings

logger = logging.getLogger(__name__)


def _clean_ag_string(val: Any) -> str:
    if val is None:
        return ""
    s_val = str(val)
    if s_val.startswith('"') and s_val.endswith('"'):
        return s_val[1:-1]
    if s_val.startswith("'") and s_val.endswith("'"):
        return s_val[1:-1]
    return s_val


class GraphSearchRepository:

    def __init__(
        self,
        async_session: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        self.async_session = async_session
        self.settings = settings
        self.graph_name: str = settings.graph_name

    def _resolve_graph_name(self) -> str:
        name = self.graph_name
        if not name.isidentifier():
            raise ValueError(
                f"Invalid graph name '{name}': must be a valid identifier"
            )
        return name

    async def search_posts_by_topics(self, topics: list[str]) -> list[int]:
        if not topics:
            return []

        graph_name = self._resolve_graph_name()
        params = json.dumps({"topics": topics})

        query = text(r"""
            SELECT * FROM ag_catalog.cypher('{graph_name}', $$
                MATCH (p\:Event)-[r]->(t\:Entity)
                WHERE (type(r) = 'ABOUT' OR type(r) = 'MENTIONS') AND t.name IN $topics
                RETURN p.db_post_id, COALESCE(p.engagement_rate, 0.0) AS er
                ORDER BY er DESC
                LIMIT 150
            $$, CAST(:params AS agtype))
            AS (db_post_id agtype, engagement_rate agtype)
        """.format(graph_name=graph_name))

        async with self.async_session() as session:
            result = await session.execute(query, {"params": params})
            seen: set[int] = set()
            post_ids: list[int] = []
            for row in result:
                raw = row[0]
                if raw is not None:
                    try:
                        pid = int(raw)
                        if pid not in seen:
                            seen.add(pid)
                            post_ids.append(pid)
                    except (ValueError, TypeError):
                        logger.warning("Could not parse db_post_id value: %s", raw)
            return post_ids

    async def search_posts_by_entities(self, label_to_ids: dict[str, list[str]]) -> dict[int, list[str]]:
        if not label_to_ids:
            return {}

        graph_name = self._resolve_graph_name()

        union_parts: list[str] = []
        params_dict: dict[str, list[str]] = {}

        for label, ids in label_to_ids.items():
            param_key = f"ids_{label}"
            union_parts.append(
                f"MATCH (p\\:Event)-[r]->(e\\:{label}) "
                f"WHERE e.id IN ${param_key} AND p.db_post_id IS NOT NULL "
                f"RETURN p.db_post_id, e.id"
            )
            params_dict[param_key] = ids

        cypher_query = " UNION ".join(union_parts)
        params_json = json.dumps(params_dict)

        query = text(
            f"SELECT * FROM ag_catalog.cypher('{graph_name}', $$ {cypher_query} $$, CAST(:params AS agtype)) "
            f"AS (db_post_id agtype, entity_id agtype)"
        )

        async with self.async_session() as session:
            result = await session.execute(query, {"params": params_json})
            matched: dict[int, list[str]] = {}
            for row in result:
                raw_db_post_id = row[0]
                raw_entity_id = row[1]
                if raw_db_post_id is not None and raw_entity_id is not None:
                    try:
                        db_post_id = int(raw_db_post_id)
                        entity_id = _clean_ag_string(raw_entity_id)
                        if db_post_id not in matched:
                            matched[db_post_id] = []
                        matched[db_post_id].append(entity_id)
                    except (ValueError, TypeError):
                        logger.warning(
                            "Could not parse search_posts_by_entities row: db_post_id=%s entity_id=%s",
                            raw_db_post_id, raw_entity_id,
                        )
            return matched

    async def fetch_subgraph_edges(self, entity_ids: list[str]) -> list[dict]:
        if not entity_ids:
            return []

        graph_name = self._resolve_graph_name()
        params = json.dumps({"e_ids": entity_ids})

        query = text(r"""
            SELECT * FROM ag_catalog.cypher('{graph_name}', $$
                MATCH (a\:Entity)-[r]->(b)
                WHERE a.id IN $e_ids
                RETURN
                    a.id AS source_id,
                    label(a) AS source_label,
                    a.name AS source_name,
                    type(r) AS relation_type,
                    b.id AS target_id,
                    label(b) AS target_label,
                    b.name AS target_name
                UNION
                MATCH (a\:Entity)<-[r]-(b)
                WHERE a.id IN $e_ids
                RETURN
                    b.id AS source_id,
                    label(b) AS source_label,
                    b.name AS source_name,
                    type(r) AS relation_type,
                    a.id AS target_id,
                    label(a) AS target_label,
                    a.name AS target_name
            $$, CAST(:params AS agtype))
            AS (
                source_id agtype, source_label agtype, source_name agtype,
                relation_type agtype,
                target_id agtype, target_label agtype, target_name agtype
            )
        """.format(graph_name=graph_name))

        async with self.async_session() as session:
            result = await session.execute(query, {"params": params})
            edges: list[dict] = []
            for row in result:
                try:
                    edges.append({
                        "source_id": _clean_ag_string(row[0]),
                        "source_label": _clean_ag_string(row[1]),
                        "source_name": _clean_ag_string(row[2]),
                        "relation_type": _clean_ag_string(row[3]),
                        "target_id": _clean_ag_string(row[4]),
                        "target_label": _clean_ag_string(row[5]),
                        "target_name": _clean_ag_string(row[6]),
                    })
                except (IndexError, TypeError):
                    continue
            return edges

    async def fetch_nodes_by_ids(self, label_to_ids: dict[str, list[str]]) -> list[dict[str, Any]]:
        if not label_to_ids:
            return []

        graph_name = self._resolve_graph_name()

        union_parts: list[str] = []
        params_dict: dict[str, list[str]] = {}

        for label, ids in label_to_ids.items():
            param_key = f"ids_{label}"
            union_parts.append(
                f"MATCH (n\\:{label}) "
                f"WHERE n.id IN ${param_key} "
                f"RETURN n.id, label(n), n.name, properties(n)"
            )
            params_dict[param_key] = ids

        cypher_query = " UNION ".join(union_parts)
        params_json = json.dumps(params_dict)

        query = text(
            f"SELECT * FROM ag_catalog.cypher('{graph_name}', $$ {cypher_query} $$, CAST(:params AS agtype)) "
            f"AS (id agtype, label agtype, name agtype, properties agtype)"
        )

        async with self.async_session() as session:
            result = await session.execute(query, {"params": params_json})
            nodes: list[dict[str, Any]] = []
            for row in result:
                try:
                    raw_props = row[3]
                    props: dict = {}
                    if raw_props is not None:
                        try:
                            if isinstance(raw_props, str):
                                props = json.loads(raw_props)
                            elif isinstance(raw_props, dict):
                                props = raw_props
                        except (json.JSONDecodeError, TypeError):
                            props = {}
                    nodes.append({
                        "id": _clean_ag_string(row[0]),
                        "label": _clean_ag_string(row[1]),
                        "name": _clean_ag_string(row[2]),
                        "properties": props,
                    })
                except (IndexError, TypeError):
                    continue
            return nodes
