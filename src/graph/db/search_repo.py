import json
import logging

from sqlalchemy import text
from sqlalchemy.engine import Result
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from src.config.config import Settings
from src.db.database import with_retry_on_deadlock

logger = logging.getLogger(__name__)


def _clean_ag_string(val: object) -> str:
    if val is None:
        return ""
    s_val = str(val)
    if s_val.startswith('"') and s_val.endswith('"'):
        return s_val[1:-1]
    if s_val.startswith("'") and s_val.endswith("'"):
        return s_val[1:-1]
    return s_val


def _parse_ag_float(val: object) -> float:
    if val is None:
        return 0.0
    try:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            s = val.strip()
            if s.startswith('"') and s.endswith('"'):
                s = s[1:-1]
            if s.startswith("'") and s.endswith("'"):
                s = s[1:-1]
            return float(s) if s else 0.0
        return 0.0
    except (ValueError, TypeError):
        return 0.0


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

    @with_retry_on_deadlock()
    async def search_posts_by_topics(self, topics: list[str]) -> list[int]:
        if not topics:
            return []

        normalized_topics = [t.lower().strip() for t in topics if t]
        if not normalized_topics:
            return []

        graph_name = self._resolve_graph_name()
        params = json.dumps({"topics": normalized_topics})

        query = text(r"""
            SELECT * FROM ag_catalog.cypher('{graph_name}', $$
                MATCH (p\:Event)-[r]->(t\:Entity)
                WHERE (type(r) = 'ABOUT' OR type(r) = 'MENTIONS') AND t.name_lower IN $topics
                RETURN p.db_post_id, COALESCE(p.engagement_rate, 0.0) AS er
                ORDER BY er DESC
                LIMIT 150
            $$, CAST(:params AS agtype))
            AS (db_post_id agtype, engagement_rate agtype)
        """.format(graph_name=graph_name))

        async with self.async_session() as session:
            result: Result = await session.execute(query, {"params": params})
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

    @with_retry_on_deadlock()
    async def search_posts_by_entities(
        self, entity_ids: list[str]
    ) -> tuple[dict[int, list[str]], dict[int, float]]:
        if not entity_ids:
            return {}, {}

        graph_name = self._resolve_graph_name()
        params = json.dumps({"ids": entity_ids})

        query = text(
            f"SELECT * FROM ag_catalog.cypher('{graph_name}', $$ "
            f"MATCH (p\\:Event)-[r]->(e) "
            f"WHERE e.id IN $ids AND p.db_post_id IS NOT NULL "
            f"RETURN p.db_post_id, e.id, p.engagement_rate"
            f" $$, CAST(:params AS agtype)) "
            f"AS (db_post_id agtype, entity_id agtype, engagement_rate agtype)"
        )

        async with self.async_session() as session:
            result: Result = await session.execute(query, {"params": params})
            matched: dict[int, list[str]] = {}
            ers: dict[int, float] = {}
            for row in result:
                raw_db_post_id = row[0]
                raw_entity_id = row[1]
                raw_er = row[2]
                if raw_db_post_id is not None and raw_entity_id is not None:
                    try:
                        db_post_id = int(raw_db_post_id)
                        entity_id = _clean_ag_string(raw_entity_id)
                        if db_post_id not in matched:
                            matched[db_post_id] = []
                        matched[db_post_id].append(entity_id)
                        ers[db_post_id] = _parse_ag_float(raw_er)
                    except (ValueError, TypeError):
                        logger.warning(
                            "Could not parse search_posts_by_entities row: db_post_id=%s entity_id=%s",
                            raw_db_post_id, raw_entity_id,
                        )
            return matched, ers

    @with_retry_on_deadlock()
    async def fetch_subgraph_edges(self, label_to_ids: dict[str, list[str]]) -> list[dict]:
        if not label_to_ids:
            return []

        graph_name = self._resolve_graph_name()

        union_parts: list[str] = []
        params_dict: dict[str, list[str]] = {}

        for label, ids in label_to_ids.items():
            param_key = f"ids_{label}"
            union_parts.append(
                f"MATCH (a\\:{label})-[r]->(b) "
                f"WHERE a.id IN ${param_key} "
                f"RETURN "
                f"  a.id AS source_id, label(a) AS source_label, a.name AS source_name, "
                f"  type(r) AS relation_type, "
                f"  b.id AS target_id, label(b) AS target_label, b.name AS target_name"
            )
            union_parts.append(
                f"MATCH (a\\:{label})<-[r]-(b) "
                f"WHERE a.id IN ${param_key} "
                f"RETURN "
                f"  b.id AS source_id, label(b) AS source_label, b.name AS source_name, "
                f"  type(r) AS relation_type, "
                f"  a.id AS target_id, label(a) AS target_label, a.name AS target_name"
            )
            params_dict[param_key] = ids

        cypher_query = " UNION ".join(union_parts)
        params_json = json.dumps(params_dict)

        query = text(
            f"SELECT * FROM ag_catalog.cypher('{graph_name}', $$ {cypher_query} $$, CAST(:params AS agtype)) "
            f"AS ("
            f"  source_id agtype, source_label agtype, source_name agtype, "
            f"  relation_type agtype, "
            f"  target_id agtype, target_label agtype, target_name agtype"
            f")"
        )

        async with self.async_session() as session:
            result: Result = await session.execute(query, {"params": params_json})
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

    @with_retry_on_deadlock()
    async def fetch_nodes_by_ids(self, label_to_ids: dict[str, list[str]]) -> list[dict]:
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
            result: Result = await session.execute(query, {"params": params_json})
            nodes: list[dict] = []
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
