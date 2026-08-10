from __future__ import annotations

import asyncio
import random
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import SessionExpired, TransientError

from src.config.config import Settings
from src.graph.ontology import EntityType, RelationType


class Neo4jClient:

    _driver: AsyncDriver | None = None

    def __init__(self, settings: Settings) -> None:
        self._url = settings.neo4j_url
        self._user = settings.neo4j_user
        self._password = settings.neo4j_password
        self._database = settings.neo4j_database
        self._max_pool_size = settings.neo4j_max_connection_pool_size

    async def connect(self) -> None:
        if self._driver is not None:
            return
        driver = AsyncGraphDatabase.driver(
            self._url,
            auth=(self._user, self._password),
            max_connection_pool_size=self._max_pool_size,
            max_connection_lifetime=3600,
            keep_alive=True,
        )
        await driver.verify_connectivity()
        self._driver = driver

    async def close(self) -> None:
        if self._driver is None:
            return
        await self._driver.close()
        self._driver = None

    async def __aenter__(self) -> Neo4jClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        await self.close()

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
        return self._driver

    @staticmethod
    def _resolve_label(label: EntityType | str) -> str:
        if isinstance(label, EntityType):
            return label.value
        if label.isidentifier():
            return label
        raise ValueError(f"Invalid Cypher identifier: {label}")

    @staticmethod
    def _resolve_rel_type(rel_type: RelationType | str) -> str:
        if isinstance(rel_type, RelationType):
            return rel_type.value
        if rel_type.isidentifier():
            return rel_type
        raise ValueError(f"Invalid Cypher identifier: {rel_type}")

    async def _execute_transaction(
        self,
        access_mode: str,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        last_exc: Exception | None = None

        async def _run(tx: Any) -> list[dict[str, Any]]:
            result = await tx.run(query, parameters or {})
            return [dict(record) async for record in result]

        for attempt in range(5):
            try:
                async with self.driver.session(
                    database=self._database,
                    default_access_mode=access_mode,
                ) as session:
                    if access_mode == "read":
                        return await session.execute_read(_run)
                    return await session.execute_write(_run)
            except (TransientError, SessionExpired) as exc:
                last_exc = exc
                if attempt < 4:
                    delay = (0.1 * (2 ** attempt)) + random.uniform(0.05, 0.25)
                    await asyncio.sleep(delay)

        raise last_exc  # type: ignore[misc]

    async def execute_read(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._execute_transaction("read", query, parameters)

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._execute_transaction("write", query, parameters)

    async def lookup_existing_ids(self, node_ids: list[str], batch_size: int = 1000) -> set[str]:
        if not node_ids:
            return set()
        result: set[str] = set()
        for i in range(0, len(node_ids), batch_size):
            batch = node_ids[i:i + batch_size]
            rows = await self.execute_read(
                "MATCH (n) WHERE n.id IN $ids RETURN DISTINCT n.id AS id",
                {"ids": batch},
            )
            result.update(row["id"] for row in rows)
        return result

    async def batch_merge_nodes(self, label: EntityType | str, nodes: list[dict[str, Any]], batch_size: int = 500) -> None:
        if not nodes:
            return
        label_str = self._resolve_label(label)
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            await self.execute_write(
                f"UNWIND $batch AS row MERGE (n:{label_str} {{id: row.id}}) SET n += row",
                {"batch": batch},
            )

    async def batch_merge_relations(
        self,
        source_label: EntityType | str,
        target_label: EntityType | str,
        rel_type: RelationType | str,
        relations: list[dict[str, Any]],
        batch_size: int = 500,
    ) -> None:
        if not relations:
            return
        src_label_str = self._resolve_label(source_label)
        tgt_label_str = self._resolve_label(target_label)
        rel_type_str = self._resolve_rel_type(rel_type)
        for i in range(0, len(relations), batch_size):
            batch = relations[i:i + batch_size]
            normalized = [
                {"source_id": r["source_id"], "target_id": r["target_id"], "properties": r.get("properties", {})}
                for r in batch
            ]
            await self.execute_write(
                f"UNWIND $batch AS row "
                f"MATCH (s:{src_label_str} {{id: row.source_id}}), "
                f"(t:{tgt_label_str} {{id: row.target_id}}) "
                f"MERGE (s)-[r:{rel_type_str}]->(t) SET r += row.properties",
                {"batch": normalized},
            )

    async def health_check(self) -> bool:
        try:
            await self.execute_read("RETURN 1 AS ok")
            return True
        except Exception:
            return False