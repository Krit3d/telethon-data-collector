from __future__ import annotations

from typing import Any, LiteralString, cast

import neo4j
from neo4j import AsyncGraphDatabase, AsyncManagedTransaction, AsyncDriver, AsyncResult

from src.config.config import Settings


class Neo4jClient:

    _driver: AsyncDriver | None = None

    def __init__(self, settings: Settings) -> None:
        self._url: str = settings.neo4j_url
        self._user: str = settings.neo4j_user
        self._password: str = settings.neo4j_password
        self._database: str = settings.neo4j_database
        self._max_pool_size: int = settings.neo4j_max_connection_pool_size

    async def connect(self) -> None:
        if self._driver is not None:
            return
        driver = AsyncGraphDatabase.driver(
            self._url,
            auth=(self._user, self._password),
            max_connection_pool_size=self._max_pool_size,
        )
        await driver.verify_connectivity()
        self._driver = driver

    async def close(self) -> None:
        if self._driver is None:
            return
        await self._driver.close()
        self._driver = None

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
        return self._driver

    async def execute_read(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        async def _read(tx: AsyncManagedTransaction) -> list[dict[str, Any]]:
            result: AsyncResult = await tx.run(cast(LiteralString, query), parameters or {})
            return [dict(record) async for record in result]

        async with self.driver.session(database=self._database, default_access_mode=neo4j.READ_ACCESS) as session:
            return await session.execute_read(_read)

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        async def _write(tx: AsyncManagedTransaction) -> list[dict[str, Any]]:
            result: AsyncResult = await tx.run(cast(LiteralString, query), parameters or {})
            return [dict(record) async for record in result]

        async with self.driver.session(database=self._database, default_access_mode=neo4j.WRITE_ACCESS) as session:
            return await session.execute_write(_write)

    async def execute_batch_unwind(
        self,
        query: str,
        batch: list[dict[str, Any]],
    ) -> None:
        async def _batch(tx: AsyncManagedTransaction) -> None:
            await tx.run(cast(LiteralString, query), {"rows": batch})

        async with self.driver.session(database=self._database, default_access_mode=neo4j.WRITE_ACCESS) as session:
            await session.execute_write(_batch)