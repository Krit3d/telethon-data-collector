import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import cast

from sqlalchemy import text
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config.config import load_settings
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

BATCH_SIZE = 50000
PROGRESS_INTERVAL = 2


async def get_vertex_labels(session: AsyncSession) -> list[str]:
    result = await session.execute(text("""
        SELECT DISTINCT l.name
        FROM ag_catalog.ag_label l
        WHERE l.kind = 'v'
          AND NOT (l.name ILIKE '%Publication%' OR l.name ILIKE '%Event%')
          AND to_regclass('social_graph."' || l.name || '"') IS NOT NULL
        ORDER BY l.name
    """))
    return [row[0] for row in result]


async def get_label_vertex_count(session: AsyncSession, label_name: str) -> int:
    try:
        result = await session.execute(
            text(f"SELECT COUNT(*) FROM social_graph.\"{label_name}\"")
        )
        return result.scalar() or 0
    except (ProgrammingError, Exception):
        logger.warning("Table social_graph.%s is missing or unreadable, returning 0", label_name)
        return 0


async def insert_1hop(session: AsyncSession, label_name: str, limit: int, last_id: str | None = None) -> tuple[int, str | None]:
    await session.execute(text("SET statement_timeout = '300s';"))
    max_id_result = await session.execute(
        text(f"""
            SELECT sub.next_last_id FROM (
                SELECT v.id::text AS next_last_id
                FROM social_graph."{label_name}" v
                WHERE (CAST(:last_id AS text) IS NULL OR v.id::text > CAST(:last_id AS text))
                ORDER BY v.id::text ASC
                LIMIT :limit
            ) sub
            ORDER BY sub.next_last_id DESC
            LIMIT 1
        """),
        {"last_id": last_id, "limit": limit},
    )
    next_last_id = max_id_result.scalar()
    if next_last_id is None:
        return (0, None)
    result = cast(CursorResult, await session.execute(
        text(f"""
            WITH vertex_subset AS (
                SELECT id, properties, tableoid
                FROM social_graph."{label_name}"
                WHERE (CAST(:last_id AS text) IS NULL OR id::text > CAST(:last_id AS text))
                ORDER BY id::text ASC
                LIMIT :limit
            ),
            raw_1hop AS (
                SELECT entity_name_lower, post_id, entity_type,
                    CASE
                        WHEN entity_type = 'Actor' THEN 1.15
                        WHEN entity_type = 'Entity' THEN 1.10
                        WHEN entity_type = 'Topic' THEN 1.00
                        WHEN entity_type = 'Place' THEN 0.95
                        WHEN entity_type = 'Event' THEN 0.95
                    END AS weight
                FROM (
                    SELECT
                        LOWER(TRIM(COALESCE((v.properties::text::jsonb)->>'name_lower', (v.properties::text::jsonb)->>'name'))) AS entity_name_lower,
                        CAST(COALESCE(
                            (e.properties::text::jsonb)->>'db_post_id',
                            (v2.properties::text::jsonb)->>'db_post_id',
                            (v.properties::text::jsonb)->>'db_post_id'
                        ) AS BIGINT) AS post_id,
                        CASE
                            WHEN l.name ILIKE '%Actor%' OR l.name ILIKE '%Author%' OR l.name ILIKE '%Person%' OR l.name ILIKE '%Brand%' OR l.name ILIKE '%Organization%' THEN 'Actor'
                            WHEN l.name ILIKE '%Topic%' OR l.name ILIKE '%Hashtag%' THEN 'Topic'
                            WHEN l.name ILIKE '%Place%' OR l.name ILIKE '%Geo%' THEN 'Place'
                            WHEN l.name ILIKE '%Event%' OR l.name ILIKE '%Publication%' THEN 'Event'
                            ELSE 'Entity'
                        END AS entity_type
                    FROM vertex_subset v
                    JOIN ag_catalog.ag_label l ON l.relation = v.tableoid
                    JOIN social_graph._ag_label_edge e ON e.start_id = v.id
                    JOIN social_graph._ag_label_vertex v2 ON v2.id = e.end_id
                    WHERE COALESCE((v.properties::text::jsonb)->>'name', (v.properties::text::jsonb)->>'name_lower', (v.properties::text::jsonb)->>'title') IS NOT NULL
                      AND COALESCE(
                          (e.properties::text::jsonb)->>'db_post_id',
                          (v2.properties::text::jsonb)->>'db_post_id',
                          (v.properties::text::jsonb)->>'db_post_id'
                      ) ~ '^[0-9]+$'
                      AND NOT (l.name ILIKE '%Event%' AND (COALESCE((v.properties::text::jsonb)->>'name', (v.properties::text::jsonb)->>'name_lower') ~ '^[0-9_]+$' OR (v.properties::text::jsonb)->>'original_id' LIKE 'event_publication_%'))

                    UNION ALL

                    SELECT
                        LOWER(TRIM(COALESCE((v.properties::text::jsonb)->>'name_lower', (v.properties::text::jsonb)->>'name'))) AS entity_name_lower,
                        CAST(COALESCE(
                            (e.properties::text::jsonb)->>'db_post_id',
                            (v2.properties::text::jsonb)->>'db_post_id',
                            (v.properties::text::jsonb)->>'db_post_id'
                        ) AS BIGINT) AS post_id,
                        CASE
                            WHEN l.name ILIKE '%Actor%' OR l.name ILIKE '%Author%' OR l.name ILIKE '%Person%' OR l.name ILIKE '%Brand%' OR l.name ILIKE '%Organization%' THEN 'Actor'
                            WHEN l.name ILIKE '%Topic%' OR l.name ILIKE '%Hashtag%' THEN 'Topic'
                            WHEN l.name ILIKE '%Place%' OR l.name ILIKE '%Geo%' THEN 'Place'
                            WHEN l.name ILIKE '%Event%' OR l.name ILIKE '%Publication%' THEN 'Event'
                            ELSE 'Entity'
                        END AS entity_type
                    FROM vertex_subset v
                    JOIN ag_catalog.ag_label l ON l.relation = v.tableoid
                    JOIN social_graph._ag_label_edge e ON e.end_id = v.id
                    JOIN social_graph._ag_label_vertex v2 ON v2.id = e.start_id
                    WHERE COALESCE((v.properties::text::jsonb)->>'name', (v.properties::text::jsonb)->>'name_lower', (v.properties::text::jsonb)->>'title') IS NOT NULL
                      AND COALESCE(
                          (e.properties::text::jsonb)->>'db_post_id',
                          (v2.properties::text::jsonb)->>'db_post_id',
                          (v.properties::text::jsonb)->>'db_post_id'
                      ) ~ '^[0-9]+$'
                      AND NOT (l.name ILIKE '%Event%' AND (COALESCE((v.properties::text::jsonb)->>'name', (v.properties::text::jsonb)->>'name_lower') ~ '^[0-9_]+$' OR (v.properties::text::jsonb)->>'original_id' LIKE 'event_publication_%'))
                ) AS inner_1hop
            )
            INSERT INTO public.graph_entity_posts (entity_name_lower, post_id, distance, weight, entity_type)
            SELECT entity_name_lower, post_id, 1 AS distance, MAX(weight) AS weight, MAX(entity_type) AS entity_type
            FROM raw_1hop
            GROUP BY entity_name_lower, post_id
            ON CONFLICT (entity_name_lower, post_id) DO UPDATE
            SET distance = LEAST(public.graph_entity_posts.distance, EXCLUDED.distance),
                weight = GREATEST(public.graph_entity_posts.weight, EXCLUDED.weight),
                entity_type = CASE
                    WHEN EXCLUDED.weight >= public.graph_entity_posts.weight THEN EXCLUDED.entity_type
                    ELSE public.graph_entity_posts.entity_type
                END
        """),
        {"limit": limit, "last_id": last_id},
    ))
    await session.commit()
    return (result.rowcount or 0, next_last_id)


async def insert_2hop(session: AsyncSession, label_name: str, limit: int, last_id: str | None = None) -> tuple[int, str | None]:
    await session.execute(text("SET statement_timeout = '300s';"))
    max_id_result = await session.execute(
        text(f"""
            SELECT sub.next_last_id FROM (
                SELECT v.id::text AS next_last_id
                FROM social_graph."{label_name}" v
                WHERE (CAST(:last_id AS text) IS NULL OR v.id::text > CAST(:last_id AS text))
                ORDER BY v.id::text ASC
                LIMIT :limit
            ) sub
            ORDER BY sub.next_last_id DESC
            LIMIT 1
        """),
        {"last_id": last_id, "limit": limit},
    )
    next_last_id = max_id_result.scalar()
    if next_last_id is None:
        return (0, None)
    result = cast(CursorResult, await session.execute(
        text(f"""
            INSERT INTO public.graph_entity_posts (entity_name_lower, post_id, distance, weight, entity_type)
            SELECT entity_name_lower, post_id, 2, MAX(weight) AS weight, MAX(entity_type) AS entity_type
            FROM (
                SELECT entity_name_lower, post_id, entity_type,
                    CASE
                        WHEN entity_type = 'Actor' THEN 0.575
                        WHEN entity_type = 'Entity' THEN 0.550
                        WHEN entity_type = 'Topic' THEN 0.500
                        WHEN entity_type = 'Place' THEN 0.475
                        WHEN entity_type = 'Event' THEN 0.475
                    END AS weight
                FROM (
                    SELECT
                        LOWER(TRIM(COALESCE((v1.properties::text::jsonb)->>'name_lower', (v1.properties::text::jsonb)->>'name'))) AS entity_name_lower,
                        gep.post_id AS post_id,
                        CASE
                            WHEN l.name ILIKE '%Actor%' OR l.name ILIKE '%Author%' OR l.name ILIKE '%Person%' OR l.name ILIKE '%Brand%' OR l.name ILIKE '%Organization%' THEN 'Actor'
                            WHEN l.name ILIKE '%Topic%' OR l.name ILIKE '%Hashtag%' THEN 'Topic'
                            WHEN l.name ILIKE '%Place%' OR l.name ILIKE '%Geo%' THEN 'Place'
                            WHEN l.name ILIKE '%Event%' OR l.name ILIKE '%Publication%' THEN 'Event'
                            ELSE 'Entity'
                        END AS entity_type
                    FROM (
                        SELECT id, properties, tableoid
                        FROM social_graph."{label_name}"
                        WHERE (CAST(:last_id AS text) IS NULL OR id::text > CAST(:last_id AS text))
                        ORDER BY id::text ASC
                        LIMIT :limit
                    ) v1
                    JOIN ag_catalog.ag_label l ON l.relation = v1.tableoid
                    JOIN social_graph._ag_label_edge e ON e.start_id = v1.id
                    JOIN social_graph._ag_label_vertex v2 ON v2.id = e.end_id
                    JOIN public.graph_entity_posts gep ON gep.entity_name_lower = LOWER(TRIM(COALESCE((v2.properties::text::jsonb)->>'name_lower', (v2.properties::text::jsonb)->>'name')))
                    WHERE COALESCE((v1.properties::text::jsonb)->>'name', (v1.properties::text::jsonb)->>'name_lower', (v1.properties::text::jsonb)->>'title') IS NOT NULL
                      AND gep.distance = 1
                      AND NOT (l.name ILIKE '%Event%' AND (COALESCE((v1.properties::text::jsonb)->>'name', (v1.properties::text::jsonb)->>'name_lower') ~ '^[0-9_]+$' OR (v1.properties::text::jsonb)->>'original_id' LIKE 'event_publication_%'))

                    UNION ALL

                    SELECT
                        LOWER(TRIM(COALESCE((v1.properties::text::jsonb)->>'name_lower', (v1.properties::text::jsonb)->>'name'))) AS entity_name_lower,
                        gep.post_id AS post_id,
                        CASE
                            WHEN l.name ILIKE '%Actor%' OR l.name ILIKE '%Author%' OR l.name ILIKE '%Person%' OR l.name ILIKE '%Brand%' OR l.name ILIKE '%Organization%' THEN 'Actor'
                            WHEN l.name ILIKE '%Topic%' OR l.name ILIKE '%Hashtag%' THEN 'Topic'
                            WHEN l.name ILIKE '%Place%' OR l.name ILIKE '%Geo%' THEN 'Place'
                            WHEN l.name ILIKE '%Event%' OR l.name ILIKE '%Publication%' THEN 'Event'
                            ELSE 'Entity'
                        END AS entity_type
                    FROM (
                        SELECT id, properties, tableoid
                        FROM social_graph."{label_name}"
                        WHERE (CAST(:last_id AS text) IS NULL OR id::text > CAST(:last_id AS text))
                        ORDER BY id::text ASC
                        LIMIT :limit
                    ) v1
                    JOIN ag_catalog.ag_label l ON l.relation = v1.tableoid
                    JOIN social_graph._ag_label_edge e ON e.end_id = v1.id
                    JOIN social_graph._ag_label_vertex v2 ON v2.id = e.start_id
                    JOIN public.graph_entity_posts gep ON gep.entity_name_lower = LOWER(TRIM(COALESCE((v2.properties::text::jsonb)->>'name_lower', (v2.properties::text::jsonb)->>'name')))
                    WHERE COALESCE((v1.properties::text::jsonb)->>'name', (v1.properties::text::jsonb)->>'name_lower', (v1.properties::text::jsonb)->>'title') IS NOT NULL
                      AND gep.distance = 1
                      AND NOT (l.name ILIKE '%Event%' AND (COALESCE((v1.properties::text::jsonb)->>'name', (v1.properties::text::jsonb)->>'name_lower') ~ '^[0-9_]+$' OR (v1.properties::text::jsonb)->>'original_id' LIKE 'event_publication_%'))
                ) AS inner_2hop
            ) AS combined
            GROUP BY entity_name_lower, post_id
            ON CONFLICT (entity_name_lower, post_id) DO UPDATE
            SET distance = LEAST(public.graph_entity_posts.distance, EXCLUDED.distance),
                weight = GREATEST(public.graph_entity_posts.weight, EXCLUDED.weight),
                entity_type = CASE
                    WHEN EXCLUDED.weight >= public.graph_entity_posts.weight THEN EXCLUDED.entity_type
                    ELSE public.graph_entity_posts.entity_type
                END
        """),
        {"limit": limit, "last_id": last_id},
    ))
    await session.commit()
    return (result.rowcount or 0, next_last_id)


async def enrich_author_flags(session: AsyncSession) -> int:
    await session.execute(text("SET statement_timeout = '300s';"))
    result = cast(CursorResult, await session.execute(
        text("""
            UPDATE public.graph_entity_posts gep
            SET is_author_blog = a.is_author_blog
            FROM public.content c
            JOIN public.accounts a ON a.id = c.account_id
            WHERE c.id = gep.post_id
              AND gep.is_author_blog IS NULL
        """)
    ))
    await session.commit()
    return result.rowcount or 0


async def process_phase(
    session: AsyncSession,
    label_names: list[str],
    batch_size: int,
    phase_name: str,
    batch_fn: Callable[[AsyncSession, str, int, str | None], Awaitable[tuple[int, str | None]]],
) -> int:
    total_rows = 0
    phase_start = time.time()

    for label_name in label_names:
        label_start = time.time()
        label_total = await get_label_vertex_count(session, label_name)

        if label_total == 0:
            logger.info("%s label=%s has 0 vertices, skipping", phase_name, label_name)
            continue

        label_rows = 0
        total_batches = 0
        current_last_id: str | None = None
        vertices_processed = 0

        while True:
            batch_start = time.time()

            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    rows, next_last_id = await batch_fn(session, label_name, batch_size, current_last_id)
                    break
                except Exception as exc:
                    logger.exception("%s label=%s last_id=%s attempt=%d failed, rolling back", phase_name, label_name, current_last_id, attempt + 1)
                    await session.rollback()
                    await session.execute(text("SET statement_timeout = '300s';"))
                    last_exc = exc
            else:
                raise last_exc  # type: ignore[misc]

            label_rows += rows
            total_rows += rows
            total_batches += 1
            vertices_processed += batch_size

            if next_last_id is None or next_last_id == current_last_id:
                logger.info(
                    "%s label=%s completed: next_last_id=%s rows=%d total_batches=%d",
                    phase_name, label_name, next_last_id, label_rows, total_batches,
                )
                break

            if rows == 0 and vertices_processed >= label_total:
                logger.info(
                    "%s label=%s completed: no rows returned, processed=%d total=%d",
                    phase_name, label_name, vertices_processed, label_total,
                )
                break

            current_last_id = next_last_id
            batch_duration = time.time() - batch_start

            if total_batches % PROGRESS_INTERVAL == 0:
                pct = (vertices_processed / label_total) * 100 if label_total > 0 else 100.0
                elapsed = round(time.time() - phase_start, 2)
                logger.info(
                    "%s label=%s progress: last_id=%s total=%d pct=%.1f%% batch_dur=%.2fs rows=%d elapsed=%.2fs",
                    phase_name, label_name, current_last_id, label_total, pct, batch_duration, rows, elapsed,
                )

        label_elapsed = round(time.time() - label_start, 2)
        logger.info(
            "%s label=%s completed: batches=%d label_rows=%d elapsed=%.2fs",
            phase_name, label_name, total_batches, label_rows, label_elapsed,
        )

    phase_elapsed = round(time.time() - phase_start, 2)
    logger.info(
        "%s all labels completed: total_rows=%d elapsed=%.2fs",
        phase_name, total_rows, phase_elapsed,
    )
    return total_rows


async def log_audit(session: AsyncSession, start_time: float) -> None:
    type_counts = await session.execute(text("""
        SELECT entity_type, COUNT(*)::int AS cnt
        FROM public.graph_entity_posts
        GROUP BY entity_type
        ORDER BY entity_type
    """))
    type_rows = type_counts.fetchall()
    for row in type_rows:
        logger.info("Audit entity_type=%s count=%d", row.entity_type, row.cnt)

    total_result = await session.execute(text("SELECT COUNT(*) FROM public.graph_entity_posts"))
    total_indexed = total_result.scalar() or 0

    unique_posts = await session.execute(text("SELECT COUNT(DISTINCT post_id) FROM public.graph_entity_posts"))
    unique_posts_count = unique_posts.scalar() or 0

    author_posts = await session.execute(text("SELECT COUNT(DISTINCT post_id) FROM public.graph_entity_posts WHERE is_author_blog = True"))
    author_posts_count = author_posts.scalar() or 0

    elapsed = round(time.time() - start_time, 2)

    logger.info(
        "Seed completed: total_indexed_pairs=%d unique_posts=%d author_posts=%d elapsed_seconds=%.2f",
        total_indexed, unique_posts_count, author_posts_count, elapsed,
    )


async def main() -> None:
    settings = load_settings()
    setup_logging(settings.log_level)
    logger.info("Starting graph projection seed for public.graph_entity_posts")

    engine = create_async_engine(
        settings.db_url,
        pool_size=5,
        max_overflow=5,
        pool_timeout=30.0,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={
            "command_timeout": 300,
            "server_settings": {
                "lock_timeout": "10000",
                "idle_in_transaction_session_timeout": "300000",
            },
        },
    )
    async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    start_time = time.time()

    async with async_session_factory() as session:
        await session.execute(text("SET statement_timeout = '300s';"))
        existing_count = await session.execute(text("SELECT COUNT(*) FROM public.graph_entity_posts"))
        logger.info("Existing rows in public.graph_entity_posts: %d", existing_count.scalar() or 0)

        label_names = await get_vertex_labels(session)
        if not label_names:
            logger.info("No vertex labels found to process, skipping")
            return

        logger.info("Vertex labels to process: %s batch_size=%d", label_names, BATCH_SIZE)

        try:
            total_1hop = await process_phase(session, label_names, BATCH_SIZE, "1-hop", insert_1hop)
        except Exception:
            raise

        try:
            total_2hop = await process_phase(session, label_names, BATCH_SIZE, "2-hop", insert_2hop)
        except Exception:
            raise

        try:
            enriched_count = await enrich_author_flags(session)
            logger.info("Author flag enrichment: updated %d rows", enriched_count)
        except Exception:
            logger.exception("Author flag enrichment failed, rolling back")
            await session.rollback()
            raise

        await log_audit(session, start_time)

    await engine.dispose()
    logger.info("Graph projection seed completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
