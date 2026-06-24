from __future__ import annotations

import asyncio
import sys
from typing import Any

from qdrant_client import AsyncQdrantClient

from src.config.config import load_settings

ENTITIES_COLLECTION: str = "social_entities"
VALID_LABELS: frozenset[str] = frozenset({"Actor", "Entity", "Place", "Event"})
GENERIC_NAMES: frozenset[str] = frozenset({"other", "unknown", "n_a", "none", ""})
PAYLOAD_FORBIDDEN_KEYS: frozenset[str] = frozenset({"category", "language"})
SCROLL_LIMIT: int = 15


def print_separator() -> None:
    print("=" * 72)


def print_entity_report(
    index: int,
    point_id: str,
    payload: dict[str, Any],
    alerts: list[str],
) -> None:
    print(f"\n--- Entity #{index + 1} ---")
    print(f"  Point ID     : {point_id}")
    print(f"  Label        : {payload.get('label', '<MISSING>')}")
    print(f"  Name         : {payload.get('name', '<MISSING>')}")
    print(f"  Original ID  : {payload.get('original_id', '<MISSING>')}")

    props: dict[str, Any] = payload.get("properties", {})
    if props:
        print("  Properties   :")
        for key, value in props.items():
            print(f"    {key}: {value}")
    else:
        print("  Properties   : (empty)")

    if alerts:
        print("  ALERTS:")
        for alert in alerts:
            print(f"    [!] {alert}")
    else:
        print("  Status       : OK")


def validate_entity(
    payload: dict[str, Any],
    alerts: list[str],
) -> None:
    label: str = str(payload.get("label", "")).strip()
    name: str = str(payload.get("name", "")).strip()
    props: dict[str, Any] = payload.get("properties", {})

    if label not in VALID_LABELS:
        alerts.append(
            f"Unexpected label '{label}'. "
            f"Expected one of: {sorted(VALID_LABELS)}"
        )

    if name.startswith("#"):
        alerts.append(
            f"Entity name starts with '#': '{name}'"
        )

    if name.lower() in GENERIC_NAMES or len(name) < 2:
        alerts.append(
            f"Entity name is generic or too short: '{name}'"
        )

    for forbidden_key in PAYLOAD_FORBIDDEN_KEYS:
        if forbidden_key in props:
            alerts.append(
                f"Forbidden key '{forbidden_key}' found in properties"
            )


async def run() -> None:
    settings = load_settings()

    if not settings.qdrant_url:
        print("ERROR: QDRANT_URL is not configured in environment / .env")
        sys.exit(1)

    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        timeout=settings.qdrant_timeout,
        api_key=settings.qdrant_api_key,
    )

    try:
        print_separator()
        print(f"Qdrant endpoint : {settings.qdrant_url}")
        print(f"Collection      : {ENTITIES_COLLECTION}")
        print(f"Scroll limit    : {SCROLL_LIMIT}")
        print_separator()

        try:
            info = await client.get_collection(ENTITIES_COLLECTION)
            total_points: int | None = info.points_count
            print(f"Total points in collection: {total_points}")
        except Exception as exc:
            print(f"ERROR: Could not retrieve collection info: {exc}")
            sys.exit(1)

        print_separator()

        result = await client.scroll(
            collection_name=ENTITIES_COLLECTION,
            limit=SCROLL_LIMIT,
            with_payload=True,
            with_vectors=False,
        )

        points = result[0]

        if not points:
            print("No points found in the collection.")
            return

        print(f"Fetched {len(points)} point(s) for inspection.\n")

        total_alerts: int = 0

        for idx, point in enumerate(points):
            point_id = str(point.id)
            payload: dict[str, Any] = point.payload or {}

            entity_alerts: list[str] = []
            validate_entity(payload, entity_alerts)
            total_alerts += len(entity_alerts)

            print_entity_report(idx, point_id, payload, entity_alerts)

        print_separator()
        print(f"Inspection complete: {len(points)} entities checked, {total_alerts} alert(s) found.")
        print_separator()

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(run())
