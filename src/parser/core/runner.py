"""Worker orchestration and lifecycle management."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from src.config.config import Settings
from .worker_base import BaseTelegramWorker

logger = logging.getLogger(__name__)


async def start_workers(
    worker_class: type[BaseTelegramWorker],
    settings: Settings,
    db: Any,
    *,
    worker_args: dict[str, Any] | None = None,
) -> None:
    """Discover sessions, read configs, and spawn worker tasks.

    This function handles the common boilerplate for running multiple workers:
    - Scanning sessions directory for .session files
    - Reading accompanying .json config files for per-session overrides
    - Creating worker instances with proper configuration
    - Running all workers concurrently with graceful shutdown

    Args:
        worker_class: The worker class to instantiate (must inherit from BaseTelegramWorker).
        settings: Global settings configuration.
        db: Database instance (will be passed to each worker).
        worker_args: Additional arguments to pass to worker constructor (beyond the standard ones).
    """
    if worker_args is None:
        worker_args = {}

    logger.info(
        "Starting workers with class %s (db=%s)",
        worker_class.__name__,
        db,
    )

    # Scan sessions directory
    sessions_dir = settings.session_dir
    if not sessions_dir.exists():
        logger.error("Sessions directory %s does not exist", sessions_dir)
        return

    session_files = sorted(sessions_dir.glob("*.session"))
    if not session_files:
        logger.error("No .session files found in %s", sessions_dir)
        return

    logger.info(
        "Found %d session files: %s",
        len(session_files),
        [f.name for f in session_files],
    )

    # Create workers
    workers = []

    for i, session_path in enumerate(session_files):
        # Look for accompanying .json config file
        json_path = session_path.with_suffix(".json")

        # Start with global settings
        api_id = settings.api_id
        api_hash = settings.api_hash
        proxy_url = settings.proxy_url
        device_model = "PC 64bit"
        system_version = "Windows 10"
        app_version = "4.16.8"
        lang_code = "en"
        system_lang_code = "en-US"

        # Override with per-session config if present
        if json_path.exists():
            try:
                with json_path.open(encoding="utf-8") as f:
                    config = json.load(f)

                api_id = (
                    config.get("api_id")
                    or config.get("app_id")
                    or settings.api_id
                )
                api_hash = (
                    config.get("api_hash")
                    or config.get("app_hash")
                    or settings.api_hash
                )
                proxy_url = (
                    config.get("proxy_url")
                    if "proxy_url" in config
                    else settings.proxy_url
                )
                device_model = (
                    config.get("device_model")
                    or config.get("device")
                    or device_model
                )
                system_version = (
                    config.get("system_version")
                    or config.get("sdk")
                    or system_version
                )
                app_version = config.get("app_version", app_version)
                lang_code = config.get("lang_code", lang_code)
                system_lang_code = config.get(
                    "system_lang_code", system_lang_code
                )

                # Validate api_id
                try:
                    api_id = int(api_id)
                except (ValueError, TypeError):
                    api_id = settings.api_id
                if not isinstance(api_hash, str):
                    api_hash = settings.api_hash

                logger.info(
                    "Loaded config for %s from %s (api_id=%s, proxy=%s, device=%s)",
                    session_path.name,
                    json_path.name,
                    api_id if api_id else "default",
                    "yes" if proxy_url else "no",
                    device_model,
                )
            except Exception as e:
                logger.warning(
                    "Failed to read %s: %s, using global settings",
                    json_path.name,
                    e,
                )
        else:
            logger.debug(
                "No config file for %s, using global settings",
                session_path.name,
            )

        # Create worker instance
        worker = worker_class(
            worker_id=i,
            session_path=session_path,
            db=db,
            settings=settings,
            api_id=api_id,
            api_hash=api_hash,
            proxy_url=proxy_url,
            device_model=device_model,
            system_version=system_version,
            app_version=app_version,
            lang_code=lang_code,
            system_lang_code=system_lang_code,
            **worker_args,
        )
        workers.append(worker)

    logger.info("Spawning %d workers", len(workers))

    # Run all workers concurrently
    tasks = [asyncio.create_task(worker.run()) for worker in workers]

    if not tasks:
        logger.error("No workers to run. Check your session directory.")
        return

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping workers...")

        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error("Global error in gather: %s", e, exc_info=True)
    finally:
        # Note: Individual workers should handle their own cleanup
        logger.info("All workers have stopped")
