"""Worker orchestration and lifecycle management with Session Pool Manager.

This module implements a robust session pool that:
- Limits concurrent workers to settings.concurrency
- Manages a pool of Telegram sessions using asyncio.Queue
- Automatically respawns workers with new sessions when one dies
- Implements cooldown for failed sessions
- Handles permanent bans by removing sessions from the pool
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from telethon.errors import AuthKeyError, UserDeactivatedError
from telethon.errors.rpcerrorlist import FloodWaitError

from src.config.config import Settings
from .exceptions import SessionExpiredError
from .worker_base import BaseTelegramWorker

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """Configuration for a single Telegram session.
    
    Attributes:
        session_path: Path to the .session file.
        api_id: Telegram API ID.
        api_hash: Telegram API hash.
        proxy_url: Optional proxy URL for this session.
        device_model: Device model string for Telethon client.
        system_version: System version string.
        app_version: App version string.
        lang_code: Language code (e.g., 'en').
        system_lang_code: System language code (e.g., 'en-US').
    """
    session_path: Path
    api_id: int
    api_hash: str
    proxy_url: str | None
    device_model: str
    system_version: str
    app_version: str
    lang_code: str
    system_lang_code: str


@dataclass
class SessionEntry:
    """A session entry in the pool with its config and cooldown state.
    
    Attributes:
        config: The session configuration.
        ready_at: Timestamp (from time.time()) when this session is ready to be used.
                  If 0 or in the past, the session is ready immediately.
    """
    
    config: SessionConfig
    ready_at: float = field(default=0.0)


class SessionPool:
    """Manages a pool of available Telegram sessions using asyncio.Queue.
    
    Sessions are loaded from the sessions directory and made available via a queue.
    Failed sessions can be put back with a cooldown period.
    Permanently banned sessions are removed from the pool.
    """
    
    def __init__(self, cooldown_period: float = 300.0) -> None:
        """Initialize the session pool.
        
        Args:
            cooldown_period: Default cooldown period in seconds for failed sessions.
        """
        self._queue: asyncio.Queue[SessionEntry] = asyncio.Queue()
        self._cooldown_period = cooldown_period
        self._loaded_count = 0
        self._banned_count = 0
        
    async def load_sessions(self, settings: Settings) -> int:
        """Scan sessions directory and load all .session files with their configs.
        
        Args:
            settings: Global settings containing session_dir and default values.
            
        Returns:
            Number of sessions successfully loaded into the pool.
        """

        sessions_dir = settings.session_dir
        if not sessions_dir.exists():
            logger.error("Sessions directory %s does not exist", sessions_dir)
            return 0
            
        session_files = sorted(sessions_dir.glob("*.session"))
        if not session_files:
            logger.error("No .session files found in %s", sessions_dir)
            return 0
            
        logger.info("Found %d session files in %s", len(session_files), sessions_dir)
        
        for session_path in session_files:
            config = self._load_session_config(session_path, settings)
            entry = SessionEntry(config=config)
            await self._queue.put(entry)
            self._loaded_count += 1
            
        logger.info("Loaded %d sessions into pool", self._loaded_count)
        return self._loaded_count
    
    def _load_session_config(self, session_path: Path, settings: Settings) -> SessionConfig:
        """Load configuration for a single session file.
        
        Reads the accompanying .json config file if present, otherwise uses global settings.
        
        Args:
            session_path: Path to the .session file.
            settings: Global settings for default values.
            
        Returns:
            SessionConfig with values from json config or defaults.
        """

        # Default values from global settings
        api_id = settings.api_id
        api_hash = settings.api_hash
        proxy_url = settings.proxy_url
        device_model = "PC 64bit"
        system_version = "Windows 10"
        app_version = "4.16.8"
        lang_code = "en"
        system_lang_code = "en-US"
        
        # Override with per-session config if present
        json_path = session_path.with_suffix(".json")
        if json_path.exists():
            try:
                with json_path.open(encoding="utf-8") as f:
                    config_data = json.load(f)
                
                # Extract API credentials
                api_id = (
                    config_data.get("api_id")
                    or config_data.get("app_id")
                    or api_id
                )
                api_hash = (
                    config_data.get("api_hash")
                    or config_data.get("app_hash")
                    or api_hash
                )
                
                # Extract proxy URL if present in config
                if "proxy_url" in config_data:
                    proxy_url = config_data["proxy_url"]
                    
                # Extract device/system info
                device_model = (
                    config_data.get("device_model")
                    or config_data.get("device")
                    or device_model
                )
                system_version = (
                    config_data.get("system_version")
                    or config_data.get("sdk")
                    or system_version
                )
                app_version = config_data.get("app_version", app_version)
                lang_code = config_data.get("lang_code", lang_code)
                system_lang_code = config_data.get(
                    "system_lang_code", system_lang_code
                )
                
                # Validate api_id
                try:
                    api_id = int(api_id)
                except (ValueError, TypeError):
                    api_id = settings.api_id
                    
                # Validate api_hash
                if not isinstance(api_hash, str):
                    api_hash = settings.api_hash
                    
                logger.debug(
                    "Loaded config for session %s (api_id=%s, proxy=%s)",
                    session_path.name,
                    api_id if api_id else "default",
                    "yes" if proxy_url else "no",
                )
            except Exception as e:
                logger.warning(
                    "Failed to read config for %s: %s, using global settings",
                    session_path.name,
                    e,
                )
        
        return SessionConfig(
            session_path=session_path,
            api_id=api_id,
            api_hash=api_hash,
            proxy_url=proxy_url,
            device_model=device_model,
            system_version=system_version,
            app_version=app_version,
            lang_code=lang_code,
            system_lang_code=system_lang_code,
        )
    
    async def get_session(self) -> SessionEntry | None:
        """Get an available session from the pool.
        
        Returns:
            SessionEntry if available, None if the pool is empty.
        """

        try:
            # Use a short timeout to check if queue has items
            return await asyncio.wait_for(self._queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            return None
    
    async def return_session(
        self, 
        entry: SessionEntry, 
        cooldown: bool = False,
        cooldown_seconds: float | None = None,
    ) -> None:
        """Return a session to the pool.
        
        Args:
            entry: The session entry to return.
            cooldown: If True, put session in cooldown before it can be reused.
                      If the session already has a future ready_at, it will be preserved.
            cooldown_seconds: Custom cooldown period. Uses default if None.
                If the entry already has a future ready_at (from FloodWait cooldown),
                that timestamp will be preserved unless cooldown_seconds is provided.
        """

        if cooldown:
            # Only set cooldown if the entry doesn't already have a future ready_at
            # This preserves the original cooldown timestamp when returning a session
            # that was already in cooldown (e.g., from FloodWaitError handling)
            current_time = time.time()
            if entry.ready_at <= current_time:
                # No existing future cooldown, set a new one
                seconds = cooldown_seconds or self._cooldown_period
                entry.ready_at = current_time + seconds
            # If entry.ready_at is already in the future, preserve it
            logger.debug(
                "Session %s returned to pool with cooldown (ready at %.0f)",
                entry.config.session_path.name,
                entry.ready_at,
            )
        else:
            entry.ready_at = 0.0  # Ready immediately
            logger.debug(
                "Session %s returned to pool (ready immediately)",
                entry.config.session_path.name,
            )
        await self._queue.put(entry)
    
    def size(self) -> int:
        """Return the number of sessions currently in the pool."""
        return self._queue.qsize()
    
    def mark_banned(self) -> None:
        """Mark a session as permanently banned (removed from pool)."""
        self._banned_count += 1
    
    @property
    def loaded_count(self) -> int:
        """Total number of sessions loaded into the pool."""
        return self._loaded_count
    
    @property
    def banned_count(self) -> int:
        """Number of sessions that have been permanently banned."""
        return self._banned_count


async def _worker_runner(
    runner_id: int,
    session_pool: SessionPool,
    worker_class: type[BaseTelegramWorker],
    settings: Settings,
    db: Any,
    worker_args: dict[str, Any] | None = None,
) -> None:
    """Worker runner task that continuously processes sessions from the pool.
    
    This function runs in a loop, picking up sessions from the pool,
    creating worker instances, and running them. If a worker completes
    normally, the session is returned to the pool for reuse. If a worker
    encounters a permanent ban error, the session is removed from the pool.
    For FloodWaitError, the session is returned with an appropriate cooldown
    so other healthy sessions can be processed in its place.
    
    Args:
        runner_id: Unique identifier for this runner (used as worker_id).
        session_pool: The session pool to pick sessions from.
        worker_class: The worker class to instantiate.
        settings: Global settings configuration.
        db: Database instance to pass to workers.
        worker_args: Additional arguments for worker constructor.
    """
    
    logger.info("Worker runner %d started", runner_id)
    
    if worker_args is None:
        worker_args = {}
    
    while True:
        # Get a session from the pool
        entry = await session_pool.get_session()
        if entry is None:
            # No sessions available right now (all checked out or in cooldown)
            # Check if all loaded sessions have been permanently banned
            active_sessions = session_pool.loaded_count - session_pool.banned_count
            if active_sessions <= 0:
                logger.info(
                    "Worker runner %d: all sessions have been permanently banned, stopping",
                    runner_id,
                )
                break
            # There are still active sessions, but they are currently checked out or in cooldown
            # Wait and try again
            logger.debug(
                "Worker runner %d: no sessions available right now, waiting (active: %d, banned: %d)",
                runner_id,
                active_sessions,
                session_pool.banned_count,
            )
            await asyncio.sleep(2.0)
            continue
        
        # Check if session is in cooldown
        now = time.time()
        if entry.ready_at > now:
            # Session is in cooldown, put it back WITHOUT modifying ready_at
            wait_time = entry.ready_at - now
            logger.debug(
                "Worker runner %d: session %s in cooldown for %.0fs, returning to pool",
                runner_id,
                entry.config.session_path.name,
                wait_time,
            )
            # Put back directly to preserve the original ready_at timestamp
            # We use return_session with cooldown=True, but the method now
            # preserves the original ready_at if it's in the future
            await session_pool.return_session(entry, cooldown=True)
            await asyncio.sleep(min(wait_time, 5.0))  # Wait up to 5 seconds
            continue
        
        # Create worker instance
        config = entry.config
        try:
            worker = worker_class(
                worker_id=runner_id,
                session_path=config.session_path,
                db=db,
                settings=settings,
                api_id=config.api_id,
                api_hash=config.api_hash,
                proxy_url=config.proxy_url,
                device_model=config.device_model,
                system_version=config.system_version,
                app_version=config.app_version,
                lang_code=config.lang_code,
                system_lang_code=config.system_lang_code,
                **worker_args,
            )
        except Exception as e:
            logger.error(
                "Worker runner %d: failed to create worker for session %s: %s",
                runner_id,
                config.session_path.name,
                e,
                exc_info=True,
            )
            # Return session to pool with cooldown
            await session_pool.return_session(entry, cooldown=True)
            continue
        
        logger.info(
            "Worker runner %d: starting session %s",
            runner_id,
            config.session_path.name,
        )
        
        try:
            await worker.run()
            # Worker completed normally
            logger.info(
                "Worker runner %d: session %s completed normally, returning to pool",
                runner_id,
                config.session_path.name,
            )
            # Return session to pool for reuse (ready immediately)
            await session_pool.return_session(entry, cooldown=False)
        except FloodWaitError as e:
            # FloodWaitError - session needs cooldown before reuse
            delay = int(getattr(e, "seconds", 0)) or 1
            logger.warning(
                "Runner %d: Session %s rate limited by TG (FloodWait %ds). "
                "Returning to pool with cooldown %ds.",
                runner_id,
                config.session_path.name,
                delay,
                delay + 10,
            )
            # Return session to pool with cooldown
            # The worker's _flood_wait_cooldown_until should already be set,
            # but we also set it here to ensure the pool has the correct ready_at
            entry.ready_at = time.time() + delay + 10
            await session_pool.return_session(entry, cooldown=True, cooldown_seconds=delay + 10)
            # Exit the current worker run cleanly - the runner will pick up
            # a different healthy session from the pool
            logger.info(
                "Worker runner %d: exiting after FloodWait, will pick new session",
                runner_id,
            )
        except (AuthKeyError, UserDeactivatedError, SessionExpiredError) as e:
            # Permanent ban - don't return session to pool
            logger.error(
                "Worker runner %d: session %s permanently banned: %s",
                runner_id,
                config.session_path.name,
                e,
            )
            session_pool.mark_banned()
        except Exception as e:
            # Transient error - return session to pool with cooldown
            logger.error(
                "Worker runner %d: session %s failed with error: %s",
                runner_id,
                config.session_path.name,
                e,
                exc_info=True,
            )
            await session_pool.return_session(entry, cooldown=True)
    
    logger.info("Worker runner %d stopped", runner_id)


async def start_workers(
    worker_class: type[BaseTelegramWorker],
    settings: Settings,
    db: Any,
    *,
    worker_args: dict[str, Any] | None = None,
    ignore_concurrency_limit: bool = False,
) -> None:
    """Start worker tasks with session pool management.
    
    This function implements a Session Pool Manager that:
    - Limits concurrent workers to settings.concurrency
    - Uses a pool of sessions that workers can pick up
    - Automatically respawns workers with new sessions when one dies
    - Implements cooldown for failed sessions
    - Handles permanent bans by removing sessions from the pool
    
    Args:
        worker_class: The worker class to instantiate (must inherit from BaseTelegramWorker).
        settings: Global settings configuration.
        db: Database instance to pass to each worker.
        worker_args: Additional arguments to pass to worker constructor.
        ignore_concurrency_limit: If True, use all available sessions ignoring settings.concurrency.
    """
    
    if worker_args is None:
        worker_args = {}
    
    if ignore_concurrency_limit:
        logger.info(
            "Starting workers with class %s (ignoring concurrency limit, using all available sessions)",
            worker_class.__name__,
        )
    else:
        logger.info(
            "Starting workers with class %s (concurrency=%d)",
            worker_class.__name__,
            settings.concurrency,
        )
    
    # Create session pool and load sessions
    session_pool = SessionPool(cooldown_period=300.0)  # 5 minute cooldown for failed sessions
    loaded = await session_pool.load_sessions(settings)
    
    if loaded == 0:
        logger.error("No sessions loaded. Check your session directory.")
        return
    
    # Determine actual concurrency (can't have more runners than sessions)
    if ignore_concurrency_limit:
        actual_concurrency = session_pool.size()
        logger.info(
            "Concurrency limit ignored: using all %d available sessions from pool (loaded: %d, banned: %d)",
            actual_concurrency,
            session_pool.loaded_count,
            session_pool.banned_count,
        )
    else:
        actual_concurrency = min(settings.concurrency, session_pool.size())
        logger.info(
            "Spawning %d worker runners (pool size: %d, loaded: %d, banned: %d)",
            actual_concurrency,
            session_pool.size(),
            session_pool.loaded_count,
            session_pool.banned_count,
        )
    
    # Spawn exactly `actual_concurrency` worker runners
    runners = [
        asyncio.create_task(
            _worker_runner(
                runner_id=i,
                session_pool=session_pool,
                worker_class=worker_class,
                settings=settings,
                db=db,
                worker_args=worker_args,
            ),
            name=f"worker-runner-{i}",
        )
        for i in range(actual_concurrency)
    ]
    
    try:
        results = await asyncio.gather(*runners, return_exceptions=True)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping workers...")
        # Cancel all runner tasks
        for task in runners:
            task.cancel()
        # Wait for all tasks to complete (with cancellation exceptions)
        results = await asyncio.gather(*runners, return_exceptions=True)
    else:
        # Log any exceptions returned by worker tasks
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.critical(
                    "Worker runner %d died with unhandled exception: %s",
                    i,
                    result,
                    exc_info=True,
                )
    finally:
        logger.info(
            "All worker runners have stopped (sessions loaded: %d, banned: %d)",
            session_pool.loaded_count,
            session_pool.banned_count,
        )
