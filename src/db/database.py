"""
Asynchronous CRUD operations for channels and posts using SQLAlchemy 2.0.
"""

import asyncio
import json
import logging
import random
import re
from typing import Any, Sequence

from sqlalchemy import bindparam, case, func, select, update, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import joinedload

from src.db.models import Base, Channel, Post

logger = logging.getLogger(__name__)


class Database:
    """Asynchronous PostgreSQL connection manager."""

    def __init__(self, db_url: str, echo: bool = False) -> None:
        """
        Initialize the database manager.

        Args:
            db_url: Connection string in postgresql+asyncpg:// format.
            echo: Enable SQL query logging (for debugging).
        """

        self.engine = create_async_engine(
            db_url,
            echo=echo,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={
                "command_timeout": 120,
                "timeout": 60,
                "server_settings": {
                    "tcp_keepalives_idle": "60",
                    "tcp_keepalives_interval": "10",
                    "tcp_keepalives_count": "5",
                },
            },
        )

        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self, max_retries: int = 5, base_delay: float = 1.0, timeout: float = 120.0) -> None:
        """
        Create all tables defined in the models (if they don't exist).

        Implements retry with exponential backoff and jitter for resilience
        against temporary network or database unavailability during startup.
        Also enforces an overall timeout to prevent indefinite blocking.

        Args:
            max_retries: Maximum number of retry attempts (default: 5).
            base_delay: Base delay in seconds for exponential backoff (default: 1.0).
            timeout: Overall timeout in seconds for the entire initialization process (default: 120.0).

        Raises:
            RuntimeError: If initialization fails after all retries or times out.
        """
        
        last_exception: Exception | None = None
        start_time = asyncio.get_event_loop().time()

        for attempt in range(1, max_retries + 1):
            try:
                async with self.engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)

                    # Initialize Apache AGE extension and graph
                    try:
                        # Create extension if not exists
                        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS age;"))
                        # Load AGE library
                        await conn.execute(text("LOAD 'age';"))
                        # Set search path for AGE
                        await conn.execute(
                            text('SET search_path = ag_catalog, "$user", public;')
                        )
                        # Create the base graph (ignore if already exists)
                        await conn.execute(
                            text(
                                "SELECT create_graph('telegram_graph') WHERE NOT EXISTS (SELECT 1 FROM ag_graph WHERE name = 'telegram_graph');"
                            )
                        )
                        logger.info("Apache AGE extension and graph initialized")
                    except Exception as e:
                        logger.error("Failed to initialize Apache AGE: %s", e)
                        raise

                logger.info("Database initialization successful")
                return  # Success, exit the retry loop

            except Exception as e:
                last_exception = e
                if attempt == max_retries:
                    logger.error(
                        "Database initialization failed after %d attempts: %s",
                        max_retries,
                        e,
                    )
                    raise RuntimeError(
                        f"Database initialization failed after {max_retries} attempts"
                    ) from e

                # Check if we've exceeded the overall timeout
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    logger.error(
                        "Database initialization exceeded timeout of %.1f seconds after %d attempts",
                        timeout,
                        attempt,
                    )
                    raise RuntimeError(
                        f"Database initialization timed out after {elapsed:.1f} seconds"
                    ) from last_exception

                # Exponential backoff with jitter to avoid thundering herd
                delay = min(base_delay * (2 ** (attempt - 1)), 60)  # Cap at 60 seconds
                jitter = random.uniform(0, delay * 0.1)  # 10% jitter
                total_delay = delay + jitter

                logger.warning(
                    "Database initialization attempt %d/%d failed: %s. Retrying in %.2f seconds...",
                    attempt,
                    max_retries,
                    e,
                    total_delay,
                )
                await asyncio.sleep(total_delay)

    async def reset_orphaned_processing_channels(self) -> None:
        """Reset all channels with status='processing' back to 'pending'.

        This recovers from crashes/restarts where channels were left in processing state.
        """
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    update(Channel)
                    .where(Channel.status == "processing")
                    .values(status="pending")
                )
                result = await session.execute(stmt)
                count = result.rowcount  # type: ignore[attr-defined]

                if count > 0:
                    logger.info(
                        "Reset orphaned processing channels back to pending: %d channels",
                        count,
                    )
                else:
                    logger.debug("No orphaned processing channels found")

    async def close(self) -> None:
        """Close all database connections."""
        await self.engine.dispose()
        logger.info("Database connections closed")

    async def upsert_channel(self, channel_data: dict[str, Any]) -> Channel:
        """
        Insert or update a channel record.

        Conflict is detected on the id field (Telegram channel_id).
        On conflict, all mutable fields except the primary key are updated.
        Status is preserved if it's already 'parsed' or 'ready_for_parsing'.

        Args:
            channel_data: Dictionary with fields matching the Channel model.

        Returns:
            The persisted Channel object.
        """

        stmt = insert(Channel).values(**channel_data)
        update_columns = {
            "username": stmt.excluded.username,
            "title": stmt.excluded.title,
            "description": stmt.excluded.description,
            "subscribers_count": stmt.excluded.subscribers_count,
            "avatar_url": stmt.excluded.avatar_url,
            "access_hash": stmt.excluded.access_hash,
            "is_author_blog": stmt.excluded.is_author_blog,
            "updated_at": stmt.excluded.updated_at,
            "status": case(
                (
                    Channel.status.in_(["parsed", "ready_for_parsing"]),
                    Channel.status,
                ),
                else_=stmt.excluded.status,
            ),
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_=update_columns,
        )

        async with self.async_session() as session:
            async with session.begin():
                await session.execute(stmt)
                # Retrieve the current record (may have existed already)
                channel = await session.get(Channel, channel_data["id"])

                if channel is None:
                    # Fallback manual creation (should not happen under normal circumstances)
                    channel = Channel(**channel_data)
                    session.add(channel)
                    await session.flush()

                logger.debug("Upserted channel: %s", channel)

                return channel

    async def upsert_post(self, post_data: dict[str, Any]) -> Post:
        """
        Insert or update a post record.

        Conflict is detected on the composite unique key (channel_id, message_id).
        On conflict, only metrics (views, comments, shares, reactions) are updated.
        Content and published_at are preserved to avoid overwriting existing data.

        Args:
            post_data: Dictionary with fields matching the Post model.

        Returns:
            The persisted Post object.
        """

        stmt = insert(Post).values(**post_data)

        update_columns = {
            "views": stmt.excluded.views,
            "comments_count": stmt.excluded.comments_count,
            "shares_count": stmt.excluded.shares_count,
            "reactions_count": stmt.excluded.reactions_count,
            "media_url": stmt.excluded.media_url,
            "updated_at": stmt.excluded.updated_at,
        }

        stmt = stmt.on_conflict_do_update(
            constraint="uq_post_channel_message",
            set_=update_columns,
        )

        async with self.async_session() as session:
            async with session.begin():
                await session.execute(stmt)

                # Retrieve the saved object
                post = await self._get_post_by_unique(
                    session, post_data["channel_id"], post_data["message_id"]
                )

                if post is None:
                    # Fallback manual creation if upsert failed unexpectedly
                    post = Post(**post_data)
                    session.add(post)
                    await session.flush()

                logger.debug("Upserted post: %s", post)
                return post

    async def _get_post_by_unique(
        self, session: AsyncSession, channel_id: int, message_id: int
    ) -> Post | None:
        """Helper method to fetch a post by its composite natural key."""
        stmt = select(Post).where(
            Post.channel_id == channel_id, Post.message_id == message_id
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_channels_batch(
        self, channel_ids: Sequence[int]
    ) -> dict[int, Channel]:
        """
        Return a dictionary of existing channels by a list of IDs.

        Useful for checking which channels are already in the DB before parsing.

        Args:
            channel_ids: List of Telegram channel IDs.

        Returns:
            Dictionary mapping channel_id to Channel object.
        """

        if not channel_ids:
            return {}

        async with self.async_session() as session:
            stmt = select(Channel).where(Channel.id.in_(channel_ids))
            result = await session.execute(stmt)
            channels = result.scalars().all()
            return {ch.id: ch for ch in channels}

    async def get_posts_by_ids(self, post_ids: list[int]) -> dict[int, Post]:
        """
        Fetch posts by a list of post IDs with their associated channels eagerly loaded.

        Args:
            post_ids: List of PostgreSQL post IDs (primary keys).

        Returns:
            Dictionary mapping post_id to Post object (with .channel populated) for efficient lookup.
        """
        if not post_ids:
            return {}

        async with self.async_session() as session:
            stmt = select(Post).options(joinedload(Post.channel)).where(Post.id.in_(post_ids))
            result = await session.execute(stmt)
            posts = result.scalars().all()
            return {post.id: post for post in posts}

    async def get_recent_posts(self, limit: int = 100) -> list[Post]:
        """Fetch recent posts from the database for indexing."""
        async with self.async_session() as session:
            stmt = select(Post).order_by(Post.id.desc()).limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_unextracted_posts(self, limit: int = 50, priority_mode: bool = False) -> list[Post]:
        """Fetch posts that have not yet been extracted to knowledge graph.

        Args:
            limit: Maximum number of posts to return.
            priority_mode: If True, order by published_at DESC (most recent first).
                          If False, order by id ASC (oldest first).

        Returns:
            List of Post objects where is_extracted is False.
        """

        async with self.async_session() as session:
            stmt = select(Post).where(Post.is_extracted == False)  # noqa: E712
            
            if priority_mode:
                # Priority mode: process most recent posts first (for search relevance)
                stmt = stmt.order_by(Post.published_at.desc())
            else:
                # Default: process oldest posts first (FIFO)
                stmt = stmt.order_by(Post.id.asc())
            
            stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def mark_post_extracted(self, post_id: int) -> None:
        """Mark a post as extracted (is_extracted = True).

        Uses a direct atomic UPDATE statement to avoid FOR UPDATE issues
        with outer joins caused by lazy="joined" relationships.

        Args:
            post_id: The database ID of the post to mark as extracted.
        """

        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    update(Post)
                    .where(Post.id == post_id)
                    .values(is_extracted=True)
                )
                result = await session.execute(stmt)

                if result.rowcount > 0:  # type: ignore[attr-defined]
                    logger.debug("Marked post id=%s as extracted", post_id)
                else:
                    logger.warning("Post id=%s not found when marking as extracted", post_id)

    async def get_random_pending_channel(
        self, require_hash: bool = False
    ) -> Channel | None:
        """
        Fetch a random channel with status='pending' and mark it as 'processing'.

        Uses FOR UPDATE SKIP LOCKED to avoid race conditions between workers.
        After marking, the transaction is committed so the channel is immediately
        visible to other workers as 'processing'.

        Args:
            require_hash: If True, only return channels with non-null access_hash.
                This allows "weak" accounts to fetch only channels that can be
                accessed directly without global search.

        Returns:
            The selected Channel entity, or None if no pending channels exist.
        """

        async with self.async_session() as session:
            # Start transaction with row-level lock
            async with session.begin():
                # Build query with optional hash requirement
                stmt = select(Channel).where(Channel.status == "pending")
                if require_hash:
                    stmt = stmt.where(Channel.access_hash.is_not(None))

                # Get random pending channel and lock it
                stmt = (
                    stmt.order_by(func.random())
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                result = await session.execute(stmt)
                channel = result.scalar_one_or_none()

                if channel is not None:
                    channel.status = "processing"
                    logger.debug(
                        "Claimed channel id=%s username=%s for processing",
                        channel.id,
                        channel.username,
                    )
                else:
                    logger.debug("No pending channels available")

                return channel

    async def mark_channel_processed(self, channel_id: int) -> None:
        """Mark a channel as successfully processed (status='parsed')."""
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Channel)
                    .where(Channel.id == channel_id)
                    .with_for_update()
                )
                result = await session.execute(stmt)
                channel = result.scalar_one_or_none()

                if channel is not None:
                    channel.status = "ready_for_parsing"
                    logger.debug("Marked channel id=%s as parsed", channel_id)

    async def mark_channel_rejected(self, channel_id: int) -> None:
        """Mark a channel as rejected (status='rejected')."""
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Channel)
                    .where(Channel.id == channel_id)
                    .with_for_update()
                )
                result = await session.execute(stmt)
                channel = result.scalar_one_or_none()

                if channel is not None:
                    channel.status = "rejected"
                    logger.debug("Marked channel id=%s as rejected", channel_id)

    async def get_channel_for_parsing(self) -> Channel | None:
        """Fetch a channel ready for POST PARSING and mark as processing.

        Only returns channels that have an access_hash to avoid global search
        rate limits and potential account bans.
        """
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Channel)
                    .where(Channel.status == "ready_for_parsing")
                    .where(Channel.is_author_blog == True)
                    .where(Channel.access_hash.is_not(None))
                    .order_by(func.random())
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                result = await session.execute(stmt)
                channel = result.scalar_one_or_none()

                if channel:
                    channel.status = "processing"
                    logger.debug(
                        "Parser claimed channel id=%s username=%s (has access_hash)",
                        channel.id,
                        channel.username,
                    )

                return channel

    async def mark_channel_parsed(self, channel_id: int) -> None:
        """Mark a channel as completely parsed (posts are saved)."""
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Channel)
                    .where(Channel.id == channel_id)
                    .with_for_update()
                )
                result = await session.execute(stmt)
                channel = result.scalar_one_or_none()

                if channel is not None:
                    channel.status = "parsed"
                    logger.debug(
                        "Marked channel id=%s as COMPLETELY PARSED", channel_id
                    )

    async def mark_channel_pending(self, channel_id: int) -> None:
        """Return a channel to pending status (e.g., if a worker failed due to a shadowban)."""
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Channel)
                    .where(Channel.id == channel_id)
                    .with_for_update()
                )
                result = await session.execute(stmt)
                channel = result.scalar_one_or_none()

                if channel is not None:
                    channel.status = "pending"
                    logger.debug(
                        "Returned channel id=%s to pending state", channel_id
                    )

    async def update_channel_access_hash(
        self, channel_id: int, access_hash: int
    ) -> None:
        """Update the access_hash for a channel.

        This is used to store the session-local correct access_hash after
        successful resolution by username.

        Args:
            channel_id: Telegram channel ID.
            access_hash: The resolved access_hash for this session.
        """
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Channel)
                    .where(Channel.id == channel_id)
                    .with_for_update()
                )
                result = await session.execute(stmt)
                channel = result.scalar_one_or_none()

                if channel is not None:
                    channel.access_hash = access_hash
                    logger.debug(
                        "Updated access_hash for channel id=%s", channel_id
                    )

    async def get_subgraph_for_entities(
        self, entity_ids: list[str]
    ) -> list[dict]:
        """
        Fetch the subgraph (direct neighbors) for a list of entity IDs from Apache AGE.

        Args:
            entity_ids: List of entity original_id values to query graph relationships for.

        Returns:
            List of dictionaries representing graph edges with keys:
            - source_id: ID of the source node
            - source_label: Label of the source node
            - source_name: Name property of the source node
            - target_id: ID of the target node
            - target_label: Label of the target node
            - target_name: Name property of the target node

        Raises:
            ValueError: If entity_ids is empty.
            RuntimeError: If query execution fails.
        """

        if not entity_ids:
            return []

        try:
            # Build the SQL query with literal Cypher string and bind parameter for entity IDs
            # The Cypher query is embedded directly using $$ delimiters (not a bind parameter)
            # The :params bind parameter contains the entity IDs as a JSON array and is cast to agtype
            query = text("""
                SELECT * FROM cypher('telegram_graph',
                    $$ UNWIND $ids AS id
                       MATCH (a)-[r]-(b)
                       WHERE a.id = id
                       RETURN a.id as a_id, label(a) as a_label, a.name as a_name,
                              type(r) as rel_type,
                              b.id as b_id, label(b) as b_label, b.name as b_name $$,
                    CAST(:params AS agtype)
                ) AS (
                    a_id agtype,
                    a_label agtype,
                    a_name agtype,
                    rel_type agtype,
                    b_id agtype,
                    b_label agtype,
                    b_name agtype
                )
            """)

            # Package entity IDs as JSON array
            params = json.dumps({"ids": entity_ids})

            async with self.async_session() as session:
                async with session.begin():
                    result = await session.execute(query, {"params": params})
                    rows = result.all()

            # Parse agtype results into dictionaries
            edges = []
            for row in rows:
                def parse_agtype(value):
                    if value is None:
                        return None
                    text = str(value)
                    if text.startswith('"') and text.endswith('"'):
                        text = text[1:-1]
                    try:
                        return json.loads(text)
                    except (json.JSONDecodeError, TypeError):
                        return text

                edge = {
                    "source_id": parse_agtype(row[0]),
                    "source_label": parse_agtype(row[1]),
                    "source_name": parse_agtype(row[2]),
                    "relation_type": parse_agtype(row[3]),
                    "target_id": parse_agtype(row[4]),
                    "target_label": parse_agtype(row[5]),
                    "target_name": parse_agtype(row[6]),
                }
                edges.append(edge)

            logger.debug(
                "Fetched subgraph for entities",
                extra={
                    "entity_count": len(entity_ids),
                    "edges_found": len(edges),
                },
            )
            return edges

        except Exception as e:
            logger.error(
                "Failed to fetch subgraph for entities",
                exc_info=e,
                extra={"entity_ids": entity_ids},
            )
            raise RuntimeError(f"Subgraph query failed: {e}") from e

    async def get_nodes_by_ids(self, node_ids: list[str], label: str | None = None) -> list[dict]:
        """
        Fetch full node details from Apache AGE graph by node IDs.

        Args:
            node_ids: List of node IDs to fetch.
            label: Optional node label filter to restrict search to a specific label.

        Returns:
            List of dictionaries with node details: id, label, name, properties (as dict).

        Raises:
            RuntimeError: If query execution fails.
        """

        if not node_ids:
            return []

        try:
            # Build Cypher query to fetch nodes with all properties
            # Using UNWIND to pass the list of IDs
            query = text("""
                SELECT * FROM cypher('telegram_graph',
                    $$ UNWIND $ids AS id
                       MATCH (n)
                       WHERE n.id = id
                       RETURN n.id as n_id, label(n) as n_label, n.name as n_name, n $$,
                    CAST(:params AS agtype)
                ) AS (
                    n_id agtype,
                    n_label agtype,
                    n_name agtype,
                    n agtype
                )
            """)

            params = json.dumps({"ids": node_ids})

            async with self.async_session() as session:
                async with session.begin():
                    result = await session.execute(query, {"params": params})
                    rows = result.all()

            nodes = []
            for row in rows:
                def parse_agtype(value):
                    if value is None:
                        return None
                    text = str(value)
                    if text.startswith('"') and text.endswith('"'):
                        text = text[1:-1]
                    try:
                        return json.loads(text)
                    except (json.JSONDecodeError, TypeError):
                        return text

                node_id = parse_agtype(row[0])
                node_label = parse_agtype(row[1])
                node_name = parse_agtype(row[2])
                node_data = parse_agtype(row[3])

                # Extract properties from node data
                properties = {}
                if isinstance(node_data, dict):
                    # Copy all keys except id, label, name (these are already extracted)
                    for key, value in node_data.items():
                        if key not in ('id', 'label', 'name'):
                            properties[key] = value

                nodes.append({
                    "id": node_id,
                    "label": node_label,
                    "name": node_name,
                    "properties": properties
                })

            logger.debug(
                "Fetched nodes from graph",
                extra={
                    "node_count": len(nodes),
                    "requested_ids": len(node_ids),
                },
            )
            return nodes

        except Exception as e:
            logger.error(
                "Failed to fetch nodes from graph",
                exc_info=e,
                extra={"node_ids": node_ids},
            )
            raise RuntimeError(f"Node fetch failed: {e}") from e

    async def execute_cypher(self, query: str) -> Any:
        """
        Execute a raw Cypher query against the Apache AGE graph.

        This is a placeholder method for future graph operations.
        The query will be executed in the context of the telegram_graph.

        Args:
            query: Cypher query string (e.g., "SELECT * FROM cypher('telegram_graph', $$ MATCH (n) RETURN n $$) AS (n agtype);")

        Returns:
            Raw query result (will be refined when graph queries are implemented).
        """

        async with self.async_session() as session:
            async with session.begin():
                result = await session.execute(text(query))
                return result.scalars().all()

    def _sanitize_properties(self, properties: dict) -> dict:
        """
        Sanitize property keys to ensure they are strictly alphanumeric snake_case.

        This utility method prevents hidden characters or invalid syntax from
        breaking the Cypher query parser. Keys are stripped, converted to
        snake_case, and validated against a strict pattern.

        Args:
            properties: Raw properties dictionary.

        Returns:
            Dictionary with sanitized keys (original values preserved).

        Raises:
            ValueError: If any key cannot be sanitized to a valid identifier.
        """
        sanitized = {}
        for key, value in properties.items():
            # Strip whitespace and convert to snake_case
            clean_key = key.strip().lower()
            # Replace any non-alphanumeric (except underscore) with underscore
            clean_key = re.sub(r'[^a-z0-9_]', '_', clean_key)
            # Collapse multiple underscores
            clean_key = re.sub(r'_+', '_', clean_key)
            # Trim leading/trailing underscores
            clean_key = clean_key.strip('_')

            # Validate final key
            if not re.match(r'^[a-z0-9_]+$', clean_key):
                raise ValueError(
                    f"Property key '{key}' could not be sanitized to a valid identifier "
                    f"(got '{clean_key}'). Only alphanumeric characters and underscores are allowed."
                )

            sanitized[clean_key] = value

        return sanitized

    async def upsert_graph_node(
        self, label: str, properties: dict, merge_key: str = "id"
    ) -> None:
        """
        Upsert a node in the Apache AGE graph.

        Uses MERGE to create or update a node with the given label and properties.
        The node is matched on the merge_key (default: 'id').

        IMPORTANT: Labels and property keys cannot be parameterized in Cypher,
        so we validate them to prevent injection attacks. Properties are set
        using an explicit SET clause with individual bind parameters to avoid
        the "SET clause expects a map" error in Apache AGE.

        Args:
            label: Graph node label (e.g., 'Channel', 'Post'). Must be alphanumeric.
            properties: Dictionary of node properties.
            merge_key: Property name to use for MERGE matching (default: 'id').

        Raises:
            ValueError: If label, merge_key, or any property key contains non-alphanumeric characters.
        """

        # Validate label to prevent Cypher injection (labels cannot be parameterized)
        if not re.match(r'^[A-Za-z0-9_]+$', label):
            raise ValueError(f"Invalid label '{label}': must be alphanumeric with underscores")

        # Validate merge_key as a simple identifier
        if not re.match(r'^[A-Za-z0-9_]+$', merge_key):
            raise ValueError(f"Invalid merge_key '{merge_key}': must be alphanumeric with underscores")

        # Sanitize all property keys to prevent Cypher injection
        props = self._sanitize_properties(properties)

        # Ensure merge_key is present in properties
        if merge_key not in props:
            props[merge_key] = properties.get(merge_key)

        # Build dynamic SET clause by iterating over properties
        # Exclude merge_key from SET to avoid conflicts (it's already used in MERGE)
        set_clauses = []
        set_params = {}
        for key, value in props.items():
            if key == merge_key:
                continue  # Skip merge_key to prevent conflicts
            # Use backticks for property names to avoid reserved keyword conflicts
            # Prefix parameter names with 'prop_' to avoid parameter name conflicts
            set_clauses.append(f"n.`{key}` = $prop_{key}")
            set_params[f"prop_{key}"] = value

        # Join SET clauses with comma, ensuring no trailing comma
        set_clause_str = ", ".join(set_clauses) if set_clauses else ""

        # Build the SQL query with explicit SET assignments
        # All parameters are passed as a single agtype map via :params
        # Inside Cypher, individual values are accessed as $key from the agtype map
        query = text(f"""
            SELECT * FROM cypher('telegram_graph',
                $$ MERGE (n:{label} {{{merge_key}: $merge_val}})
                   {f"SET {set_clause_str}" if set_clause_str else ""}
                   RETURN n $$,
                CAST(:params AS agtype)
            ) AS (v agtype)
        """)

        # Package all parameters as a single JSON object
        # The agtype map contains all values referenced in the Cypher query
        params_dict = {"merge_val": props.get(merge_key), **set_params}
        params = json.dumps(params_dict)

        async with self.async_session() as session:
            async with session.begin():
                await session.execute(query, {"params": params})

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
    ) -> None:
        """
        Upsert an edge in the Apache AGE graph.

        Matches start and end nodes by label and merge keys, then MERGEs the edge.
        Edge properties are set using an explicit SET clause with individual bind
        parameters to avoid the "SET clause expects a map" error in Apache AGE.

        IMPORTANT: Labels and relationship types cannot be parameterized in Cypher,
        so we validate them to prevent injection attacks. All property keys are
        sanitized to valid identifiers. Properties are assigned individually.

        Args:
            start_label: Label of the start node. Must be alphanumeric.
            start_merge_key: Property name to match start node.
            start_merge_val: Value for start node matching.
            edge_label: Relationship type label. Must be alphanumeric.
            end_label: Label of the end node. Must be alphanumeric.
            end_merge_key: Property name to match end node.
            end_merge_val: Value for end node matching.
            edge_properties: Optional dictionary of edge properties.

        Raises:
            ValueError: If any label, edge_label, merge key, or property key contains non-alphanumeric characters.
        """
        
        # Validate all alphanumeric identifiers to prevent Cypher injection
        for identifier_name, identifier_value in [
            ("start_label", start_label),
            ("edge_label", edge_label),
            ("end_label", end_label),
        ]:
            if not re.match(r'^[A-Za-z0-9_]+$', identifier_value):
                raise ValueError(
                    f"Invalid {identifier_name} '{identifier_value}': must be alphanumeric with underscores"
                )

        # Merge keys are also identifiers in the Cypher query, validate them
        if not re.match(r'^[A-Za-z0-9_]+$', start_merge_key):
            raise ValueError(f"Invalid start_merge_key '{start_merge_key}': must be alphanumeric with underscores")
        if not re.match(r'^[A-Za-z0-9_]+$', end_merge_key):
            raise ValueError(f"Invalid end_merge_key '{end_merge_key}': must be alphanumeric with underscores")

        # Sanitize all edge property keys to prevent Cypher injection
        edge_properties = edge_properties or {}
        props = self._sanitize_properties(edge_properties)

        # Build dynamic SET clause by iterating over edge properties
        set_clauses = []
        set_params = {}
        for key, value in props.items():
            # Use backticks for property names to avoid reserved keyword conflicts
            # Prefix parameter names with 'prop_' to avoid parameter name conflicts
            set_clauses.append(f"r.`{key}` = $prop_{key}")
            set_params[f"prop_{key}"] = value

        # Join SET clauses with comma, ensuring no trailing comma
        set_clause_str = ", ".join(set_clauses) if set_clauses else ""

        # Build the SQL query with explicit SET assignments
        # All parameters are passed as a single agtype map via :params
        # Inside Cypher, individual values are accessed as $key from the agtype map
        query = text(f"""
            SELECT * FROM cypher('telegram_graph',
                $$ MATCH (a:{start_label} {{{start_merge_key}: $sid}})
                   MATCH (b:{end_label} {{{end_merge_key}: $eid}})
                   MERGE (a)-[r:{edge_label}]->(b)
                   {f"SET {set_clause_str}" if set_clause_str else ""}
                   RETURN r $$,
                CAST(:params AS agtype)
            ) AS (v agtype)
        """)

        # Package all parameters as a single JSON object
        # The agtype map contains all values referenced in the Cypher query
        params_dict = {"sid": start_merge_val, "eid": end_merge_val, **set_params}
        params = json.dumps(params_dict)

        async with self.async_session() as session:
            async with session.begin():
                await session.execute(query, {"params": params})
