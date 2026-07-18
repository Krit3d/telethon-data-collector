from __future__ import annotations

import asyncio
import csv
import logging
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from sqlalchemy import select, update

from src.config.config import Settings, load_settings
from src.db.database import Database, with_retry_on_deadlock
from src.db.models import Account, Content

logger = logging.getLogger(__name__)


class CategorySelection(BaseModel):
    category_id: str


class EnrichmentWorker:

    def __init__(
        self,
        settings: Settings,
        poll_interval: int = 5,
        iab_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.poll_interval = poll_interval
        self._shutdown_event = asyncio.Event()
        self._db = Database(settings.db_url)
        self._llm_client = AsyncOpenAI(
            api_key=settings.cloud_ru_api_key,
            base_url=settings.cloud_ru_base_url,
            timeout=60.0,
        )
        self._llm_model = settings.cloud_ru_llm_model

        if iab_path is None:
            script_dir = Path(__file__).resolve().parent
            found: Path | None = None
            for _ in range(4):
                candidate = script_dir / "src" / "config" / "Content Taxonomy 3.1.tsv"
                if candidate.exists():
                    found = candidate
                    break
                script_dir = script_dir.parent
            if found is not None:
                iab_path = found
            else:
                iab_path = (
                    Path(__file__).resolve().parent.parent.parent.parent
                    / "src"
                    / "config"
                    / "Content Taxonomy 3.1.tsv"
                )
        self._iab_path = Path(iab_path)

        self._cached_yaml_taxonomy, self._taxonomy_dict = self._parse_taxonomy()

    def _parse_taxonomy(self) -> tuple[str, dict[str, dict[str, str]]]:
        path = self._iab_path
        if not path.exists():
            logger.warning("Taxonomy file not found at %s", path)
            return ("", {})

        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            all_rows = list(reader)

        for row in all_rows:
            if not row or not row[0].strip() or not row[0].strip().isdigit():
                continue
            unique_id = row[0].strip()
            parent_id = row[1].strip() if len(row) > 1 and row[1].strip() else ""
            name = row[2].strip() if len(row) > 2 else ""
            extension = row[7].strip() if len(row) > 7 else ""
            if not name:
                continue
            rows.append({
                "unique_id": unique_id,
                "parent_id": parent_id,
                "name": name,
                "extension": extension,
            })

        logger.info("Parsed %d taxonomy entries from %s", len(rows), path)

        tree_dict = self._build_tree(rows)
        yaml_str = self._dict_to_yaml(tree_dict)
        lookup_dict = self._build_lookup(rows)

        logger.info(
            "Built taxonomy YAML (%d chars) and lookup dict with %d entries",
            len(yaml_str),
            len(lookup_dict),
        )

        return yaml_str, lookup_dict

    def _build_tree(self, rows: list[dict[str, str]]) -> dict[str, dict | str]:
        node_map: dict[str, dict] = {}
        children_map: dict[str, list[str]] = {}

        for row in rows:
            uid = row["unique_id"]
            node_map[uid] = {"name": row["name"], "parent_id": row["parent_id"]}
            children_map[uid] = []

        for uid, node in node_map.items():
            parent_id = node["parent_id"]
            if parent_id and parent_id in children_map:
                children_map[parent_id].append(uid)

        def _to_dict(uid: str) -> dict | str:
            children = children_map.get(uid, [])
            if not children:
                return node_map[uid]["name"]
            result: dict[str, dict | str] = {}
            for child_uid in sorted(children, key=lambda x: int(x) if x.isdigit() else x):
                result[child_uid] = _to_dict(child_uid)
            return result

        tree: dict[str, dict | str] = {}
        for uid, node in node_map.items():
            if not node["parent_id"]:
                tree[uid] = _to_dict(uid)

        return tree

    def _dict_to_yaml(self, d: dict[str, dict | str], indent: int = 0) -> str:
        lines: list[str] = []
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, str):
                lines.append(f'{prefix}{key}: {value}')
            elif isinstance(value, dict):
                lines.append(f'{prefix}{key}:')
                lines.append(self._dict_to_yaml(value, indent + 1))
        return "\n".join(lines)

    def _build_lookup(self, rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
        node_map: dict[str, dict[str, str]] = {}
        for row in rows:
            node_map[row["unique_id"]] = {
                "name": row["name"],
                "parent_id": row["parent_id"],
                "extension": row.get("extension", ""),
            }

        def _build_path(uid: str) -> str:
            parts: list[str] = []
            current = uid
            while current and current in node_map:
                parts.insert(0, node_map[current]["name"])
                current = node_map[current]["parent_id"]
            return " > ".join(parts)

        lookup: dict[str, dict[str, str]] = {}
        for uid in node_map:
            lookup[uid] = {
                "path": _build_path(uid),
                "extension": node_map[uid]["extension"],
            }

        return lookup

    async def _generate_explanation(self, context: str) -> str | None:
        system_prompt = (
            "You are an expert analyst. Based on the following context about an author's "
            "activity and content, generate a concise description of the author's expertise "
            "and activity in Russian. The output must be approximately 150 words (4-5 sentences). "
            "Do not include filler words or fluff. Be specific and factual."
        )
        try:
            response = await self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            explanation = response.choices[0].message.content
            if explanation and explanation.strip():
                return explanation.strip()
            logger.warning("LLM returned empty explanation")
            return None
        except Exception as e:
            logger.error("Failed to generate explanation: %s", e)
            return None

    async def _identify_category(
        self, explanation: str
    ) -> tuple[str | None, str | None, str | None]:
        system_prompt = (
            "You are a content categorization assistant. Your task is to choose strictly "
            "one category ID from the provided YAML taxonomy that best matches the author's "
            "expertise described below.\n\n"
            f"Taxonomy:\n{self._cached_yaml_taxonomy}\n\n"
            "Return ONLY the numeric ID as a string. Do not invent new IDs. "
            "Choose strictly from the provided taxonomy."
        )

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": explanation},
        ]

        for attempt in range(1, 3):
            try:
                response = await self._llm_client.beta.chat.completions.parse(
                    model=self._llm_model,
                    messages=messages,
                    temperature=0.1,
                    response_format=CategorySelection,
                )
                parsed = response.choices[0].message.parsed
                if parsed is not None and parsed.category_id:
                    category_id = parsed.category_id.strip()
                    if category_id in self._taxonomy_dict:
                        entry = self._taxonomy_dict[category_id]
                        logger.info(
                            "Category validation succeeded on attempt %d: id=%s path=%s extension=%s",
                            attempt,
                            category_id,
                            entry["path"],
                            entry["extension"],
                        )
                        return category_id, entry["path"], entry["extension"]

                    logger.warning(
                        "Attempt %d: LLM returned invalid category_id=%s, not found in taxonomy",
                        attempt,
                        category_id,
                    )

                if attempt == 1:
                    invalid_id = parsed.category_id.strip() if parsed is not None and parsed.category_id else "null"
                    messages.append({"role": "assistant", "content": invalid_id})
                    messages.append({
                        "role": "user",
                        "content": (
                            "WARNING: In your previous attempt, you returned an invalid ID. "
                            "This is a critical error. Choose STRICTLY from the allowed YAML "
                            "structure and return ONLY the valid numeric ID. Do not invent new IDs."
                        ),
                    })
            except Exception as e:
                logger.error("LLM category identification failed on attempt %d: %s", attempt, e)
                if attempt == 1:
                    messages.append({"role": "assistant", "content": "null"})
                    messages.append({
                        "role": "user",
                        "content": (
                            "WARNING: In your previous attempt, you returned an invalid ID. "
                            "This is a critical error. Choose STRICTLY from the allowed YAML "
                            "structure and return ONLY the valid numeric ID. Do not invent new IDs."
                        ),
                    })

        logger.error(
            "Category validation failed after 2 attempts for explanation: %.100s",
            explanation,
        )
        return None, None, None

    def _calculate_static_avg_er(
        self, posts: list[dict[str, Any]], subscribers_count: int | None
    ) -> float:
        if not subscribers_count or subscribers_count <= 0:
            return 0.0
        if not posts:
            return 0.0
        total_er = 0.0
        for post in posts:
            reactions = post.get("reactions_count", 0) or 0
            comments = post.get("comments_count", 0) or 0
            shares = post.get("shares_count", 0) or 0
            total_er += (reactions + comments + shares) / subscribers_count
        return total_er / len(posts)

    @with_retry_on_deadlock()
    async def _process_single_account(self) -> None:
        async with self._db.async_session() as session:
            async with session.begin():
                subq = (
                    select(Content.account_id)
                    .where(Content.is_enriched == False)
                    .distinct()
                    .subquery()
                )
                stmt = (
                    select(Account)
                    .where(Account.status == "verified")
                    .where(Account.id.in_(select(subq.c.account_id)))
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                result = await session.execute(stmt)
                account = result.scalar_one_or_none()

                if account is None:
                    logger.debug("No verified accounts with unenriched content available for processing")
                    return

                logger.info("Locked account %d (%s) for enrichment", account.id, account.title)

                content_stmt = (
                    select(Content)
                    .where(Content.account_id == account.id)
                    .where(Content.is_enriched == False)
                )
                content_result = await session.execute(content_stmt)
                posts = list(content_result.scalars().all())

                if not posts:
                    logger.warning("Account %d has no unenriched posts, skipping", account.id)
                    return

                account.status = "processing"
                account_id = account.id
                subscribers_count = account.subscribers_count

                account_data = {
                    "id": account.id,
                    "subscribers_count": account.subscribers_count,
                    "title": account.title,
                    "description": account.description,
                }

                posts_data = [
                    {
                        "id": p.id,
                        "content": p.content,
                        "transcription": p.transcription,
                        "reactions_count": p.reactions_count,
                        "comments_count": p.comments_count,
                        "shares_count": p.shares_count,
                    }
                    for p in posts
                ]

        context_parts: list[str] = []
        if account_data["title"]:
            context_parts.append(f"Title: {account_data['title']}")
        if account_data["description"]:
            context_parts.append(f"Description: {account_data['description']}")
        for i, post_data in enumerate(posts_data):
            post_parts: list[str] = []
            if post_data.get("content"):
                post_parts.append(post_data["content"])
            if post_data.get("transcription"):
                post_parts.append(post_data["transcription"])
            if post_parts:
                context_parts.append(f"Post {i + 1}: {' '.join(post_parts)}")

        context = "\n\n".join(context_parts)

        try:
            explanation = await self._generate_explanation(context)
            if explanation is None:
                raise RuntimeError(f"Failed to generate explanation for account {account_id}")

            category_id, category_path, category_extension = await self._identify_category(explanation)
            static_avg_er = self._calculate_static_avg_er(posts_data, subscribers_count)
        except Exception:
            logger.exception("Heavy computation failed for account %d, rolling back status to verified", account_id)
            async with self._db.async_session() as session:
                async with session.begin():
                    await session.execute(
                        update(Account)
                        .where(Account.id == account_id)
                        .values(status="verified")
                    )
            return

        async with self._db.async_session() as session:
            async with session.begin():
                result = await session.execute(
                    select(Account).where(Account.id == account_id)
                )
                account = result.scalar_one()

                account.explanation = explanation
                account.category_id = category_id
                account.category_path = category_path
                account.category_extension = category_extension
                account.static_avg_er = static_avg_er
                account.updated_at = datetime.now(timezone.utc)
                account.status = "verified"

                post_ids = [p["id"] for p in posts_data]
                await session.execute(
                    update(Content)
                    .where(Content.id.in_(post_ids))
                    .values(is_enriched=True, updated_at=datetime.now(timezone.utc))
                )

                logger.info(
                    "Enriched account %d: category_id=%s category_path=%s category_extension=%s "
                    "static_avg_er=%.6f posts=%d",
                    account_id,
                    category_id or "none",
                    category_path or "none",
                    category_extension or "none",
                    static_avg_er,
                    len(post_ids),
                )

    async def _process_iteration(self) -> None:
        try:
            await self._process_single_account()
        except Exception as e:
            logger.error("Unexpected error during enrichment iteration: %s", e, exc_info=True)

    def handle_shutdown(self, *args: object) -> None:
        logger.info("Shutdown signal received, stopping enrichment worker...")
        self._shutdown_event.set()

    async def start(self) -> None:
        logger.info("Enrichment worker started")
        await self._db.init_db()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.handle_shutdown)
            except NotImplementedError:
                pass

        while not self._shutdown_event.is_set():
            await self._process_iteration()

            if self._shutdown_event.is_set():
                break

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.poll_interval,
                )
            except asyncio.TimeoutError:
                pass

        await self._llm_client.close()
        await self._db.close()
        logger.info("Enrichment worker stopped")


async def main() -> None:
    settings = load_settings()
    worker = EnrichmentWorker(settings=settings)
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
