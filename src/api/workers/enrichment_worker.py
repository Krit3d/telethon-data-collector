import asyncio
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from sqlalchemy import select, update

from src.config.config import Settings, load_settings
from src.db.database import Database
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
        self._processing_ids: set[int] = set()
        self._acquire_lock = asyncio.Lock()
        self._concurrency = getattr(settings, 'enrichment_concurrency', 15)

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

    def _parse_taxonomy(self) -> tuple[str, dict[str, dict[str, str | None]]]:
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

    def _build_lookup(self, rows: list[dict[str, str]]) -> dict[str, dict[str, str | None]]:
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

        lookup: dict[str, dict[str, str | None]] = {}
        for uid in node_map:
            ext = node_map[uid]["extension"]
            lookup[uid] = {
                "path": _build_path(uid),
                "extension": None if ext.strip() == "" else ext,
            }

        return lookup

    async def _generate_explanation(self, context: str) -> str | None:
        system_prompt = (
            "Analyze the author profile and publications to write a factual, dense summary "
            "of their professional field of expertise in Russian, adhering to these strict rules:\n"
            "1. GROUNDING: Rely strictly on explicit facts. Do not speculate, extrapolate, "
            "or assume unstated credentials (e.g., calling them 'expert' or 'doctor' without explicit proof).\n"
            "2. NOISE FILTER: Ignore social media boilerplate (likes, links, subscribe), transcript artifacts, "
            "and background song lyrics. Focus only on professional, business, or thematic semantic content.\n"
            "3. NO DATA FALLBACK: If the context is empty, generic, or lacks thematic substance, output "
            "exactly: 'Данных для анализа автора недостаточно.' Do not guess.\n"
            "4. STYLE: Use a formal, objective, third-person tone (e.g., 'Автор специализируется на...', "
            "'В публикациях рассматриваются...'). Avoid fluff, generalizations, or praise (e.g., 'делится советами').\n"
            "5. OUTPUT FORMAT: ~150 words (4-5 cohesive sentences) covering: core professional domain, "
            "key specialized topics/services, and target audience or practical application."
        )
        last_exception: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = await self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context},
                    ],
                    temperature=0.15,
                    max_tokens=300,
                )
                explanation = response.choices[0].message.content
                if explanation and explanation.strip():
                    return explanation.strip()
                logger.warning("LLM returned empty explanation on attempt %d", attempt)
                return None
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                if "429" in error_str or "too many requests" in error_str or "timeout" in error_str or "timed out" in error_str:
                    delay = 1.0 * (2 ** (attempt - 1))
                    logger.warning(
                        "Rate limit or timeout on explanation attempt %d/3, retrying in %.1fs: %s",
                        attempt, delay, e,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error("Non-retryable error in explanation generation: %s", e)
                return None
        logger.error("Explanation generation failed after 3 retries: %s", last_exception)
        return None

    async def _call_llm_with_network_retries(self, messages: list[ChatCompletionMessageParam]) -> CategorySelection | None:
        last_exception: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = await self._llm_client.beta.chat.completions.parse(
                    model=self._llm_model,
                    messages=messages,
                    temperature=0.0,
                    response_format=CategorySelection,
                )
                parsed = response.choices[0].message.parsed
                if parsed is not None:
                    return parsed
                return None
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                if "429" in error_str or "too many requests" in error_str or "timeout" in error_str or "timed out" in error_str:
                    delay = 1.0 * (2 ** (attempt - 1))
                    logger.warning(
                        "Network retry %d/3 for category LLM call, waiting %.1fs: %s",
                        attempt, delay, e,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error("Non-retryable error in category LLM call: %s", e)
                return None
        logger.error("Category LLM call failed after 3 network retries: %s", last_exception)
        return None

    async def _identify_category(self, explanation: str) -> tuple[str | None, str | None, str | None]:
        system_prompt = (
            "You are a deterministic classification assistant. Map the provided Russian explanation "
            "to exactly one category ID from the YAML taxonomy."
            "CRITICAL: You are strictly restricted to the exact IDs present in the YAML taxonomy. "
            "Never invent or hallucinate any ID. If the explanation matches no category or contains "
            "'Данных для анализа автора недостаточно', return an empty string for category_id."
        )

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt + "\n\nTaxonomy:\n" + self._cached_yaml_taxonomy},
            {"role": "user", "content": explanation},
        ]

        for attempt in range(1, 3):
            parsed = await self._call_llm_with_network_retries(messages)
            if parsed is None:
                return None, None, None

            category_id = parsed.category_id.strip() if parsed.category_id else ""

            if category_id == "" or category_id == "none":
                return None, None, None

            if category_id in self._taxonomy_dict:
                entry = self._taxonomy_dict[category_id]
                return category_id, entry["path"], entry["extension"]

            if attempt == 1:
                logger.warning(
                    "Invalid category ID on attempt %d for explanation: %.100s, id=%s",
                    attempt, explanation, category_id,
                )
                messages.append({"role": "assistant", "content": f'{{"category_id": "{category_id}"}}'})
                messages.append({
                    "role": "user",
                    "content": (
                        "WARNING: In your previous attempt, you returned an invalid category ID. This is a critical error. Choose STRICTLY from the existing numeric keys inside the provided YAML taxonomy. Do not invent new IDs, and do not write the category name instead of the numeric ID."
                    ),
                })
                continue

            logger.critical(
                "Category validation failed after fallback for account. explanation=%.100s invalid_id=%s",
                explanation, category_id,
            )
            return None, None, None

        logger.error(
            "Category validation failed after 2 attempts for explanation: %.100s",
            explanation,
        )
        return None, None, None

    def _calculate_static_avg_er(
        self, posts: list[dict[str, Any]], subscribers_count: int | None
    ) -> float:
        if not posts:
            return 0.0
        total_er = 0.0
        for post in posts:
            reactions = post.get("reactions_count") or 0
            comments = post.get("comments_count") or 0
            shares = post.get("shares_count") or 0
            views_count = post.get("views_count")
            if views_count is not None and views_count > 0:
                post_er = ((reactions + comments + shares) / views_count) * 100
            elif subscribers_count is not None and subscribers_count > 0:
                post_er = ((reactions + comments + shares) / subscribers_count) * 100
            else:
                post_er = 0.0
            total_er += min(100.0, post_er)
        return total_er / len(posts)

    async def _try_acquire_account(self) -> dict[str, Any] | None:
        async with self._acquire_lock:
            async with self._db.async_session() as session:
                async with session.begin():
                    stmt = (
                        select(Account)
                        .where(Account.status == "verified")
                        .where(
                            select(Content.id)
                            .where(Content.account_id == Account.id)
                            .where(Content.is_enriched == False)
                            .exists()
                        )
                    )
                    if self._processing_ids:
                        stmt = stmt.where(Account.id.notin_(self._processing_ids))
                    stmt = stmt.limit(1)
                    result = await session.execute(stmt)
                    account = result.scalar_one_or_none()

                    if account is None:
                        return None

                    account_id = account.id
                    self._processing_ids.add(account_id)

                    content_stmt = (
                        select(Content)
                        .where(Content.account_id == account_id)
                        .where(Content.is_enriched == False)
                    )
                    content_result = await session.execute(content_stmt)
                    posts = list(content_result.scalars().all())

                    if not posts:
                        self._processing_ids.discard(account_id)
                        return None

                    return {
                        "account_id": account_id,
                        "title": account.title,
                        "description": account.description,
                        "subscribers_count": account.subscribers_count,
                        "posts_data": [
                            {
                                "id": p.id,
                                "content": p.content,
                                "transcription": p.transcription,
                                "reactions_count": p.reactions_count,
                                "comments_count": p.comments_count,
                                "shares_count": p.shares_count,
                                "views_count": p.views,
                            }
                            for p in posts
                        ],
                    }

    async def _process_single_unit(self) -> None:
        account_data = await self._try_acquire_account()
        if account_data is None:
            return

        account_id = account_data["account_id"]
        subscribers_count = account_data["subscribers_count"]
        posts_data = account_data["posts_data"]

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

        explanation: str | None = None
        category_id: str | None = None
        category_path: str | None = None
        category_extension: str | None = None
        static_avg_er: float = 0.0

        try:
            explanation = await self._generate_explanation(context)
            if explanation is None:
                logger.error("Skipping save for account %d due to failed explanation generation. Will retry later.", account_id)
                self._processing_ids.discard(account_id)
                return
            if "Данных для анализа автора недостаточно" in explanation:
                category_id = None
                category_path = None
                category_extension = None
            else:
                category_id, category_path, category_extension = await self._identify_category(explanation)
            static_avg_er = self._calculate_static_avg_er(posts_data, subscribers_count)
        except Exception:
            logger.exception("LLM processing failed for account %d", account_id)
            self._processing_ids.discard(account_id)
            return

        try:
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
        except Exception:
            logger.exception("Failed to save enrichment results for account %d", account_id)
        finally:
            self._processing_ids.discard(account_id)

    async def _worker_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                await self._process_single_unit()
            except Exception as e:
                logger.error("Unexpected error in worker loop: %s", e, exc_info=True)

            if self._shutdown_event.is_set():
                break

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.poll_interval,
                )
            except asyncio.TimeoutError:
                pass

    def handle_shutdown(self, *args: object) -> None:
        logger.info("Shutdown signal received, stopping enrichment worker...")
        self._shutdown_event.set()

    async def start(self) -> None:
        logger.info(
            "Enrichment worker starting with concurrency=%d poll_interval=%ds",
            self._concurrency,
            self.poll_interval,
        )
        await self._db.init_db()

        tasks = [asyncio.create_task(self._worker_loop()) for _ in range(self._concurrency)]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Worker tasks cancelled, shutting down...")
        finally:
            self._shutdown_event.set()
            await self._llm_client.close()
            await self._db.close()
            logger.info("Enrichment worker stopped")


async def main() -> None:
    settings = load_settings()
    worker = EnrichmentWorker(settings=settings)
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
