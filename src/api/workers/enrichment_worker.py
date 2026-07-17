from __future__ import annotations

import asyncio
import csv
import difflib
import logging
import signal
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel
from sqlalchemy import or_, select, update

from src.config.config import Settings, load_settings
from src.db.database import Database, with_retry_on_deadlock
from src.db.models import Account, Content

logger = logging.getLogger(__name__)


class EnrichmentResponse(BaseModel):
    tier_1: str
    leaf_name: str
    explanation: str
    sentiment: str


@dataclass
class PostData:
    id: int
    content: str | None
    transcription: str | None
    reactions_count: int | None
    comments_count: int | None
    shares_count: int | None


@dataclass
class AccountData:
    id: int
    description: str | None
    subscribers_count: int | None
    posts: list[PostData] = field(default_factory=list)


class EnrichmentWorker:

    def __init__(
        self,
        settings: Settings,
        poll_interval: int = 5,
        batch_size: int = 5,
        iab_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self._shutdown_event = asyncio.Event()
        self._db = Database(settings.db_url)
        self._llm_client = AsyncOpenAI(
            api_key=settings.cloud_ru_api_key,
            base_url=settings.cloud_ru_base_url,
            timeout=60.0,
        )
        self._llm_model = settings.cloud_ru_llm_model
        self._max_concurrent_llm = getattr(settings, 'max_concurrent_llm_requests', None) or 3
        self._llm_semaphore = asyncio.Semaphore(self._max_concurrent_llm)
        if iab_path is None:
            script_dir = Path(__file__).resolve().parent
            found: Path | None = None
            for _ in range(4):
                candidate1 = script_dir / "src" / "config" / "Content Taxonomy 3.1.tsv"
                candidate2 = script_dir / "config" / "Content Taxonomy 3.1.tsv"
                if candidate1.exists():
                    found = candidate1
                    break
                if candidate2.exists():
                    found = candidate2
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
        self._iab_taxonomy: list[dict[str, str]] = []
        self._iab_loaded = False

    def _ensure_iab_loaded(self) -> None:
        if self._iab_loaded:
            return
        if not self._iab_path.exists():
            logger.warning("IAB taxonomy file not found at %s", self._iab_path)
            self._iab_loaded = True
            return
        with self._iab_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            rows = list(reader)
        data_rows = rows[2:]
        for row in data_rows:
            if not row or not row[0].strip():
                continue
            unique_id = row[0].strip()
            tier_1 = row[3].strip() if len(row) > 3 else ""
            tier_2 = row[4].strip() if len(row) > 4 else ""
            tier_3 = row[5].strip() if len(row) > 5 else ""
            tier_4 = row[6].strip() if len(row) > 6 else ""
            parts = [p for p in [tier_1, tier_2, tier_3, tier_4] if p]
            path = " > ".join(parts) if parts else tier_1
            self._iab_taxonomy.append({
                "id": unique_id,
                "path": path,
                "tier_1": tier_1,
                "leaf": parts[-1] if parts else tier_1,
            })
        self._iab_loaded = True
        logger.info("Loaded %d IAB taxonomy entries", len(self._iab_taxonomy))

    def _fuzzy_match_taxonomy(self, tier_1: str, leaf_name: str) -> tuple[str, str]:
        self._ensure_iab_loaded()
        if not self._iab_taxonomy:
            return ("", tier_1)

        all_paths = [entry["path"] for entry in self._iab_taxonomy]
        tier_1_set: set[str] = set()
        tier_1_defaults: dict[str, str] = {}
        for entry in self._iab_taxonomy:
            t1 = entry["tier_1"]
            tier_1_set.add(t1)
            if t1 not in tier_1_defaults:
                tier_1_defaults[t1] = entry["id"]

        path_to_id = {entry["path"]: entry["id"] for entry in self._iab_taxonomy}

        matches = difflib.get_close_matches(leaf_name, all_paths, n=1, cutoff=0.6)
        if matches:
            matched_path = matches[0]
            return (path_to_id[matched_path], matched_path)

        leaf_to_paths: dict[str, list[str]] = {}
        for entry in self._iab_taxonomy:
            leaf = entry["leaf"]
            if leaf not in leaf_to_paths:
                leaf_to_paths[leaf] = []
            leaf_to_paths[leaf].append(entry["path"])

        leaf_matches = difflib.get_close_matches(leaf_name, list(leaf_to_paths.keys()), n=1, cutoff=0.6)
        if leaf_matches:
            best_path = leaf_to_paths[leaf_matches[0]][0]
            return (path_to_id.get(best_path, ""), best_path)

        tier_matches = difflib.get_close_matches(tier_1, list(tier_1_set), n=1, cutoff=0.5)
        if tier_matches:
            matched_t1 = tier_matches[0]
            return (tier_1_defaults.get(matched_t1, ""), matched_t1)

        return ("", tier_1)

    async def _call_llm(self, messages: list[dict[str, str]]) -> EnrichmentResponse | None:
        try:
            response = await self._llm_client.beta.chat.completions.parse(
                model=self._llm_model,
                messages=messages,  # type: ignore
                temperature=0.2,
                response_format=EnrichmentResponse,
            )
            message = response.choices[0].message
            if message.parsed is not None:
                return message.parsed
            if message.refusal:
                logger.warning("LLM refused to respond: %s", message.refusal)
            return None
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return None

    def _calculate_static_er(
        self, posts: list[PostData], subscribers_count: int | None
    ) -> float:
        if not subscribers_count or subscribers_count <= 0:
            return 0.0
        total_er = 0.0
        valid_count = 0
        for post in posts:
            reactions = post.reactions_count or 0
            comments = post.comments_count or 0
            shares = post.shares_count or 0
            er = ((reactions + comments + shares) / subscribers_count) * 100.0
            total_er += er
            valid_count += 1
        if valid_count == 0:
            return 0.0
        return total_er / valid_count

    @with_retry_on_deadlock()
    async def _handle_processing_failure(self, account_data: AccountData) -> None:
        async with self._db.async_session() as session:
            async with session.begin():
                stmt = (
                    select(Account)
                    .where(Account.id == account_data.id)
                    .with_for_update()
                )
                result = await session.execute(stmt)
                account = result.scalar_one_or_none()

                if account is None:
                    return

                raw_metadata: dict[str, object] = dict(account.raw_metadata) if account.raw_metadata else {}
                enrichment_attempts = raw_metadata.get("enrichment_attempts", 0)
                if not isinstance(enrichment_attempts, int):
                    enrichment_attempts = 0
                enrichment_attempts += 1
                raw_metadata["enrichment_attempts"] = enrichment_attempts
                account.raw_metadata = raw_metadata

                now = datetime.now(timezone.utc)

                if enrichment_attempts >= 3:
                    logger.warning(
                        "Account %d has hit the maximum limit of failed attempts and is being parked",
                        account_data.id,
                    )
                    account.status = "rejected"
                    account.updated_at = now

                    post_ids = [p.id for p in account_data.posts]
                    if post_ids:
                        stmt_content = (
                            update(Content)
                            .where(Content.id.in_(post_ids))
                            .values(is_enriched=True)
                        )
                        await session.execute(stmt_content)
                else:
                    account.status = "pending"
                    account.updated_at = now

    async def _process_account_dto(self, account_data: AccountData) -> None:
        if not account_data.posts:
            return

        try:
            static_avg_er = self._calculate_static_er(account_data.posts, account_data.subscribers_count)

            system_prompt = (
                "You are a content categorization assistant. "
                "Analyze the provided author profile and posts, then return a JSON object with exactly four keys: "
                '"tier_1" (best-fitting top-level IAB 3.1 category, e.g. "Medical Health", "Finance"), '
                '"leaf_name" (specific subcategory name, e.g. "Dental Health" or "Personal Finance"), '
                '"explanation" (concise 1-2 sentence Russian description of the author\'s primary focus and expertise), '
                '"sentiment" (string: "positive", "neutral", or "negative" reflecting prevailing sentiment). '
                "Output strictly valid JSON, no markdown, no explanation."
            )

            bio = account_data.description or ""
            context_parts: list[str] = [f"Bio: {bio}"]
            for i, post in enumerate(account_data.posts):
                parts: list[str] = []
                if post.content:
                    parts.append(post.content)
                if post.transcription:
                    parts.append(post.transcription)
                text = " ".join(parts)
                if text:
                    context_parts.append(f"Post {i + 1}: {text[:2000]}")
            user_prompt = "\n\n".join(context_parts)

            llm_result = await self._call_llm([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])

            if llm_result is None:
                await self._handle_processing_failure(account_data)
                return

            llm_tier_1 = llm_result.tier_1
            llm_leaf = llm_result.leaf_name
            explanation = llm_result.explanation
            sentiment = llm_result.sentiment

            category_id, category_path = self._fuzzy_match_taxonomy(llm_tier_1, llm_leaf)
            if not category_id:
                category_path = llm_tier_1

            post_ids = [p.id for p in account_data.posts]
            await self._update_account_enrichment(
                account_id=account_data.id,
                post_ids=post_ids,
                category_id=category_id if category_id else None,
                category_path=category_path if category_path else None,
                explanation=explanation if explanation else None,
                static_avg_er=static_avg_er,
                sentiment=sentiment,
            )

            logger.info(
                "Enriched account %d: category=%s, er=%.4f, sentiment=%s, posts=%d",
                account_data.id,
                category_path or "none",
                static_avg_er,
                sentiment,
                len(post_ids),
            )
        except Exception:
            await self._handle_processing_failure(account_data)

    @with_retry_on_deadlock()
    async def _update_account_enrichment(
        self,
        account_id: int,
        post_ids: list[int],
        category_id: str | None,
        category_path: str | None,
        explanation: str | None,
        static_avg_er: float,
        sentiment: str,
    ) -> None:
        async with self._db.async_session() as session:
            async with session.begin():
                stmt_account = (
                    update(Account)
                    .where(Account.id == account_id)
                    .values(
                        category_id=category_id,
                        category_path=category_path,
                        explanation=explanation,
                        static_avg_er=static_avg_er,
                        static_sentiment=sentiment,
                        status="parsed",
                        updated_at=datetime.now(timezone.utc),
                    )
                )
                await session.execute(stmt_account)

                stmt_content = (
                    update(Content)
                    .where(Content.id.in_(post_ids))
                    .values(is_enriched=True)
                )
                await session.execute(stmt_content)

    async def _process_batch(self) -> None:
        account_dtos: list[AccountData] = []

        async with self._db.async_session() as session:
            async with session.begin():
                subq = (
                    select(Content.account_id)
                    .where(Content.is_enriched == False)
                    .distinct()
                    .subquery()
                )
                lease_threshold = datetime.now(timezone.utc) - timedelta(minutes=5)
                stmt = (
                    select(Account)
                    .where(Account.id.in_(select(subq.c.account_id)))
                    .where(
                        or_(
                            Account.status != "processing",
                            Account.updated_at < lease_threshold,
                        )
                    )
                    .limit(self.batch_size)
                    .with_for_update(skip_locked=True)
                )
                result = await session.execute(stmt)
                accounts = list(result.scalars().all())

                if not accounts:
                    logger.debug("No accounts with unenriched content found")
                    return

                now = datetime.now(timezone.utc)
                for account in accounts:
                    account.status = "processing"
                    account.updated_at = now

                await session.flush()

                account_ids = [a.id for a in accounts]
                posts_stmt = (
                    select(Content)
                    .where(Content.account_id.in_(account_ids))
                    .where(Content.is_enriched == False)
                    .order_by(Content.published_at.desc())
                )
                posts_result = await session.execute(posts_stmt)
                all_posts = list(posts_result.scalars().all())

                posts_by_account: dict[int, list[Content]] = {}
                for post in all_posts:
                    if post.account_id not in posts_by_account:
                        posts_by_account[post.account_id] = []
                    if len(posts_by_account[post.account_id]) < 12:
                        posts_by_account[post.account_id].append(post)

                for account in accounts:
                    account_posts = posts_by_account.get(account.id, [])
                    post_dtos = [
                        PostData(
                            id=post.id,
                            content=post.content,
                            transcription=post.transcription,
                            reactions_count=post.reactions_count,
                            comments_count=post.comments_count,
                            shares_count=post.shares_count,
                        )
                        for post in account_posts
                    ]
                    account_dtos.append(
                        AccountData(
                            id=account.id,
                            description=account.description,
                            subscribers_count=account.subscribers_count,
                            posts=post_dtos,
                        )
                    )

        logger.info("Processing enrichment batch of %d accounts", len(account_dtos))

        async def process_with_semaphore(account_data: AccountData) -> None:
            async with self._llm_semaphore:
                if self._shutdown_event.is_set():
                    return
                try:
                    await self._process_account_dto(account_data)
                except Exception as e:
                    logger.error(
                        "Failed to enrich account %d: %s",
                        account_data.id,
                        e,
                        exc_info=True,
                    )

        tasks = [process_with_semaphore(adto) for adto in account_dtos]
        await asyncio.gather(*tasks)

    def handle_shutdown(self, *args: object) -> None:
        logger.info("Shutdown signal received, stopping enrichment worker...")
        self._shutdown_event.set()

    async def start(self) -> None:
        logger.info("Enrichment worker started")
        await self._db.init_db()
        self._ensure_iab_loaded()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.handle_shutdown)

        while not self._shutdown_event.is_set():
            try:
                await self._process_batch()
            except Exception as e:
                logger.error(
                    "Unexpected error in enrichment worker: %s",
                    e,
                    exc_info=True,
                )

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