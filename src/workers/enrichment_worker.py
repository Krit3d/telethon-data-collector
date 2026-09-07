import asyncio
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError, field_validator
from sqlalchemy import select, update

from src.config.config import Settings, load_settings
from src.db.database import Database
from src.db.models import Account, Content
from src.utils.cities import canonicalize_city
from src.utils.countries import canonicalize_country

logger = logging.getLogger(__name__)


class EnrichmentLLMResult(BaseModel):
    is_author_blog: bool
    country: str | None = None
    city: str | None = None
    gender: Literal["male", "female", "duo", "neutral", "unknown"]
    explanation: str | None = None

    @field_validator("gender", mode="before")
    @classmethod
    def _normalize_gender(cls, val: Any) -> str:
        if not isinstance(val, str):
            raise ValueError("Gender must be a valid non-empty string")
        cleaned = val.strip().lower()
        if cleaned not in {"male", "female", "duo", "neutral", "unknown"}:
            raise ValueError(
                f"Invalid gender '{val}'. Must strictly be one of: male, female, duo, neutral, unknown"
            )
        return cleaned

    @field_validator("country", mode="before")
    @classmethod
    def _normalize_country(cls, val: Any) -> str | None:
        if val is None or (isinstance(val, str) and not val.strip()):
            return None
        if not isinstance(val, str):
            raise ValueError("Country must be a string or null")
        canonical = canonicalize_country(val)
        if canonical is None:
            raise ValueError(f"Invalid country '{val}'. Must be a valid ISO country or null")
        return canonical

    @field_validator("city", mode="before")
    @classmethod
    def _normalize_city(cls, val: Any) -> str | None:
        if val is None or (isinstance(val, str) and not val.strip()):
            return None
        if not isinstance(val, str):
            raise ValueError("City must be a string or null")
        return canonicalize_city(val)


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

        self._needs_llm = (
            settings.enrichment_enable_demographics
            or settings.enrichment_enable_explanation
            or settings.enrichment_enable_categories
        )
        if self._needs_llm:
            if not settings.deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY must be configured for enrichment worker")
            self._llm_client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                timeout=60.0,
            )
            self._llm_model = settings.deepseek_llm_model
        else:
            self._llm_client = None
            self._llm_model = ""

        self._processing_ids: set[int] = set()
        self._acquire_lock = asyncio.Lock()
        self._concurrency = getattr(settings, 'enrichment_concurrency', 15)

        if settings.enrichment_enable_categories:
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
        else:
            self._iab_path = Path(iab_path) if iab_path else Path("")
            self._cached_yaml_taxonomy = ""
            self._taxonomy_dict = {}

        self._system_prompt = self._build_enrichment_prompt()

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
            if not row or not row[0].strip():
                continue
            header_check = row[0].strip().lower()
            if header_check in ("unique id", "relational id system", "unique_id", "id"):
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
            for child_uid in sorted(children, key=lambda x: (0, int(x)) if x.isdigit() else (1, str(x))):
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

    def _build_enrichment_prompt(self) -> str:
        parts: list[str] = [
            "You are an expert profile analyzer. Produce a structured JSON summary based strictly on context.",
            "CRITICAL LANGUAGE RULE:",
            "All textual values MUST BE 100% WRITTEN IN RUSSIAN. If the source context is in Ukrainian, Kazakh, English, Bulgarian, or any other language, YOU MUST TRANSLATE ALL FACTS INTO RUSSIAN. Outputting in Ukrainian or any non-Russian language is a critical failure.",
            "RULES:",
            "1. GROUNDING: Rely strictly on explicit facts. Do not speculate, extrapolate, or assume unstated credentials.",
            "2. NOISE FILTER: Ignore social media boilerplate (likes, links, subscribe), transcript artifacts, and background song lyrics. Focus only on professional, business, or thematic semantic content.",
            "3. MAXIMUM RECOVERY: Even if provided metadata is minimal, ALWAYS synthesize a full factual account summary based on available context. But NEVER hallucinate.",
            "4. Determine 'is_author_blog': Set true for personal blogs, lifestyle creators, or an individual experts promoting their own professional services/consultations (e.g., a real estate realtor, private doctor, or lawyer). Set false if the account represents a corporate brand, a local business storefront, a retail shop, an educational institution (e.g., a university), or a community platform/club. Note that sponsored content or product advertisements do NOT change a personal/creator blog into a corporate business.",
        ]

        if self.settings.enrichment_enable_demographics:
            parts.extend([
                "5. GENDER: Must be STRICTLY one of: ['male', 'female', 'duo', 'neutral', 'unknown'].",
                "   - 'male': one or more male creators/presenters (solo male creator, brothers, all-male team).",
                "   - 'female': one or more female creators/presenters (solo female creator, sisters, all-female team).",
                "   - 'duo': mixed-gender team, couple, family, or co-hosts of different genders (e.g., husband and wife, family business).",
                "   - 'neutral': strictly faceless accounts with no human host or creator on camera (corporations, network retailers, news aggregators, meme channels).",
                "   - 'unknown': creator presence exists, but the gender cannot be reliably determined from context.",
                "6. COUNTRY: 2-letter ISO 3166-1 alpha-2 code in UPPERCASE (KZ, RU, BY, UZ, etc.) based on toponyms, cities, currencies, phone codes, and geotags. If impossible to determine -> set null.",
                "7. CITY: normalized canonical city name strictly in RUSSIAN orthography (e.g., translate 'Київ' -> 'Киев', 'Одеса' -> 'Одесса', 'Харків' -> 'Харьков', 'Tbilisi' -> 'Тбилиси'). NEVER copy non-Russian spelling verbatim.",
                "   - SOURCES: Identify the city from direct mentions, local addresses, or unique city-specific landmarks (e.g., 'Храм Христа Спасителя' -> 'Москва', 'Красная Поляна' -> 'Сочи', 'Медеу' -> 'Алматы').",
                "   - MULTIPLE CITIES: If the author travels or visits multiple locations, select their primary home base/city of residence (home life, permanent studio/work, family events) rather than tour/vacation spots.",
                "   - FALLBACK: Set null if no city or unique local landmark is mentioned. Do not guess a city from nationwide retail chains (e.g., bookstores, supermarkets) or language alone.",
            ])

        if self.settings.enrichment_enable_explanation:
            parts.extend([
                "8. EXPLANATION: compose a factual, objective, dense professional summary strictly between 100 and 150 words (NEVER below 100 words or 500 characters), strictly in Russian. It must always end with a complete sentence; do not cut off mid-word.",
                "   - STYLE: Do not use forced prefix templates like 'Автор специализируется...' or 'Автор делает...'. Write naturally based on the entity type, for example: 'Официальный профиль Алматинского университета...', 'Экспертный блог практикующего риелтора...'.",
                "   - CONTENT & RATIONALE: Cover in depth:\n"
                "   (a) ENTITY CLASSIFICATION & NATURE: Explicitly state whether the profile is a personal/creator blog or a corporate/business account, explaining HOW it is run (e.g., whether it combines personal experience with business, acts as a personal expert brand, or serves strictly as a corporate storefront/brand account).\n"
                "   (b) CORE TOPICS & SERVICES: Highlight specific products, services, or themes covered in recent posts.\n"
                "   (c) AUDIENCE & BALANCE: Describe the target audience and how personal storytelling vs direct sales/promotions are balanced.\n"
                "   (d) PRACTICAL VALUE: Summarize the practical takeaways or insights provided to followers.\n"
            ])

        example_fields: list[str] = ['"is_author_blog": true']
        if self.settings.enrichment_enable_demographics:
            example_fields.append('"gender": "female"')
            example_fields.append('"country": "KZ"')
            example_fields.append('"city": "Алматы"')
        if self.settings.enrichment_enable_explanation:
            example_fields.append('"explanation": "..."')

        example_json = "{\n  " + ",\n  ".join(example_fields) + "\n}"
        parts.append("EXAMPLE JSON OUTPUT:\n" + example_json)

        return "\n\n".join(parts)

    async def _generate_enrichment(self, context: str) -> EnrichmentLLMResult | None:
        assert self._llm_client is not None
        system_prompt = self._system_prompt
        last_exception: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = await self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"CRITICAL INSTRUCTION: TRANSLATE ALL FACTS TO RUSSIAN. Output language MUST be 100% Russian.\n\nSource context:\n{context}"},
                    ],
                    temperature=0.15,
                    max_tokens=1000,
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "disabled"}},
                )
                content = (response.choices[0].message.content or "").strip()
                parsed = EnrichmentLLMResult.model_validate_json(content)
                if parsed is not None:
                    return parsed
                logger.warning("LLM returned empty enrichment result on attempt %d", attempt)
                return None
            except (ValidationError, ValueError):
                logger.warning("Failed to parse enrichment JSON on attempt %d", attempt)
                if attempt < 3:
                    continue
                return None
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                if "429" in error_str or "too many requests" in error_str or "timeout" in error_str or "timed out" in error_str:
                    delay = 1.0 * (2 ** (attempt - 1))
                    logger.warning(
                        "Rate limit or timeout on enrichment attempt %d/3, retrying in %.1fs: %s",
                        attempt, delay, e,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error("Non-retryable error in enrichment generation: %s", e)
                return None
        logger.error("Enrichment generation failed after 3 retries: %s", last_exception)
        return None

    async def _call_llm_with_network_retries(self, messages: list[ChatCompletionMessageParam]) -> CategorySelection | None:
        assert self._llm_client is not None
        prompt_messages: list[ChatCompletionMessageParam] = list(messages)
        if prompt_messages and isinstance(prompt_messages[0], dict):
            prompt_messages[0] = {  # type: ignore[assignment]
                **prompt_messages[0],
                "content": str(prompt_messages[0].get("content", "")) + (
                    "\n\nEXAMPLE JSON OUTPUT:\n"
                    '{\n'
                    '  "category_id": "EXACT_CATEGORY_ID"\n'
                    '}'
                ),
            }

        last_exception: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = await self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    messages=prompt_messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "disabled"}},
                )
                content = (response.choices[0].message.content or "").strip()
                parsed = CategorySelection.model_validate_json(content)
                if parsed is not None:
                    return parsed
                return None
            except (ValidationError, ValueError):
                logger.warning("Failed to parse category selection JSON on attempt %d", attempt)
                if attempt < 3:
                    continue
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

    async def _identify_category(self, text: str) -> tuple[str | None, str | None, str | None]:
        system_prompt = (
            "You are a deterministic IAB classification assistant. Map the provided Russian profile context "
            "to EXACTLY ONE best-fitting category ID from the provided YAML taxonomy.\n\n"
            "CRITICAL RULES:\n"
            "1. 100% MANDATORY CLASSIFICATION: Every single profile MUST be assigned its single closest matching best-fitting category ID from the taxonomy. Never return an empty string or null category.\n"
            "2. STRICT IDS ONLY: You are strictly restricted to exact category keys existing in the taxonomy. Never invent or hallucinate new IDs."
        )

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt + "\n\nTaxonomy:\n" + self._cached_yaml_taxonomy},
            {"role": "user", "content": text},
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
                    "Invalid category ID on attempt %d for context: %.100s, id=%s",
                    attempt, text, category_id,
                )
                messages.append({"role": "assistant", "content": f'{{"category_id": "{category_id}"}}'})
                messages.append({
                    "role": "user",
                    "content": (
                        "WARNING: In your previous attempt, you returned an invalid category ID. This is a critical error. Choose STRICTLY from the exact existing keys inside the provided YAML taxonomy. Do not invent new IDs, and do not write the category name instead of the exact ID key."
                    ),
                })
                continue

            logger.critical(
                "Category validation failed after fallback for account. context=%.100s invalid_id=%s",
                text, category_id,
            )
            return None, None, None

        logger.error(
            "Category validation failed after 2 attempts for context: %.100s",
            text,
        )
        return None, None, None

    def _calculate_static_avg_er(self, posts: list[dict[str, Any]]) -> float:
        valid_ers: list[float] = []
        for post in posts:
            reactions = post.get("reactions_count")
            views = post.get("views_count")
            if reactions is None or views is None or views < 50:
                continue
            comments = post.get("comments_count") or 0
            shares = post.get("shares_count") or 0
            post_er = min(30.0, ((reactions + comments + shares) / views) * 100.0)
            valid_ers.append(post_er)
        if not valid_ers:
            return 0.0
        return sum(valid_ers) / len(valid_ers)

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
                        .order_by(Content.published_at.desc())
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
                        "is_author_blog": account.is_author_blog,
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

    async def _process_single_unit(self) -> bool:
        account_data = await self._try_acquire_account()
        if account_data is None:
            return False

        account_id = account_data["account_id"]
        posts_data = account_data["posts_data"]
        existing_is_author_blog = account_data["is_author_blog"]

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

        llm_result: EnrichmentLLMResult | None = None
        category_id: str | None = None
        category_path: str | None = None
        category_extension: str | None = None
        static_avg_er: float | None = None

        try:
            if self.settings.enrichment_enable_demographics or self.settings.enrichment_enable_explanation:
                llm_result = await self._generate_enrichment(context)
                if llm_result is None:
                    logger.error("Skipping save for account %d due to failed LLM generation. Will retry later.", account_id)
                    self._processing_ids.discard(account_id)
                    return True

            if self.settings.enrichment_enable_categories:
                category_id, category_path, category_extension = await self._identify_category(context)
                if category_id is None:
                    logger.error("Skipping enrichment save for account %d because category identification failed.", account_id)
                    self._processing_ids.discard(account_id)
                    return True

            if self.settings.enrichment_enable_er:
                static_avg_er = self._calculate_static_avg_er(posts_data)
        except Exception:
            logger.exception("LLM processing failed for account %d", account_id)
            self._processing_ids.discard(account_id)
            return True

        now = datetime.now(timezone.utc)
        values: dict[str, Any] = {"updated_at": now}
        if llm_result is not None:
            if existing_is_author_blog is None:
                values["is_author_blog"] = llm_result.is_author_blog
            if self.settings.enrichment_enable_demographics:
                values["gender"] = llm_result.gender
                values["country"] = llm_result.country
                values["city"] = llm_result.city
            if self.settings.enrichment_enable_explanation:
                values["explanation"] = llm_result.explanation
        if self.settings.enrichment_enable_categories:
            values["category_id"] = category_id
            values["category_path"] = category_path
            values["category_extension"] = category_extension
        if self.settings.enrichment_enable_er:
            values["static_avg_er"] = static_avg_er

        try:
            async with self._db.async_session() as session:
                async with session.begin():
                    await session.execute(
                        update(Account).where(Account.id == account_id).values(**values)
                    )

                    await session.execute(
                        update(Content)
                        .where(Content.account_id == account_id)
                        .values(is_enriched=True, updated_at=now)
                    )

                    logger.info(
                        "Enriched account %d: %s posts=%d",
                        account_id,
                        values,
                        len(posts_data),
                    )
        except Exception:
            logger.exception("Failed to save enrichment results for account %d", account_id)
        finally:
            self._processing_ids.discard(account_id)

        return True

    async def _worker_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                found = await self._process_single_unit()
            except Exception as e:
                logger.error("Unexpected error in worker loop: %s", e, exc_info=True)
                found = False

            if self._shutdown_event.is_set():
                break

            if found:
                await asyncio.sleep(0)
                continue

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
            if self._llm_client is not None:
                await self._llm_client.close()
            await self._db.close()
            logger.info("Enrichment worker stopped")


async def main() -> None:
    settings = load_settings()
    worker = EnrichmentWorker(settings=settings)
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
