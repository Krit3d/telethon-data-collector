import argparse
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from src.config.config import load_settings
from src.db.database import Database
from src.db.models import Content
from src.graph.client import Neo4jClient

logger = logging.getLogger(__name__)

HORMONE_KEYS: list[str] = [
    "score_dopamine",
    "score_oxytocin",
    "score_serotonin",
    "score_cortisol",
    "score_adrenaline",
    "score_endorphin",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and validate psychographic / hormone characteristics of extracted posts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Random sample size of posts to inspect.",
    )
    parser.add_argument(
        "--account-id",
        type=int,
        default=None,
        help="Optional filter by specific account ID.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hormones_inspection.md"),
        help="Path for the output Markdown report.",
    )
    parser.add_argument(
        "--min-hormone-score",
        type=float,
        default=None,
        help="Optional minimum threshold for at least one hormone score.",
    )
    return parser.parse_args()


def _build_cypher_query(account_id: int | None, min_hormone_score: float | None) -> tuple[str, dict]:
    conditions: list[str] = ["p.primary_hormone IS NOT NULL"]
    params: dict = {}

    if account_id is not None:
        conditions.append("p.account_id = $account_id")
        params["account_id"] = account_id

    if min_hormone_score is not None:
        hormone_filters = " OR ".join(
            f"p.{key} >= $min_hormone_score" for key in HORMONE_KEYS
        )
        conditions.append(f"({hormone_filters})")
        params["min_hormone_score"] = min_hormone_score

    where_clause = " AND ".join(conditions)

    query = (
        f"MATCH (p:Post) "
        f"WHERE {where_clause} "
        f"RETURN p.content_id AS content_id, "
        f"p.account_id AS account_id, "
        f"p.language AS language, "
        f"p.tone AS tone, "
        f"p.secondary_tone AS secondary_tone, "
        f"p.primary_hormone AS primary_hormone, "
        f"p.secondary_hormone AS secondary_hormone, "
        f"p.score_dopamine AS score_dopamine, "
        f"p.score_oxytocin AS score_oxytocin, "
        f"p.score_serotonin AS score_serotonin, "
        f"p.score_cortisol AS score_cortisol, "
        f"p.score_adrenaline AS score_adrenaline, "
        f"p.score_endorphin AS score_endorphin, "
        f"p.is_spam_or_gambling AS is_spam_or_gambling, "
        f"p.published_at AS published_at "
        f"ORDER BY rand() "
        f"LIMIT $limit"
    )
    return query, params


def _fmt(val: Any) -> str:
    if val is None:
        return "-"
    return f"{float(val):.1f}"


def _preview(text: str | None) -> str:
    if not text:
        return "-"
    cleaned = text.replace("\n", " ").replace("\r", " ")
    if len(cleaned) > 50:
        return cleaned[:50] + "..."
    return cleaned


def _build_summary_matrix(posts: list[dict], content_map: dict[int, Content]) -> list[str]:
    lines: list[str] = []
    lines.append("## Summary Matrix")
    lines.append("")
    lines.append(
        "| Content ID | Author | Lang | Tone | Primary | Secondary | Dop | Oxy | Ser | Cor | Adr | End | Spam | Preview |"
    )
    lines.append(
        "|-----------:|:------:|:----:|:----:|:-------:|:---------:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|:--------|"
    )
    for post in posts:
        cid = post.get("content_id")
        content_row = content_map.get(int(cid)) if cid is not None else None
        author = f"@{content_row.account.username}" if content_row and content_row.account else "?"
        lang = post.get("language", "?")
        tone = post.get("tone", "?")
        primary = post.get("primary_hormone", "?")
        secondary = post.get("secondary_hormone") or "-"
        dop = _fmt(post.get("score_dopamine"))
        oxy = _fmt(post.get("score_oxytocin"))
        ser = _fmt(post.get("score_serotonin"))
        cor = _fmt(post.get("score_cortisol"))
        adr = _fmt(post.get("score_adrenaline"))
        end = _fmt(post.get("score_endorphin"))
        spam = "S" if post.get("is_spam_or_gambling") else "-"
        preview = _preview(content_row.content if content_row else None)
        lines.append(
            f"| {cid} | {author} | {lang} | {tone} | {primary} | {secondary} | {dop} | {oxy} | {ser} | {cor} | {adr} | {end} | {spam} | {preview} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def _build_post_block(post: dict, content_row: Content | None) -> list[str]:
    lines: list[str] = []

    content_id = post.get("content_id")
    account_id = post.get("account_id")
    username = content_row.account.username if content_row and content_row.account else "?"
    account_title = content_row.account.title if content_row and content_row.account else "?"

    lines.append(f"## Post #{content_id} | @{username} ({account_title}) [ID: {account_id}]")
    lines.append("")

    language = post.get("language", "?")
    tone = post.get("tone", "?")
    secondary_tone = post.get("secondary_tone")
    primary_hormone = post.get("primary_hormone", "?")
    secondary_hormone = post.get("secondary_hormone")
    spam_flag = post.get("is_spam_or_gambling", False)

    tone_str = f"{tone} / {secondary_tone}" if secondary_tone else tone
    hormone_str = f"{primary_hormone} / {secondary_hormone}" if secondary_hormone else primary_hormone

    lines.append(
        f"**Language:** {language} | **Tone:** {tone_str} | "
        f"**Hormones:** {hormone_str} | **Spam:** {spam_flag}"
    )
    lines.append("")

    lines.append("| Dopamine | Oxytocin | Serotonin | Cortisol | Adrenaline | Endorphin |")
    lines.append("|:--------:|:--------:|:---------:|:--------:|:----------:|:---------:|")
    lines.append(
        f"| {_fmt(post.get('score_dopamine'))} "
        f"| {_fmt(post.get('score_oxytocin'))} "
        f"| {_fmt(post.get('score_serotonin'))} "
        f"| {_fmt(post.get('score_cortisol'))} "
        f"| {_fmt(post.get('score_adrenaline'))} "
        f"| {_fmt(post.get('score_endorphin'))} |"
    )
    lines.append("")

    content_text = content_row.content if content_row else None
    if content_text:
        lines.append("### Post Text")
        lines.append("")
        lines.append("```")
        lines.append(content_text)
        lines.append("```")
    else:
        lines.append("### Post Text")
        lines.append("")
        lines.append("*None*")
    lines.append("")

    transcription = content_row.transcription if content_row else None
    if transcription:
        lines.append("### Transcription")
        lines.append("")
        lines.append("```")
        lines.append(transcription)
        lines.append("```")
    else:
        lines.append("### Transcription")
        lines.append("")
        lines.append("*None*")
    lines.append("")

    lines.append("---")
    lines.append("")

    return lines


async def main() -> None:
    args = _parse_args()
    settings = load_settings()

    db = Database(settings.db_url)
    neo4j = Neo4jClient(settings)

    try:
        await neo4j.connect()

        cypher_query, cypher_params = _build_cypher_query(
            args.account_id, args.min_hormone_score,
        )
        cypher_params["limit"] = args.limit

        neo4j_posts = await neo4j.execute_read(cypher_query, cypher_params)
        logger.info("Fetched %d posts from Neo4j", len(neo4j_posts))

        if not neo4j_posts:
            logger.warning("No posts found matching the criteria.")
            output_path = args.output
            if str(output_path.parent) not in ("", "."):
                output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                "# Hormones Inspection Report\n\nNo posts found matching the specified criteria.\n",
                encoding="utf-8",
            )
            print(f"Empty report written to {output_path}")
            return

        content_ids: list[int] = []
        for p in neo4j_posts:
            cid = p.get("content_id")
            if cid is not None:
                content_ids.append(int(cid))

        content_map: dict[int, Content] = {}
        if content_ids:
            async with db.async_session() as session:
                stmt = (
                    select(Content)
                    .options(joinedload(Content.account))
                    .where(Content.id.in_(content_ids))
                )
                result = await session.execute(stmt)
                for row in result.unique().scalars().all():
                    content_map[row.id] = row

        logger.info("Mapped %d content records from PostgreSQL", len(content_map))

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        output_path = args.output
        if str(output_path.parent) not in ("", "."):
            output_path.parent.mkdir(parents=True, exist_ok=True)

        report_lines: list[str] = []
        report_lines.append("# Hormones Inspection Report")
        report_lines.append("")
        report_lines.append(f"- **Generated at:** {now_utc}")
        report_lines.append(f"- **Sample size:** {len(neo4j_posts)} posts")
        report_lines.append(f"- **Limit:** {args.limit}")
        if args.account_id is not None:
            report_lines.append(f"- **Account ID filter:** {args.account_id}")
        if args.min_hormone_score is not None:
            report_lines.append(f"- **Min hormone score filter:** {args.min_hormone_score}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

        report_lines.extend(_build_summary_matrix(neo4j_posts, content_map))

        for post in neo4j_posts:
            cid = post.get("content_id")
            content_row = content_map.get(int(cid)) if cid is not None else None
            report_lines.extend(_build_post_block(post, content_row))

        output_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"Report written to {output_path} ({len(neo4j_posts)} posts)")

    except Exception:
        logger.exception("Unexpected error during hormone inspection")
        raise
    finally:
        await neo4j.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())