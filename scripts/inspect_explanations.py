import asyncio

from sqlalchemy import func, select
from sqlalchemy.orm import joinedload

from src.config.config import Settings
from src.db.database import Database
from src.db.models import Account


async def main() -> None:
    settings = Settings()  # type: ignore[call-arg]
    db = Database(settings.db_url)

    async with db.async_session() as session:
        stmt = (
            select(Account)
            .options(joinedload(Account.content))
            .where(
                Account.status == "verified",
                Account.is_author_blog == True,
                Account.explanation.isnot(None),
                Account.category_id.isnot(None),
            )
            .order_by(func.random())
            .limit(30)
        )
        result = await session.execute(stmt)
        accounts = result.unique().scalars().all()

    lines: list[str] = []
    lines.append("# Explanations Audit Report")
    lines.append("")
    lines.append(
        f"Audit of **{len(accounts)}** verified author blogs with generated explanations and category assignments."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    for account in accounts:
        subscribers = account.subscribers_count or 0

        lines.append(f"## Account #{account.id}")
        lines.append("")
        lines.append(f"- **Username:** @{account.username or 'N/A'}")
        lines.append(f"- **Title:** {account.title}")
        lines.append("")
        lines.append("### Category Details")
        lines.append("")
        lines.append(f"- **Category ID:** {account.category_id}")
        lines.append(f"- **Category Path:** {account.category_path or 'N/A'}")
        lines.append(f"- **Category Extension:** {account.category_extension or 'N/A'}")
        lines.append("")
        lines.append("### Generated Explanation")
        lines.append("")
        lines.append(f"{account.explanation}")
        lines.append("")
        lines.append("### ER Calculation Breakdown")
        lines.append("")
        lines.append(f"- **Subscribers Count:** {subscribers}")
        lines.append("")

        posts = account.content
        lines.append("| Post ID | Reactions | Comments | Shares | Views | Post ER |")
        lines.append("|---------|-----------|----------|--------|-------|---------|")

        post_ers: list[float] = []
        for post in posts:
            reactions = post.reactions_count or 0
            comments = post.comments_count or 0
            shares = post.shares_count or 0
            views = post.views
            total_engagement = reactions + comments + shares
            if views is not None and views > 0:
                post_er = min(30.0, (total_engagement / views) * 100)
            elif subscribers > 0:
                post_er = min(30.0, (total_engagement / subscribers) * 100)
            else:
                post_er = 0.0
            post_ers.append(post_er)
            lines.append(
                f"| {post.id} | {reactions} | {comments} | {shares} | {views or 0} | {post_er:.6f} |"
            )

        lines.append("")

        computed_avg_er = sum(post_ers) / len(post_ers) if post_ers else 0.0
        stored_avg_er = account.static_avg_er or 0.0

        lines.append(f"- **Computed Average Post ER:** {computed_avg_er:.6f}%")
        lines.append(f"- **Stored `static_avg_er`:** {stored_avg_er:.6f}%")
        lines.append("")
        lines.append("---")
        lines.append("")

    output = "\n".join(lines)

    with open("explanations_audit.md", "w", encoding="utf-8") as f:
        f.write(output)

    print(f"explanations_audit.md has been successfully written with {len(accounts)} accounts.")

    await db.engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())