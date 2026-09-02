from src.api.schemas import GraphAuthorEvidence
from src.graph.search_repo import Neo4jSearchRepository


class GraphReasoner:

    def __init__(self, graph_repo: Neo4jSearchRepository) -> None:
        self._graph_repo = graph_repo

    async def reason_author_evidences(
        self,
        account_ids: list[int],
        graph_entities: list[str],
        semantic_topics: list[str],
        search_tokens: list[str] | None = None,
        target_languages: list[str] | None = None,
        negative_topics: list[str] | None = None,
        negative_entities: list[str] | None = None,
    ) -> dict[int, GraphAuthorEvidence]:
        if not account_ids:
            return {}

        token_source = search_tokens if search_tokens is not None else list(graph_entities) + list(semantic_topics)
        normalized_tokens = list(dict.fromkeys(token.lower().strip() for token in token_source if token and token.strip()))
        negative_tokens = list(dict.fromkeys(token.lower().strip() for token in (negative_topics or []) + (negative_entities or []) if token and token.strip()))

        return await self._graph_repo.get_authors_graph_evidence(
            account_ids=account_ids,
            search_tokens=normalized_tokens,
            target_languages=target_languages,
            negative_tokens=negative_tokens,
        )