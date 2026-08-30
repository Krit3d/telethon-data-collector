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
        target_languages: list[str] | None = None,
    ) -> dict[int, GraphAuthorEvidence]:
        if not account_ids:
            return {}

        search_tokens = list(set(graph_entities) | set(semantic_topics))

        return await self._graph_repo.get_authors_graph_evidence(
            account_ids=account_ids,
            search_tokens=search_tokens,
            target_languages=target_languages,
        )