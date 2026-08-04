import logging
import time

from src.api.schemas import QueryMetadata, SearchRequest, SearchResponse
from src.api.services.search.query_parser import QueryParser
from src.api.services.search.ranker import SearchRanker
from src.api.services.search.retriever import SearchRetriever

logger = logging.getLogger(__name__)


class SearchService:

    def __init__(self, query_parser: QueryParser, retriever: SearchRetriever, ranker: SearchRanker) -> None:
        self._query_parser = query_parser
        self._retriever = retriever
        self._ranker = ranker

    async def execute_search(self, request: SearchRequest) -> SearchResponse:
        start_time = time.perf_counter()

        reformulated = await self._query_parser.parse(request)

        if reformulated is None:
            total_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResponse(
                items=[],
                total=0,
                query_metadata=QueryMetadata(
                    original_query=request.query,
                    dense_query="",
                    graph_entities=[],
                    target_iab_ids=[],
                    resolved_profile_type=request.author_type,
                    execution_time_ms=total_ms,
                    timings={"total_ms": total_ms},
                ),
            )

        candidates, timings, counts = await self._retriever.retrieve_candidates(request, reformulated)

        total_ms = (time.perf_counter() - start_time) * 1000.0

        response = self._ranker.rank_and_format(
            candidates, request, reformulated,
            execution_time_ms=total_ms,
            timings=timings,
            counts=counts,
        )

        return response