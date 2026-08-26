import time

from src.api.schemas import AuthorSearchResultItem, QueryMetadata, SearchRequest, SearchResponse
from src.api.services.search.query_parser import QueryParser
from src.api.services.search.retriever import VectorRetriever
from src.api.services.search.graph_reasoner import GraphReasoner
from src.api.services.search.dbsf_engine import DbsfRankingEngine
from src.api.services.search.hydrator import PostgresHydrator


class SearchService:

    def __init__(self, query_parser: QueryParser, retriever: VectorRetriever, graph_reasoner: GraphReasoner, dbsf_engine: DbsfRankingEngine, hydrator: PostgresHydrator) -> None:
        self._query_parser = query_parser
        self._retriever = retriever
        self._graph_reasoner = graph_reasoner
        self._dbsf_engine = dbsf_engine
        self._hydrator = hydrator

    async def execute_search(self, request: SearchRequest) -> SearchResponse:
        start_time = time.perf_counter()
        timings: dict[str, float] = {}

        planning_start = time.perf_counter()
        reformulated = await self._query_parser.parse(request)
        timings["planning_ms"] = (time.perf_counter() - planning_start) * 1000.0

        if not reformulated.dense_query:
            total_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResponse(
                items=[],
                total=0,
                query_metadata=QueryMetadata(
                    original_query=request.query,
                    dense_query="",
                    graph_entities=[],
                    semantic_topics=[],
                    resolved_profile_type=request.author_type,
                    execution_time_ms=total_ms,
                    timings=timings,
                ),
            )

        is_author_blog: bool | None = (
            True if request.author_type == "expert"
            else False if request.author_type == "business"
            else None
        )
        vector_aggregates, vector_timings = await self._retriever.retrieve_vector_candidates(
            dense_query=reformulated.dense_query, is_author_blog=is_author_blog,
        )
        timings.update(vector_timings)

        if not vector_aggregates:
            total_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResponse(
                items=[],
                total=0,
                query_metadata=QueryMetadata(
                    original_query=request.query,
                    dense_query=reformulated.dense_query,
                    graph_entities=reformulated.graph_entities,
                    semantic_topics=reformulated.semantic_topics,
                    resolved_profile_type=request.author_type,
                    execution_time_ms=total_ms,
                    timings=timings,
                ),
                confidence_level="NONE",
                message="No relevant content found matching the query topic.",
            )

        graph_start = time.perf_counter()
        account_ids = list(vector_aggregates.keys())
        graph_evidences = await self._graph_reasoner.reason_author_evidences(
            account_ids=account_ids,
            graph_entities=reformulated.graph_entities,
            semantic_topics=reformulated.semantic_topics,
        )
        timings["graph_reasoning_ms"] = (time.perf_counter() - graph_start) * 1000.0

        dbsf_start = time.perf_counter()
        scored_candidates = self._dbsf_engine.rank_candidates(
            vector_aggregates=vector_aggregates,
            graph_evidences=graph_evidences,
        )
        timings["dbsf_fusion_ms"] = (time.perf_counter() - dbsf_start) * 1000.0

        hydration_start = time.perf_counter()
        hydrated_pairs = await self._hydrator.hydrate_and_filter_candidates(
            candidates=scored_candidates,
            request=request,
        )
        timings["db_hydration_ms"] = (time.perf_counter() - hydration_start) * 1000.0

        items: list[AuthorSearchResultItem] = []
        for scored, hydrated in hydrated_pairs:
            items.append(
                AuthorSearchResultItem(
                    account_id=hydrated.account_id,
                    platform=hydrated.platform,
                    username=hydrated.username,
                    title=hydrated.title,
                    url=hydrated.profile_url,
                    final_score=scored.final_score,
                    vector_score=scored.normalized_vector_score if request.include_analytics else None,
                    graph_score=scored.normalized_graph_score if request.include_analytics else None,
                    static_avg_er=hydrated.static_avg_er if request.include_analytics else None,
                    category_path=hydrated.category_path,
                    explanation=hydrated.explanation,
                    contacts=hydrated.contacts,
                    has_contacts=hydrated.has_contacts,
                    subscribers_count=hydrated.subscribers_count,
                )
            )

        total_ms = (time.perf_counter() - start_time) * 1000.0

        query_metadata = QueryMetadata(
            original_query=request.query,
            dense_query=reformulated.dense_query,
            graph_entities=reformulated.graph_entities,
            semantic_topics=reformulated.semantic_topics,
            resolved_profile_type=request.author_type,
            execution_time_ms=total_ms,
            timings=timings,
            qdrant_candidates_count=len(vector_aggregates) if request.include_analytics else None,
            graph_evidences_count=len(graph_evidences) if request.include_analytics else None,
            total_candidates_count=len(scored_candidates) if request.include_analytics else None,
        )

        if not items:
            confidence_level = "NONE"
            warning_message = "No relevant authors found matching your query topic in the current database."
        elif max(item.final_score for item in items) < 0.40:
            confidence_level = "LOW"
            warning_message = "Low confidence match: showing closest available author profiles."
        else:
            confidence_level = "HIGH"
            warning_message = None

        return SearchResponse(
            items=items,
            total=len(items),
            query_metadata=query_metadata,
            confidence_level=confidence_level,
            warning_message=warning_message,
        )