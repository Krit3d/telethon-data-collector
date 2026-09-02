import asyncio
import time

from src.api.schemas import AuthorSearchResultItem, AuthorVectorAggregate, BrandAnalysisRequest, BrandAnalysisResponse, DbsfScoredCandidate, GraphAuthorEvidence, QueryMetadata, SearchPlanRequest, SearchPlanResponse, SearchRequest, SearchResponse, coerce_hormone, coerce_tone
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

    async def plan_campaign(self, request: SearchPlanRequest) -> SearchPlanResponse:
        return await self._query_parser.plan(request)

    async def analyze_brand(self, payload: BrandAnalysisRequest) -> BrandAnalysisResponse:
        return await self._query_parser.analyze_brand(payload)

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
                    target_languages=reformulated.target_languages,
                    resolved_profile_type=request.author_type,
                    execution_time_ms=total_ms,
                    timings=timings,
                    affinity_dense_query=reformulated.affinity_dense_query,
                    negative_topics=reformulated.negative_topics,
                ),
                inferred_filters=reformulated.inferred_filters,
            )

        is_author_blog: bool | None = (
            True if request.author_type == "expert"
            else False if request.author_type == "business"
            else None
        )

        audience_clusters = request.audience_clusters if request.audience_clusters else reformulated.audience_clusters
        direct_query = request.direct_cluster.dense_query if request.direct_cluster and request.direct_cluster.dense_query else reformulated.dense_query

        retrieval_tasks = [
            self._retriever.retrieve_vector_candidates(
                dense_query=direct_query, is_author_blog=is_author_blog,
            )
        ]
        for cluster in audience_clusters:
            retrieval_tasks.append(
                self._retriever.retrieve_vector_candidates(
                    dense_query=cluster.dense_query, is_author_blog=is_author_blog,
                )
            )

        retrieval_start = time.perf_counter()
        retrieval_results = await asyncio.gather(*retrieval_tasks)
        timings["vector_retrieval_ms"] = (time.perf_counter() - retrieval_start) * 1000.0

        direct_aggregates, direct_timings = retrieval_results[0]
        timings.update(direct_timings)
        cluster_aggregates: list[dict[int, AuthorVectorAggregate]] = []
        for result in retrieval_results[1:]:
            aggregates, cluster_timings = result
            timings.update(cluster_timings)
            cluster_aggregates.append(aggregates)

        all_aggregates = [direct_aggregates, *cluster_aggregates]
        if not any(all_aggregates):
            total_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResponse(
                items=[],
                total=0,
                query_metadata=QueryMetadata(
                    original_query=request.query,
                    dense_query=reformulated.dense_query,
                    graph_entities=reformulated.graph_entities,
                    semantic_topics=reformulated.semantic_topics,
                    target_languages=reformulated.target_languages,
                    resolved_profile_type=request.author_type,
                    execution_time_ms=total_ms,
                    timings=timings,
                    affinity_dense_query=reformulated.affinity_dense_query,
                    negative_topics=reformulated.negative_topics,
                ),
                confidence_level="NONE",
                message="No relevant content found matching the query topic.",
                inferred_filters=reformulated.inferred_filters,
            )

        graph_start = time.perf_counter()
        direct_semantic_topics = request.direct_cluster.semantic_topics if request.direct_cluster and request.direct_cluster.semantic_topics else reformulated.semantic_topics
        graph_tasks = [
            self._graph_reasoner.reason_author_evidences(
                account_ids=list(direct_aggregates.keys()),
                graph_entities=reformulated.graph_entities,
                semantic_topics=direct_semantic_topics,
                target_languages=reformulated.target_languages,
                negative_topics=reformulated.negative_topics,
                negative_entities=reformulated.negative_entities,
            )
        ]
        for cluster, aggregates in zip(audience_clusters, cluster_aggregates):
            cluster_tokens = cluster.semantic_topics or cluster.dense_query.split()
            graph_tasks.append(
                self._graph_reasoner.reason_author_evidences(
                    account_ids=list(aggregates.keys()),
                    graph_entities=reformulated.graph_entities,
                    semantic_topics=cluster.semantic_topics,
                    search_tokens=cluster_tokens,
                    target_languages=reformulated.target_languages,
                    negative_topics=reformulated.negative_topics,
                    negative_entities=reformulated.negative_entities,
                )
            )
        graph_results = await asyncio.gather(*graph_tasks)
        timings["graph_reasoning_ms"] = (time.perf_counter() - graph_start) * 1000.0

        direct_evidences: dict[int, GraphAuthorEvidence] = graph_results[0]
        cluster_evidences: list[dict[int, GraphAuthorEvidence]] = list(graph_results[1:])

        dbsf_start = time.perf_counter()
        target_tone = reformulated.target_tone.value if reformulated.target_tone else None
        target_hormones = [h.value if hasattr(h, 'value') else str(h) for h in reformulated.target_hormones] if reformulated.target_hormones else None

        direct_candidates = self._dbsf_engine.rank_candidates(
            vector_aggregates=direct_aggregates,
            graph_evidences=direct_evidences,
            match_type="direct",
            affinity_reason=None,
            target_languages=reformulated.target_languages,
            target_tone=target_tone,
            target_hormones=target_hormones,
        )

        cluster_candidate_pools: list[list[DbsfScoredCandidate]] = []
        for cluster, aggregates, evidences in zip(audience_clusters, cluster_aggregates, cluster_evidences):
            if not aggregates:
                cluster_candidate_pools.append([])
                continue
            cluster_candidates = self._dbsf_engine.rank_candidates(
                vector_aggregates=aggregates,
                graph_evidences=evidences,
                match_type="affinity",
                affinity_reason=cluster.name,
                affinity_mode=True,
                target_languages=reformulated.target_languages,
                target_tone=target_tone,
                target_hormones=target_hormones,
            )
            cluster_candidate_pools.append(cluster_candidates)

        direct_ids = {c.account_id for c in direct_candidates}

        affinity_candidates = self._dbsf_engine.combine_multi_cluster_candidates(
            audience_cluster_pools=cluster_candidate_pools,
            direct_account_ids=direct_ids,
        )
        timings["dbsf_fusion_ms"] = (time.perf_counter() - dbsf_start) * 1000.0

        graph_evidences = dict(direct_evidences)
        for evidences in cluster_evidences:
            graph_evidences.update(evidences)

        hydration_start = time.perf_counter()
        hydrated_pairs = await self._hydrator.hydrate_and_filter_candidates(
            direct_candidates=direct_candidates,
            affinity_candidates=affinity_candidates,
            request=request,
            direct_limit=20,
            affinity_limit=20,
        )
        timings["db_hydration_ms"] = (time.perf_counter() - hydration_start) * 1000.0

        items: list[AuthorSearchResultItem] = []
        for scored, hydrated in hydrated_pairs:
            evidence = graph_evidences.get(hydrated.account_id)
            location = (evidence.location_name if evidence and evidence.location_name else (hydrated.raw_metadata.get("location") if hydrated.raw_metadata else None))
            primary_language = evidence.primary_language if evidence else None
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
                    location=location,
                    primary_language=primary_language,
                    match_type=scored.match_type,
                    affinity_reason=scored.affinity_reason,
                    primary_tone=coerce_tone(evidence.primary_tone) if evidence else None,
                    primary_hormone=coerce_hormone(evidence.primary_hormone) if evidence else None,
                )
            )

        total_ms = (time.perf_counter() - start_time) * 1000.0

        query_metadata = QueryMetadata(
            original_query=request.query,
            dense_query=reformulated.dense_query,
            graph_entities=reformulated.graph_entities,
            semantic_topics=reformulated.semantic_topics,
            target_languages=reformulated.target_languages,
            resolved_profile_type=request.author_type,
            execution_time_ms=total_ms,
            timings=timings,
            qdrant_candidates_count=sum(len(aggregates) for aggregates in all_aggregates) if request.include_analytics else None,
            graph_evidences_count=len(graph_evidences) if request.include_analytics else None,
            total_candidates_count=len(direct_candidates) + len(affinity_candidates) if request.include_analytics else None,
            affinity_dense_query=reformulated.affinity_dense_query,
            negative_topics=reformulated.negative_topics,
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
            inferred_filters=reformulated.inferred_filters,
        )