from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.schemas import (
    AuthorVectorAggregate,
    DbsfScoredCandidate,
    GraphAuthorEvidence,
    HydratedAuthorRecord,
    ReformulatedQuery,
    SearchRequest,
    SearchResponse,
)
from src.api.services.search.dbsf_engine import DbsfRankingEngine
from src.api.services.search.graph_reasoner import GraphReasoner
from src.api.services.search.hydrator import PostgresHydrator
from src.api.services.search.search_service import SearchService


class TestDbsfRankingEngine:

    def test_aggregate_author_vector_score_decay(self):
        scores = [0.90, 0.50, 0.40]
        result = DbsfRankingEngine.aggregate_author_vector_score(scores)
        expected = 0.90 + max(0.0, 0.50 - 0.30) / (2.0 ** 0.70) + max(0.0, 0.40 - 0.30) / (3.0 ** 0.70)
        assert result == pytest.approx(expected, rel=1e-4)
        assert result < 0.90 + 0.20 + 0.10
        assert result > 0.90

    def test_aggregate_author_vector_score_empty(self):
        assert DbsfRankingEngine.aggregate_author_vector_score([]) == 0.0

    def test_aggregate_author_vector_score_single(self):
        assert DbsfRankingEngine.aggregate_author_vector_score([0.85]) == 0.85

    def test_calculate_raw_graph_score_priors(self):
        evidence = GraphAuthorEvidence(
            account_id=1,
            topic_coverage_weight=0.50,
            matched_categories=["IAB1"],
            matched_entities_count=3,
            direct_mentions_count=2,
            has_role_relation=True,
            has_tech_relation=True,
            is_creator=True,
            is_promoter=False,
            is_spam_or_gambling=False,
            graph_tms_score=0.0,
            raw_graph_score=0.0,
        )
        mention_signal = min(1.0, 2 * 0.20 + 3 * 0.10)
        ontological_priors = 0.25 + 0.20 + 0.15
        expected_raw = 0.45 * 0.50 + 0.35 * mention_signal + ontological_priors
        raw_score, tms_score = DbsfRankingEngine.calculate_raw_graph_score(evidence)
        assert raw_score == pytest.approx(expected_raw, rel=1e-4)
        assert tms_score == 0.0

    def test_calculate_raw_graph_score_promoter_penalty(self):
        evidence = GraphAuthorEvidence(
            account_id=2,
            topic_coverage_weight=0.50,
            matched_categories=[],
            matched_entities_count=0,
            direct_mentions_count=0,
            has_role_relation=False,
            has_tech_relation=False,
            is_creator=False,
            is_promoter=True,
            is_spam_or_gambling=False,
            graph_tms_score=0.0,
            raw_graph_score=0.0,
        )
        mention_signal = min(1.0, 0 * 0.20 + 0 * 0.10)
        ontological_priors = -0.10
        expected_raw = max(0.0, 0.45 * 0.50 + 0.35 * mention_signal + ontological_priors)
        raw_score, tms_score = DbsfRankingEngine.calculate_raw_graph_score(evidence)
        assert raw_score == pytest.approx(expected_raw, rel=1e-4)

    def test_calculate_raw_graph_score_none(self):
        raw_score, tms_score = DbsfRankingEngine.calculate_raw_graph_score(None)
        assert raw_score == 0.0
        assert tms_score == 0.10

    def test_calculate_raw_graph_score_spam(self):
        evidence = GraphAuthorEvidence(
            account_id=3,
            topic_coverage_weight=0.80,
            matched_categories=[],
            matched_entities_count=0,
            direct_mentions_count=0,
            has_role_relation=False,
            has_tech_relation=False,
            is_creator=False,
            is_promoter=False,
            is_spam_or_gambling=True,
            graph_tms_score=0.0,
            raw_graph_score=0.0,
        )
        raw_score, tms_score = DbsfRankingEngine.calculate_raw_graph_score(evidence)
        assert raw_score == 0.0
        assert tms_score == 0.0

    def test_normalize_distribution_edge_cases(self):
        assert DbsfRankingEngine._normalize_distribution([]) == []
        assert DbsfRankingEngine._normalize_distribution([0.0]) == [0.0]
        assert DbsfRankingEngine._normalize_distribution([5.0]) == [1.0]
        result_all_same = DbsfRankingEngine._normalize_distribution([1.0, 1.0, 1.0])
        assert result_all_same == [1.0, 1.0, 1.0]
        result_all_zero = DbsfRankingEngine._normalize_distribution([0.0, 0.0, 0.0])
        assert result_all_zero == [0.0, 0.0, 0.0]

    def test_rank_candidates_sorting(self):
        vector_aggregates = {
            1: AuthorVectorAggregate(account_id=1, post_scores=[0.9], max_vector_score=0.9, matched_posts_count=1),
            2: AuthorVectorAggregate(account_id=2, post_scores=[0.8], max_vector_score=0.8, matched_posts_count=1),
            3: AuthorVectorAggregate(account_id=3, post_scores=[0.7], max_vector_score=0.7, matched_posts_count=1),
        }
        graph_evidences = {
            1: GraphAuthorEvidence(account_id=1, topic_coverage_weight=0.5, matched_categories=["IAB1"], matched_entities_count=1, direct_mentions_count=1, has_role_relation=True, has_tech_relation=False, is_creator=False, is_promoter=False, is_spam_or_gambling=False, graph_tms_score=0.0, raw_graph_score=0.0),
            2: GraphAuthorEvidence(account_id=2, topic_coverage_weight=0.3, matched_categories=[], matched_entities_count=0, direct_mentions_count=0, has_role_relation=False, has_tech_relation=False, is_creator=False, is_promoter=False, is_spam_or_gambling=False, graph_tms_score=0.0, raw_graph_score=0.0),
            3: GraphAuthorEvidence(account_id=3, topic_coverage_weight=0.1, matched_categories=[], matched_entities_count=0, direct_mentions_count=0, has_role_relation=False, has_tech_relation=False, is_creator=False, is_promoter=False, is_spam_or_gambling=False, graph_tms_score=0.0, raw_graph_score=0.0),
        }
        candidates = DbsfRankingEngine.rank_candidates(vector_aggregates, graph_evidences)
        assert len(candidates) == 3
        for i in range(len(candidates) - 1):
            assert candidates[i].final_score >= candidates[i + 1].final_score
        assert candidates[0].account_id == 1
        assert candidates[2].account_id == 3

    def test_rank_candidates_spam_zeroed(self):
        vector_aggregates = {
            1: AuthorVectorAggregate(account_id=1, post_scores=[0.9], max_vector_score=0.9, matched_posts_count=1),
        }
        graph_evidences = {
            1: GraphAuthorEvidence(account_id=1, topic_coverage_weight=0.0, matched_categories=[], matched_entities_count=0, direct_mentions_count=0, has_role_relation=False, has_tech_relation=False, is_creator=False, is_promoter=False, is_spam_or_gambling=True, graph_tms_score=0.0, raw_graph_score=0.0),
        }
        candidates = DbsfRankingEngine.rank_candidates(vector_aggregates, graph_evidences)
        assert len(candidates) == 1
        assert candidates[0].final_score == 0.0


class TestGraphReasoner:

    @pytest.fixture
    def reasoner(self):
        mock_repo = AsyncMock()
        return GraphReasoner(mock_repo)

    def test_tms_direct_category_match(self, reasoner):
        author_categories = ["IAB1-1", "IAB2"]
        target_iab_ids = ["IAB1-1", "IAB3"]
        ancestor_map = {"IAB1-1": ["IAB1"], "IAB3": ["IAB2"]}
        tms = reasoner._calculate_tms_score(author_categories, target_iab_ids, ancestor_map)
        assert tms == 1.0

    def test_tms_ancestor_category_match(self, reasoner):
        author_categories = ["IAB1"]
        target_iab_ids = ["IAB1-1", "IAB3"]
        ancestor_map = {"IAB1-1": ["IAB1", "IAB0"], "IAB3": ["IAB2"]}
        tms = reasoner._calculate_tms_score(author_categories, target_iab_ids, ancestor_map)
        assert tms == 0.70

    def test_tms_no_match(self, reasoner):
        author_categories = ["IAB5"]
        target_iab_ids = ["IAB1-1", "IAB3"]
        ancestor_map = {"IAB1-1": ["IAB1"], "IAB3": ["IAB2"]}
        tms = reasoner._calculate_tms_score(author_categories, target_iab_ids, ancestor_map)
        assert tms == 0.0

    def test_tms_empty_categories(self, reasoner):
        assert reasoner._calculate_tms_score([], ["IAB1"], {}) == 0.0
        assert reasoner._calculate_tms_score(["IAB1"], [], {}) == 0.0
        assert reasoner._calculate_tms_score([], [], {}) == 0.0


class TestPostgresHydrator:

    def test_extract_contacts_with_contacts(self):
        raw_metadata = {
            "contacts": {
                "emails": ["test@example.com"],
                "phones": ["+79123456789"],
                "telegram_handles": [],
                "telegram_channels": [],
                "telegram_personal": [],
                "advertising_emails": [],
                "advertising_telegrams": [],
            }
        }
        contacts, has_contacts = PostgresHydrator._extract_contacts(raw_metadata)
        assert has_contacts is True
        assert contacts is not None
        assert contacts["emails"] == ["test@example.com"]
        assert contacts["phones"] == ["+79123456789"]

    def test_extract_contacts_no_contacts(self):
        raw_metadata = {
            "contacts": {
                "emails": [],
                "phones": [],
                "telegram_handles": [],
                "telegram_channels": [],
                "telegram_personal": [],
                "advertising_emails": [],
                "advertising_telegrams": [],
            }
        }
        contacts, has_contacts = PostgresHydrator._extract_contacts(raw_metadata)
        assert has_contacts is False
        assert contacts is None

    def test_extract_contacts_none(self):
        contacts, has_contacts = PostgresHydrator._extract_contacts(None)
        assert has_contacts is False
        assert contacts is None

    def test_extract_contacts_empty_dict(self):
        contacts, has_contacts = PostgresHydrator._extract_contacts({})
        assert has_contacts is False
        assert contacts is None

    def test_extract_contacts_fallback_to_root(self):
        raw_metadata = {
            "emails": ["admin@example.com"],
            "phones": [],
            "telegram_handles": [],
            "telegram_channels": [],
            "telegram_personal": [],
            "advertising_emails": [],
            "advertising_telegrams": [],
        }
        contacts, has_contacts = PostgresHydrator._extract_contacts(raw_metadata)
        assert has_contacts is True
        assert contacts is not None
        assert contacts["emails"] == ["admin@example.com"]

    def test_extract_profile_url_instagram(self):
        url = PostgresHydrator._extract_profile_url("instagram", "testuser", None)
        assert url == "https://instagram.com/testuser"

    def test_extract_profile_url_telegram(self):
        url = PostgresHydrator._extract_profile_url("telegram", "testchannel", None)
        assert url == "https://t.me/testchannel"

    def test_extract_profile_url_case_insensitive(self):
        url = PostgresHydrator._extract_profile_url("Instagram", "TestUser", None)
        assert url == "https://instagram.com/TestUser"

    def test_extract_profile_url_from_metadata(self):
        raw_metadata = {"profile_url": "https://custom.url/profile"}
        url = PostgresHydrator._extract_profile_url("instagram", "testuser", raw_metadata)
        assert url == "https://custom.url/profile"

    def test_extract_profile_url_no_username(self):
        url = PostgresHydrator._extract_profile_url("instagram", None, None)
        assert url is None

    def test_extract_profile_url_unknown_platform(self):
        url = PostgresHydrator._extract_profile_url("tiktok", "testuser", None)
        assert url is None


class TestSearchService:

    @pytest.fixture
    def mock_query_parser(self):
        parser = MagicMock()
        parser.parse = AsyncMock()
        return parser

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.retrieve_vector_candidates = AsyncMock()
        return retriever

    @pytest.fixture
    def mock_graph_reasoner(self):
        reasoner = AsyncMock()
        return reasoner

    @pytest.fixture
    def mock_dbsf_engine(self):
        engine = MagicMock()
        engine.rank_candidates = MagicMock()
        return engine

    @pytest.fixture
    def mock_hydrator(self):
        hydrator = MagicMock()
        hydrator.hydrate_and_filter_candidates = AsyncMock()
        return hydrator

    @pytest.fixture
    def search_service(self, mock_query_parser, mock_retriever, mock_graph_reasoner, mock_dbsf_engine, mock_hydrator):
        return SearchService(
            query_parser=mock_query_parser,
            retriever=mock_retriever,
            graph_reasoner=mock_graph_reasoner,
            dbsf_engine=mock_dbsf_engine,
            hydrator=mock_hydrator,
        )

    @pytest.mark.asyncio
    async def test_execute_search_empty_query(self, search_service, mock_query_parser):
        mock_query_parser.parse.return_value = ReformulatedQuery(
            dense_query="",
            graph_entities=[],
            target_topics=[],
            target_iab_ids=[],
            profile_type_intent="expert",
        )
        request = SearchRequest(query="  ", limit=10)
        response = await search_service.execute_search(request)
        assert isinstance(response, SearchResponse)
        assert response.total == 0
        assert response.items == []
        assert response.query_metadata is not None
        assert response.query_metadata.original_query == ""
        assert response.query_metadata.dense_query == ""

    @pytest.mark.asyncio
    async def test_execute_search_full_pipeline(
        self,
        search_service,
        mock_query_parser,
        mock_retriever,
        mock_graph_reasoner,
        mock_dbsf_engine,
        mock_hydrator,
    ):
        mock_query_parser.parse.return_value = ReformulatedQuery(
            dense_query="test query",
            graph_entities=["entity1"],
            target_topics=["topic1"],
            target_iab_ids=["IAB1"],
            profile_type_intent="expert",
        )

        vector_aggregates = {
            1: AuthorVectorAggregate(
                account_id=1,
                post_scores=[0.95, 0.85],
                max_vector_score=0.95,
                matched_posts_count=2,
            ),
            2: AuthorVectorAggregate(
                account_id=2,
                post_scores=[0.80],
                max_vector_score=0.80,
                matched_posts_count=1,
            ),
        }
        vector_timings = {"embedding_ms": 15.0, "qdrant_posts_ms": 45.0}
        mock_retriever.retrieve_vector_candidates.return_value = (vector_aggregates, vector_timings)

        graph_evidences = {
            1: GraphAuthorEvidence(
                account_id=1,
                topic_coverage_weight=0.5,
                matched_categories=["IAB1"],
                matched_entities_count=1,
                direct_mentions_count=2,
                has_role_relation=True,
                has_tech_relation=False,
                is_creator=False,
                is_promoter=False,
                is_spam_or_gambling=False,
                graph_tms_score=0.0,
                raw_graph_score=0.0,
            ),
            2: GraphAuthorEvidence(
                account_id=2,
                topic_coverage_weight=0.3,
                matched_categories=[],
                matched_entities_count=0,
                direct_mentions_count=0,
                has_role_relation=False,
                has_tech_relation=False,
                is_creator=False,
                is_promoter=False,
                is_spam_or_gambling=False,
                graph_tms_score=0.0,
                raw_graph_score=0.0,
            ),
        }
        mock_graph_reasoner.reason_author_evidences.return_value = graph_evidences

        scored_candidates = [
            DbsfScoredCandidate(
                account_id=1,
                raw_vector_score=0.95,
                raw_graph_score=0.50,
                normalized_vector_score=0.85,
                normalized_graph_score=0.75,
                tms_score=1.0,
                final_score=0.85,
            ),
            DbsfScoredCandidate(
                account_id=2,
                raw_vector_score=0.80,
                raw_graph_score=0.30,
                normalized_vector_score=0.60,
                normalized_graph_score=0.40,
                tms_score=0.0,
                final_score=0.50,
            ),
        ]
        mock_dbsf_engine.rank_candidates.return_value = scored_candidates

        hydrated_record = HydratedAuthorRecord(
            account_id=1,
            platform="telegram",
            username="test_channel",
            title="Test Channel",
            category_id="IAB1",
            category_path="Technology",
            explanation="Relevant author",
            static_avg_er=0.05,
            subscribers_count=10000,
            is_author_blog=True,
            raw_metadata={"key": "value"},
            contacts=None,
            has_contacts=False,
            profile_url="https://t.me/test_channel",
        )
        mock_hydrator.hydrate_and_filter_candidates.return_value = [
            (scored_candidates[0], hydrated_record),
        ]

        request = SearchRequest(
            query="test query",
            limit=10,
            author_type="expert",
            include_contacts=False,
            include_analytics=True,
        )
        response = await search_service.execute_search(request)

        assert isinstance(response, SearchResponse)
        assert response.total == 1
        assert len(response.items) == 1
        assert response.items[0].account_id == 1
        assert response.items[0].platform == "telegram"
        assert response.items[0].username == "test_channel"
        assert response.items[0].title == "Test Channel"
        assert response.items[0].url == "https://t.me/test_channel"
        assert response.items[0].final_score == 0.85
        assert response.items[0].vector_score == 0.85
        assert response.items[0].graph_score == 0.75
        assert response.items[0].tms_score == 1.0
        assert response.items[0].has_contacts is False
        assert response.items[0].contacts is None
        assert response.items[0].subscribers_count == 10000
        assert response.items[0].category_path == "Technology"
        assert response.items[0].explanation == "Relevant author"
        assert response.query_metadata is not None
        assert response.query_metadata.original_query == "test query"
        assert response.query_metadata.dense_query == "test query"
        assert response.query_metadata.graph_entities == ["entity1"]
        assert response.query_metadata.target_iab_ids == ["IAB1"]
        assert response.query_metadata.resolved_profile_type == "expert"
        assert response.confidence_level == "HIGH"
        assert response.warning_message is None

        mock_query_parser.parse.assert_awaited_once_with(request)
        mock_retriever.retrieve_vector_candidates.assert_awaited_once_with(dense_query="test query")
        mock_graph_reasoner.reason_author_evidences.assert_awaited_once()
        mock_dbsf_engine.rank_candidates.assert_called_once_with(
            vector_aggregates=vector_aggregates,
            graph_evidences=graph_evidences,
        )
        mock_hydrator.hydrate_and_filter_candidates.assert_awaited_once_with(
            candidates=scored_candidates,
            request=request,
        )

    @pytest.mark.asyncio
    async def test_execute_search_no_vector_results(self, search_service, mock_query_parser, mock_retriever):
        mock_query_parser.parse.return_value = ReformulatedQuery(
            dense_query="some query",
            graph_entities=[],
            target_topics=[],
            target_iab_ids=[],
            profile_type_intent="expert",
        )
        mock_retriever.retrieve_vector_candidates.return_value = ({}, {"embedding_ms": 5.0, "qdrant_posts_ms": 10.0})
        request = SearchRequest(query="some query", limit=10)
        response = await search_service.execute_search(request)
        assert response.total == 0
        assert response.items == []
        assert response.query_metadata is not None
        assert response.query_metadata.dense_query == "some query"
        assert response.confidence_level == "NONE"
        assert response.message == "No relevant content found matching the query topic."
        assert response.warning_message is None
