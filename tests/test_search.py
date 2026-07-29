import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from src.api.schemas import AuthorSearchResultItem, SearchRequest, SearchResponse
from src.api.services.search import SearchService
from src.api.services.search.query_parser import QueryParser, ParsedQuerySchema
from src.api.services.search.retriever import SearchRetriever, RetrievalResult
from src.api.services.search.ranker import SearchRanker
from src.db.database import Database
from src.embeddings.qdrant_service import QdrantService
from src.graph.db.search_repo import GraphSearchRepository


class MockSettings:
    cloud_ru_api_key = "test-api-key"
    cloud_ru_base_url = "https://test-api.example.com"
    cloud_ru_llm_model = "test-model"
    deepseek_api_key = "test-deepseek-key"
    deepseek_base_url = "https://api.deepseek.com/v1"
    deepseek_llm_model = "deepseek-chat"
    qdrant_collection_name = "social_posts"
    graph_name = "social_graph"


def _create_mock_llm_response(content: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=content))
    ]
    return mock_response


@pytest.fixture
def mock_settings():
    return MockSettings()


@pytest.fixture
def mock_db():
    return AsyncMock(spec=Database)


@pytest.fixture
def mock_qdrant():
    return AsyncMock(spec=QdrantService)


@pytest.fixture
def mock_graph_repo():
    return AsyncMock(spec=GraphSearchRepository)


@pytest.fixture
def search_service(mock_settings, mock_db, mock_qdrant, mock_graph_repo):
    mock_llm_client = MagicMock()
    mock_llm_client.chat.completions.create = AsyncMock()

    mock_query_parser = MagicMock(spec=QueryParser)
    mock_query_parser._llm_client = mock_llm_client
    mock_query_parser._llm_model = mock_settings.deepseek_llm_model
    mock_query_parser._cached_yaml_taxonomy = ""
    mock_query_parser._taxonomy_dict = {}
    mock_query_parser.parse_query = AsyncMock(
        return_value=ParsedQuerySchema(dense_query="test query")
    )
    mock_query_parser.calculate_taxonomy_match_score = MagicMock(return_value=0.0)

    mock_retriever = MagicMock(spec=SearchRetriever)
    mock_retriever._qdrant = mock_qdrant
    mock_retriever._graph_search_repo = mock_graph_repo
    mock_retriever._db = mock_db
    mock_retriever.retrieve_candidates = AsyncMock()

    mock_ranker = MagicMock(spec=SearchRanker)
    mock_ranker._query_parser = mock_query_parser
    mock_ranker.rank_and_fuse = AsyncMock()

    service = SearchService(
        query_parser=mock_query_parser,
        retriever=mock_retriever,
        ranker=mock_ranker,
    )
    service._query_parser = mock_query_parser
    service._retriever = mock_retriever
    service._ranker = mock_ranker
    return service


@pytest.mark.asyncio
async def test_short_query_returns_empty(search_service):
    payload = SearchRequest(query="insurance", limit=10)
    result = await search_service.execute_search(payload)
    assert isinstance(result, SearchResponse)
    assert result.results == []
    search_service._query_parser.parse_query.assert_not_called()
    search_service._retriever.retrieve_candidates.assert_not_called()
    search_service._ranker.rank_and_fuse.assert_not_called()


@pytest.mark.asyncio
async def test_stopwords_only_query_returns_empty(search_service):
    payload = SearchRequest(query="the a is", limit=10)
    result = await search_service.execute_search(payload)
    assert isinstance(result, SearchResponse)
    assert result.results == []
    search_service._query_parser.parse_query.assert_not_called()
    search_service._retriever.retrieve_candidates.assert_not_called()
    search_service._ranker.rank_and_fuse.assert_not_called()


@pytest.mark.asyncio
async def test_happy_path_search_with_mmr_and_contact_enrichment(
    search_service, mock_db, mock_qdrant, mock_graph_repo
):
    mock_query_parser = search_service._query_parser
    mock_retriever = search_service._retriever
    mock_ranker = search_service._ranker

    parsed_query = ParsedQuerySchema(
        dense_query="test query",
        lexical_queries=["tech", "content"],
        graph_entities=["tech"],
        core_tech_entities=["tech"],
        target_domains=["content"],
        negative_domains=[],
        target_iab_ids=[],
        profile_type_intent="both",
    )
    mock_query_parser.parse_query = AsyncMock(return_value=parsed_query)

    mock_qdrant.search_posts.return_value = [
        {"post_id": 101, "score": 0.85, "engagement_rate": 5.5},
        {"post_id": 102, "score": 0.75, "engagement_rate": 3.2},
    ]
    mock_qdrant.search_entities.return_value = []

    mock_graph_repo.search_posts_by_topics.return_value = {}
    mock_graph_repo.search_posts_by_entities.return_value = {}

    current_time = datetime.now(timezone.utc)
    five_days_ago = datetime.fromtimestamp(current_time.timestamp() - 5 * 24 * 3600, tz=timezone.utc)

    mock_db.get_search_candidates.return_value = [
        {
            "id": 101,
            "account_id": 1001,
            "username": "author_one",
            "account_title": "Tech Content Creator",
            "description": "Creating tech content",
            "subscribers_count": 50000,
            "platform": "TELEGRAM",
            "content": "Latest tech trends and reviews",
            "transcription": None,
            "published_at": five_days_ago,
            "created_at": five_days_ago,
            "message_id": 10001,
            "platform_content_id": None,
            "raw_metadata": json.dumps({"contacts": {"email": "author@example.com", "telegram": "@author_one"}}),
            "static_avg_er": 5.5,
            "category_id": None,
            "category_path": None,
            "is_enriched": True,
            "explanation": None,
            "category_extension": None,
            "is_author_blog": None,
        },
        {
            "id": 102,
            "account_id": 1002,
            "username": "author_two",
            "account_title": "Gadget Reviewer",
            "description": "Reviewing the latest gadgets",
            "subscribers_count": 75000,
            "platform": "TELEGRAM",
            "content": "In-depth gadget reviews and comparisons",
            "transcription": None,
            "published_at": five_days_ago,
            "created_at": five_days_ago,
            "message_id": 10002,
            "platform_content_id": None,
            "raw_metadata": json.dumps({"contacts": {}}),
            "static_avg_er": 3.2,
            "category_id": None,
            "category_path": None,
            "is_enriched": True,
            "explanation": None,
            "category_extension": None,
            "is_author_blog": None,
        },
    ]

    retrieval_result = RetrievalResult(
        vector_scores={101: 0.85, 102: 0.75},
        graph_scores={101: 0.5, 102: 0.3},
        graph_post_entities={101: ["ent1"], 102: ["ent2"]},
        topic_post_weights={},
        intersection_post_ids=set(),
        entity_id_to_score={"ent1": 0.9, "ent2": 0.8},
        candidates_rows=mock_db.get_search_candidates.return_value,
        parsed_query=parsed_query,
    )
    mock_retriever.retrieve_candidates = AsyncMock(return_value=retrieval_result)

    mock_ranker.rank_and_fuse.return_value = SearchResponse(
        results=[
            AuthorSearchResultItem(
                author_id="1001",
                username="author_one",
                title="Tech Content Creator",
                description="Creating tech content",
                subscribers_count=50000,
                platform="TELEGRAM",
                final_score=0.85,
                vector_score=0.85,
                graph_score=0.0,
                avg_engagement_rate=5.5,
                explanation="Excellent match for tech project",
                contacts={"email": "author@example.com", "telegram": "@author_one"},
                has_contacts=True,
                is_dormant=False,
                most_recent_post_at=five_days_ago.isoformat(),
            ),
            AuthorSearchResultItem(
                author_id="1002",
                username="author_two",
                title="Gadget Reviewer",
                description="Reviewing the latest gadgets",
                subscribers_count=75000,
                platform="TELEGRAM",
                final_score=0.75,
                vector_score=0.75,
                graph_score=0.0,
                avg_engagement_rate=3.2,
                explanation="Good gadget reviewer",
                contacts=None,
                has_contacts=False,
                is_dormant=False,
                most_recent_post_at=five_days_ago.isoformat(),
            ),
        ],
    )

    payload = SearchRequest(query="looking for tech content creators for product launch", limit=10)
    result = await search_service.execute_search(payload)

    assert isinstance(result, SearchResponse)
    assert len(result.results) == 2

    author_one = next(r for r in result.results if r.author_id == "1001")
    author_two = next(r for r in result.results if r.author_id == "1002")

    assert author_one.final_score == pytest.approx(0.85, abs=0.01)
    assert "Excellent match" in author_one.explanation
    assert author_two.final_score == pytest.approx(0.75, abs=0.01)
    assert "Good gadget reviewer" in author_two.explanation


@pytest.mark.asyncio
async def test_safety_prefiltering_discards_gambling_authors(
    search_service, mock_db, mock_qdrant, mock_graph_repo
):
    mock_retriever = search_service._retriever
    mock_ranker = search_service._ranker

    mock_qdrant.search_posts.return_value = [
        {"post_id": 201, "score": 0.9, "engagement_rate": 10.0},
    ]
    mock_qdrant.search_entities.return_value = []

    mock_graph_repo.search_posts_by_entities.return_value = {}

    current_time = datetime.now(timezone.utc)
    one_day_ago = datetime.fromtimestamp(current_time.timestamp() - 1 * 24 * 3600, tz=timezone.utc)

    candidates_rows = [
        {
            "id": 201,
            "account_id": 2001,
            "username": "gambling_promoter",
            "account_title": "Best 1xbet predictions and casino wins",
            "description": "Daily casino and 1xbet tips",
            "subscribers_count": 100000,
            "platform": "TELEGRAM",
            "content": "Join our casino channel for daily 1xbet wins and kazino bonuses",
            "transcription": None,
            "published_at": one_day_ago,
            "created_at": one_day_ago,
            "message_id": 20001,
            "platform_content_id": None,
            "raw_metadata": json.dumps({}),
            "static_avg_er": 0.0,
            "category_id": None,
            "category_path": None,
            "is_enriched": True,
            "explanation": None,
            "category_extension": None,
            "is_author_blog": None,
        },
    ]
    mock_db.get_search_candidates.return_value = candidates_rows

    parsed_query = ParsedQuerySchema(dense_query="test query")
    retrieval_result = RetrievalResult(
        vector_scores={201: 0.9},
        graph_scores={},
        graph_post_entities={},
        topic_post_weights={},
        intersection_post_ids=set(),
        entity_id_to_score={},
        candidates_rows=candidates_rows,
        parsed_query=parsed_query,
    )
    mock_retriever.retrieve_candidates = AsyncMock(return_value=retrieval_result)

    mock_ranker.rank_and_fuse = AsyncMock(
        return_value=SearchResponse(
            results=[],
            message="По вашему запросу не найдено подходящих авторов. Попробуйте переформулировать запрос или расширить описание проекта.",
        )
    )

    payload = SearchRequest(query="looking for content creators", limit=10)
    result = await search_service.execute_search(payload)

    assert isinstance(result, SearchResponse)
    assert result.results == []


@pytest.mark.asyncio
async def test_external_api_failure_graceful_fallback(
    search_service, mock_db, mock_qdrant, mock_graph_repo
):
    mock_query_parser = search_service._query_parser
    mock_retriever = search_service._retriever
    mock_ranker = search_service._ranker

    parsed_query = ParsedQuerySchema(dense_query="test query")
    mock_query_parser.parse_query = AsyncMock(return_value=parsed_query)

    mock_qdrant.search_posts.return_value = [
        {"post_id": 501, "score": 0.85, "engagement_rate": 6.0},
    ]
    mock_qdrant.search_entities.return_value = []

    mock_graph_repo.search_posts_by_entities.return_value = {}

    current_time = datetime.now(timezone.utc)
    three_days_ago = datetime.fromtimestamp(current_time.timestamp() - 3 * 24 * 3600, tz=timezone.utc)

    candidates_rows = [
        {
            "id": 501,
            "account_id": 5001,
            "username": "test_author",
            "account_title": "Test Content Creator",
            "description": "Creating test content",
            "subscribers_count": 40000,
            "platform": "TELEGRAM",
            "content": "Test content for various projects",
            "transcription": None,
            "published_at": three_days_ago,
            "created_at": three_days_ago,
            "message_id": 50001,
            "platform_content_id": None,
            "raw_metadata": json.dumps({}),
            "static_avg_er": 6.0,
            "category_id": None,
            "category_path": None,
            "is_enriched": True,
            "explanation": None,
            "category_extension": None,
            "is_author_blog": None,
        },
    ]
    mock_db.get_search_candidates.return_value = candidates_rows

    retrieval_result = RetrievalResult(
        vector_scores={501: 0.85},
        graph_scores={},
        graph_post_entities={},
        topic_post_weights={},
        intersection_post_ids=set(),
        entity_id_to_score={},
        candidates_rows=candidates_rows,
        parsed_query=parsed_query,
    )
    mock_retriever.retrieve_candidates = AsyncMock(return_value=retrieval_result)

    mock_ranker.rank_and_fuse = AsyncMock(
        return_value=SearchResponse(
            results=[
                AuthorSearchResultItem(
                    author_id="5001",
                    username="test_author",
                    title="Test Content Creator",
                    description="Creating test content",
                    subscribers_count=40000,
                    platform="TELEGRAM",
                    final_score=0.7,
                    vector_score=0.85,
                    graph_score=0.0,
                    avg_engagement_rate=6.0,
                    explanation="Test explanation",
                    has_contacts=False,
                    is_dormant=False,
                    most_recent_post_at=three_days_ago.isoformat(),
                ),
            ],
        )
    )

    payload = SearchRequest(query="need content creators for project", limit=10)
    result = await search_service.execute_search(payload)

    assert isinstance(result, SearchResponse)
    assert len(result.results) == 1
    assert result.results[0].final_score > 0.0
