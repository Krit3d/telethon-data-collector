import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest  # type: ignore

from src.api.schemas import SearchRequest, SearchResponse
from src.api.services.search_service import SearchService
from src.db.database import Database
from src.embeddings.qdrant_service import QdrantService
from src.graph.db.search_repo import GraphSearchRepository


class MockSettings:
    cloud_ru_api_key = "test-api-key"
    cloud_ru_base_url = "https://test-api.example.com"
    cloud_ru_llm_model = "test-model"


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
    db = AsyncMock(spec=Database)
    return db


@pytest.fixture
def mock_qdrant():
    qdrant = AsyncMock(spec=QdrantService)
    return qdrant


@pytest.fixture
def mock_graph_repo():
    graph_repo = AsyncMock(spec=GraphSearchRepository)
    return graph_repo


@pytest.fixture
def search_service(mock_settings, mock_db, mock_qdrant, mock_graph_repo):
    service = SearchService(
        settings=mock_settings,
        qdrant=mock_qdrant,
        db=mock_db,
        graph_search_repo=mock_graph_repo,
    )
    service._llm_client = MagicMock()
    service._llm_client.chat.completions.create = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_short_query_returns_empty(search_service):
    payload = SearchRequest(query="insurance", limit=10)
    result = await search_service.execute_search(payload)
    assert isinstance(result, SearchResponse)
    assert result.results == []
    search_service._qdrant.search_posts.assert_not_called()
    search_service._graph_search_repo.search_posts_by_entities.assert_not_called()
    search_service._llm_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_stopwords_only_query_returns_empty(search_service):
    payload = SearchRequest(query="the a is", limit=10)
    result = await search_service.execute_search(payload)
    assert isinstance(result, SearchResponse)
    assert result.results == []
    search_service._qdrant.search_posts.assert_not_called()
    search_service._graph_search_repo.search_posts_by_entities.assert_not_called()
    search_service._llm_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_happy_path_search_with_mmr_and_contact_enrichment(
    search_service, mock_db, mock_qdrant, mock_graph_repo
):
    search_service._reformulate_query = AsyncMock(return_value=("test query", ["entity1", "entity2"]))

    mock_qdrant.search_posts.return_value = [
        {"post_id": 101, "score": 0.85, "engagement_rate": 5.5},
        {"post_id": 102, "score": 0.75, "engagement_rate": 3.2},
    ]
    mock_qdrant.search_entities.return_value = [
        {"entity_id": "ent1", "score": 0.9, "label": "Person"},
        {"entity_id": "ent2", "score": 0.8, "label": "Brand"},
    ]

    mock_graph_repo.search_posts_by_entities.return_value = (
        {101: ["ent1", "ent2"], 102: ["ent1"]},
        {101: 0.06, 102: 0.04},
    )

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
        },
    ]

    search_service._llm_client.chat.completions.create.return_value = _create_mock_llm_response(
        json.dumps([
            {"author_id": 1001, "final_score": 0.77, "explanation": "Excellent match for tech project"},
            {"author_id": 1002, "final_score": 0.85, "explanation": "Good gadget reviewer"},
        ])
    )

    payload = SearchRequest(query="looking for tech content creators for product launch", limit=10)
    result = await search_service.execute_search(payload)

    assert isinstance(result, SearchResponse)
    assert len(result.results) == 2

    author_one = next(r for r in result.results if r.author_id == 1001)
    author_two = next(r for r in result.results if r.author_id == 1002)

    assert author_one.final_score == pytest.approx(0.92, abs=0.01)
    assert "В профиле автора найдены контактные данные" in author_one.explanation
    assert author_two.final_score == pytest.approx(0.85, abs=0.01)
    assert "В профиле автора найдены контактные данные" not in author_two.explanation


@pytest.mark.asyncio
async def test_safety_prefiltering_discards_gambling_authors(
    search_service, mock_db, mock_qdrant, mock_graph_repo
):
    search_service._reformulate_query = AsyncMock(return_value=("test query", []))

    mock_qdrant.search_posts.return_value = [
        {"post_id": 201, "score": 0.9, "engagement_rate": 10.0},
    ]
    mock_qdrant.search_entities.return_value = []

    mock_graph_repo.search_posts_by_entities.return_value = ({}, {})

    current_time = datetime.now(timezone.utc)
    one_day_ago = datetime.fromtimestamp(current_time.timestamp() - 1 * 24 * 3600, tz=timezone.utc)

    mock_db.get_search_candidates.return_value = [
        {
            "id": 201,
            "account_id": 2001,
            "username": "gambling_promoter",
            "account_title": "Best 1xbet predictions and casino wins",
            "description": "Daily casino and 1xbet tips",
            "subscribers_count": 100000,
            "platform": "TELEGRAM",
            "content": "Join our casino channel for daily 1xbet wins and kазино bonuses",
            "transcription": None,
            "published_at": one_day_ago,
            "created_at": one_day_ago,
            "message_id": 20001,
            "platform_content_id": None,
            "raw_metadata": json.dumps({}),
        },
    ]

    payload = SearchRequest(query="looking for content creators", limit=10)
    result = await search_service.execute_search(payload)

    assert isinstance(result, SearchResponse)
    assert result.results == []


@pytest.mark.asyncio
async def test_dormant_accounts_filtering_active_author_only(
    search_service, mock_db, mock_qdrant, mock_graph_repo
):
    search_service._reformulate_query = AsyncMock(return_value=("test query", []))

    mock_qdrant.search_posts.return_value = [
        {"post_id": 301, "score": 0.8, "engagement_rate": 4.0},
        {"post_id": 302, "score": 0.7, "engagement_rate": 3.0},
    ]
    mock_qdrant.search_entities.return_value = []

    mock_graph_repo.search_posts_by_entities.return_value = ({}, {})

    current_time = datetime.now(timezone.utc)
    five_days_ago = datetime.fromtimestamp(current_time.timestamp() - 5 * 24 * 3600, tz=timezone.utc)
    two_hundred_days_ago = datetime.fromtimestamp(current_time.timestamp() - 200 * 24 * 3600, tz=timezone.utc)

    mock_db.get_search_candidates.return_value = [
        {
            "id": 301,
            "account_id": 3001,
            "username": "active_author",
            "account_title": "Active Content Creator",
            "description": "Regular content creator",
            "subscribers_count": 30000,
            "platform": "TELEGRAM",
            "content": "Regular content about technology",
            "transcription": None,
            "published_at": five_days_ago,
            "created_at": five_days_ago,
            "message_id": 30001,
            "platform_content_id": None,
            "raw_metadata": json.dumps({}),
        },
        {
            "id": 302,
            "account_id": 3002,
            "username": "dormant_author",
            "account_title": "Inactive Creator",
            "description": "Former content creator",
            "subscribers_count": 50000,
            "platform": "TELEGRAM",
            "content": "Old content about various topics",
            "transcription": None,
            "published_at": two_hundred_days_ago,
            "created_at": two_hundred_days_ago,
            "message_id": 30002,
            "platform_content_id": None,
            "raw_metadata": json.dumps({}),
        },
    ]

    payload = SearchRequest(query="content creators wanted", limit=1)
    result = await search_service.execute_search(payload)

    assert isinstance(result, SearchResponse)
    assert len(result.results) == 1
    assert result.results[0].author_id == 3001


@pytest.mark.asyncio
async def test_dormant_accounts_penalty_when_no_active_authors(
    search_service, mock_db, mock_qdrant, mock_graph_repo
):
    search_service._reformulate_query = AsyncMock(return_value=("test query", []))

    mock_qdrant.search_posts.return_value = [
        {"post_id": 401, "score": 0.9, "engagement_rate": 5.0},
    ]
    mock_qdrant.search_entities.return_value = []

    mock_graph_repo.search_posts_by_entities.return_value = ({}, {})

    current_time = datetime.now(timezone.utc)
    two_hundred_days_ago = datetime.fromtimestamp(current_time.timestamp() - 200 * 24 * 3600, tz=timezone.utc)

    mock_db.get_search_candidates.return_value = [
        {
            "id": 401,
            "account_id": 4001,
            "username": "dormant_only_author",
            "account_title": "Former Tech Creator",
            "description": "Previously active tech creator",
            "subscribers_count": 80000,
            "platform": "TELEGRAM",
            "content": "Old technology content",
            "transcription": None,
            "published_at": two_hundred_days_ago,
            "created_at": two_hundred_days_ago,
            "message_id": 40001,
            "platform_content_id": None,
            "raw_metadata": json.dumps({}),
        },
    ]

    search_service._llm_client.chat.completions.create.return_value = _create_mock_llm_response(
        json.dumps([
            {"author_id": 4001, "final_score": 0.88, "explanation": "Former tech creator"},
        ])
    )

    payload = SearchRequest(query="tech content needed", limit=10)
    result = await search_service.execute_search(payload)

    assert isinstance(result, SearchResponse)
    assert len(result.results) == 1
    dormant_author = result.results[0]
    assert dormant_author.author_id == 4001
    assert dormant_author.final_score < 0.88
    assert dormant_author.final_score > 0.0


@pytest.mark.asyncio
async def test_external_api_failure_graceful_fallback(
    search_service, mock_db, mock_qdrant, mock_graph_repo
):
    search_service._reformulate_query = AsyncMock(return_value=("test query", []))

    mock_qdrant.search_posts.return_value = [
        {"post_id": 501, "score": 0.85, "engagement_rate": 6.0},
    ]
    mock_qdrant.search_entities.return_value = []

    mock_graph_repo.search_posts_by_entities.return_value = ({}, {})

    current_time = datetime.now(timezone.utc)
    three_days_ago = datetime.fromtimestamp(current_time.timestamp() - 3 * 24 * 3600, tz=timezone.utc)

    mock_db.get_search_candidates.return_value = [
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
        },
    ]

    search_service._llm_client.chat.completions.create.side_effect = openai.APIError(
        message="Test API error",
        request=MagicMock(),
        body=MagicMock(),
    )

    payload = SearchRequest(query="need content creators for project", limit=10)
    result = await search_service.execute_search(payload)

    assert isinstance(result, SearchResponse)
    assert len(result.results) == 1
    assert result.results[0].explanation == ""
    assert result.results[0].final_score > 0.0
