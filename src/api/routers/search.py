from fastapi import APIRouter, Depends

from src.api.dependencies import get_search_service
from src.api.schemas import SearchRequest, SearchResponse
from src.api.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("/")
async def search_content(
    payload: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    return await service.execute_search(payload)
