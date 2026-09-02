from fastapi import APIRouter, Depends

from src.api.dependencies import get_search_service
from src.api.schemas import BrandAnalysisRequest, BrandAnalysisResponse, SearchPlanRequest, SearchPlanResponse, SearchRequest, SearchResponse
from src.api.services.search import SearchService

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("/", response_model=SearchResponse, response_model_exclude_none=True)
async def search_content(
    payload: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    return await service.execute_search(payload)


@router.post("/plan", response_model=SearchPlanResponse, response_model_exclude_none=True)
async def plan_search(
    payload: SearchPlanRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchPlanResponse:
    return await service.plan_campaign(payload)


@router.post("/analyze-brand", response_model=BrandAnalysisResponse, response_model_exclude_none=True)
async def analyze_brand(
    payload: BrandAnalysisRequest,
    service: SearchService = Depends(get_search_service),
) -> BrandAnalysisResponse:
    return await service.analyze_brand(payload)
