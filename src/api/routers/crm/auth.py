import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select

from src.api.dependencies import get_crm_client, get_db
from src.api.schemas import CrmLoginRequest, CrmLoginResponse, CrmRegisterRequest
from src.api.services.crm_client import TwentyCrmClient
from src.db.database import Database
from src.db.models import User
from src.utils.rate_limiter import rate_limit
from src.utils.security import create_access_token, hash_password, verify_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["CRM Auth"])


@router.post("/login", response_model=CrmLoginResponse)
async def crm_login(
    payload: CrmLoginRequest,
    request: Request,
    db: Database = Depends(get_db),
    crm_client: TwentyCrmClient = Depends(get_crm_client),
    _rate_limit: None = Depends(rate_limit(max_requests=5, window_seconds=60)),
) -> CrmLoginResponse:
    email = payload.email.strip()
    async with db.async_session() as session:
        stmt = select(User).where(func.lower(User.email) == email.lower())
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()

    if user is not None and bool(user.password_hash):
        if not verify_password(payload.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Неверный email или пароль")
        access_token = create_access_token(
            {"sub": user.email, "user_id": user.id},
            request.app.state.settings.secret_key,
        )
        return CrmLoginResponse(
            token=access_token,
            user={"id": user.id, "email": user.email, "name": user.name or user.email.split("@")[0]},
        )

    try:
        data = await crm_client.authenticate(payload.email, payload.password)
    except ValueError:
        raise HTTPException(status_code=401, detail="Неверный email или пароль")
    except Exception as exc:
        logger.exception("Twenty login failed")
        raise HTTPException(status_code=500, detail=f"Twenty service error: {exc}")
    token = data.get("token")
    user_data = data.get("user")
    if not isinstance(token, str) or not token or not isinstance(user_data, dict):
        raise HTTPException(status_code=401, detail="Неверный email или пароль")
    user_email = user_data.get("email")
    if not isinstance(user_email, str) or not user_email:
        user_email = email
    user_name = user_data.get("name")
    if not isinstance(user_name, str) or not user_name:
        user_name = user_email.split("@")[0]
    async with db.async_session() as session:
        stmt = select(User).where(func.lower(User.email) == user_email.lower())
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        if user is None:
            user = User(email=user_email, name=user_name, password_hash="")
            session.add(user)
        else:
            if user.email != user_email:
                user.email = user_email
            if not user.name and user_name:
                user.name = user_name
        await session.commit()
        user_id = user.id
        user_email = user.email
        user_name = user.name or user_email.split("@")[0]
    access_token = create_access_token(
        {"sub": user_email, "user_id": user_id},
        request.app.state.settings.secret_key,
    )
    return CrmLoginResponse(
        token=access_token,
        user={"id": user_id, "email": user_email, "name": user_name},
    )


@router.post("/register", response_model=CrmLoginResponse)
async def crm_register(
    payload: CrmRegisterRequest,
    request: Request,
    db: Database = Depends(get_db),
    _rate_limit: None = Depends(rate_limit(max_requests=5, window_seconds=60)),
) -> CrmLoginResponse:
    email = payload.email.strip()
    async with db.async_session() as session:
        stmt = select(User).where(func.lower(User.email) == email.lower())
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing is not None:
            raise HTTPException(status_code=400, detail="Регистрация с указанными данными невозможна")

        user = User(
            email=email,
            name=payload.name,
            password_hash=hash_password(payload.password),
        )
        session.add(user)
        await session.commit()
        user_id = user.id
        user_email = user.email
        user_name = user.name or user_email.split("@")[0]

    access_token = create_access_token(
        {"sub": user_email, "user_id": user_id},
        request.app.state.settings.secret_key,
    )
    return CrmLoginResponse(
        token=access_token,
        user={"id": user_id, "email": user_email, "name": user_name},
    )