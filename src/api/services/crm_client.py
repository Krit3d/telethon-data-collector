import logging
from typing import Any

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class TwentyCrmClient:

    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(10.0, connect=10.0, read=10.0, write=10.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            headers={"Authorization": f"Bearer {self._api_key}"} if self._api_key else {},
        )
        self._auth_client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(10.0, connect=10.0, read=10.0, write=10.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            headers={"Content-Type": "application/json"},
        )

    async def authenticate(self, email: str, password: str) -> dict[str, Any]:
        query = """
        mutation GetLoginToken($email: String!, $password: String!, $origin: String!) {
          getLoginTokenFromCredentials(email: $email, password: $password, origin: $origin) {
            loginToken {
              token
            }
          }
        }
        """
        payload = {
            "query": query,
            "variables": {"email": email.strip(), "password": password, "origin": self._base_url},
        }
        headers = {"Content-Type": "application/json"}
        response = await self._auth_client.post("/metadata", json=payload, headers=headers)
        if response.status_code != 200 or response.json().get("errors"):
            response = await self._auth_client.post("/graphql", json=payload, headers=headers)
        if response.status_code != 200:
            raise ValueError(f"Twenty auth failed with status {response.status_code}")
        data = response.json()
        if data.get("errors"):
            err_msg = data["errors"][0].get("message", "Auth failed")
            raise ValueError(err_msg)
        token = data["data"]["getLoginTokenFromCredentials"]["loginToken"]["token"]
        return {"token": token, "user": {"email": email, "name": email.split("@")[0]}}

    async def get_creators(self, limit: int = 100) -> list[dict[str, Any]]:
        response = await self._client.get(
            "/rest/creators",
            params={"limit": limit, "order": "createdAt:desc"},
        )
        response.raise_for_status()
        data = response.json()
        records = data.get("data")
        if isinstance(records, dict):
            creators = records.get("creators")
            if isinstance(creators, list):
                return creators
            for value in records.values():
                if isinstance(value, list):
                    return value
            return []
        if isinstance(records, list):
            return records
        return []

    async def update_creator_status(self, creator_id: str, status: str) -> dict[str, Any]:
        response = await self._client.patch(
            f"/rest/creators/{creator_id}",
            json={"status": status},
        )
        response.raise_for_status()
        return response.json()

    async def find_creator_by_account_id(self, account_id: str) -> dict[str, Any] | None:
        response = await self._client.get(
            "/rest/creators",
            params={"filter[accountid][eq]": account_id},
        )
        response.raise_for_status()
        data = response.json()
        records = data.get("data")
        if isinstance(records, dict):
            for value in records.values():
                if isinstance(value, list) and value:
                    return value[0]
            return None
        if isinstance(records, list) and records:
            return records[0]
        return None

    async def upsert_creator(self, payload: dict[str, Any]) -> dict[str, Any]:
        account_id = str(payload.get("accountid") or payload.get("accountId"))
        logger.info("Upsert creator: url=%s payload=%s", f"{self._base_url}/rest/creators", payload)
        existing = await self.find_creator_by_account_id(account_id)
        if existing is not None:
            existing_id = existing.get("id")
            response = await self._client.patch(
                f"/rest/creators/{existing_id}",
                json=payload,
            )
            if response.status_code >= 400:
                logger.error("Twenty error %d: %s", response.status_code, response.text)
                raise HTTPException(status_code=response.status_code, detail=f"Twenty CRM error: {response.text}")
            return response.json()
        response = await self._client.post(
            "/rest/creators",
            json=payload,
        )
        if response.status_code >= 400:
            logger.error("Twenty error %d: %s", response.status_code, response.text)
            raise HTTPException(status_code=response.status_code, detail=f"Twenty CRM error: {response.text}")
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._auth_client.aclose()