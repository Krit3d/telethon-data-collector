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

    async def register_user(self, email: str, password: str) -> dict[str, Any]:
        email = email.strip()
        query = """
        mutation SignUp($email: String!, $password: String!) {
          signUp(email: $email, password: $password) {
            loginToken {
              token
            }
          }
        }
        """
        payload = {
            "query": query,
            "variables": {"email": email, "password": password},
        }
        headers = {"Content-Type": "application/json"}
        response = await self._auth_client.post("/graphql", json=payload, headers=headers)
        token: str | None = None
        if response.status_code == 200:
            data = response.json()
            if not data.get("errors"):
                sign_up = data.get("data", {}).get("signUp")
                if isinstance(sign_up, dict):
                    login_token = sign_up.get("loginToken")
                    if isinstance(login_token, dict):
                        candidate = login_token.get("token")
                        if isinstance(candidate, str) and candidate:
                            token = candidate
        if token is None:
            token = await self._fallback_register(email, password)
        return {"token": token, "user": {"email": email, "name": email.split("@")[0]}}

    async def _fallback_register(self, email: str, password: str) -> str:
        create_query = """
        mutation CreateUser($email: String!, $password: String!) {
          createUser(data: { email: $email, password: $password }) {
            id
          }
        }
        """
        payload = {
            "query": create_query,
            "variables": {"email": email, "password": password},
        }
        headers = {"Content-Type": "application/json"}
        response = await self._auth_client.post("/graphql", json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if not data.get("errors") and data.get("data", {}).get("createUser"):
                return await self._login_token(email, password)
        invite_query = """
        mutation InviteUser($email: String!) {
          inviteUser(email: $email) {
            id
          }
        }
        """
        invite_payload = {
            "query": invite_query,
            "variables": {"email": email},
        }
        response = await self._auth_client.post("/graphql", json=invite_payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if not data.get("errors") and data.get("data", {}).get("inviteUser"):
                return await self._login_token(email, password)
        raise ValueError("Twenty registration failed: signUp mutation unavailable")

    async def _login_token(self, email: str, password: str) -> str:
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
            "variables": {"email": email, "password": password, "origin": self._base_url},
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
        return token

    async def get_creators(self, limit: int = 100, user_email: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit, "order": "createdAt:desc"}
        if user_email:
            params["filter[useremail][eq]"] = user_email
        response = await self._client.get(
            "/rest/creators",
            params=params,
        )
        response.raise_for_status()
        data = response.json()
        records = data.get("data")
        if isinstance(records, dict):
            creators = records.get("creators")
            if isinstance(creators, list):
                logger.info("Twenty CRM: found %d creators", len(creators))
                return creators
            for value in records.values():
                if isinstance(value, list):
                    logger.info("Twenty CRM: found %d creators", len(value))
                    return value
            logger.info("Twenty CRM: found 0 creators")
            return []
        if isinstance(records, list):
            logger.info("Twenty CRM: found %d creators", len(records))
            return records
        logger.info("Twenty CRM: found 0 creators")
        return []

    async def get_creator_by_id(self, creator_id: str) -> dict[str, Any] | None:
        try:
            response = await self._client.get(f"/rest/creators/{creator_id.strip()}")
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        data = response.json()
        record = data.get("data")
        if isinstance(record, dict):
            creator = record.get("creator")
            if isinstance(creator, dict):
                return creator
            return record
        return None

    async def update_creator_status(self, creator_id: str, status: str) -> dict[str, Any]:
        response = await self._client.patch(
            f"/rest/creators/{creator_id}",
            json={"status": status},
        )
        if response.status_code == 404:
            existing = await self.find_creator_by_account_id(creator_id)
            if existing is not None:
                existing_id = existing.get("id")
                if existing_id:
                    response = await self._client.patch(
                        f"/rest/creators/{existing_id}",
                        json={"status": status},
                    )
        response.raise_for_status()
        return response.json()

    async def find_creator_by_account_id(self, account_id: str) -> dict[str, Any] | None:
        params: dict[str, Any] = {"filter[accountid][eq]": str(account_id).strip()}
        response = await self._client.get(
            "/rest/creators",
            params=params,
        )
        response.raise_for_status()
        data = response.json()
        records = data.get("data")
        creators: list[dict[str, Any]] = []
        if isinstance(records, dict):
            for value in records.values():
                if isinstance(value, list):
                    creators = value
                    break
        elif isinstance(records, list):
            creators = records
        target = str(account_id).strip()
        for c in creators:
            c_acc = str(c.get("accountid") or c.get("accountId") or "").strip()
            if c_acc == target:
                return c
        return None

    async def delete_creator(self, identifier: str) -> bool:
        target = str(identifier).strip()
        if not target:
            return False
        response = await self._client.delete(f"/rest/creators/{target}")
        if response.status_code in {200, 204}:
            return True
        if response.status_code == 404:
            existing = await self.find_creator_by_account_id(target)
            if existing is not None:
                existing_id = existing.get("id")
                if existing_id:
                    retry = await self._client.delete(f"/rest/creators/{existing_id}")
                    if retry.status_code in {200, 204}:
                        return True
        return False

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