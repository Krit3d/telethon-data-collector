import hashlib
import hmac
from typing import Any

import httpx


class SlideColdClient:

    def __init__(
        self,
        api_key: str,
        account_id: str | None = None,
        base_url: str = "https://slidecold.com",
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._account_id = account_id
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "SlideColdClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    async def send_message(
        self,
        recipient: str,
        text: str,
        media_url: str | None = None,
        account_id: str | None = None,
    ) -> str:
        recipient = recipient.lstrip("@").strip()
        acc_id = account_id or self._account_id
        payload: dict[str, Any] = {"account_id": acc_id, "recipient": recipient, "text": text}
        if media_url:
            payload["media_url"] = media_url
        response = await self._client.post("/api/v1/messages/send", json=payload)
        if response.status_code not in (200, 201):
            raise RuntimeError(f"Send message failed with status {response.status_code}: {response.text}")
        data = response.json()
        message_id = data.get("id") or data.get("message_id") or data.get("data", {}).get("id")
        return str(message_id)

    async def get_replies(self, account_id: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if account_id:
            params["account_id"] = account_id
        response = await self._client.get("/api/v1/replies", params=params)
        if response.status_code != 200:
            response = await self._client.get("/api/v1/conversations/replies", params=params)
        if response.status_code != 200:
            raise RuntimeError(f"Get replies failed with status {response.status_code}: {response.text}")
        data = response.json()
        if isinstance(data, list):
            return data
        replies = data.get("data")
        if isinstance(replies, list):
            return replies
        return []

    @staticmethod
    def verify_webhook_signature(raw_body: bytes, signature: str | None, secret: str) -> bool:
        if not signature or not secret:
            return False
        if signature.startswith("sha256="):
            signature = signature[len("sha256="):]
        digest = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(digest, signature)