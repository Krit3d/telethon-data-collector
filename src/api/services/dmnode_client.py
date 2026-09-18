import hashlib
import hmac
from datetime import datetime, timezone
from typing import Any

import httpx


class DMnodeClient:

    def __init__(self, api_key: str, base_url: str = "https://dmnode.com", timeout: float = 30.0) -> None:
        self._api_key = api_key
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

    async def __aenter__(self) -> "DMnodeClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    async def run_safety_check(self, payload: dict[str, Any]) -> str:
        response = await self._client.post("/api/campaigns/dry-run", json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Safety check failed with status {response.status_code}: {response.text}")
        data = response.json()
        if isinstance(data, list):
            data = data[0] if data else {}
        approval_token = data.get("approvalToken")
        if approval_token is None:
            approval_token = data.get("data", {}).get("approvalToken")
        if approval_token is None:
            raise RuntimeError("Safety check response missing approvalToken")
        return str(approval_token)

    async def create_campaign(self, payload: dict[str, Any], approval_token: str) -> str:
        cloned = dict(payload)
        cloned["dryRunApprovalToken"] = approval_token
        response = await self._client.post("/api/campaigns", json=cloned)
        if response.status_code != 200:
            raise RuntimeError(f"Create campaign failed with status {response.status_code}: {response.text}")
        data = response.json()
        campaign = data.get("campaign", {})
        campaign_id = campaign.get("id") if isinstance(campaign, dict) else None
        if campaign_id is None:
            campaign_id = data.get("id")
        if campaign_id is None:
            raise RuntimeError("Create campaign response missing campaign id")
        return str(campaign_id)

    async def start_campaign(self, campaign_id: str) -> bool:
        response = await self._client.post(f"/api/campaigns/{campaign_id}/start", json={})
        return response.status_code in (200, 201, 204)

    async def get_campaign_status(self, campaign_id: str) -> dict[str, Any]:
        response = await self._client.get(f"/api/campaigns/{campaign_id}")
        if response.status_code != 200:
            raise RuntimeError(f"Get campaign status failed with status {response.status_code}: {response.text}")
        return response.json()

    async def get_replies(self) -> list[dict[str, Any]]:
        response = await self._client.get("/api/replies")
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
    def verify_webhook_signature(raw_body: bytes, header: str | None, secret: str) -> bool:
        if not header or not secret:
            return False
        parts = header.split(",")
        t: str | None = None
        v1: str | None = None
        for part in parts:
            pair = part.strip().split("=", 1)
            if len(pair) != 2:
                continue
            key, value = pair[0].strip(), pair[1].strip()
            if key == "t":
                t = value
            elif key == "v1":
                v1 = value
        if t is None or v1 is None:
            return False
        try:
            timestamp = float(t)
            if timestamp > 1e12:
                timestamp = timestamp / 1000.0
            ts = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except ValueError:
            try:
                ts = datetime.fromisoformat(t)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except ValueError:
                return False
        if abs((datetime.now(timezone.utc) - ts).total_seconds()) > 300:
            return False
        digest = hmac.new(secret.encode("utf-8"), f"{t}.{raw_body.decode('utf-8')}".encode("utf-8"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(digest, v1)