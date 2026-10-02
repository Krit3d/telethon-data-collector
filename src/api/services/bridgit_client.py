from typing import Any

import httpx


class BridgitAPIError(RuntimeError):
    pass


class BridgitClient:

    def __init__(
        self,
        api_key: str,
        account_login: str | None = None,
        base_url: str = "https://app.bridgit.me/api/public/v3",
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._account_login = (
            account_login.replace("@", "").strip() if account_login else account_login
        )
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "BridgitClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    async def resolve_account_login(self) -> str:
        if self._account_login:
            return self._account_login
        params: dict[str, Any] = {"api_key": self._api_key}
        response = await self._client.get("/account/my", params=params)
        if response.status_code != 200:
            raise BridgitAPIError(
                f"Get account profile failed with status {response.status_code}: {response.text}"
            )
        data = response.json()
        if not isinstance(data, dict):
            raise BridgitAPIError(f"Unexpected account profile payload: {data}")
        accounts = data.get("accounts")
        if not isinstance(accounts, list) or not accounts:
            raise BridgitAPIError("No active accounts found in Bridgit profile")
        chosen: str | None = None
        for account in accounts:
            if not isinstance(account, dict):
                continue
            login_value = account.get("login")
            if not login_value:
                continue
            candidate = str(login_value).replace("@", "").strip()
            if not candidate:
                continue
            if account.get("status") == "ok":
                chosen = candidate
                break
            if chosen is None:
                chosen = candidate
        if not chosen:
            raise BridgitAPIError("No active accounts found in Bridgit profile")
        self._account_login = chosen
        return chosen

    async def get_account_status(self, login: str | None = None) -> dict[str, Any]:
        target_login = (login or self._account_login or "").replace("@", "").strip()
        if not target_login:
            target_login = await self.resolve_account_login()
        params: dict[str, Any] = {"api_key": self._api_key}
        if target_login:
            params["login"] = target_login
        response = await self._client.get("/account/currentstatus", params=params)
        if response.status_code != 200:
            raise BridgitAPIError(
                f"Get account status failed with status {response.status_code}: {response.text}"
            )
        data = response.json()
        if not isinstance(data, dict):
            raise BridgitAPIError(f"Unexpected account status payload: {data}")
        if data.get("status") != "success":
            raise BridgitAPIError(
                f"Account status error: {data.get('message') or data.get('explain') or data}"
            )
        return data

    async def send_direct_message(
        self,
        recipient: str,
        text: str,
        media_url: str | None = None,
        login: str | None = None,
    ) -> list[str]:
        clean_recipient = recipient.replace("@", "").strip()
        full_text = f"{text}\n{media_url}" if media_url else text
        sender_login = (login or self._account_login or "").replace("@", "").strip()
        if not sender_login:
            sender_login = await self.resolve_account_login()
        data_payload: dict[str, Any] = {
            "api_key": self._api_key,
            "login": sender_login,
            "list": clean_recipient,
            "message": full_text,
        }
        response = await self._client.post("/direct/createlist", data=data_payload)
        if response.status_code != 200:
            raise BridgitAPIError(
                f"Send direct message failed with status {response.status_code}: {response.text}"
            )
        data = response.json()
        if not isinstance(data, dict) or data.get("status") != "success":
            raise BridgitAPIError(
                f"Send direct message error: {data.get('message') or data.get('explain') or data}"
            )
        log = [
            str(x).replace("@", "").strip().lower() for x in data.get("log") or []
        ]
        if clean_recipient.lower() not in log:
            raise BridgitAPIError(
                f"Recipient {clean_recipient} was not queued by Bridgit: {data}"
            )
        return [str(item) for item in data.get("log") or []]

    async def get_direct_stats(
        self,
        login: str | None = None,
        start: int | str | None = None,
        end: int | str | None = None,
    ) -> dict[str, Any]:
        target_login = (login or self._account_login or "").replace("@", "").strip()
        if not target_login:
            target_login = await self.resolve_account_login()
        params: dict[str, Any] = {"api_key": self._api_key}
        if target_login:
            params["login"] = target_login
        start_value = self._coerce_time_param(start)
        if start_value is not None:
            params["start"] = start_value
        end_value = self._coerce_time_param(end)
        if end_value is not None:
            params["end"] = end_value
        response = await self._client.get("/direct/stats", params=params)
        if response.status_code != 200:
            raise BridgitAPIError(
                f"Get direct stats failed with status {response.status_code}: {response.text}"
            )
        data = response.json()
        if not isinstance(data, dict):
            raise BridgitAPIError(f"Unexpected direct stats payload: {data}")
        if data.get("status") != "success":
            raise BridgitAPIError(
                f"Direct stats error: {data.get('message') or data.get('explain') or data}"
            )
        return data

    @staticmethod
    def _coerce_time_param(value: int | str | None) -> int | str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        text = str(value).strip()
        if not text:
            return None
        try:
            return int(float(text))
        except (TypeError, ValueError):
            return text

    async def get_direct_waiting(self, login: str | None = None) -> dict[str, Any]:
        target_login = (login or self._account_login or "").replace("@", "").strip()
        if not target_login:
            target_login = await self.resolve_account_login()
        params: dict[str, Any] = {"api_key": self._api_key}
        if target_login:
            params["login"] = target_login
        response = await self._client.get("/direct/waiting", params=params)
        if response.status_code != 200:
            raise BridgitAPIError(
                f"Get direct waiting failed with status {response.status_code}: {response.text}"
            )
        data = response.json()
        if not isinstance(data, dict):
            raise BridgitAPIError(f"Unexpected direct waiting payload: {data}")
        if data.get("status") != "success":
            raise BridgitAPIError(
                f"Direct waiting error: {data.get('message') or data.get('explain') or data}"
            )
        return data
