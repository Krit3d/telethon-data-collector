import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._buckets: defaultdict[str, deque[float]] = defaultdict(deque)
        self._last_cleanup: float = time.monotonic()
        self._cleanup_interval: float = 300.0

    def get_client_ip(self, request: Request) -> str:
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            cleaned = real_ip.strip()
            if cleaned:
                return cleaned
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            parts = forwarded.split(",")
            candidate = parts[-1].strip() if len(parts) > 1 else parts[0].strip()
            if candidate:
                return candidate
        client = request.client
        if client is not None and client.host:
            return client.host
        return "unknown"

    def _cleanup(self, now: float, window_seconds: int) -> None:
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        stale_keys = [
            key
            for key, bucket in self._buckets.items()
            if not bucket or now - bucket[-1] > window_seconds
        ]
        for key in stale_keys:
            del self._buckets[key]

    async def is_rate_limited(self, key: str, max_requests: int, window_seconds: int) -> bool:
        now = time.monotonic()
        self._cleanup(now, window_seconds)
        bucket = self._buckets[key]
        threshold = now - window_seconds
        while bucket and bucket[0] <= threshold:
            bucket.popleft()
        if len(bucket) >= max_requests:
            return True
        bucket.append(now)
        return False


_limiter = InMemoryRateLimiter()


def rate_limit(max_requests: int, window_seconds: int):
    async def dependency(request: Request) -> None:
        client_ip = _limiter.get_client_ip(request)
        normalized_path = request.url.path.rstrip("/") or "/"
        key = f"{client_ip}:{normalized_path}"
        limited = await _limiter.is_rate_limited(key, max_requests, window_seconds)
        if limited:
            raise HTTPException(
                status_code=429,
                detail="Слишком много запросов. Попробуйте позже.",
                headers={"Retry-After": str(window_seconds)},
            )

    return dependency
