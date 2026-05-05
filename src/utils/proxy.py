import base64
import binascii
from typing import Any
from urllib.parse import urlparse

from python_socks import ProxyType


def build_telethon_proxy(proxy_url: str | None) -> dict[str, Any] | None:
    """Build Telethon proxy configuration from URL string."""
    if not proxy_url:
        return None

    if proxy_url.startswith("mtproxy://"):
        proxy_str = proxy_url[10:]

        if "@" not in proxy_str:
            raise ValueError(
                "Invalid MTProxy URL format. Expected: mtproxy://secret@host:port"
            )

        secret, host_port = proxy_str.split("@", 1)

        if ":" not in host_port:
            raise ValueError("Invalid MTProxy URL: missing port")

        addr, port_str = host_port.rsplit(":", 1)
        addr = addr.strip("[]")

        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid port: {port_str}")

        hex_secret = secret

        # If secret contains non-hex chars, assume it's URL-safe base64
        if not all(c in "0123456789abcdefABCDEF" for c in secret):
            try:
                padding = 4 - len(secret) % 4
                secret_padded = secret + "=" * (padding % 4)
                decoded = base64.urlsafe_b64decode(secret_padded)
                hex_secret = binascii.hexlify(decoded).decode()
            except Exception:
                raise ValueError(
                    "Invalid MTProxy secret format (failed base64 decode)."
                )

        return {
            "addr": addr,
            "port": port,
            "secret": hex_secret,
            "is_mtproxy": True,
        }

    parsed = urlparse(proxy_url)
    if not parsed.scheme or not parsed.hostname or not parsed.port:
        raise ValueError(
            "Invalid PROXY_URL. Use e.g. socks5://user:pass@ip:port, http://ip:port, or mtproxy://secret@ip:port"
        )

    scheme = parsed.scheme.lower()
    if scheme in {"socks5", "socks5h"}:
        proxy_type = ProxyType.SOCKS5
    elif scheme == "socks4":
        proxy_type = ProxyType.SOCKS4
    elif scheme in {"http", "https"}:
        proxy_type = ProxyType.HTTP
    else:
        raise ValueError(f"Unsupported proxy scheme: {scheme}")

    proxy_dict = {
        "proxy_type": proxy_type,
        "addr": parsed.hostname.strip("[]"),
        "port": parsed.port,
        "username": parsed.username,
        "password": parsed.password,
    }
    # For SOCKS5/5h, enable remote DNS resolution to avoid client-side DNS leaks
    if scheme in {"socks5", "socks5h"}:
        proxy_dict["rdns"] = True

    return proxy_dict
