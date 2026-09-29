"""Renewable gateway access tokens published by an operator-owned OIDC broker.

The broker owns login/refresh and atomically replaces its private token file.
Sandhi validates the bearer token; metadata here pins the intended broker identity.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Any
from urllib.parse import urlsplit

from victor.core.identity.protocols import AccessToken


class GatewayTokenFileCredential:
    """Read a fresh, private broker result per call; never retain refresh tokens."""

    def __init__(self, config: dict[str, Any]) -> None:
        if set(config) != {"token_file", "issuer", "audience", "subject"}:
            raise ValueError("OIDC gateway requires token_file, issuer, audience and subject")
        if not all(isinstance(v, str) and v.strip() for v in config.values()):
            raise ValueError("OIDC gateway identity must be explicit")
        issuer = urlsplit(config["issuer"])
        if (
            issuer.scheme != "https"
            or not issuer.hostname
            or issuer.username
            or issuer.password
            or issuer.query
            or issuer.fragment
        ):
            raise ValueError("OIDC gateway issuer must use HTTPS")
        self._path = Path(config["token_file"]).expanduser()
        if not self._path.is_absolute():
            raise ValueError("OIDC token_file must be absolute")
        self._identity = {k: config[k] for k in ("issuer", "audience", "subject")}

    async def get_token(self, *scopes: str) -> AccessToken:
        """Read an access token for the configured audience, rejecting stale files."""
        if scopes != (self._identity["audience"],):
            raise ValueError("unexpected gateway audience")
        return await asyncio.to_thread(self._read)

    def _read(self) -> AccessToken:
        if os.name != "posix":
            raise ValueError("private token files require POSIX ownership checks")
        # O_NONBLOCK avoids hanging on a substituted FIFO before fstat can reject it.
        fd = os.open(self._path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_size > 65536
            ):
                raise ValueError("gateway token file must be private and bounded")
            raw = source.read(65537)
        if len(raw) > 65536:
            raise ValueError("gateway token file exceeds limit")

        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate credential field")
                result[key] = value
            return result

        data = json.loads(raw, object_pairs_hook=pairs)
        fields = {"access_token", "token_type", "expires_on", *self._identity}
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("invalid gateway token contract")
        if any(data[k] != v for k, v in self._identity.items()):
            raise ValueError("gateway identity changed")
        token, expiry = data["access_token"], data["expires_on"]
        if (
            data["token_type"] != "Bearer"
            or not isinstance(token, str)
            or not 0 < len(token) <= 16384
            or not re.fullmatch(r"[A-Za-z0-9._~+/-]+=*", token)
            or type(expiry) not in (int, float)
            or not math.isfinite(expiry)
            or not 30 < expiry - time.time() <= 3600
        ):
            raise ValueError("invalid or expired gateway token")
        return AccessToken(token, float(expiry))
