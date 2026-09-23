"""Opt-in OAuth boundary for local validation observers.

The operator owns the profile and broker command. Bootstrap credentials stay with
that broker; only short-lived access tokens enter this process. This is not an OS
sandbox against members with unrestricted same-user tools.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import ssl
import sys
import time
from typing import Any, AsyncIterator, cast
from urllib.parse import urlsplit

from aiohttp import ClientResponse, ClientSession

from victor.core.identity.cache import CachingTokenCredential
from victor.core.identity.protocols import AccessToken

MAX_BYTES = 65536
MINIMUM_LIFETIME = 150  # Preserve the 120-second gateway deadline plus clock margin.
ACQUISITION_TIMEOUT = 20.0
PRIVATE_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "x-api-key",
        "api-key",
        "x-goog-api-key",
        "x-sandhi-grant",
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
    }
)


class GatewayOAuthError(ValueError):
    """Redacted, terminal local credential/configuration failure."""

    http_status = 401


class GatewayBrokerUnavailable(GatewayOAuthError):
    """Acquisition failed without establishing that an identity is invalid."""

    http_status = 503


def require(condition: Any, message: str) -> None:
    if not condition:
        raise GatewayOAuthError(message)


def strict_json(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "duplicate JSON field")
            result[key] = value
        return result

    def constant(_: str) -> Any:
        raise GatewayOAuthError("non-finite JSON number")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError):
        raise GatewayOAuthError("invalid structured JSON") from None


def fields(value: Any, names: set[str]) -> None:
    require(isinstance(value, dict) and set(value) == names, "unexpected contract fields")


def text(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and len(value) <= 4096


class BrokerCredential:
    """Strict fixed-command credential source; caching belongs to the core decorator."""

    def __init__(self, config: dict[str, Any]):
        fields(config, {"command", "issuer", "audience", "subject"})
        command = config["command"]
        require(
            isinstance(command, list) and command and all(text(x) for x in command),
            "broker command must be a fixed argv list",
        )
        require(Path(command[0]).is_absolute(), "broker executable must be absolute")
        require(
            all(text(config[k]) for k in ("issuer", "audience", "subject")),
            "broker identity must be explicit",
        )
        self._command = tuple(command)
        self.identity = tuple(config[k] for k in ("issuer", "audience", "subject"))

    async def get_token(self, *scopes: str) -> AccessToken:
        require(scopes == (self.identity[1],), "unexpected credential audience")
        require(os.name == "posix", "OAuth broker requires POSIX process groups")
        process: asyncio.subprocess.Process | None = None
        finished = False
        try:
            # Never forward provider credentials from the harness environment.
            environment = {
                key: os.environ[key]
                for key in ("PATH", "HOME", "LANG", "SSH_AUTH_SOCK")
                if key in os.environ
            }
            process = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=environment,
                start_new_session=True,
            )
            assert process.stdout is not None
            output = bytearray()
            while chunk := await process.stdout.read(min(4096, MAX_BYTES + 1 - len(output))):
                output.extend(chunk)
                require(len(output) <= MAX_BYTES, "broker response exceeds limit")
            returncode = await process.wait()
            finished = True
            if returncode != 0:
                raise GatewayBrokerUnavailable("credential broker unavailable")
            result = strict_json(bytes(output))
            fields(
                result,
                {"access_token", "token_type", "expires_on", "issuer", "audience", "subject"},
            )
            require(
                tuple(result[k] for k in ("issuer", "audience", "subject")) == self.identity,
                "credential identity changed",
            )
            token = result["access_token"]
            require(
                result["token_type"] == "Bearer"
                and isinstance(token, str)
                and re.fullmatch(r"[A-Za-z0-9._~+/-]+=*", token)
                and len(token) <= 16384,
                "invalid bearer credential",
            )
            expires = result["expires_on"]
            require(
                type(expires) in (float, int)
                and math.isfinite(expires)
                and MINIMUM_LIFETIME < expires - time.time() <= 3600,
                "credential lifetime outside accepted bounds",
            )
            return AccessToken(token, float(expires))
        except asyncio.CancelledError:
            raise
        except GatewayOAuthError:
            raise
        except Exception:
            raise GatewayBrokerUnavailable("credential broker unavailable") from None
        finally:
            if process is not None and not finished:
                primary = sys.exception()
                cleanup_errors: list[str] = []
                # Once the leader is reaped its PID/group can be reused. Detached
                # descendants are outside this broker's containment guarantee.
                if process.returncode is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except Exception as exc:
                        cleanup_errors.append("signal:" + type(exc).__name__)
                # asyncio exposes no public reader-close API. Closing the owned
                # subprocess transport releases inherited pipes without signaling
                # an already-exited leader's potentially recycled process group.
                try:
                    cast(Any, process)._transport.close()
                except Exception as exc:
                    cleanup_errors.append("close:" + type(exc).__name__)
                try:
                    await asyncio.wait_for(process.wait(), timeout=1)
                except Exception as exc:
                    cleanup_errors.append("reap:" + type(exc).__name__)
                if cleanup_errors:
                    note = "credential broker cleanup failed (" + ",".join(cleanup_errors) + ")"
                    if primary is not None:
                        primary.add_note(note)
                    else:
                        raise GatewayBrokerUnavailable(note) from None


class GatewayOAuth:
    """Fixed route bindings and separate inference/accounting credentials."""

    @classmethod
    def load(cls, path: Path) -> GatewayOAuth:
        try:
            with path.open("rb") as stream:
                raw = stream.read(MAX_BYTES + 1)
            require(len(raw) <= MAX_BYTES, "OAuth profile exceeds limit")
            return cls(strict_json(raw))
        except (OSError, ssl.SSLError, ValueError, TypeError):
            raise GatewayOAuthError("OAuth profile could not be loaded") from None

    def __init__(self, config: dict[str, Any]):
        fields(
            config,
            {
                "schema_version",
                "gateway_url",
                "ca_file",
                "database",
                "credentials",
                "accounting",
                "routes",
            },
        )
        require(
            type(config["schema_version"]) is int and config["schema_version"] == 1,
            "unsupported OAuth profile version",
        )
        require(text(config["gateway_url"]), "gateway URL required")
        target = urlsplit(config["gateway_url"])
        require(
            target.scheme == "https"
            and target.hostname
            and not target.username
            and not target.password
            and target.path in ("", "/")
            and not target.query
            and not target.fragment,
            "gateway must be an HTTPS origin",
        )
        self.gateway_url = config["gateway_url"].rstrip("/")
        require(
            text(config["database"]) and Path(config["database"]).is_absolute(),
            "absolute accounting database path required",
        )
        self.database = Path(config["database"])
        ca = config["ca_file"]
        require(ca is None or (text(ca) and Path(ca).is_absolute()), "invalid CA path")
        self._tls = ssl.create_default_context(cafile=ca)
        require(
            isinstance(config["credentials"], dict) and config["credentials"],
            "explicit credentials required",
        )
        sources = {name: BrokerCredential(value) for name, value in config["credentials"].items()}
        self._credentials = {
            name: CachingTokenCredential(source, refresh_before=MINIMUM_LIFETIME)
            for name, source in sources.items()
        }
        self._audiences = {name: source.identity[1] for name, source in sources.items()}
        self._accounting = config["accounting"]
        require(
            isinstance(self._accounting, str) and self._accounting in sources,
            "separate accounting credential required",
        )
        require(isinstance(config["routes"], dict) and config["routes"], "routes required")
        self._routes: dict[str, dict[str, Any]] = {}
        self._capabilities: dict[str, str] = {}
        for provider, route in config["routes"].items():
            fields(route, {"credential", "grant", "models"})
            require(provider in ("zai", "inferflux"), "unsupported validation provider")
            credential = route["credential"]
            require(
                isinstance(credential, str) and credential in sources, "unknown route credential"
            )
            require(
                sources[credential].identity != sources[self._accounting].identity,
                "accounting and inference principals must differ",
            )
            require(
                text(route["grant"]) and re.fullmatch(r"[A-Za-z0-9_-]+", route["grant"]),
                "invalid fixed grant",
            )
            models = route["models"]
            require(
                isinstance(models, list)
                and models
                and all(text(m) for m in models)
                and len(set(models)) == len(models),
                "unique model allowlist required",
            )
            self._routes[provider] = dict(route)
            self._capabilities[provider] = secrets.token_urlsafe(32)

    def capability(self, provider: str) -> str:
        require(provider in self._capabilities, "provider has no OAuth route")
        return self._capabilities[provider]

    @asynccontextmanager
    async def request(
        self,
        client: ClientSession,
        method: str,
        url: str,
        *,
        purpose: str,
        headers: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ClientResponse]:
        target = urlsplit(url)
        require(
            url == self.gateway_url + target.path and not target.query and not target.fragment,
            "request must target the fixed gateway without query or fragment",
        )
        items = list((headers or {}).items())
        outgoing = {key: value for key, value in items if key.lower() not in PRIVATE_HEADERS}
        outgoing["Accept-Encoding"] = "identity"
        if purpose == "inference":
            require(
                method == "POST" and target.path == "/v1/chat/completions", "inference route denied"
            )
            authorizations = [v for k, v in items if k.lower() == "authorization"]
            require(
                len(authorizations) == 1
                and isinstance(authorizations[0], str)
                and authorizations[0].isascii(),
                "one loopback capability required",
            )
            matches = [
                name
                for name, capability in self._capabilities.items()
                if hmac.compare_digest(authorizations[0], "Bearer " + capability)
            ]
            require(len(matches) == 1, "invalid loopback capability")
            route = self._routes[matches[0]]
            body = kwargs.get("data")
            require(
                isinstance(body, bytes) and "json" not in kwargs, "exact request bytes required"
            )
            payload = strict_json(cast(bytes, body))
            require(
                isinstance(payload, dict) and payload.get("model") in route["models"],
                "model outside configured route",
            )
            credential = route["credential"]
            outgoing["x-sandhi-grant"] = route["grant"]
        elif purpose == "accounting":
            allowed_get = target.path in ("/admin/version", "/dashboard/api/usage") or bool(
                re.fullmatch(r"/admin/usage/run/[A-Za-z0-9_-]{1,256}", target.path)
            )
            require(
                (method == "GET" and allowed_get)
                or (method == "POST" and target.path == "/admin/usage/diagnostics"),
                "accounting route denied",
            )
            credential = self._accounting
        else:
            raise GatewayOAuthError("unknown credential purpose")
        # Overall acquisition deadline includes waiting for the core cache's lock.
        try:
            token = await asyncio.wait_for(
                self._credentials[credential].get_token(self._audiences[credential]),
                timeout=ACQUISITION_TIMEOUT,
            )
        except asyncio.CancelledError:
            raise
        except GatewayOAuthError:
            raise
        except Exception:
            raise GatewayBrokerUnavailable("credential broker unavailable") from None
        outgoing["Authorization"] = "Bearer " + token.token
        async with client.request(
            method, url, headers=outgoing, ssl=self._tls, allow_redirects=False, **kwargs
        ) as response:
            yield response
