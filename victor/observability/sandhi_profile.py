# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gateway profile client — victor's consumption side of the sandhi per-user profile.

The sandhi proxy (contract frozen) serves ``GET /auth/profile``: one admin
configures per-group defaults centrally, and a victor user downloads their
profile **once** with an out-of-band bootstrap token, then runs locally with
layered override. This module is the HTTP client + translator; the CLI surface
is ``victor profiles pull`` (``victor/ui/commands/profiles.py``) and the merge
point is ``Settings.load_profiles`` (bundled < central ``~/.victor/profiles.d``
< user ``~/.victor/profiles.yaml`` — the user file still wins key-by-key).

Wire contract (sandhi-proxy ``GET /auth/profile``):

- Auth: ``Authorization: Bearer skp_...`` — a per-user **bootstrap profile
  token** the operator mints and hands over out-of-band (shown once,
  revocable). It is not a virtual key and never goes near provider traffic.
- 200 → ``{"mode": "gateway"|"local", "endpoint": ..., "group": ...,
  "defaults": {"timeout_seconds": ..., "max_tokens": ...}, "models": [...],
  "credentials": {"virtual_key": "vk_..."}}``. ``endpoint`` and
  ``credentials`` are gateway-mode only, and the virtual key is **shown once
  by the server** — after the first download the proxy never serves it
  again, so victor persists it only in the 0600 central profile file
  (``~/.victor/profiles.d/sandhi.yaml``) and never logs or echoes it.
- 401 → token rejected or revoked; 404 → no profile configured for the
  user's group; 503 → gateway store unavailable.

FFI note: a **local**-mode profile carries no secrets at all — provider keys
resolve through the standing chain (provider env var > keyring >
``~/.victor/api_keys.yaml``), untouched by this feature; the profile only
records the env-var *names* as guidance. Metering stays **embedded** via
``SandhiMeter`` (``victor/observability/sandhi_meter.py``) per the owner's
central decision: remote reporting is deliberately NOT wired here. Units are
neutral tokens (sandhi ADR-0001); no dollars originate in this module.

Style follows ``sandhi_admin.py``: short timeout, ``httpx``, and every
failure mapped to a value the caller can render — never a raise into the
CLI's control flow it did not choose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# One profile document; anything slower means the operator's gateway is down.
_PROFILE_TIMEOUT_SECONDS = 3.0

# The central drop file — one per user, fixed name so re-pull overwrites it
# and the settings layer picks it up deterministically.
CENTRAL_PROFILE_FILENAME = "sandhi.yaml"
CENTRAL_PROFILES_DIRNAME = "profiles.d"
CENTRAL_PROFILE_NAME = "sandhi"

_VALID_MODES = ("gateway", "local")

# Mirror of ``provider_config_registry._normalize_gateway_root`` /
# ``sandhi_admin._gateway_root``: operators paste either the root or a /v1
# form; /auth/profile lives at the root, so a suffixed value would 404.
_URL_SUFFIXES = ("/v1", "/v1beta")


def normalize_gateway_root(url: str) -> str:
    """Canonicalize a gateway URL to its ROOT (no trailing /v1, /v1beta, slash)."""
    root = (url or "").strip().rstrip("/")
    for suffix in _URL_SUFFIXES:
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    return root.rstrip("/")


@dataclass(frozen=True)
class ProfileFetch:
    """Outcome of one ``GET /auth/profile`` attempt.

    ``status`` is the HTTP status code, or ``None`` for a transport failure
    (DNS, TLS, connect, timeout, malformed URL). ``payload`` is the decoded
    JSON document on 200 and ``None`` otherwise.
    """

    status: Optional[int]
    payload: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and isinstance(self.payload, dict)


def fetch_gateway_profile(gateway_url: str, token: str) -> ProfileFetch:
    """Fetch the caller's profile from ``GET {root}/auth/profile`` — never raises.

    Every failure mode lands in the return value so the CLI can render the
    right operator guidance (401 → mint a new token, 404 → ask the operator,
    503 / transport → retry later) without this module deciding policy.
    """
    root = normalize_gateway_root(gateway_url)
    if not root or not (token or "").strip():
        return ProfileFetch(status=None)
    try:
        response = httpx.get(
            f"{root}/auth/profile",
            headers={"Authorization": f"Bearer {token.strip()}"},
            timeout=_PROFILE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        # HTTPError covers transport faults; InvalidURL (a malformed gateway
        # URL) is NOT an HTTPError subclass and must not escape either.
        logger.debug("gateway profile fetch failed: %s", exc)
        return ProfileFetch(status=None)
    if response.status_code != 200:
        return ProfileFetch(status=response.status_code)
    try:
        payload = response.json()
    except ValueError:
        return ProfileFetch(status=200)
    if not isinstance(payload, dict):
        return ProfileFetch(status=200)
    return ProfileFetch(status=200, payload=payload)


def provider_for_model(model: str) -> str:
    """Map a central model slug to victor's provider transport slug.

    The proxy speaks one dialect per provider slug, so the profile's models
    list picks the dialect: ``claude-*`` → anthropic, ``gemini-*`` → google,
    ``command-*`` → cohere, everything else defaults to the openai-compatible
    dialect (the widest family on the proxy, ≈20 providers).
    """
    m = (model or "").lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gemini", "gemma")):
        return "google"
    if m.startswith("command"):
        return "cohere"
    return "openai"


@dataclass(frozen=True)
class TranslatedProfile:
    """A fetched server profile translated into victor's native config shape.

    ``document`` is the YAML document written to
    ``~/.victor/profiles.d/sandhi.yaml``: a ``profiles`` entry plus, in
    gateway mode, the ``providers.<slug>.gateway`` block whose ``virtual_key``
    is the shown-once secret. ``env_var`` is local-mode guidance only (the
    provider key env-var NAME — never a value).
    """

    mode: str
    group: str
    provider: str
    model: str
    document: Dict[str, Any]
    endpoint: Optional[str] = None
    virtual_key: Optional[str] = None
    env_var: Optional[str] = None


def translate_gateway_profile(payload: Dict[str, Any]) -> TranslatedProfile:
    """Translate a ``GET /auth/profile`` payload into the central YAML document.

    Raises ``ValueError`` with an operator-facing message when the payload
    does not match the frozen contract (the CLI renders it, never the raw
    exception).
    """
    mode = str(payload.get("mode") or "").strip()
    if mode not in _VALID_MODES:
        raise ValueError(
            f"gateway returned an unknown profile mode {mode!r} "
            f"(expected one of {', '.join(_VALID_MODES)}) — ask the operator to "
            "check the gateway deployment"
        )
    group = str(payload.get("group") or "").strip()
    models: List[str] = [str(m) for m in (payload.get("models") or []) if str(m).strip()]
    defaults = payload.get("defaults") or {}
    if not isinstance(defaults, dict):
        defaults = {}

    provider = provider_for_model(models[0]) if models else "openai"
    entry: Dict[str, Any] = {
        "provider": provider,
        "model": models[0] if models else "auto",
        "description": (
            f"Central profile pulled from the sandhi gateway (group {group or 'unknown'}) "
            "— re-run `victor profiles pull` after the operator changes central defaults."
        ),
    }
    # Defaults are advisory ceilings from the operator; either may be
    # null/absent on the wire and then the victor defaults stand.
    max_tokens = defaults.get("max_tokens")
    if max_tokens is not None:
        entry["max_tokens"] = int(max_tokens)
    timeout_seconds = defaults.get("timeout_seconds")
    if timeout_seconds is not None:
        entry["timeout"] = int(timeout_seconds)

    endpoint: Optional[str] = None
    virtual_key: Optional[str] = None
    document: Dict[str, Any] = {"profiles": {CENTRAL_PROFILE_NAME: entry}}

    if mode == "gateway":
        endpoint = str(payload.get("endpoint") or "").strip()
        credentials = payload.get("credentials") or {}
        if not isinstance(credentials, dict):
            credentials = {}
        virtual_key = str(credentials.get("virtual_key") or "").strip()
        if not endpoint:
            raise ValueError(
                "gateway profile is mode 'gateway' but carries no endpoint — "
                "ask the operator to check the gateway deployment"
            )
        if not virtual_key:
            raise ValueError(
                "gateway profile carries no virtual key — the key is shown once at "
                "grant time; ask the operator to re-issue it"
            )
        document["providers"] = {
            provider: {"gateway": {"url": endpoint, "virtual_key": virtual_key}},
        }

    env_var: Optional[str] = None
    if mode == "local":
        # Guidance only: record the env-var NAME the provider's standing key
        # chain (env > keyring > ~/.victor/api_keys.yaml) resolves first.
        from victor.config.api_keys import PROVIDER_ENV_VARS

        env_var = PROVIDER_ENV_VARS.get(provider)

    return TranslatedProfile(
        mode=mode,
        group=group,
        provider=provider,
        model=entry["model"],
        document=document,
        endpoint=endpoint,
        virtual_key=virtual_key,
        env_var=env_var,
    )


def mask_secret(secret: Optional[str]) -> str:
    """Render a secret safe for status output: first 6 chars + ellipsis."""
    if not secret:
        return "—"
    return f"{secret[:6]}…" if len(secret) > 6 else "•••"
