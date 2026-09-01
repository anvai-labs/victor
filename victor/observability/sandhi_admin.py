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

"""Read-only client for a sandhi proxy's admin usage surface (wire-truth cost).

Sandhi owns the run cost tree (``RunCostTreeV1``, ADR-0005 D7): every usage
event sharing a ``run_id``, folded per ``step_id`` and assembled by
``parent_id``. This module is victor's first consumer of it — the missing
read side that turns the tree victor already populates (``x-sandhi-run-id``
+ per-turn ``x-sandhi-step-id``) into renderable wire truth, replacing the
hardcoded per-1k price-table estimates in the turn tracker.

Contract with the proxy (crates/sandhi-proxy/src/operator.rs ``usage_run``):

- ``GET /admin/usage/run/{run_id}`` with ``Authorization: Bearer <admin>``;
- 200 → ``{"run": RunCostTreeV1}``; 403 when no admin token is configured
  or the presented one is wrong; 404 when no usage event carries the run id.

Every failure maps to ``None`` — attribution must never raise into the turn
path (the standing rule from ``sandhi_transport``). Units are neutral
tokens (ADR-0001): no dollars originate here; conversion is the consumer's
policy and happens only where real prices exist.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# How long an admin fetch may take before we give up and fall back to estimates.
# The tree is one aggregate query; anything slower means the proxy is unreachable.
_ADMIN_TIMEOUT_SECONDS = 3.0


def _admin_token() -> str:
    """The proxy's admin bearer (``SANDHI_ADMIN_TOKEN`` — the same env the proxy reads)."""
    return os.environ.get("SANDHI_ADMIN_TOKEN", "")


def _gateway_root() -> str:
    """The proxy root (``SANDHI_GATEWAY_URL``), no trailing slash; empty when unset."""
    return os.environ.get("SANDHI_GATEWAY_URL", "").rstrip("/")


def fetch_run_cost_tree(
    run_id: str,
    *,
    base_url: Optional[str] = None,
    admin_token: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch one run's cost tree as raw JSON, or ``None`` — never raises.

    ``base_url``/``admin_token`` default to the ``SANDHI_GATEWAY_URL`` and
    ``SANDHI_ADMIN_TOKEN`` environments. ``None`` means: no route configured,
    no token, run unknown (404), admin refused (403), store absent (503), or
    the fetch failed — in every case the caller falls back to its estimate
    path rather than surfacing an error mid-session.
    """
    root = (base_url or _gateway_root()).rstrip("/")
    token = admin_token if admin_token is not None else _admin_token()
    if not root or not token or not run_id:
        return None
    try:
        response = httpx.get(
            f"{root}/admin/usage/run/{run_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=_ADMIN_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        logger.debug("run cost tree fetch failed for %s: %s", run_id, exc)
        return None
    if response.status_code != 200:
        # 403/404/503 are expected operator states, not errors to surface.
        logger.debug("run cost tree fetch for %s returned %d", run_id, response.status_code)
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    tree = payload.get("run")
    return tree if isinstance(tree, dict) else None


def step_rows(tree: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Flatten a run cost tree into one row per step, ``own`` (not rollup).

    ``own`` is what a step itself spent; ``rollup`` includes descendants and
    would double-count across a flat listing. Each row carries the neutral
    split (tokens in/out, cache read/creation, billable, calls, latency)
    exactly as the proxy folded it.
    """
    rows: list[Dict[str, Any]] = []

    def walk(node: Dict[str, Any]) -> None:
        own = node.get("own") or {}
        rows.append(
            {
                "step_id": node.get("step_id"),
                "parent_id": node.get("parent_id"),
                "calls": own.get("calls", 0),
                "tokens_in": own.get("tokens_in", 0),
                "tokens_out": own.get("tokens_out", 0),
                "cache_read_tokens": own.get("cache_read_tokens", 0),
                "cache_creation_tokens": own.get("cache_creation_tokens", 0),
                "billable_tokens": own.get("billable_tokens", 0),
            }
        )
        for child in node.get("children") or []:
            walk(child)

    for root_node in tree.get("roots") or []:
        walk(root_node)
    return rows


def find_step(tree: Dict[str, Any], step_id: str) -> Optional[Dict[str, Any]]:
    """The ``own`` row for one step id, or ``None`` when the tree lacks it."""
    for row in step_rows(tree):
        if row.get("step_id") == step_id:
            return row
    return None
