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

"""Wire-truth cost client (victor ↔ sandhi admin surface)."""

from __future__ import annotations

from typing import Optional

import httpx
import pytest

from victor.observability.sandhi_admin import (
    fetch_run_cost_tree,
    find_step,
    step_rows,
)

TREE = {
    "run_id": "run-1",
    "total": {"calls": 3, "tokens_in": 10, "tokens_out": 4, "billable_tokens": 18},
    "roots": [
        {
            "step_id": "(none)",
            "own": {"calls": 1, "tokens_in": 4, "tokens_out": 1, "billable_tokens": 5},
            "rollup": {"calls": 1, "tokens_in": 4, "tokens_out": 1, "billable_tokens": 5},
            "children": [],
        },
        {
            "step_id": "turn-a",
            "parent_id": None,
            "own": {
                "calls": 1,
                "tokens_in": 3,
                "tokens_out": 2,
                "cache_read_tokens": 7,
                "cache_creation_tokens": 0,
                "billable_tokens": 12,
            },
            "rollup": {
                "calls": 2,
                "tokens_in": 6,
                "tokens_out": 3,
                "billable_tokens": 13,
            },
            "children": [
                {
                    "step_id": "turn-b",
                    "parent_id": "turn-a",
                    "own": {
                        "calls": 1,
                        "tokens_in": 3,
                        "tokens_out": 1,
                        "billable_tokens": 1,
                    },
                    "rollup": {
                        "calls": 1,
                        "tokens_in": 3,
                        "tokens_out": 1,
                        "billable_tokens": 1,
                    },
                    "children": [],
                }
            ],
        },
    ],
}


def _stub(monkeypatch, status: int = 200, body: Optional[dict] = None) -> None:
    """Patch httpx.get to answer one canned response, asserting the request shape."""

    def fake_get(url: str, headers=None, timeout=None):
        assert url == "http://gw.example/admin/usage/run/run-1"
        assert headers == {"Authorization": "Bearer tok"}
        return httpx.Response(status, json=body if body is not None else {"run": TREE})

    monkeypatch.setattr(httpx, "get", fake_get)


def test_happy_path_returns_the_tree(monkeypatch):
    monkeypatch.setenv("SANDHI_GATEWAY_URL", "http://gw.example")
    monkeypatch.setenv("SANDHI_ADMIN_TOKEN", "tok")
    _stub(monkeypatch)
    tree = fetch_run_cost_tree("run-1")
    assert tree == TREE


@pytest.mark.parametrize("status", [403, 404, 503, 500])
def test_expected_failures_map_to_none(monkeypatch, status):
    monkeypatch.setenv("SANDHI_GATEWAY_URL", "http://gw.example")
    monkeypatch.setenv("SANDHI_ADMIN_TOKEN", "tok")
    _stub(monkeypatch, status=status, body={"error": "x"})
    assert fetch_run_cost_tree("run-1") is None


def test_missing_route_or_token_is_none_without_a_call(monkeypatch):
    monkeypatch.delenv("SANDHI_GATEWAY_URL", raising=False)
    monkeypatch.delenv("SANDHI_ADMIN_TOKEN", raising=False)
    called = []

    def boom(*a, **kw):
        called.append(1)
        raise AssertionError("must not fetch")

    monkeypatch.setattr(httpx, "get", boom)
    assert fetch_run_cost_tree("run-1") is None
    assert called == []


def test_transport_error_is_none(monkeypatch):
    monkeypatch.setenv("SANDHI_GATEWAY_URL", "http://gw.example")
    monkeypatch.setenv("SANDHI_ADMIN_TOKEN", "tok")

    def boom(*a, **kw):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", boom)
    assert fetch_run_cost_tree("run-1") is None


def test_step_rows_use_own_not_rollup():
    rows = step_rows(TREE)
    by_step = {r["step_id"]: r for r in rows}
    # own, not rollup: turn-a's row must NOT include turn-b's spend.
    assert by_step["turn-a"]["billable_tokens"] == 12
    assert by_step["turn-a"]["cache_read_tokens"] == 7
    assert by_step["turn-b"]["billable_tokens"] == 1
    assert by_step["turn-b"]["parent_id"] == "turn-a"
    assert len(rows) == 3


def test_find_step_and_missing():
    assert find_step(TREE, "turn-b")["parent_id"] == "turn-a"
    assert find_step(TREE, "nope") is None


def test_gateway_root_strips_v1_suffixes(monkeypatch):
    from victor.observability.sandhi_admin import _gateway_root

    for raw, expected in [
        ("http://gw:8600", "http://gw:8600"),
        ("http://gw:8600/v1", "http://gw:8600"),
        ("http://gw:8600/v1beta/", "http://gw:8600"),
        ("", ""),
    ]:
        monkeypatch.setenv("SANDHI_GATEWAY_URL", raw)
        assert _gateway_root() == expected, raw


def test_non_dict_json_200_maps_to_none(monkeypatch):
    monkeypatch.setenv("SANDHI_GATEWAY_URL", "http://gw.example")
    monkeypatch.setenv("SANDHI_ADMIN_TOKEN", "tok")

    def fake_get(url, headers=None, timeout=None):
        return httpx.Response(200, content=b'"just a string"')

    monkeypatch.setattr(httpx, "get", fake_get)
    assert fetch_run_cost_tree("run-1") is None


def test_malformed_url_does_not_raise(monkeypatch):
    monkeypatch.setenv("SANDHI_GATEWAY_URL", "not a url")
    monkeypatch.setenv("SANDHI_ADMIN_TOKEN", "tok")

    def real_get(url, **kw):
        # Delegate to real httpx so InvalidURL genuinely fires.
        import httpx as _h

        return _h.get(url, **kw)

    monkeypatch.setattr(httpx, "get", real_get)
    assert fetch_run_cost_tree("run-1") is None
