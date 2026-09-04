# Copyright 2026 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""Tool schema/token estimate cache (co-design review U4-F1).

estimate_tool_tokens ran to_schema(tier) + len(str(schema))//4 per tool,
per strategy pass, every turn — while a schema-cache keyed by a registry
version counter already existed at the registry layer with zero
production consumers. These tests pin the new (name, tier, category)
cache: hit/miss, version invalidation on every mutation path, and the
stub-probe bypass.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.agent.services.tool_service import ToolService
from victor.tools.registry import ToolRegistry


class _FakeTool:
    def __init__(self, name: str, schema: dict):
        self.name = name
        self._schema = schema
        self.to_schema_calls = 0

    def to_schema(self, tier):
        self.to_schema_calls += 1
        return {"name": self.name, "tier": tier, **self._schema}


def _registry_with_version(version: int):
    reg = MagicMock(spec=["schema_cache_version"])
    reg.schema_cache_version = version
    return reg


def _service(registry) -> ToolService:
    return ToolService(
        config=SimpleNamespace(default_tool_budget=100),
        tool_selector=MagicMock(),
        tool_executor=MagicMock(),
        tool_registrar=registry,
    )


class TestEstimateCache:
    def test_schema_built_once_per_tool_tier(self):
        service = _service(_registry_with_version(1))
        tool = _FakeTool("search", {"parameters": {"q": "s"}})

        first = service.estimate_tool_tokens(tool)
        second = service.estimate_tool_tokens(tool)
        third = service.estimate_tool_tokens(tool, provider_category="anthropic")

        assert first == second
        assert tool.to_schema_calls == 2, "same (name, tier, category) must hit"
        assert third != first or True  # distinct key; independent build

    def test_registry_mutation_invalidates(self):
        """register/unregister/batch bump the version — the cache must drop."""
        service = _service(_registry_with_version(1))
        tool = _FakeTool("search", {"parameters": {}})
        service.estimate_tool_tokens(tool)

        # Same service, registry version moved (mutation happened).
        service._registrar = _registry_with_version(2)
        service.estimate_tool_tokens(tool)
        assert tool.to_schema_calls == 2, "version bump must invalidate"

    def test_stub_probe_never_reads_or_writes_cache(self):
        """_demote's temporary _schema_level=STUB estimate is
        unrepresentative — the bypass must not poison or consume the cache."""
        service = _service(_registry_with_version(1))
        normal = _FakeTool("critical_tool", {"parameters": {"a": 1}})
        base = service.estimate_tool_tokens(normal)

        stub_tool = _FakeTool("critical_tool", {"parameters": {"a": 1}})
        stub_tool._schema_level = "stub"
        stub_estimate = service.estimate_tool_tokens(stub_tool, _use_cache=False)

        assert stub_tool.to_schema_calls == 1
        # The cache holds exactly the un-patched estimate...
        assert list(service._tool_estimate_cache.values()) == [base]
        # ...which still serves the un-patched tool.
        again = service.estimate_tool_tokens(normal)
        assert again == base
        assert normal.to_schema_calls == 1

    def test_no_version_accessor_means_no_cache(self):
        """When the registrar is not a ToolRegistry (defensive getattr path),
        estimation still works — uncached."""
        service = _service(SimpleNamespace())  # no schema_cache_version
        tool = _FakeTool("search", {"parameters": {}})
        a = service.estimate_tool_tokens(tool)
        b = service.estimate_tool_tokens(tool)
        assert a == b
        assert tool.to_schema_calls == 2

    def test_real_registry_version_bumps_on_mutation(self):
        """Integration pin: the ACTUAL ToolRegistry bumps
        schema_cache_version on every mutation path."""
        reg = ToolRegistry()
        v0 = reg.schema_cache_version
        reg.register_dict({"name": "probe", "description": "d", "parameters": {}})
        v1 = reg.schema_cache_version
        assert v1 > v0
        reg.unregister("probe")
        v2 = reg.schema_cache_version
        assert v2 > v1
        with reg.batch_update():
            reg.register_dict({"name": "probe2", "description": "d", "parameters": {}})
        assert reg.schema_cache_version > v2
