# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""tool-supply P6 — resolve_contract is the single tool-metadata fusion authority."""

from __future__ import annotations

from typing import Any, Dict

from victor.tools.base import BaseTool, ToolResult
from victor.tools.contract import resolve_contract
from victor.tools.enums import CostTier
from victor.tools.metadata import ToolMetadata


class _ExplicitTool:
    name = "x"
    description = "d"
    parameters: Dict[str, Any] = {}
    cost_tier = CostTier.FREE
    metadata = ToolMetadata(category="search", keywords=["k"])


class _AutogenTool:
    name = "grepper"
    description = "search files by pattern"
    parameters: Dict[str, Any] = {"type": "object"}
    cost_tier = CostTier.LOW
    metadata = None


def test_explicit_metadata_returned_as_is():
    t = _ExplicitTool()
    assert resolve_contract(t) is t.metadata


def test_autogen_matches_generate_from_tool():
    t = _AutogenTool()
    expected = ToolMetadata.generate_from_tool(
        name=t.name,
        description=t.description,
        parameters=t.parameters,
        cost_tier=t.cost_tier,
    )
    assert resolve_contract(t).to_dict() == expected.to_dict()


def test_autogen_result_is_cached_per_instance():
    t = _AutogenTool()
    first = resolve_contract(t)
    assert resolve_contract(t) is first  # cached, not recomputed


def test_non_weakreferenceable_tool_still_resolves():
    class _Slotted:
        __slots__ = ("name", "description", "parameters", "cost_tier", "metadata")

        def __init__(self):
            self.name = "s"
            self.description = "d"
            self.parameters = {}
            self.cost_tier = CostTier.FREE
            self.metadata = None

    # __slots__ without __weakref__ -> not weak-referenceable; must not raise.
    result = resolve_contract(_Slotted())
    assert isinstance(result, ToolMetadata)


class _RealTool(BaseTool):
    @property
    def name(self) -> str:
        return "real_read"

    @property
    def description(self) -> str:
        return "read a file from disk"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(
        self, _exec_ctx: Dict[str, Any], **kwargs: Any
    ) -> ToolResult:  # pragma: no cover
        return ToolResult(success=True, output="")


def test_base_tool_get_metadata_delegates_to_resolve_contract():
    t = _RealTool()
    # get_metadata now delegates; result must equal resolve_contract for the same tool.
    assert t.get_metadata().to_dict() == resolve_contract(t).to_dict()


# ---------------------------------------------------------------------------
# FEP-0009 Phase 2 — SDK contract bridge (duck-typed; no victor_contracts import)
# ---------------------------------------------------------------------------

import dataclasses  # noqa: E402

import pytest  # noqa: E402

from victor.tools.enums import AccessMode, DangerLevel, ExecutionCategory  # noqa: E402


@dataclasses.dataclass(frozen=True)
class _StubContract:
    """A stand-in for victor_contracts.tools.ToolContract (duck-typed via plain strings)."""

    category: Any = "git"
    access_mode: Any = "write"
    danger_level: Any = "high"
    execution_category: Any = "write"
    cost_tier: Any = "medium"
    keywords: tuple = ()
    use_cases: tuple = ()
    task_types: tuple = ()
    stages: tuple = ()


class _ContractTool:
    name = "deployer"
    description = "deploy the application"
    parameters: Dict[str, Any] = {}
    metadata = None
    contract = _StubContract()


def test_contract_traits_override_autogen():
    md = resolve_contract(_ContractTool())
    assert md.category == "git"
    assert md.access_mode == AccessMode.WRITE
    assert md.danger_level == DangerLevel.HIGH
    assert md.execution_category == ExecutionCategory.WRITE


def test_contract_hints_override_only_when_nonempty():
    class _WithHints:
        name = "t"
        description = "does a thing"
        parameters: Dict[str, Any] = {}
        metadata = None
        contract = _StubContract(keywords=("alpha", "beta"), stages=("reading",))

    md = resolve_contract(_WithHints())
    assert md.keywords == ["alpha", "beta"]
    assert md.stages == ["reading"]

    # Empty declared hints fall back to the autogen baseline (non-empty keywords).
    md_default = resolve_contract(_ContractTool())
    assert md_default.keywords  # autogen produced keywords from name/description


def test_explicit_metadata_wins_over_contract():
    class _Both:
        name = "x"
        description = "d"
        parameters: Dict[str, Any] = {}
        metadata = ToolMetadata(category="search")
        contract = _StubContract(category="git")

    assert resolve_contract(_Both()).category == "search"  # metadata tier wins


def test_contract_cost_tier_drives_priority_hints():
    md = resolve_contract(_ContractTool())  # cost_tier="medium"
    assert any("API" in hint for hint in md.priority_hints)


def test_unknown_enum_value_falls_back_to_default():
    class _Bad:
        name = "t"
        description = "d"
        parameters: Dict[str, Any] = {}
        metadata = None
        contract = _StubContract(access_mode="not_a_real_mode")

    md = resolve_contract(_Bad())
    assert md.access_mode == AccessMode.READONLY  # default, not a crash


def test_contract_result_is_cached_per_instance():
    t = _ContractTool()
    first = resolve_contract(t)
    assert resolve_contract(t) is first


def test_real_tool_contract_bridges():
    tools_mod = pytest.importorskip("victor_contracts.tools")

    class _RealContractTool:
        name = "git_push"
        description = "push commits to a remote"
        parameters: Dict[str, Any] = {}
        metadata = None
        contract = tools_mod.ToolContract(
            category=tools_mod.ToolCategory.GIT,
            access_mode=tools_mod.AccessMode.WRITE,
            danger_level=tools_mod.DangerLevel.HIGH,
            execution_category=tools_mod.ExecutionCategory.NETWORK,
        )

    md = resolve_contract(_RealContractTool())
    assert md.category == "git"
    assert md.access_mode == AccessMode.WRITE
    assert md.danger_level == DangerLevel.HIGH
    assert md.execution_category == ExecutionCategory.NETWORK


def test_every_contract_field_maps_to_metadata():
    tools_mod = pytest.importorskip("victor_contracts.tools")
    metadata_fields = {f.name for f in dataclasses.fields(ToolMetadata)}
    # cost_tier is intentionally mapped to priority_hints, not stored as a field.
    indirect = {"cost_tier"}
    for f in dataclasses.fields(tools_mod.ToolContract):
        assert (
            f.name in metadata_fields or f.name in indirect
        ), f"ToolContract.{f.name} has no ToolMetadata target (orphan trait)"
