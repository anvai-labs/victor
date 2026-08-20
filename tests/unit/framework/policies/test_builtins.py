# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Tests for built-in governance policies."""

from victor.framework.policies import (
    AllowToolsPolicy,
    AskOnToolsPolicy,
    CostBudgetPolicy,
    DenyToolsPolicy,
    MaxToolCallsPolicy,
    Phase,
    PolicyContext,
    PolicyEvent,
)


def _event(tool_name="write_file", *, cost=0.0, model=None):
    return PolicyEvent(
        phase=Phase.TOOL_CALL,
        tool_name=tool_name,
        context=PolicyContext(cost_usd=cost, model=model),
    )


# -- CostBudgetPolicy -------------------------------------------------------


async def test_cost_budget_allows_under_cap():
    policy = CostBudgetPolicy(max_cost_usd=5.0)
    verdict = await policy.evaluate(_event(cost=1.0, model="opus"))
    assert verdict.is_allow


async def test_cost_budget_denies_at_cap_on_expensive_model():
    policy = CostBudgetPolicy(max_cost_usd=5.0, expensive_models=["opus"])
    verdict = await policy.evaluate(_event(cost=5.0, model="claude-opus-4-8"))
    assert verdict.is_deny
    assert "5.00" in verdict.reason


async def test_cost_budget_allows_over_cap_on_cheap_model():
    # Downgrade-gate semantics: over budget but on a cheap model -> allowed.
    policy = CostBudgetPolicy(max_cost_usd=5.0, expensive_models=["opus"])
    verdict = await policy.evaluate(_event(cost=9.0, model="haiku"))
    assert verdict.is_allow


async def test_cost_budget_fail_open_when_cost_unavailable():
    policy = CostBudgetPolicy(max_cost_usd=5.0)
    verdict = await policy.evaluate(_event(cost=0.0, model="opus"))
    assert verdict.is_allow


async def test_cost_budget_ask_threshold_once():
    policy = CostBudgetPolicy(max_cost_usd=100.0, ask_thresholds_usd=[3.0])
    # First crossing asks.
    first = await policy.evaluate(_event(cost=3.5, model="opus"))
    assert first.is_ask
    # Second call past the same threshold no longer asks.
    second = await policy.evaluate(_event(cost=4.0, model="opus"))
    assert second.is_allow


async def test_cost_budget_empty_expensive_models_applies_to_all():
    policy = CostBudgetPolicy(max_cost_usd=2.0, expensive_models=[])
    verdict = await policy.evaluate(_event(cost=2.0, model="some-cheap-model"))
    assert verdict.is_deny


# -- MaxToolCallsPolicy -----------------------------------------------------


async def test_max_tool_calls_allows_up_to_limit_then_denies():
    policy = MaxToolCallsPolicy(limit=2)
    assert (await policy.evaluate(_event())).is_allow  # call 1
    assert (await policy.evaluate(_event())).is_allow  # call 2
    third = await policy.evaluate(_event())  # call 3 exceeds
    assert third.is_deny
    assert "2" in third.reason


async def test_max_tool_calls_none_limit_never_denies():
    policy = MaxToolCallsPolicy(limit=None)
    for _ in range(10):
        assert (await policy.evaluate(_event())).is_allow


# -- AskOnToolsPolicy -------------------------------------------------------


async def test_ask_on_tools_only_applies_to_configured_tools():
    policy = AskOnToolsPolicy(["run_command", "delete_file"])
    assert policy.applies_to("run_command") is True
    assert policy.applies_to("read_file") is False


async def test_ask_on_tools_returns_ask():
    policy = AskOnToolsPolicy(["run_command"])
    verdict = await policy.evaluate(_event(tool_name="run_command"))
    assert verdict.is_ask
    assert "run_command" in verdict.reason


# -- DenyToolsPolicy --------------------------------------------------------


async def test_deny_tools_applies_only_to_listed():
    policy = DenyToolsPolicy(["run_command", "write_file"])
    assert policy.applies_to("run_command") is True
    assert policy.applies_to("read_file") is False


async def test_deny_tools_denies_listed_tool():
    policy = DenyToolsPolicy(["run_command"])
    verdict = await policy.evaluate(_event(tool_name="run_command"))
    assert verdict.is_deny
    assert "run_command" in verdict.reason


# -- AllowToolsPolicy -------------------------------------------------------


async def test_allow_tools_applies_to_unlisted_only():
    policy = AllowToolsPolicy(["read_file", "list_files"])
    # Allowed tools are NOT subject to the policy (it only fires on others).
    assert policy.applies_to("read_file") is False
    # Unlisted tools ARE subject to the policy (will be denied).
    assert policy.applies_to("run_command") is True


async def test_allow_tools_denies_unlisted_tool():
    policy = AllowToolsPolicy(["read_file"])
    verdict = await policy.evaluate(_event(tool_name="run_command"))
    assert verdict.is_deny
    assert "allowed-tools" in verdict.reason


async def test_allow_tools_empty_disables_policy():
    # Empty allowlist => policy applies to nothing (disabled).
    policy = AllowToolsPolicy([])
    assert policy.applies_to("anything") is False
