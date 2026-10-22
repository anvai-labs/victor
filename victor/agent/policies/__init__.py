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

"""Governance policy engine for Victor.

A declarative ALLOW / DENY / ASK governance layer over agent tool execution
and message phases, assembled from existing framework primitives (the
middleware chain, the turn boundary, HITL approvals, and token/cost
accounting). See ``feps/fep-0005-policy-engine.md``.

Typical wiring (done by the agent factory when the feature is enabled)::

    from victor.agent.policies import (
        PolicyEngine, PolicyEngineMiddleware, CostBudgetPolicy, MaxToolCallsPolicy,
    )

    engine = PolicyEngine([
        CostBudgetPolicy(max_cost_usd=5.0, ask_thresholds_usd=[3.0]),
        MaxToolCallsPolicy(limit=50),
    ])
    middleware = PolicyEngineMiddleware(engine, context_provider=provide_ctx)
    chain.add(middleware)
"""

from victor.agent.policies.base import Policy
from victor.agent.policies.builtins import (
    AllowToolsPolicy,
    AskOnToolsPolicy,
    BlockPatternPolicy,
    CostBudgetPolicy,
    DenyToolsPolicy,
    MaxToolCallsPolicy,
    RedactContentPolicy,
)
from victor.agent.policies.engine import EventEmitter, PolicyEngine
from victor.agent.policies.gate import GateResult, MessagePolicyGate
from victor.agent.policies.handlers import (
    ApprovalHandlerFn,
    PolicyApprovalHandler,
    console_approval_handler,
    make_console_approval_handler,
    register_policy_approval_handler,
)
from victor.agent.policies.middleware import (
    ContextProvider,
    PolicyEngineMiddleware,
    resolve_policy_ask,
)
from victor.agent.policies.types import (
    Phase,
    PolicyAction,
    PolicyContext,
    PolicyEvent,
    PolicyVerdict,
)

__all__ = [
    # Core
    "PolicyEngine",
    "PolicyEngineMiddleware",
    "MessagePolicyGate",
    "GateResult",
    "Policy",
    # Types
    "Phase",
    "PolicyAction",
    "PolicyContext",
    "PolicyEvent",
    "PolicyVerdict",
    # Builtins
    "CostBudgetPolicy",
    "AskOnToolsPolicy",
    "DenyToolsPolicy",
    "AllowToolsPolicy",
    "MaxToolCallsPolicy",
    "RedactContentPolicy",
    "BlockPatternPolicy",
    # Approval handlers
    "console_approval_handler",
    "make_console_approval_handler",
    "PolicyApprovalHandler",
    "register_policy_approval_handler",
    "ApprovalHandlerFn",
    "resolve_policy_ask",
    # Aliases
    "EventEmitter",
    "ContextProvider",
]
