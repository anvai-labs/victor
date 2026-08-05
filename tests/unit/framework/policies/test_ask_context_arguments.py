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

"""The policy ASK context must carry the tool arguments (for informed approvals)."""

from __future__ import annotations

from types import SimpleNamespace

from victor.framework.policies.middleware import PolicyEngineMiddleware


async def test_ask_context_includes_arguments(monkeypatch):
    captured = {}

    async def _fake_resolve(approval_handler, **kwargs):
        captured.update(kwargs)
        return True  # approve

    monkeypatch.setattr("victor.framework.policies.middleware.resolve_policy_ask", _fake_resolve)

    # An engine whose evaluate() returns an ASK verdict.
    verdict = SimpleNamespace(
        is_deny=False,
        is_ask=True,
        reason="needs review",
        policy_name="ask_on_tools",
        modified_arguments=None,
    )

    class _Engine:
        async def evaluate(self, event):
            return verdict

    mw = PolicyEngineMiddleware(_Engine(), ask_fallback="deny")

    result = await mw.before_tool_call("bash", {"command": "rm -rf build"})

    assert result.proceed is True
    ctx = captured["context"]
    assert ctx["tool_name"] == "bash"
    assert ctx["arguments"] == {"command": "rm -rf build"}
