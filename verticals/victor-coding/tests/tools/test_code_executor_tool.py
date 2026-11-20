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

"""Tests for the code executor sandbox tool (Docker-based code execution).

FOU-S2: the lifecycle quality bar (lazy acquisition, ownership,
self-expiry, owner-aware reaper) is owned by core
``victor.tools.sandbox.container_sandbox`` and pinned there
(tests/unit/tools/sandbox/test_container_sandbox.py). This module pins the
VERTICAL side of the promotion: the delegation identity (CodeSandbox IS the
core class, not a copy), the compat re-export surface, and the @tool entry
point contract. Docker is always mocked here — patching the CORE module;
tests/integration/infra/test_docker_availability.py exercises a real daemon.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import victor_coding.tools.code_executor_tool as cet
import victor.tools.sandbox.container_sandbox as core_cs
from victor.tools.sandbox.container_sandbox import ContainerSandbox

MODULE = "victor.tools.sandbox.container_sandbox"


# ---------------------------------------------------------------------------
# Delegation identity: the vertical delegates to core, not a copy
# ---------------------------------------------------------------------------


class TestVerticalDelegation:
    def test_codesandbox_is_the_core_class(self):
        assert cet.CodeSandbox is core_cs.ContainerSandbox
        assert cet.CodeSandbox is ContainerSandbox

    def test_resource_error_is_the_core_typed_error(self):
        assert cet.ResourceError is core_cs.ContainerSandboxUnavailableError

    def test_compat_names_are_core_identities(self):
        for name in (
            "DOCKER_AVAILABLE",
            "SANDBOX_CONTAINER_LABEL",
            "SANDBOX_CONTAINER_VALUE",
            "IDLE_TTL_LABEL",
            "DEFAULT_IMAGE",
            "DEFAULT_IDLE_TTL_SECONDS",
            "LEASE_PATH",
            "watchdog_command",
            "find_orphaned_sandboxes",
            "reap_orphaned_sandboxes",
            "host_identity",
            "owner_alive",
            "owner_gone",
            "owner_labels",
            "process_start_marker",
            "created_age_seconds",
        ):
            assert getattr(cet, name) is getattr(core_cs, name), name

    def test_core_behavior_flows_through_the_vertical_alias(self, docker_env):
        """A vertical-constructed CodeSandbox exercises the core lifecycle."""
        sandbox = cet.CodeSandbox(idle_ttl_seconds=120)
        sandbox.start()
        kwargs = docker_env.client.containers.run.call_args.kwargs
        assert kwargs["auto_remove"] is True
        assert kwargs["network_disabled"] is True  # network disabled by default
        assert kwargs["command"] == core_cs.watchdog_command(120)
        sandbox.stop()
        docker_env.created[0].remove.assert_called_once_with(force=True)

    def test_typed_refusal_reason_through_the_vertical_alias(self):
        with patch(f"{MODULE}.DOCKER_AVAILABLE", False):
            with pytest.raises(cet.ResourceError) as excinfo:
                cet.CodeSandbox(require_docker=True)
        assert excinfo.value.reason == "binary_missing"


# ---------------------------------------------------------------------------
# Mocked Docker (patched at the CORE module the vertical delegates to)
# ---------------------------------------------------------------------------


def _mock_container(container_id="c0ffee"):
    container = MagicMock()
    container.id = container_id
    container.short_id = container_id[:6]
    container.status = "running"
    container.exec_run.return_value = MagicMock(exit_code=0, output=(b"hello world\n", b""))
    return container


@pytest.fixture
def docker_env():
    """Mocked docker module whose client hands out a fresh container per run()."""
    with (
        patch(f"{MODULE}.DOCKER_AVAILABLE", True),
        patch(f"{MODULE}.docker") as mock_docker,
        patch(f"{MODULE}._reap_once"),
    ):
        client = mock_docker.from_env.return_value
        created = []

        def _run(*_args, **_kwargs):
            created.append(_mock_container(f"c{len(created)}ffee"))
            return created[-1]

        client.containers.run.side_effect = _run
        yield SimpleNamespace(docker=mock_docker, client=client, created=created)


# ---------------------------------------------------------------------------
# Async helpers and the @tool entry point (reserved _exec_ctx injection,
# typed failures — see TestSandboxTool)
# ---------------------------------------------------------------------------


class TestAsyncHelpers:
    def test_execute_code_formats_output(self):
        sandbox = MagicMock()
        sandbox.execute.return_value = {
            "exit_code": 0,
            "stdout": "42\n",
            "stderr": "",
        }
        result = asyncio.run(cet._execute_code(sandbox, "print(42)"))
        assert "Exit Code: 0" in result
        assert "42" in result

    def test_upload_files_success(self):
        sandbox = MagicMock()
        result = asyncio.run(cet._upload_files(sandbox, ["/a.py", "/b.py"]))
        assert "2 files" in result

    def test_upload_files_error(self):
        sandbox = MagicMock()
        sandbox.put_files.side_effect = RuntimeError("disk full")
        result = asyncio.run(cet._upload_files(sandbox, ["/a.py"]))
        assert "disk full" in result


class TestSandboxTool:
    """The @tool entry point receives its context the way the tool executor
    injects it (the reserved ``_exec_ctx`` parameter) and reports typed
    failures.

    Codex review (code executor context contract): the tool used to declare
    an ordinary ``context`` parameter the executor never supplies, so a live
    dispatch returned SUCCESS with the output "Error: Context not provided."
    while a mock manager received zero calls. The adapter-level tests below
    dispatch through ``Tool.execute(_exec_ctx=...)`` — the executor's normal
    injection — and pin success-on-execution plus typed (never
    success-shaped) failures.
    """

    def _make_ctx(self, sandbox_instance):
        return {"code_manager": sandbox_instance}

    def _exec_sandbox_mock(self):
        sandbox_mock = MagicMock()
        sandbox_mock.execute.return_value = {
            "exit_code": 0,
            "stdout": "ok",
            "stderr": "",
        }
        return sandbox_mock

    async def _dispatch(self, _exec_ctx=None, **kwargs):
        """Mirror the canonical executor dispatch of a registered tool."""
        return await cet.sandbox.Tool.execute(_exec_ctx=_exec_ctx, **kwargs)

    # -- executor injection: the normal dispatch path actually executes ----

    def test_execute_operation_via_executor_injection(self):
        sandbox_mock = self._exec_sandbox_mock()
        result = asyncio.run(
            self._dispatch(
                _exec_ctx=self._make_ctx(sandbox_mock),
                operation="execute",
                code="print(1)",
            )
        )
        assert result.success is True
        assert "ok" in result.output
        # The worst failure class was success-without-execution: pin that the
        # sandbox manager actually received the work.
        sandbox_mock.execute.assert_called_once()

    def test_upload_operation_via_executor_injection(self):
        sandbox_mock = MagicMock()
        result = asyncio.run(
            self._dispatch(
                _exec_ctx=self._make_ctx(sandbox_mock),
                operation="upload",
                file_paths=["/a.py"],
            )
        )
        assert result.success is True
        assert "uploaded" in result.output.lower()
        sandbox_mock.put_files.assert_called_once()

    # -- typed failures: never success-shaped -------------------------------

    def test_missing_context_is_typed_failure_through_executor(self):
        """No runtime context at all (the historical live-path bug): the
        adapter reports success=False with a typed error, not a string body."""
        result = asyncio.run(self._dispatch(_exec_ctx=None, operation="execute", code="print(1)"))
        assert result.success is False
        assert "_exec_ctx" in result.error or "execution context" in result.error
        assert result.metadata["category"] == "tool_execution"

    def test_missing_sandbox_handle_is_typed_failure_through_executor(self):
        result = asyncio.run(
            self._dispatch(
                _exec_ctx={"code_manager": None},
                operation="execute",
                code="print(1)",
            )
        )
        assert result.success is False
        assert "code_manager" in result.error
        assert result.metadata["category"] == "tool_execution"

    def test_raw_function_returns_typed_failure_without_context(self):
        """Raw (undecorated-path) calls report the structured failure dict —
        the vertical's sanctioned failure contract, never a bare string."""
        result = asyncio.run(cet.sandbox(operation="execute", code="print(1)", _exec_ctx=None))
        assert result["success"] is False
        assert result["category"] == "tool_execution"
        assert "recovery_hint" in result

    # -- argument validation is typed too -----------------------------------

    def test_execute_requires_code(self):
        result = asyncio.run(
            cet.sandbox(operation="execute", code="", _exec_ctx=self._make_ctx(MagicMock()))
        )
        assert result["success"] is False
        assert "required" in result["error"].lower()

    def test_unknown_operation(self):
        result = asyncio.run(
            cet.sandbox(operation="frobnicate", _exec_ctx=self._make_ctx(MagicMock()))
        )
        assert result["success"] is False
        assert "Unknown operation" in result["error"]

    def test_schema_does_not_advertise_context_to_the_model(self):
        """The reserved context parameter is framework-internal: it must not
        appear in the LLM-facing schema (the model could never supply a real
        session context; advertising it invites hallucinated arguments)."""
        properties = cet.sandbox.Tool.parameters["properties"]
        assert "context" not in properties
        assert "_exec_ctx" not in properties
        assert set(properties) == {"operation", "code", "file_paths"}
