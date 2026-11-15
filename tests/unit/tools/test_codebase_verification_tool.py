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

"""codebase_verify / codebase_verify_batch: importable, registered, working.

The module failed at import (AccessMode.READ; an unsupported @tool kwarg) and
tool discovery swallowed the error, so both tools were silently absent while
configuration referenced them. These tests pin that every tool module the
registry scans imports, and that the two tools register and behave.
"""

import asyncio
import importlib
import inspect
import os
import re

import pytest

from victor.agent.shared_tool_registry import SharedToolRegistry
from victor.tools.base import AccessMode, DangerLevel


def _discovered_module_files():
    source = inspect.getsource(SharedToolRegistry._discover_tools)
    excluded = set(re.findall(r'"([a-z_]+\.py)"', source))
    tools_dir = os.path.dirname(importlib.import_module("victor.tools").__file__)
    return sorted(f for f in os.listdir(tools_dir) if f.endswith(".py") and f not in excluded)


def test_every_discovered_tool_module_imports():
    """Discovery logs import failures at debug level only; a broken module
    silently drops its tools. Fail loudly here instead."""
    failures = {}
    for filename in _discovered_module_files():
        try:
            importlib.import_module(f"victor.tools.{filename[:-3]}")
        except Exception as exc:  # noqa: BLE001 - collecting every failure
            failures[filename] = f"{type(exc).__name__}: {exc}"
    assert failures == {}


@pytest.fixture
def tools():
    module = importlib.import_module("victor.tools.codebase_verification_tool")
    return module.codebase_verify, module.codebase_verify_batch


def test_both_tools_are_read_only_and_safe(tools):
    for fn in tools:
        assert fn.Tool.access_mode is AccessMode.READONLY
        assert fn.Tool.danger_level is DangerLevel.SAFE
    assert [fn.Tool.name for fn in tools] == ["codebase_verify", "codebase_verify_batch"]
    assert tools[0].Tool.description.startswith("Verify codebase claims")


def test_registry_discovers_both_tools():
    # Discovery runs once per singleton; a sibling test in the shard may have
    # left a pre-populated instance, so discover on a fresh one.
    SharedToolRegistry.reset_instance()
    registry = SharedToolRegistry.get_instance()
    registry.get_tool_classes()
    try:
        assert {"codebase_verify", "codebase_verify_batch"} <= set(registry._tool_classes)
    finally:
        SharedToolRegistry.reset_instance()


@pytest.fixture
def project(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "io.py").write_text(
        "def load(path):\n    f = open(path)\n    return f.read()\n"
    )
    return {"workspace_root": str(tmp_path)}


def test_claim_about_missing_file_has_zero_confidence(tools, project):
    result = asyncio.run(
        tools[0]("SQL injection in query builder", file_path="pkg/missing.py", exec_ctx=project)
    )
    assert result["success"] is True
    assert result["verified"] is False
    assert result["confidence"] == 0.0
    assert "not found" in result["reason"].lower()


def test_claim_about_existing_file_gathers_evidence(tools, project):
    result = asyncio.run(
        tools[0](
            "Resource leak: file opened without context manager",
            file_path="pkg/io.py",
            line_number=2,
            exec_ctx=project,
        )
    )
    assert result["success"] is True
    assert result["confidence"] > 0.0
    sources = result["evidence"]["sources"]
    assert sources[0]["source_file"] == "pkg/io.py"
    assert sources[0]["context"]["exists"] is True


def test_batch_summarises_and_tolerates_sparse_entries(tools, project):
    result = asyncio.run(
        tools[1](
            [
                {"description": "Resource leak in load", "file_path": "pkg/io.py"},
                {"issue_type": "security", "file_path": "pkg/missing.py"},
                {},
            ],
            exec_ctx=project,
        )
    )
    assert result["success"] is True and result["total"] == 3
    assert sum(result["summary"]["severity_distribution"].values()) == 3
    assert len(result["issues"]) == 3


def test_batch_of_nothing(tools, project):
    result = asyncio.run(tools[1]([], exec_ctx=project))
    assert result["total"] == 0
    assert result["summary"]["verification_rate"] == 0.0


def test_temporal_analysis_receives_framework_execution_snapshot(tools, project, monkeypatch):
    admitted_policy = object()
    captured = {}

    class FakeTemporalAnalyzer:
        def __init__(self, project_root, run_policy_snapshot=None):
            captured["project_root"] = project_root
            captured["run_policy_snapshot"] = run_policy_snapshot

        def analyze_issue_temporal_context(self, _issue):
            return {"temporal_nature": None}

    import victor.tools.verification as verification_package

    monkeypatch.setattr(verification_package, "TemporalContextAnalyzer", FakeTemporalAnalyzer)
    result = asyncio.run(
        tools[0](
            "Resource leak in load",
            file_path="pkg/io.py",
            enable_temporal_analysis=True,
            _exec_ctx={
                "workspace_root": project["workspace_root"],
                "run_policy_snapshot": admitted_policy,
            },
        )
    )

    assert result["success"] is True
    assert captured["run_policy_snapshot"] is admitted_policy


def test_legacy_exec_ctx_snapshot_cannot_weaken_required_policy(tools, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from victor.config.sandbox_settings import SandboxSettings
    from victor.config.settings import get_settings
    from victor.runtime.policy_snapshot import RunPolicySnapshot
    from victor.tools.sandbox import backends

    (tmp_path / ".git").mkdir()
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "io.py").write_text("def load(path):\n    return open(path).read()\n")

    disabled_settings = SimpleNamespace(
        sandbox=SandboxSettings(sandbox_enabled=True, sandbox_policy="disabled")
    )
    monkeypatch.setattr("victor.config.settings.get_settings", lambda: disabled_settings)
    caller_snapshot = RunPolicySnapshot.from_settings(get_settings())
    required_settings = SimpleNamespace(
        sandbox=SandboxSettings(sandbox_enabled=True, sandbox_policy="required")
    )
    monkeypatch.setattr("victor.config.settings.get_settings", lambda: required_settings)
    monkeypatch.setattr(backends.platform, "system", lambda: "Linux")
    monkeypatch.setattr(backends.shutil, "which", lambda _name: None)

    launched = []

    def _run(argv, **_kwargs):
        launched.append([str(value) for value in argv])
        return None

    monkeypatch.setattr("subprocess.run", _run)

    result = asyncio.run(
        tools[0](
            "Resource leak in load",
            file_path="pkg/io.py",
            enable_temporal_analysis=True,
            exec_ctx={
                "workspace_root": str(tmp_path),
                "run_policy_snapshot": caller_snapshot,
            },
        )
    )

    assert result["success"] is True
    assert launched == []
