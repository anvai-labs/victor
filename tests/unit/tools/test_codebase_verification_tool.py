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
    registry = SharedToolRegistry.get_instance()
    registry.get_tool_classes()
    assert {"codebase_verify", "codebase_verify_batch"} <= set(registry._tool_classes)


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
