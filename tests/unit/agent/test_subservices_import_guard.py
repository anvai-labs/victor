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

"""Import boundary guard tests for chat and tool subservices.

Migration Note (2026-05-01):
- victor/agent/services/chat/* subservices removed (unused parallel architecture)
- victor/agent/services/tools/* subservices removed (unused parallel architecture)
- Canonical ChatService is in victor/agent/services/chat_service.py
- Canonical ToolService is in victor/agent/services/tool_service.py
- Both chat/ and tools/ subdirectories were completely deleted

These tests prevent regression by ensuring no new internal code imports
from the removed subservice packages.
"""

import ast
import os
from pathlib import Path

import pytest

REMOVED_SUBSERVICE_PACKAGES = (
    "victor.agent.services.chat",
    "victor.agent.services.tools",
)


def _get_python_files(directory: Path) -> list[Path]:
    """Get all Python files in a directory recursively."""
    files = []
    for root, dirs, filenames in os.walk(directory):
        # Skip test files and __pycache__
        dirs[:] = [d for d in dirs if d not in ["__pycache__", "test", "tests"]]

        for filename in filenames:
            if filename.endswith(".py"):
                files.append(Path(root) / filename)
    return files


def _check_subservice_import(file_path: Path) -> bool:
    """Check if a file imports from removed subservice packages.

    Returns True if the forbidden import is found.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "victor.agent.services":
                    if any(alias.name in {"chat", "tools"} for alias in node.names):
                        return True

                if node.module and any(
                    node.module == package or node.module.startswith(f"{package}.")
                    for package in REMOVED_SUBSERVICE_PACKAGES
                ):
                    return True

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(
                        alias.name == package or alias.name.startswith(f"{package}.")
                        for package in REMOVED_SUBSERVICE_PACKAGES
                    ):
                        return True

        return False
    except (SyntaxError, UnicodeDecodeError):
        # Skip files that can't be parsed
        return False


@pytest.mark.parametrize(
    "source",
    [
        "import victor.agent.services.chat\n",
        "import victor.agent.services.tools as removed_tools\n",
        "from victor.agent.services.chat import ChatFlowService\n",
        "from victor.agent.services.tools import ToolExecutorService\n",
        "from victor.agent.services import chat\n",
        "from victor.agent.services import tools as removed_tools\n",
        "from victor.agent.services.chat.protocols import ChatRuntimeProtocol\n",
        "from victor.agent.services.tools.protocols import ToolExecutorProtocol\n",
    ],
)
def test_removed_subservice_import_patterns_are_rejected(tmp_path: Path, source: str) -> None:
    """Catch exact package imports and submodule imports for removed subservices."""
    sample = tmp_path / "sample.py"
    sample.write_text(source, encoding="utf-8")

    assert _check_subservice_import(sample) is True


@pytest.mark.parametrize(
    "source",
    [
        "import victor.agent.services.chat_service\n",
        "import victor.agent.services.tool_service as tool_service\n",
        "from victor.agent.services.chat_service import ChatService\n",
        "from victor.agent.services.tool_service import ToolService\n",
    ],
)
def test_canonical_service_import_patterns_remain_allowed(tmp_path: Path, source: str) -> None:
    """Allow canonical service imports that replaced the removed subservices."""
    sample = tmp_path / "sample.py"
    sample.write_text(source, encoding="utf-8")

    assert _check_subservice_import(sample) is False


def test_internal_code_does_not_import_removed_subservices():
    """Test that internal production code does not import removed subservices.

    Subservices removed 2026-05-01:
    - victor/agent/services/chat/* (complete directory removed)
    - victor/agent/services/tools/* (complete directory removed)

    Canonical services:
    - victor/agent/services/chat_service.py (ChatService)
    - victor/agent/services/tool_service.py (ToolService)

    This is an AST-based guard test to prevent regression.
    """
    repo_root = Path(__file__).resolve().parents[3]
    victor_dir = repo_root / "victor"

    violations = []

    for py_file in _get_python_files(victor_dir):
        relative_path = py_file.relative_to(repo_root)
        if _check_subservice_import(py_file):
            violations.append(str(relative_path))

    assert not violations, (
        f"Found {len(violations)} file(s) importing removed subservices.\n"
        f"Use canonical ChatService/ToolService instead.\n"
        f"Violations: {violations}"
    )


def test_removed_chat_package_import_fails():
    """Test that importing removed victor.agent.services.chat package fails.

    The entire chat/ subservice directory was removed 2026-05-01.
    Importing should raise ModuleNotFoundError.
    """
    with pytest.raises(ModuleNotFoundError, match="victor.agent.services.chat"):
        import victor.agent.services.chat  # noqa: F401


def test_removed_tools_package_import_fails():
    """Test that importing removed victor.agent.services.tools package fails.

    The entire tools/ subservice directory was removed 2026-05-01.
    Importing should raise ModuleNotFoundError.
    """
    with pytest.raises(ModuleNotFoundError, match="victor.agent.services.tools"):
        import victor.agent.services.tools  # noqa: F401


def test_canonical_chat_service_import_works():
    """Test that canonical ChatService can still be imported.

    Canonical path: victor.agent.services.chat_service.ChatService
    """
    from victor.agent.services.chat_service import ChatService

    assert ChatService is not None


def test_canonical_tool_service_import_works():
    """Test that canonical ToolService can still be imported.

    Canonical path: victor.agent.services.tool_service.ToolService
    """
    from victor.agent.services.tool_service import ToolService

    assert ToolService is not None
