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

"""Guards for the lazy (PEP 562) top-level ``victor`` package.

``import victor`` must stay cheap: no orchestrator/framework/provider stack
may load until a public name is actually accessed. These tests run in a clean
subprocess because the pytest worker has already imported most of victor.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Subtrees that must NOT be imported by a bare `import victor`. Keep this list
# in sync with the lazy-import contract in victor/__init__.py.
_HEAVY_SUBTREES = [
    "victor.agent",
    "victor.framework",
    "victor.providers",
    "victor.tools",
    "victor.workflows",
    "victor.core.verticals",
    "victor.config.settings",
]


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )


class TestLazyTopLevelImport:
    """`import victor` stays cheap; the public API still resolves."""

    def test_bare_import_loads_no_heavy_subtrees(self) -> None:
        script = f"""
import sys
import victor

heavy = {_HEAVY_SUBTREES!r}
loaded = sorted(m for m in sys.modules if any(m == h or m.startswith(h + ".") for h in heavy))
if loaded:
    print("eagerly imported:", ", ".join(loaded))
    raise SystemExit(1)
"""
        result = _run(script)
        if result.returncode != 0:
            pytest.fail(
                "`import victor` pulled in heavy subtrees — the lazy __init__ regressed:\n"
                f"{result.stdout.strip() or result.stderr.strip()}"
            )

    def test_public_api_resolves_lazily(self) -> None:
        script = """
import victor

# Attribute access and from-import forms must both resolve.
assert victor.Agent.__name__ == "Agent"
assert victor.Settings.__name__ == "Settings"
assert victor.AgentOrchestrator.__name__ == "AgentOrchestrator"
from victor import task, UnifiedAgentConfig  # noqa: F401

# Unknown names must raise AttributeError (not ImportError/None).
try:
    victor.DefinitelyNotAnExport
except AttributeError:
    pass
else:
    raise SystemExit("AttributeError not raised for unknown attribute")

# dir() must advertise the public API without forcing imports first.
d = dir(victor)
for name in ("Agent", "agent", "task", "Settings"):
    assert name in d, f"{name} missing from dir(victor)"
"""
        result = _run(script)
        assert result.returncode == 0, result.stdout.strip() or result.stderr.strip()

    def test_agent_decorator_survives_submodule_import(self) -> None:
        """`victor.agent` is the real subpackage AND stays callable.

        The import machinery binds the submodule onto the parent package on
        first import — the decorator lives on the (callable) module itself,
        so importing any victor.agent submodule first must not break it.
        """
        script = """
import victor.agent.orchestrator  # bind the submodule onto the parent first
import victor

assert callable(victor.agent), type(victor.agent).__name__
import victor.agent as agent_pkg
assert victor.agent is agent_pkg  # mock.patch("victor.agent...") depends on this
"""
        result = _run(script)
        assert result.returncode == 0, result.stdout.strip() or result.stderr.strip()


# Direct-entry imports that historically deadlocked on import cycles once the
# top-level victor/__init__ went lazy (the eager init always entered via
# framework first, masking them). Each runs in a fresh interpreter because the
# pytest worker has already imported most of victor.
# - victor.core.verticals.package_schema: vertical-validation CI entry
# - victor.classification: cycle via framework.task.complexity
# - victor.protocols: cycle via agent.runtime_intelligence_pipeline
_DIRECT_ENTRY_MODULES = [
    "victor.core.verticals.package_schema",
    "victor.core.verticals",
    "victor.classification",
    "victor.protocols",
]


@pytest.mark.parametrize("module", _DIRECT_ENTRY_MODULES)
def test_direct_submodule_entry_has_no_import_cycle(module: str) -> None:
    """Importing a subpackage FIRST (no prior `import victor`) must work."""
    result = _run(f"import {module}")
    assert result.returncode == 0, (
        f"direct `import {module}` failed — an import cycle regressed:\n"
        f"{result.stdout.strip() or result.stderr.strip()}"
    )
