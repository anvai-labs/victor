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

"""Web server architectural boundary guard (P0-A).

Enforces the workspace's "UI layer MUST use VictorClient, NEVER import
AgentOrchestrator" mandate for the FastAPI web surface under ``web/server/``.

The existing ``test_architectural_boundaries.py`` scans ``victor/ui`` and
``victor/commands`` but not the out-of-package ``web/server`` tree, so this
test closes that gap. It AST-walks every Python file under ``web/server/``
and fails on any import of ``victor.agent.orchestrator`` (direct or aliased),
matching the canonical scanner pattern.

Run with: pytest tests/unit/framework/test_web_server_boundary.py -v
"""

import ast
from pathlib import Path

import pytest

# tests/unit/framework/test_web_server_boundary.py -> repo root
REPO_ROOT = Path(__file__).parent.parent.parent.parent
WEB_SERVER_DIR = REPO_ROOT / "web" / "server"

# Modules the web layer is forbidden to import directly. The web server must
# route through ``victor.framework.client.VictorClient`` (the same seam the
# CLI/TUI use) rather than re-implementing agent lifecycle on top of the
# orchestrator.
FORBIDDEN_MODULES = {"victor.agent.orchestrator", "victor.agent.orchestrator_properties"}


@pytest.fixture
def web_server_files():
    """All Python files under web/server/, if the directory exists."""
    if not WEB_SERVER_DIR.exists():
        pytest.skip(f"web server directory not found at {WEB_SERVER_DIR}")
    return list(WEB_SERVER_DIR.rglob("*.py"))


class TestWebServerArchitecturalBoundary:
    """Guard: the web layer must not bypass the VictorClient seam."""

    def test_web_server_does_not_import_orchestrator(self, web_server_files):
        """web/server/** MUST NOT import victor.agent.orchestrator.

        The web server is a UI-layer surface and must go through
        ``VictorClient`` (created via ``Agent.create(session_config=...)``),
        not construct ``AgentOrchestrator`` directly. See the P0-A task in
        ``docs/ux-adoption-action-plan.md`` and the "Design Mandates" section
        of CLAUDE.md.
        """
        assert web_server_files, "no Python files found under web/server/"

        violations = []

        for file_path in web_server_files:
            # Skip vendored/generated or test files if any are ever added here.
            if "test_" in file_path.name:
                continue

            try:
                source = file_path.read_text()
                tree = ast.parse(source, filename=str(file_path))
            except (OSError, SyntaxError) as exc:
                pytest.fail(f"could not parse {file_path}: {exc}")

            for node in ast.walk(tree):
                # `from victor.agent.orchestrator import AgentOrchestrator`
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module in FORBIDDEN_MODULES:
                        for alias in node.names:
                            violations.append(
                                {
                                    "file": str(file_path.relative_to(REPO_ROOT)),
                                    "line": node.lineno,
                                    "import": f"from {module} import {alias.name}",
                                }
                            )

                # `import victor.agent.orchestrator` (bare/aliased)
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in FORBIDDEN_MODULES:
                            violations.append(
                                {
                                    "file": str(file_path.relative_to(REPO_ROOT)),
                                    "line": node.lineno,
                                    "import": f"import {alias.name}",
                                }
                            )

        if violations:
            details = "\n".join(f"  - {v['file']}:{v['line']}  {v['import']}" for v in violations)
            pytest.fail(
                "web/server/ must not import the orchestrator directly.\n"
                "Route the web layer through victor.framework.client.VictorClient "
                "instead.\n\nViolations:\n" + details
            )

    def test_web_server_uses_victor_client_seam(self, web_server_files):
        """web/server/** SHOULD import the VictorClient seam.

        A soft positive guard: if the web layer is present, at least one file
        must wire up ``VictorClient`` so the server is exercising the
        framework-owned lifecycle rather than re-implementing it.
        """
        assert web_server_files, "no Python files found under web/server/"

        uses_client = False
        for file_path in web_server_files:
            try:
                source = file_path.read_text()
            except OSError:
                continue
            if "VictorClient" in source and "victor.framework.client" in source:
                uses_client = True
                break

        assert uses_client, (
            "web/server/ should import VictorClient from victor.framework.client "
            "to honor the UI-layer seam."
        )


class TestWebServerSessionStateBoundary:
    """Guard (P0-B): session state must live in the injectable store.

    ``main.py`` previously held sessions in module-level mutable dicts
    (``SESSION_AGENTS``/``SESSION_TOKENS``) guarded by a module-level lock —
    unrestartable, single-worker state that also serialized agent creation
    behind a global lock. State now lives behind the ``SessionStore``
    protocol (``web/server/session_store.py``); this ratchet keeps the dicts
    from coming back.
    """

    LEGACY_SESSION_GLOBALS = {"SESSION_AGENTS", "SESSION_TOKENS", "SESSION_LOCK"}

    def test_no_module_level_session_dicts_in_main(self):
        main_py = WEB_SERVER_DIR / "main.py"
        if not main_py.exists():
            pytest.skip("web/server/main.py not found")

        tree = ast.parse(main_py.read_text(), filename=str(main_py))
        offenders = []
        for node in tree.body:  # module level only
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in self.LEGACY_SESSION_GLOBALS:
                    offenders.append(f"{target.id} (line {node.lineno})")

        assert not offenders, (
            "Legacy module-level session state reintroduced in web/server/main.py: "
            f"{offenders}. Use the SessionStore protocol "
            "(web/server/session_store.py) instead."
        )

    def test_main_uses_session_store_seam(self):
        main_py = WEB_SERVER_DIR / "main.py"
        if not main_py.exists():
            pytest.skip("web/server/main.py not found")
        source = main_py.read_text()
        assert "session_store" in source, (
            "web/server/main.py no longer references the session_store seam — "
            "if session management moved, update this guard deliberately."
        )
