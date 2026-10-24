"""Naming-collision fixes (self-analysis round 2 pause, 2026-10-24).

The Agent.__init__ class-NAME gate forced every test double in the repo to
be NAMED "AgentOrchestrator" (5 files), shadowing the real symbol in the
code graph. These tests pin the three layers of the fix: structural
validation, non-shadowing doubles, and ambiguity-tolerant impact
resolution.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class TestStructuralOrchestratorValidation:
    def test_rejects_casual_misuse(self):
        from victor.framework.agent import Agent

        for bad in ("a string", None, {"chat": True}, 42, object()):
            with pytest.raises(ValueError, match="requires an AgentOrchestrator"):
                Agent(bad)

    def test_accepts_orchestrator_shaped_stand_ins_without_name_matching(self):
        """The gate is structural: a stand-in named ANYTHING passes — the
        old gate forced doubles to be named 'AgentOrchestrator'."""
        from victor.framework.agent import Agent

        stand_in = MagicMock()
        stand_in.chat = MagicMock()
        stand_in.stream_chat = MagicMock()
        stand_in.messages = []
        # Must not raise on construction; abort before side effects.
        try:
            agent = Agent(stand_in)
        except Exception as exc:  # noqa: BLE001 - construction may wire more
            assert not isinstance(exc, ValueError), f"orchestrator-shaped stand-in rejected: {exc}"
        else:
            assert agent is not None

    def test_real_orchestrator_class_satisfies_the_duck_surface(self):
        """The check uses the surface the REAL class demonstrably has — not
        OrchestratorProtocol, which the concrete class does not satisfy at
        runtime (model/provider_name/tool_budget are dynamic getattr
        surfaces). Pinned so a future property removal that would break the
        gate surfaces here first."""
        from victor.agent.orchestrator import AgentOrchestrator

        assert callable(getattr(AgentOrchestrator, "chat", None))
        assert callable(getattr(AgentOrchestrator, "stream_chat", None))
        assert hasattr(AgentOrchestrator, "messages")


class TestNoProductionSymbolShadowing:
    def test_test_classes_shadowing_production_symbols_are_aliased_or_renamed(self):
        """GENERAL collision guard — no hardcoded symbol names (the tool runs
        across many repos). Scans test files for top-level classes whose
        names are ALSO production classes under victor/; anything not in the
        explicit alias allowlist is a shadowing bug. The allowlist documents
        INTENTIONAL aliases (with reasons), not suppression — the default is
        a semantically unique rename (…Mock / Fake…)."""
        import ast
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]

        # Intentional aliases: production class name -> reason the test-local
        # class of the same name is deliberate. The entries below are TRACKED
        # DEBT (semantically-unique renames queued in the naming-collisions
        # TD backlog, docs/development/naming-collisions.md), listed per-name
        # so nothing passes silently; a NEW shadow not listed here FAILS.
        # (Function-nested test doubles — Config/State/Agent-style — cannot
        # shadow imports and are handled by graph resolution; not listed.)
        allowlist = {
            "MockProvider": "test-local provider fixtures colliding with the "
            "distinct production MockProvider; rename queued",
            "ModelFamily": "test-local enum fixture; rename queued",
            "TestErrorHandling": "test-suffixed classes colliding with the "
            "production TestErrorHandling reporter; rename queued (13 files)",
            "TestStatus": "test-local status enum colliding with the "
            "production TestStatus; rename queued",
            "User": "test-local RBAC fixtures; rename queued",
        }

        production_classes: set[str] = set()
        for path in (repo_root / "victor").rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    production_classes.add(node.name)

        shadows = []
        for path in (repo_root / "tests").rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError):
                continue
            for node in tree.body:  # top-level only: fixture doubles
                if (
                    isinstance(node, ast.ClassDef)
                    and node.name in production_classes
                    and node.name not in allowlist
                ):
                    shadows.append(f"{path.relative_to(repo_root)}:{node.lineno} {node.name}")

        assert not shadows, (
            "test classes shadow production symbols (rename the double — "
            "AgentOrchestratorMock-style — or add an intentional-alias entry "
            "with a reason):\n  " + "\n  ".join(shadows)
        )


class TestAmbiguityTolerantResolution:
    def test_resolution_prefers_non_test_candidates(self, tmp_path, monkeypatch):
        """A bare name matching test doubles + the real symbol resolves to
        the non-test definition."""
        from victor.ui.commands import graph as graph_cmd

        async def find_nodes(name):
            assert name == "AgentOrchestrator"
            return [
                SimpleNamespace(
                    node_id="test-double-1",
                    name="AgentOrchestrator",
                    type="class",
                    file="tests/unit/test_a.py",
                    line=10,
                ),
                SimpleNamespace(
                    node_id="the-real-one",
                    name="AgentOrchestrator",
                    type="class",
                    file="victor/agent/orchestrator.py",
                    line=449,
                ),
            ]

        store = SimpleNamespace(find_nodes=find_nodes)

        async def get_node_by_id(node_id):
            return None

        store.get_node_by_id = get_node_by_id

        import asyncio

        resolved = asyncio.run(graph_cmd._resolve_target("AgentOrchestrator", store))
        assert resolved == "the-real-one"

    def test_resolution_lists_candidates_when_still_ambiguous(self, monkeypatch):
        """Multiple non-test definitions print the candidates instead of a
        bare refusal — the user keeps the information needed to proceed."""
        from rich.console import Console

        from victor.ui.commands import graph as graph_cmd

        async def find_nodes(name):
            return [
                SimpleNamespace(
                    node_id="a",
                    name="helper",
                    type="function",
                    file="victor/core/x.py",
                    line=1,
                ),
                SimpleNamespace(
                    node_id="b",
                    name="helper",
                    type="function",
                    file="victor/tools/y.py",
                    line=2,
                ),
            ]

        store = SimpleNamespace(find_nodes=find_nodes)

        async def get_node_by_id(node_id):
            return None

        store.get_node_by_id = get_node_by_id

        output = Console(record=True, width=120)
        import asyncio

        with monkeypatch.context() as m:
            m.setattr(graph_cmd, "console", output)
            resolved = asyncio.run(graph_cmd._resolve_target("helper", store))
        assert resolved is None  # genuinely ambiguous — no guessing
        rendered = output.export_text()
        assert "Ambiguous target 'helper'" in rendered
        assert "victor/core/x.py:1" in rendered
        assert "victor/tools/y.py:2" in rendered
        assert "file:line" in rendered
