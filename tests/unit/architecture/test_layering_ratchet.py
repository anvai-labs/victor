"""Layering ratchet (scripts/ci/layering_ratchet.py).

The live check runs against the committed baseline: a new concrete upward
import fails, and a removed one fails until the baseline is regenerated.
The rule tests drive the scanner over a synthetic package so each sanctioned
exception (abstractions, TYPE_CHECKING, design-sanctioned edges) is pinned.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "layering_ratchet",
    Path(__file__).resolve().parents[3] / "scripts" / "ci" / "layering_ratchet.py",
)
assert _SPEC and _SPEC.loader
ratchet = importlib.util.module_from_spec(_SPEC)
sys.modules["layering_ratchet"] = ratchet
_SPEC.loader.exec_module(ratchet)


def test_live_tree_matches_committed_baseline():
    current = ratchet.current_pairs()
    baseline = ratchet.load_baseline()
    assert current - baseline == set(), (
        "new upward concrete imports (move code down, depend on a protocol, or "
        f"inject): {sorted(current - baseline)}"
    )
    assert baseline - current == set(), (
        "upward leaks were fixed — run `python scripts/ci/layering_ratchet.py "
        f"--update-baseline` to lock in: {sorted(baseline - current)}"
    )


@pytest.mark.parametrize(
    "module,layer",
    [
        ("victor.ui.commands.chat", "client"),
        ("victor.framework.agent", "framework"),
        ("victor.agent.orchestrator", "runtime"),
        ("victor.workflows.graph", "runtime"),
        ("victor.core.container", "infrastructure"),
        # longest prefix wins: the MCP client is infrastructure, its server
        # (victor exposed as an MCP server) stays a client surface
        ("victor.integrations.mcp.client", "infrastructure"),
        ("victor.integrations.mcp.server", "client"),
        ("victor.integrations.api.server", "client"),
        ("victor.observability.metrics", None),
        ("victor_contracts.rl", None),
    ],
)
def test_layer_of(module, layer):
    assert ratchet.layer_of(module) == layer


@pytest.fixture
def fake_tree(tmp_path, monkeypatch):
    root = tmp_path
    (root / "victor").mkdir()

    def write(rel: str, src: str) -> Path:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        for parent in path.relative_to(root).parents:
            init = root / parent / "__init__.py"
            if str(parent) != "." and not init.exists():
                init.write_text("", encoding="utf-8")
        path.write_text(src, encoding="utf-8")
        return path

    monkeypatch.setattr(ratchet, "ROOT", root)
    monkeypatch.setattr(ratchet, "PKG", root / "victor")
    ratchet._module_file.cache_clear()
    ratchet._abstract_names.cache_clear()
    yield write
    ratchet._module_file.cache_clear()
    ratchet._abstract_names.cache_clear()


def _leaks(write, files):
    return set(ratchet.iter_upward_leaks(files))


def test_concrete_upward_import_is_a_leak(fake_tree):
    fake_tree("victor/framework/builder.py", "class Builder:\n    pass\n")
    f = fake_tree("victor/agent/user.py", "from victor.framework.builder import Builder\n")
    assert _leaks(fake_tree, [f]) == {("victor.agent.user", "victor.framework.builder")}


def test_function_local_import_still_counts(fake_tree):
    fake_tree("victor/framework/builder.py", "def build():\n    pass\n")
    f = fake_tree(
        "victor/core/thing.py",
        "def run():\n    from victor.framework.builder import build\n    return build()\n",
    )
    assert _leaks(fake_tree, [f]) == {("victor.core.thing", "victor.framework.builder")}


def test_protocol_and_contract_imports_are_dependency_inversion(fake_tree):
    fake_tree(
        "victor/framework/protocols.py",
        "from typing import Protocol, runtime_checkable\n"
        "from dataclasses import dataclass\nfrom enum import Enum\n"
        "class Port(Protocol):\n    def go(self) -> None: ...\n"
        "@runtime_checkable\nclass Checked(Protocol):\n    pass\n"
        "@dataclass\nclass Payload:\n    x: int\n"
        "class Kind(Enum):\n    A = 1\n"
        "class Impl:\n    pass\n",
    )
    ok = fake_tree(
        "victor/agent/ok.py",
        "from victor.framework.protocols import Port, Checked, Payload, Kind\n",
    )
    mixed = fake_tree(
        "victor/agent/mixed.py", "from victor.framework.protocols import Port, Impl\n"
    )
    assert _leaks(fake_tree, [ok]) == set()
    assert _leaks(fake_tree, [mixed]) == {("victor.agent.mixed", "victor.framework.protocols")}


def test_type_checking_imports_are_ignored(fake_tree):
    fake_tree("victor/framework/builder.py", "class Builder:\n    pass\n")
    f = fake_tree(
        "victor/agent/typed.py",
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    from victor.framework.builder import Builder\n",
    )
    assert _leaks(fake_tree, [f]) == set()


def test_downward_and_same_layer_imports_are_fine(fake_tree):
    fake_tree("victor/core/util.py", "def helper():\n    pass\n")
    fake_tree("victor/agent/peer.py", "def peer():\n    pass\n")
    down = fake_tree("victor/framework/top.py", "from victor.core.util import helper\n")
    same = fake_tree("victor/agent/other.py", "from victor.agent.peer import peer\n")
    assert _leaks(fake_tree, [down, same]) == set()


def test_sanctioned_edges(fake_tree):
    fake_tree("victor/framework/graph.py", "class StateGraph:\n    pass\n")
    fake_tree("victor/framework/client.py", "class VictorClient:\n    pass\n")
    team = fake_tree("victor/teams/coord.py", "from victor.framework.graph import StateGraph\n")
    root = fake_tree(
        "victor/core/bootstrap.py", "from victor.framework.client import VictorClient\n"
    )
    assert _leaks(fake_tree, [team, root]) == set()


def test_runtime_may_drive_the_rl_engine(fake_tree):
    fake_tree("victor/framework/rl/__init__.py", "def analyze():\n    pass\n")
    fake_tree("victor/framework/rl/coordinator.py", "def get_rl_coordinator():\n    pass\n")
    agent = fake_tree(
        "victor/agent/rl_user.py",
        "from victor.framework.rl.coordinator import get_rl_coordinator\n"
        "from victor.framework.rl import analyze\n",
    )
    team = fake_tree(
        "victor/teams/rl_user.py",
        "from victor.framework.rl.coordinator import get_rl_coordinator\n",
    )
    assert _leaks(fake_tree, [agent, team]) == set()


def test_rl_sanction_is_runtime_only_and_package_bounded(fake_tree):
    fake_tree("victor/framework/rl/coordinator.py", "def get_rl_coordinator():\n    pass\n")
    fake_tree("victor/framework/rlx.py", "def lookalike():\n    pass\n")
    fake_tree("victor/framework/agentic_loop.py", "def run():\n    pass\n")
    tool = fake_tree(
        "victor/tools/rl_user.py",
        "from victor.framework.rl.coordinator import get_rl_coordinator\n",
    )
    lookalike = fake_tree(
        "victor/agent/lookalike.py", "from victor.framework.rlx import lookalike\n"
    )
    other = fake_tree(
        "victor/agent/loop_user.py", "from victor.framework.agentic_loop import run\n"
    )
    assert _leaks(fake_tree, [tool, lookalike, other]) == {
        ("victor.tools.rl_user", "victor.framework.rl.coordinator"),
        ("victor.agent.lookalike", "victor.framework.rlx"),
        ("victor.agent.loop_user", "victor.framework.agentic_loop"),
    }


def test_relative_import_resolution(fake_tree):
    fake_tree("victor/framework/rl/hooks.py", "def get_rl_hooks():\n    pass\n")
    # a relative import inside the framework package resolves within it: no leak
    inner = fake_tree("victor/framework/rl/user.py", "from .hooks import get_rl_hooks\n")
    assert _leaks(fake_tree, [inner]) == set()


def test_check_reports_new_and_fixed(capsys):
    assert ratchet.check({"a -> b"}, {"a -> b"}) == 0
    assert ratchet.check({"a -> b", "c -> d"}, {"a -> b"}) == 1
    assert "NEW upward leak: c -> d" in capsys.readouterr().out
    assert ratchet.check(set(), {"a -> b"}) == 1
    assert "--update-baseline" in capsys.readouterr().out


def test_exception_classes_are_concrete_coupling(fake_tree):
    """Catching a higher layer's exception by class is coupling, not inversion."""
    fake_tree("victor/framework/errors.py", "class Paused(BaseException):\n    pass\n")
    f = fake_tree("victor/agent/catcher.py", "from victor.framework.errors import Paused\n")
    assert _leaks(fake_tree, [f]) == {("victor.agent.catcher", "victor.framework.errors")}
