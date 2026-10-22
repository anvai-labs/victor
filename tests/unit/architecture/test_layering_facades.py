"""Layering round 2: moved modules and their compatibility facades.

Each module moved down a layer leaves a typed alias facade at its old path.
The contract pinned here: the old dotted path IS the moved module object
(so module state, singletons and ContextVars stay one object), and an
old-path ``mock.patch`` reaches the code that actually runs.
"""

from __future__ import annotations

import importlib
from unittest import mock

import pytest

MOVES = [
    ("victor.framework.request_scope_heuristics", "victor.agent.request_scope_heuristics"),
    ("victor.framework.search_novelty", "victor.agent.search_novelty"),
    ("victor.framework.preload", "victor.agent.preload"),
    ("victor.framework.tool_progress", "victor.core.tool_progress"),
    ("victor.framework.graph_algorithms", "victor.core.graph_algorithms"),
    ("victor.framework.tool_naming", "victor.core.tool_naming"),
    ("victor.framework.contextual_errors", "victor.core.contextual_errors"),
    ("victor.agent.tool_sequence_tracker", "victor.tools.selection.sequence_tracker"),
]

# Lower-layer modules retargeted to the new homes (module-level imports).
RETARGETED = [
    "victor.tools.http_pool",
    "victor.tools.bash",
    "victor.tools.unified._search_helpers",
    "victor.core.verticals.defaults.tool_defaults",
    "victor.core.tool_dependency_loader",
    "victor.workflows.registry",
    "victor.workflows.graph",
    "victor.tools.semantic_selector",
]


@pytest.mark.parametrize("old,new", MOVES)
def test_old_path_is_the_moved_module(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)


@pytest.mark.parametrize("old,new", MOVES)
def test_parent_package_attribute_resolves_to_moved_module(old, new):
    parent, leaf = old.rsplit(".", 1)
    importlib.import_module(old)
    assert getattr(importlib.import_module(parent), leaf) is importlib.import_module(new)


def test_old_path_patch_reaches_the_running_code():
    import victor.core.tool_naming as moved

    with mock.patch("victor.framework.tool_naming.get_canonical_name", return_value="PATCHED"):
        assert moved.get_canonical_name("read_file") == "PATCHED"


def test_module_state_is_shared_through_the_facade():
    import victor.core.tool_progress as new
    import victor.framework.tool_progress as old

    sentinel = object()
    with mock.patch.object(new, "_sink", sentinel):
        assert old._sink is sentinel


@pytest.mark.parametrize("module", RETARGETED)
def test_retargeted_importers_still_import(module):
    importlib.import_module(module)
