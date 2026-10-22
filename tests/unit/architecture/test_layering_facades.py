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
    ("victor.framework.entry_point_registry", "victor.core.entry_point_registry"),
    ("victor.framework.module_loader", "victor.core.module_loader"),
    ("victor.runtime.trace_context", "victor.core.trace_context"),
    ("victor.agent.change_tracker", "victor.core.change_tracker"),
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
    "victor.core.chunking.registry",
    "victor.core.plugins.registry",
    "victor.core.utils.capability_loader",
    "victor.core.verticals.vertical_loader",
    "victor.tools.registry",
    "victor.tools.plugin_registry",
    "victor.tools.patch_tool",
    "victor.integrations.api.change_tracker_ops",
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


def test_singleton_is_shared_through_the_facade():
    """The FileChangeHistory singleton tools write to is the one clients read."""
    from victor.agent.change_tracker import get_change_tracker as old_get
    from victor.core.change_tracker import get_change_tracker as new_get
    from victor.core.change_tracker import reset_change_tracker

    reset_change_tracker()
    try:
        assert old_get() is new_get()
    finally:
        reset_change_tracker()


def test_old_path_patch_reaches_moved_module_internals():
    """test_vertical_registry_reset patches module_loader through its old path."""
    import victor.core.module_loader as moved

    sentinel = object()
    with mock.patch("victor.framework.module_loader.get_entry_point_cache", return_value=sentinel):
        assert moved.get_entry_point_cache() is sentinel


def test_contextvar_is_one_object_through_the_facade():
    import victor.core.trace_context as new
    import victor.runtime.trace_context as old

    assert old._current_trace is new._current_trace
    assert old.get_correlation_id is new.get_correlation_id


@pytest.mark.parametrize("module", RETARGETED)
def test_retargeted_importers_still_import(module):
    importlib.import_module(module)


def test_client_change_ops_read_the_tracker_patched_through_the_old_path():
    """API routes (client layer) and tools now import the core path; a patch
    through the old runtime path must still be what they see."""
    from victor.integrations.api import change_tracker_ops as ops

    tracker = mock.Mock()
    tracker.undo.return_value = (True, "undone", ["a.py"])
    tracker.redo.return_value = (False, "nothing to redo", [])
    tracker.get_history.return_value = [{"id": "g1"}]
    with mock.patch("victor.agent.change_tracker.get_change_tracker", return_value=tracker):
        assert ops.undo_last_change() == {"success": True, "message": "undone", "files": ["a.py"]}
        assert ops.redo_last_change() == {
            "success": False,
            "message": "nothing to redo",
            "files": [],
        }
        assert ops.change_history(5) == {"history": [{"id": "g1"}]}
    tracker.get_history.assert_called_once_with(limit=5)
