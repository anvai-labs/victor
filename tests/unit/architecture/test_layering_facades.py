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
    ("victor.framework.prompt_document", "victor.agent.prompting.document"),
    ("victor.framework.prompt_sections", "victor.agent.prompting.sections"),
    ("victor.framework.prompt_builder", "victor.agent.prompting.builder"),
    (
        "victor.framework.prompt_dictionary_compressor",
        "victor.agent.prompting.dictionary_compressor",
    ),
    ("victor.framework.action_recovery", "victor.core.action_recovery"),
    ("victor.framework.approval_binding", "victor.agent.approval_binding"),
    ("victor.framework.approval_pause", "victor.agent.approval_pause"),
    ("victor.framework.execution_checkpoint", "victor.agent.execution_checkpoint"),
    ("victor.framework.hitl", "victor.agent.hitl"),
    ("victor.framework.policies", "victor.agent.policies"),
    ("victor.framework.policies.base", "victor.agent.policies.base"),
    ("victor.framework.policies.builtins", "victor.agent.policies.builtins"),
    ("victor.framework.policies.engine", "victor.agent.policies.engine"),
    ("victor.framework.policies.gate", "victor.agent.policies.gate"),
    ("victor.framework.policies.handlers", "victor.agent.policies.handlers"),
    ("victor.framework.policies.middleware", "victor.agent.policies.middleware"),
    ("victor.framework.policies.types", "victor.agent.policies.types"),
    ("victor.framework.task.protocols", "victor.core.task.protocols"),
    ("victor.framework.task.direct_response", "victor.core.task.direct_response"),
    ("victor.framework.task_types", "victor.core.task_types"),
    ("victor.framework.loop.guards", "victor.agent.loop_guards"),
    ("victor.framework.perception_integration", "victor.agent.perception_integration"),
    ("victor.framework.rl.hooks", "victor.core.rl.hooks"),
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
    "victor.agent.prompt_normalizer",
    "victor.agent.prompt_orchestrator",
    "victor.agent.system_prompt_policy",
    "victor.agent.services.prompt_runtime",
    "victor.agent.action_observation",
    "victor.agent.durable_resume",
    "victor.agent.tool_pipeline",
    "victor.agent.middleware_chain",
    "victor.agent.services.planning_runtime",
    "victor.teams.unified_coordinator",
    "victor.storage.embeddings.task_classifier",
    "victor.classification.pattern_registry",
    "victor.agent.query_classifier",
    "victor.agent.continuation_strategy",
    "victor.agent.services.runtime_intelligence",
    "victor.storage.cache.tiered_cache",
    "victor.teams.mixins.rl",
    "victor.agent.mode_controller",
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


def test_framework_public_prompt_api_is_the_moved_class():
    """victor.framework's lazy public export still serves PromptBuilder."""
    import victor.framework as framework
    from victor.agent.prompting.builder import PromptBuilder

    assert framework.PromptBuilder is PromptBuilder


def test_old_policies_submodule_path_never_loads_a_second_copy():
    """A cold import of an old package-submodule path must hit the alias.

    Aliasing only the package would make Python search the moved package's
    __path__ and execute engine.py again under the old name: two PolicyEngine
    classes, two module states. Run in a fresh interpreter so it is truly cold.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import victor.framework.policies.engine as old\n"
        "import victor.agent.policies.engine as new\n"
        "assert old is new\n"
        "files = {m.__file__ for k, m in list(sys.modules.items()) if k.endswith('policies.engine')}\n"
        "assert len(files) == 1, files\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]


def test_approval_contextvars_are_one_object_through_the_facade():
    import victor.agent.approval_binding as new
    import victor.framework.approval_binding as old

    assert old.current_approval_call is new.current_approval_call
    assert old.current_approval_grant is new.current_approval_grant
    token = new.current_approval_call.set("call-1")
    try:
        assert old.current_approval_call.get() == "call-1"
    finally:
        new.current_approval_call.reset(token)


def test_approval_pause_caught_by_old_name():
    """Framework code catches ApprovalPause by class; both names are one class."""
    from victor.agent.approval_pause import ApprovalPause as New
    from victor.framework.approval_pause import ApprovalPause as Old

    assert Old is New


def test_framework_task_package_reexports_the_moved_vocabulary():
    """victor.framework.task re-exports protocols/direct_response via facades."""
    import victor.framework.task as task
    from victor.core.task.direct_response import classify_direct_response_prompt
    from victor.core.task.protocols import TaskComplexity

    assert task.TaskComplexity is TaskComplexity
    assert task.classify_direct_response_prompt is classify_direct_response_prompt


def test_loop_guards_contextvar_reaches_the_perception_hook():
    """The framework loop installs guards through the old path; the moved
    perception seam (runtime) must count against those same guards."""
    import victor.framework.loop.guards as old
    from victor.agent import loop_guards as new

    assert old._CURRENT_GUARDS is new._CURRENT_GUARDS
    calls = []
    fake = mock.Mock(note_perception=lambda: calls.append(1))
    token = old._CURRENT_GUARDS.set(fake)
    try:
        new.note_perception()
    finally:
        old._CURRENT_GUARDS.reset(token)
    assert calls == [1]


def test_rl_hook_registry_singleton_is_shared_with_the_coordinator_wiring():
    """RLCoordinator wires itself in through the old framework path; emitters
    (caches, selectors, runtime) now import the core path. One registry."""
    import victor.framework.rl.hooks as old
    from victor.core.rl import hooks as new

    new.reset_rl_hooks()
    try:
        coordinator = mock.Mock(is_closed=False)
        registry = old.get_rl_hooks(coordinator=coordinator)
        assert new.get_rl_hooks() is registry
        assert registry._coordinator is coordinator
        assert old.RLEventType is new.RLEventType and old.RLEvent is new.RLEvent
    finally:
        new.reset_rl_hooks()


def test_no_sys_modules_seam_is_keyed_on_a_facade_path():
    """``patch.dict(sys.modules, {old_path: fake})`` cannot reach moved code.

    The alias facade makes old-path ``mock.patch`` targets work, but a fake
    injected under the OLD key in sys.modules is never imported by code that
    now imports the new path, so the test silently stops exercising anything.
    Key such seams on the new module path.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    old_paths = {old for old, _ in MOVES}
    offenders = []
    for path in (root / "tests").rglob("*.py"):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "sys.modules" not in text:
            continue
        for old in old_paths:
            if f'"{old}"' in text or f"'{old}'" in text:
                offenders.append(f"{path.relative_to(root)}: {old}")
    assert offenders == [], offenders
