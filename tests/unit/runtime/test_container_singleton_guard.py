"""Guard tests for ServiceContainer singleton elimination (GS-3).

Tracks get_container() call sites and prevents new ones from being added.
Allowed locations: core/container.py (definition), core/bootstrap.py (init),
core/plugins/ (plugin context), framework/service_provider.py (framework init).
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VICTOR_ROOT = REPO_ROOT / "victor"

# These locations are allowed to call get_container() — they are infrastructure
# bootstrap code, not business logic. Everything else should receive the
# container via constructor injection or ExecutionContext.
ALLOWED_GET_CONTAINER_PREFIXES = (
    "victor/core/container.py",  # Definition
    "victor/core/bootstrap.py",  # Bootstrap infrastructure
    "victor/core/service_resolution.py",  # Compatibility-only service resolution
    "victor/core/plugins/",  # Plugin context (framework init)
    "victor/core/events/",  # Event backend factory functions
    "victor/framework/service_provider.py",  # Service provider (framework init)
    "victor/framework/policies/handlers.py",  # Approval-handler registration into the container
    "victor/agent/subagents/base.py",  # Member ASK handler resolution (FEP-0028 slice 2a, #740)
    # Singleton accessor functions (DI migration compatibility shims)
    "victor/agent/task_analyzer.py",  # get_task_analyzer() singleton accessor
    "victor/agent/mode_controller.py",  # get_mode_controller() singleton accessor
    "victor/agent/tool_call_tracker.py",  # get_tool_call_tracker() singleton accessor
    "victor/storage/embeddings/service.py",  # get_embedding_service() singleton accessor
    # Service infrastructure (self-configuration)
    "victor/agent/services/tiered_decision_service.py",  # Provider detection
    # Framework infrastructure (optional service lookups)
    "victor/agent/tool_selection/selector.py",  # decision-service lazy lookup (F-004 move, #484)
    "victor/framework/agentic_loop.py",  # Decision service lookup
    "victor/framework/rl/learners/prompt_optimizer.py",  # Credit tracking service lookup
    "victor/storage/embeddings/intent_classifier.py",  # Tiered service lazy lookup
    "victor/storage/embeddings/task_classifier.py",  # Tiered service lazy lookup
    "victor/tools/semantic_selector.py",  # Tiered service lazy lookup
    # Entry points (CLI, evaluation, testing infrastructure)
    "victor/evaluation/agent_adapter.py",  # Evaluation entry point
    "victor/evaluation/real_run_runner.py",  # Benchmark runner entry point
    "victor/ui/commands/benchmark.py",  # CLI command
    "victor/ui/commands/utils.py",  # CLI utility
)


def _count_get_container_calls(root: Path) -> list:
    """Find executable get_container() calls via AST.

    Only counts calls that reference victor.core.container.get_container,
    not other get_container methods/functions (false positives).
    """
    calls = []
    for py_file in root.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        rel = str(py_file.relative_to(REPO_ROOT))
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel)
        except (SyntaxError, UnicodeDecodeError):
            continue

        # Track imports of get_container from victor.core.container
        imports_get_container = False
        import_aliases = set()  # Track 'as' aliases

        for node in ast.walk(tree):
            # Check for: from victor.core import get_container
            if isinstance(node, ast.ImportFrom):
                if node.module and "victor.core" in node.module:
                    for alias in node.names:
                        if alias.name == "get_container":
                            imports_get_container = True
                            if alias.asname:
                                import_aliases.add(alias.asname)
            # Check for: import victor.core.container as ... (then container.get_container())
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name and "victor.core.container" in alias.name:
                        # Will catch container.get_container() calls
                        pass

        # Now find actual calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Direct call: get_container()
                if isinstance(func, ast.Name) and func.id == "get_container":
                    if imports_get_container or func.id in import_aliases:
                        calls.append((rel, node.lineno))
                # Attribute call: container.get_container() or module.get_container()
                elif isinstance(func, ast.Attribute):
                    if func.attr == "get_container":
                        # Check if value is 'container' from victor.core import
                        calls.append((rel, node.lineno))
    return calls


class TestContainerSingletonGuard:
    """Prevent proliferation of get_container() calls in business logic."""

    def test_get_container_call_count_capped(self):
        """Track get_container() calls outside allowed infrastructure.

        Current known non-infrastructure callers (to be migrated):
        - agent/ coordinators, prompt pipeline, etc.
        - framework/ step handlers, capability helpers, etc.

        This test tracks the count to ensure it doesn't INCREASE.
        As call sites are migrated to DI, update the cap downward.
        """
        calls = _count_get_container_calls(VICTOR_ROOT)
        non_infra_calls = [
            (f, line)
            for f, line in calls
            if not any(f.startswith(p) for p in ALLOWED_GET_CONTAINER_PREFIXES)
        ]
        # Current count: 0 non-infrastructure calls.
        # All remaining get_container() calls are in infrastructure modules.
        # This cap prevents new business logic from using get_container() directly.
        MAX_ALLOWED = 0
        assert len(non_infra_calls) <= MAX_ALLOWED, (
            f"Found {len(non_infra_calls)} get_container() calls outside infrastructure "
            f"(cap is {MAX_ALLOWED}). Use constructor injection or ExecutionContext instead.\n"
            f"New calls:\n" + "\n".join(f"  {f}:{line}" for f, line in sorted(non_infra_calls))
        )

    def test_no_get_container_in_new_runtime_module(self):
        """The new victor/runtime/ module must not use get_container()."""
        runtime_root = VICTOR_ROOT / "runtime"
        if not runtime_root.exists():
            pytest.skip("victor/runtime/ not yet created")

        calls = _count_get_container_calls(runtime_root)
        assert not calls, (
            "victor/runtime/ must not use get_container(). "
            "Use ExecutionContext.container instead:\n"
            + "\n".join(f"  {f}:{line}" for f, line in calls)
        )

    def test_orchestrator_uses_instance_container(self):
        """Orchestrator must use self._container, not get_container()."""
        import inspect

        from victor.agent.orchestrator import AgentOrchestrator

        source = inspect.getsource(AgentOrchestrator)
        # Count get_container() calls in orchestrator source
        import ast as _ast

        tree = _ast.parse(source)
        get_container_calls = 0
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Call):
                func = node.func
                if isinstance(func, _ast.Name) and func.id == "get_container":
                    get_container_calls += 1
                elif isinstance(func, _ast.Attribute) and func.attr == "get_container":
                    get_container_calls += 1

        assert get_container_calls == 0, (
            f"AgentOrchestrator has {get_container_calls} get_container() call(s). "
            f"Use self._container or self._execution_context.container instead."
        )
