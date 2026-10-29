"""Guard tests for global state elimination (GS-2).

Tracks get_global_manager() call sites and ensures the count does not
increase. As sites are migrated to ExecutionContext, the expected count
should decrease.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VICTOR_ROOT = REPO_ROOT / "victor"


def _count_global_manager_calls(root: Path) -> list:
    """Count executable get_global_manager() calls (not in docstrings/comments)."""
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

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "get_global_manager":
                    calls.append((rel, node.lineno))
                elif isinstance(func, ast.Attribute) and func.attr == "get_global_manager":
                    calls.append((rel, node.lineno))
    return calls


class TestGlobalStateGuard:
    """Prevent new get_global_manager() call sites from being added."""

    def test_global_manager_call_count_does_not_increase(self):
        """Track executable get_global_manager() calls.

        Current known call sites (all in state/ factory internals):
        - victor/state/factory.py (internal implementation)

        This test fails if new calls are added outside state/.
        Migrate existing calls to ExecutionContext instead.
        """
        calls = _count_global_manager_calls(VICTOR_ROOT)
        # Allowed locations:
        # - victor/state/ — the factory implementation itself
        # - victor/runtime/context.py — transitional bridge in ExecutionContext.create()
        allowed_prefixes = ("victor/state/", "victor/runtime/")
        non_allowed_calls = [
            (f, line) for f, line in calls if not any(f.startswith(p) for p in allowed_prefixes)
        ]
        assert not non_allowed_calls, (
            f"Found {len(non_allowed_calls)} get_global_manager() call(s) outside allowed locations. "
            f"Use ExecutionContext instead:\n"
            + "\n".join(f"  {f}:{line}" for f, line in non_allowed_calls)
        )

    def test_orchestrator_has_execution_context(self):
        """Orchestrator must create _execution_context during init."""
        import inspect

        from victor.agent.orchestrator import AgentOrchestrator

        source = inspect.getsource(AgentOrchestrator)
        assert "_execution_context" in source, (
            "AgentOrchestrator must have _execution_context attribute. "
            "See victor/runtime/context.py."
        )
        assert (
            "_create_execution_context" in source
        ), "AgentOrchestrator must have _create_execution_context() method."

    def test_execution_context_importable_from_runtime(self):
        """ExecutionContext must be importable from victor.runtime."""
        from victor.runtime import ExecutionContext

        assert ExecutionContext is not None
