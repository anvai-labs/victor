"""Guard test for singleton proliferation (GS-5).

Tracks the count of singleton patterns (_instance: Optional + get_instance)
across the codebase and prevents new ones from being added.

As singletons are converted to DI (via ExecutionContext or ServiceContainer),
the cap should be lowered.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VICTOR_ROOT = REPO_ROOT / "victor"


def _count_singleton_classes(root: Path) -> list:
    """Find classes with _instance: Optional pattern (singleton indicator)."""
    singletons = []
    for py_file in root.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        rel = str(py_file.relative_to(REPO_ROOT))
        try:
            source = py_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        # Simple text search for the pattern — faster than full AST
        if "_instance: Optional" in source or "_instance:Optional" in source:
            # Count occurrences in this file
            count = source.count("_instance: Optional") + source.count("_instance:Optional")
            singletons.append((rel, count))
    return singletons


class TestSingletonGuard:
    """Prevent new singletons from being added to the codebase."""

    def test_singleton_count_does_not_increase(self):
        """Track files with singleton pattern.

        Current audited count: 70 files with _instance: Optional.
        This cap prevents new singletons from being added.
        As singletons are migrated to DI, lower this number.
        """
        singletons = _count_singleton_classes(VICTOR_ROOT)
        # +1: ModelCapabilityLoader (tool_calling/capabilities.py);
        # -1: chain/persona registries share NamespacedRegistry's singleton.
        MAX_FILES = 70
        assert len(singletons) <= MAX_FILES, (
            f"Found {len(singletons)} files with singleton pattern "
            f"(cap is {MAX_FILES}). Use DI via ExecutionContext or "
            f"ServiceContainer instead of creating new singletons.\n"
            f"New singleton files:\n"
            + "\n".join(f"  {f} ({c} instances)" for f, c in sorted(singletons))
        )

    def test_new_runtime_module_has_no_singletons(self):
        """The new victor/runtime/ module should avoid singletons.

        CacheRegistry uses singleton for backward compat but new code
        should prefer DI.
        """
        runtime_root = VICTOR_ROOT / "runtime"
        if not runtime_root.exists():
            pytest.skip("victor/runtime/ not yet created")

        singletons = _count_singleton_classes(runtime_root)
        # Allow CacheRegistry (1 file) as transitional
        MAX_RUNTIME_SINGLETONS = 1
        assert len(singletons) <= MAX_RUNTIME_SINGLETONS, (
            f"victor/runtime/ has {len(singletons)} singleton files "
            f"(cap is {MAX_RUNTIME_SINGLETONS}). Prefer DI.\n"
            + "\n".join(f"  {f}" for f, _ in singletons)
        )

    def test_conftest_has_singleton_reset_fixtures(self):
        """Autouse fixtures must exist for major singletons to prevent test leakage."""
        # Autouse fixtures live in tests/unit/conftest.py (not tests/conftest.py)
        conftest = REPO_ROOT / "tests" / "unit" / "conftest.py"
        if not conftest.exists():
            conftest = REPO_ROOT / "tests" / "conftest.py"
        assert conftest.exists(), "conftest.py not found"

        source = conftest.read_text(encoding="utf-8")
        # Key singletons that MUST have reset fixtures
        required_resets = [
            "reset_service_container",
            "reset_global_state_manager",
            "reset_embedding_singleton",
        ]
        for reset_name in required_resets:
            assert reset_name in source, (
                f"conftest.py missing autouse fixture '{reset_name}'. "
                f"Singletons leak state between tests without reset fixtures."
            )
