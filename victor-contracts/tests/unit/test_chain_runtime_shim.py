"""victor_contracts.chain_runtime: inert, host-free deprecated shim (0.14.0, FEP-0042).

Published victor-coding 0.13.0 registers chains at import through this module,
so it must keep importing and accepting registrations without the host.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

DEPRECATION = (
    r"chain_runtime is an inert compatibility shim since victor-contracts 0\.14\.0.*0\.15\.0"
)


@pytest.mark.parametrize("name", ["ChainRegistry", "get_chain_registry"])
def test_access_warns_once_at_caller(name: str) -> None:
    chain_runtime = importlib.import_module("victor_contracts.chain_runtime")
    with pytest.warns(DeprecationWarning, match=DEPRECATION) as record:
        getattr(chain_runtime, name)
    assert len(record) == 1
    assert record[0].filename == __file__


def test_from_import_warns_at_caller() -> None:
    with pytest.warns(DeprecationWarning, match=DEPRECATION) as record:
        from victor_contracts.chain_runtime import get_chain_registry  # noqa: F401
    assert record[0].filename == __file__


def test_module_import_alone_does_not_warn() -> None:
    sys.modules.pop("victor_contracts.chain_runtime", None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        importlib.import_module("victor_contracts.chain_runtime")


def test_published_vertical_registration_pattern_is_accepted_and_discarded() -> None:
    with pytest.warns(DeprecationWarning, match=DEPRECATION):
        from victor_contracts.chain_runtime import ChainRegistry, get_chain_registry

    registry = get_chain_registry()
    assert isinstance(registry, ChainRegistry)
    assert get_chain_registry() is registry
    result = registry.register_chain(
        name="explore_file",
        version="0.5.0",
        chain=object(),
        category="exploration",
        description="Read and analyze a file",
        tags=["coding", "exploration"],
        author="victor-coding",
    )
    assert result is None
    registry.register_chain("positional", "1.0.0", object())
    assert registry.get_chain("explore_file") is None
    assert registry.get_chain("explore_file", version="0.5.0") is None
    assert registry.get_chain_metadata("explore_file") is None
    assert registry.list_chains() == []
    assert registry.list_chain_versions("explore_file") == []
    assert registry.get_registry_stats() == {
        "total_chains": 0,
        "total_versions": 0,
        "category_counts": {},
    }


def test_unknown_name_is_attribute_error() -> None:
    chain_runtime = importlib.import_module("victor_contracts.chain_runtime")
    with pytest.raises(AttributeError, match="has no attribute 'not_a_name'"):
        _ = chain_runtime.not_a_name


def test_shim_never_imports_the_host() -> None:
    import victor_contracts.chain_runtime as chain_runtime

    tree = ast.parse(Path(chain_runtime.__file__).read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name == "victor" or name.startswith("victor.") for name in imported)
    assert "importlib" not in imported


def test_works_without_the_host_installed() -> None:
    """Block every ``victor`` / ``victor.*`` import and replay the published pattern."""
    script = textwrap.dedent("""
        import importlib.abc, sys, warnings

        class BlockHost(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "victor" or name.startswith("victor."):
                    raise ModuleNotFoundError(name)
                return None

        sys.meta_path.insert(0, BlockHost())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from victor_contracts.chain_runtime import get_chain_registry
            get_chain_registry().register_chain(
                name="n", version="1.0.0", chain=object(), category="c",
                description="d", tags=["t"], author="a",
            )
        assert any(issubclass(w.category, DeprecationWarning) for w in caught), caught
        assert not any(m == "victor" or m.startswith("victor.") for m in sys.modules)
        """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
