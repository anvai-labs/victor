"""Tests for the shared PEP 562 lazy-import helper and the bridges built on it."""

from __future__ import annotations

import importlib
import json
import types

import pytest

from victor_contracts._lazy import make_lazy_getattr

# Every bridge module whose module-level ``__getattr__`` comes from the helper.
LAZY_BRIDGES = (
    "capability_runtime",
    "chain_runtime",
    "database_runtime",
    "embeddings_runtime",
    "enrichment_runtime",
    "feature_flag_runtime",
    "graph_rag_runtime",
    "indexing_runtime",
    "init_runtime",
    "lsp_runtime",
    "middleware_runtime",
    "processing_runtime",
    "provider_runtime",
    "rl_runtime",
    "safety_patterns",
    "safety_policy",
    "search_runtime",
    "workflow_runtime",
)

# Deprecated in 0.9.0, removed in 0.12.0 (CONTRACT_STABILITY.md).
REMOVED_BRIDGES = (
    "agent_spec_runtime",
    "graph_runtime",
    "handler_runtime",
    "subagent_runtime",
    "tool_runtime",
    "workflow_executor_runtime",
)


def _module_with(lazy_imports: dict[str, str]) -> types.ModuleType:
    module = types.ModuleType("victor_contracts.fake_bridge")
    module.__getattr__ = make_lazy_getattr(module.__name__, lazy_imports)  # type: ignore[method-assign]
    return module


def test_resolves_mapped_name_from_target_module() -> None:
    module = _module_with({"dumps": "json"})
    assert module.dumps is json.dumps


def test_unknown_name_raises_attribute_error_with_module_name() -> None:
    module = _module_with({"dumps": "json"})
    with pytest.raises(AttributeError) as excinfo:
        _ = module.loads
    assert str(excinfo.value) == "module 'victor_contracts.fake_bridge' has no attribute 'loads'"
    assert not hasattr(module, "loads")


def test_missing_target_module_propagates_import_error() -> None:
    module = _module_with({"thing": "victor_contracts_tests_no_such_module"})
    with pytest.raises(ModuleNotFoundError):
        _ = module.thing


def test_mapping_is_read_on_every_access_and_results_are_not_cached() -> None:
    lazy_imports = {"dumps": "json"}
    module = _module_with(lazy_imports)
    assert module.dumps is json.dumps
    assert "dumps" not in vars(module)

    lazy_imports["dumps"] = "pickle"
    import pickle

    assert module.dumps is pickle.dumps


def test_returned_callable_is_named_like_a_module_getattr() -> None:
    getter = make_lazy_getattr("victor_contracts.x", {})
    assert getter.__name__ == "__getattr__"
    assert getter.__qualname__ == "__getattr__"
    assert "victor_contracts.x" in (getter.__doc__ or "")


@pytest.mark.parametrize("bridge", LAZY_BRIDGES)
def test_bridges_use_the_shared_helper(bridge: str) -> None:
    module = importlib.import_module(f"victor_contracts.{bridge}")
    lazy_imports = module._LAZY_IMPORTS
    assert lazy_imports and all(
        isinstance(k, str) and isinstance(v, str) for k, v in lazy_imports.items()
    )
    getter = module.__getattr__
    assert getter.__qualname__ == "__getattr__"
    assert getter.__module__ == "victor_contracts._lazy"
    with pytest.raises(AttributeError) as excinfo:
        getter("definitely_not_exported")
    assert str(excinfo.value) == (
        f"module 'victor_contracts.{bridge}' has no attribute 'definitely_not_exported'"
    )


@pytest.mark.parametrize("bridge", REMOVED_BRIDGES)
def test_removed_bridges_are_not_importable(bridge: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"victor_contracts.{bridge}")
