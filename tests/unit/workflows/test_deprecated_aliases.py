"""make_deprecated_alias_getattr and the five TeamNode* shim modules."""

from __future__ import annotations

import importlib
import warnings
from typing import Any

import pytest

from victor.workflows.deprecated_aliases import make_deprecated_alias_getattr

# Byte-for-byte warning text the TeamNode* shims published before the shared factory.
_EXPECTED_TEMPLATE = (
    "{alias} is deprecated; use {target} instead. "
    "This compatibility alias remains supported through "
    "v0.9.0 (2027-03-31) and "
    "will be removed after that milestone. "
    "See docs/architecture/migration.md for migration guidance."
)

_SHIMS = [
    ("victor.framework.workflows.nodes", "TeamNode", "TeamStep"),
    ("victor.framework.workflows.nodes", "TeamNodeConfig", "TeamStepConfig"),
    ("victor.workflows", "TeamNodeWorkflow", "TeamStepWorkflow"),
    ("victor.workflows.definition", "TeamNodeWorkflow", "TeamStepWorkflow"),
    ("victor.workflows.executors", "TeamNodeExecutor", "TeamStepExecutor"),
    ("victor.workflows.executors.team", "TeamNodeExecutor", "TeamStepExecutor"),
]


class _Target:
    pass


def _hooks(module_globals: dict[str, Any] | None = None) -> tuple[Any, Any, dict[str, Any]]:
    module_globals = {"existing": 1} if module_globals is None else module_globals
    getattr_hook, dir_hook = make_deprecated_alias_getattr(
        "pkg.mod",
        module_globals,
        {"OldName": _Target},
        removal_version="v9",
        removal_date="2099-01-01",
        migration_guide="docs/guide.md",
    )
    return getattr_hook, dir_hook, module_globals


def test_alias_returns_target_with_contract_warning() -> None:
    getattr_hook, _, _ = _hooks()
    with pytest.warns(DeprecationWarning) as record:
        assert getattr_hook("OldName") is _Target
    assert len(record) == 1
    assert str(record[0].message) == (
        "OldName is deprecated; use _Target instead. "
        "This compatibility alias remains supported through v9 (2099-01-01) and "
        "will be removed after that milestone. See docs/guide.md for migration guidance."
    )
    # stacklevel=2 attributes the warning to the accessing code (this file).
    assert record[0].filename == __file__


def test_unknown_name_raises_plain_attribute_error_without_warning() -> None:
    getattr_hook, _, _ = _hooks()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(AttributeError) as excinfo:
            getattr_hook("Missing")
    assert str(excinfo.value) == "module 'pkg.mod' has no attribute 'Missing'"


def test_dir_lists_live_globals_and_aliases_sorted() -> None:
    _, dir_hook, module_globals = _hooks()
    assert dir_hook() == ["OldName", "existing"]
    module_globals["added_later"] = 2
    assert dir_hook() == ["OldName", "added_later", "existing"]


@pytest.mark.parametrize("module_name,alias,target", _SHIMS)
def test_shim_modules_keep_byte_identical_warnings(
    module_name: str, alias: str, target: str
) -> None:
    module = importlib.import_module(module_name)
    with pytest.warns(DeprecationWarning) as record:
        resolved = getattr(module, alias)
    assert resolved is getattr(module, target)
    assert str(record[0].message) == _EXPECTED_TEMPLATE.format(alias=alias, target=target)
    assert alias in dir(module) and target in dir(module)


@pytest.mark.parametrize("module_name", sorted({m for m, _, _ in _SHIMS}))
def test_shim_modules_reject_unknown_attributes(module_name: str) -> None:
    module = importlib.import_module(module_name)
    with pytest.raises(
        AttributeError, match=rf"^module '{module_name}' has no attribute 'NoSuch'$"
    ):
        _ = module.NoSuch
