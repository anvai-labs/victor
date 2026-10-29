# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Import smoke for import-only source changes (ci-fast)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load():
    module_path = Path(__file__).resolve().parents[3] / "scripts" / "ci" / "import_smoke.py"
    spec = importlib.util.spec_from_file_location("import_smoke", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load().smoke


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    root = tmp_path / "smokepkg"
    root.mkdir()
    (root / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield root
    for name in [m for m in sys.modules if m.startswith("smokepkg")]:
        del sys.modules[name]


def _write(pkg, name, src):
    (pkg / f"{name}.py").write_text(src)
    return f"smokepkg/{name}.py"


def test_importable_modules_pass(pkg):
    path = _write(pkg, "ok", "import os\nX = os.sep\n")
    assert smoke([path], read_base=lambda p: None, root=pkg.parent) == ([], [])


def test_new_failure_is_reported(pkg):
    # the change removed an import the module still uses
    path = _write(pkg, "broke", "X = os.sep\n")
    base = "import os\nX = os.sep\n"
    new, pre = smoke([path], read_base=lambda p: base, root=pkg.parent)
    assert [m for m, _ in new] == ["smokepkg.broke"] and pre == []
    assert "NameError" in new[0][1]


def test_failure_present_at_base_is_pre_existing(pkg):
    src = "import enum\nclass A(enum.Enum):\n    B = 1\nY = A.READ\n"
    path = _write(pkg, "legacy", src)
    base = "import enum, sys\nclass A(enum.Enum):\n    B = 1\nY = A.READ\n"
    new, pre = smoke([path], read_base=lambda p: base, root=pkg.parent)
    assert new == [] and [m for m, _ in pre] == ["smokepkg.legacy"]


def test_different_failure_than_base_is_new(pkg):
    path = _write(pkg, "worse", "X = os.sep\nY = missing\n")
    base = "import os\nX = os.sep\nY = missing\n"  # base fails too, but on NameError 'missing'
    new, pre = smoke([path], read_base=lambda p: base, root=pkg.parent)
    assert [m for m, _ in new] == ["smokepkg.worse"]


def test_package_init_paths_map_to_the_package(pkg):
    sub = pkg / "sub"
    sub.mkdir()
    (sub / "__init__.py").write_text("VALUE = 1\n")
    assert smoke(["smokepkg/sub/__init__.py"], read_base=lambda p: None, root=pkg.parent) == (
        [],
        [],
    )
