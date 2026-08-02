# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Changed-file -> mirror-test selection for the lightweight develop gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    module_path = Path(__file__).resolve().parents[3] / "scripts" / "ci" / "select_changed_tests.py"
    spec = importlib.util.spec_from_file_location("select_changed_tests", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


select = _load().select


def test_source_file_maps_to_mirror_tests():
    # victor/agent/fast_pruning.py -> its mirror tests (base + suffixed variants that exist).
    out = select(["victor/agent/fast_pruning.py"])
    assert "tests/unit/agent/test_fast_pruning.py" in out
    assert "tests/unit/agent/test_fast_pruning_reference.py" in out


def test_changed_test_file_runs_directly():
    out = select(["tests/unit/agent/test_metrics_runtime.py"])
    assert out == ["tests/unit/agent/test_metrics_runtime.py"]


def test_non_python_and_unmapped_yield_nothing():
    # Docs/CI/config changes map to no unit tests -> empty (caller treats as pass).
    assert select(["README.md", ".github/workflows/ci-fast.yml", "Makefile"]) == []


def test_source_without_mirror_test_is_skipped():
    # A source file whose mirror test doesn't exist contributes nothing (no crash).
    out = select(["victor/this_module_has_no_test_zzz.py"])
    assert out == []


def test_results_are_deduped_and_sorted():
    out = select(
        [
            "victor/framework/client.py",
            "victor/framework/client.py",  # duplicate input
            "tests/unit/framework/test_client.py",  # also a direct hit
        ]
    )
    assert out == sorted(set(out))
    assert "tests/unit/framework/test_client.py" in out
