# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The native wheel version must ride the victor-ai release version.

0.11.0's release 400'd on PyPI because the native line stayed at an
already-published version while everything else bumped. These tests pin the
two mechanisms that prevent a recurrence: sync_version.py moves all four
native version spots together, and check_version_sync.py fails the release
when the native version drifts from the release version.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo_tree(tmp_path: Path) -> Path:
    """A minimal fake repo root with every native version spot at 0.8.2."""
    (tmp_path / "rust" / "crates" / "python-bindings").mkdir(parents=True)
    (tmp_path / "VERSION").write_text("0.8.2\n")
    (tmp_path / "pyproject.toml").write_text(
        'version = "0.8.2"\n\n'
        "[project.optional-dependencies]\n"
        'native = [\n    "victor-native>=0.8.2",\n]\n'
    )
    (tmp_path / "rust" / "pyproject.toml").write_text('[project]\nversion = "0.8.2"\n')
    (tmp_path / "rust" / "crates" / "python-bindings" / "Cargo.toml").write_text(
        '[package]\nname = "victor_native"\nversion = "0.8.2"\n'
    )
    (tmp_path / "rust" / "Cargo.lock").write_text(
        '[[package]]\nname = "victor_native"\nversion = "0.8.2"\n'
        'dependencies = []\n\n[[package]]\nname = "serde"\nversion = "1.0.210"\n'
    )
    return tmp_path


def test_sync_ai_moves_all_native_spots_together(repo_tree: Path):
    mod = _load("sync_version")
    # Release-prep bumps VERSION first; sync fans it out.
    (repo_tree / "VERSION").write_text("0.11.0\n")

    mod.sync_ai(repo_tree)

    version = "0.11.0"
    assert 'version = "0.11.0"' in (repo_tree / "rust" / "pyproject.toml").read_text()
    crate = (repo_tree / "rust" / "crates" / "python-bindings" / "Cargo.toml").read_text()
    assert 'name = "victor_native"' in crate and 'version = "0.11.0"' in crate
    lock = (repo_tree / "rust" / "Cargo.lock").read_text()
    assert 'name = "victor_native"\nversion = "0.11.0"' in lock
    # Unrelated locked package is untouched.
    assert 'name = "serde"\nversion = "1.0.210"' in lock
    root = (repo_tree / "pyproject.toml").read_text()
    assert '"victor-native>=0.11.0"' in root
    assert 'version = "0.11.0"' in root.split("[project.optional-dependencies]")[0]


def test_sync_is_idempotent(repo_tree: Path):
    mod = _load("sync_version")
    (repo_tree / "VERSION").write_text("0.11.0\n")

    mod.sync_ai(repo_tree)
    mod.sync_ai(repo_tree)

    errors = _load("check_version_sync").native_version_errors(repo_tree, "0.11.0")
    assert errors == []


def test_gate_rejects_native_version_drift_from_release(repo_tree: Path):
    mod = _load("check_version_sync")

    errors = mod.native_version_errors(repo_tree, "0.11.0")

    assert any("must equal the victor-ai" in err for err in errors)
    assert any("0.8.2" in err and "0.11.0" in err for err in errors)


def test_gate_passes_when_native_matches_release(repo_tree: Path):
    _load("sync_version").sync_native(repo_tree, "0.11.0")

    errors = _load("check_version_sync").native_version_errors(repo_tree, "0.11.0")

    assert errors == []


def test_gate_still_catches_partial_native_bumps(repo_tree: Path):
    """Internal-agreement checks survive the new release-version check."""
    mod = _load("check_version_sync")
    # Bump only rust/pyproject.toml, leave the crate manifest behind.
    (repo_tree / "rust" / "pyproject.toml").write_text('[project]\nversion = "0.9.0"\n')

    errors = mod.native_version_errors(repo_tree, "0.9.0")

    assert any("must agree" in err for err in errors)
