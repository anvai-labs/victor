"""Unit tests for the keybindings loader (pure, no Textual)."""

from __future__ import annotations

import json
from pathlib import Path

from victor.ui.tui.keybindings import (
    DEFAULT_BINDINGS,
    default_override_path,
    load_keybindings,
)


def test_defaults_when_file_missing(tmp_path: Path) -> None:
    assert load_keybindings(tmp_path / "absent.json") == list(DEFAULT_BINDINGS)


def test_override_replaces_only_that_action(tmp_path: Path) -> None:
    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"toggle_sidebar": "f5"}), encoding="utf-8")
    bindings = load_keybindings(path)
    keys = {action: key for key, action, _ in bindings}
    assert keys["toggle_sidebar"] == "f5"
    assert keys["quit"] == "ctrl+q"  # untouched default


def test_invalid_json_falls_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "kb.json"
    path.write_text("{ not json", encoding="utf-8")
    assert load_keybindings(path) == list(DEFAULT_BINDINGS)


def test_non_object_json_falls_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    assert load_keybindings(path) == list(DEFAULT_BINDINGS)


def test_non_string_values_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"quit": 123, "help": "f9"}), encoding="utf-8")
    keys = {action: key for key, action, _ in load_keybindings(path)}
    assert keys["quit"] == "ctrl+q"  # invalid value ignored
    assert keys["help"] == "f9"


def test_default_override_path_points_at_dotvictor() -> None:
    path = default_override_path()
    assert path.name == "keybindings.json"
    assert path.parent.name == ".victor"


def test_copy_action_bound_to_ctrl_c_by_default(tmp_path: Path) -> None:
    keys = {action: key for key, action, _ in load_keybindings(tmp_path / "absent.json")}
    assert keys["copy"] == "ctrl+c"


def test_copy_binding_is_reboundable(tmp_path: Path) -> None:
    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"copy": "ctrl+y"}), encoding="utf-8")
    keys = {action: key for key, action, _ in load_keybindings(path)}
    assert keys["copy"] == "ctrl+y"
