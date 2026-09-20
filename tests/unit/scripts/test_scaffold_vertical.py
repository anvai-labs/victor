from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_scaffold_module():
    module_path = Path(__file__).resolve().parents[3] / "scripts" / "scaffold_vertical.py"
    spec = importlib.util.spec_from_file_location("scaffold_vertical", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scaffold_vertical = load_scaffold_module()


def test_scaffold_vertical_emits_plugin_only_entry_point(tmp_path: Path) -> None:
    """Scaffolded packages should follow the canonical plugin-first contract."""

    pkg_dir = scaffold_vertical.scaffold("security", tmp_path)

    pyproject = (pkg_dir / "pyproject.toml").read_text(encoding="utf-8")
    package_init = (pkg_dir / "victor_security" / "__init__.py").read_text(encoding="utf-8")
    assistant = (pkg_dir / "victor_security" / "assistant.py").read_text(encoding="utf-8")
    generated_test = (pkg_dir / "tests" / "test_security.py").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.12"' in pyproject
    assert '[project.entry-points."victor.plugins"]' in pyproject
    assert 'security = "victor_security:plugin"' in pyproject
    assert '[project.entry-points."victor.verticals"]' not in pyproject
    assert "victor-contracts>=0.7.0" in pyproject
    assert "victor_contracts" in package_init
    assert "victor_contracts" in assistant
    assert "victor_sdk" not in package_init
    assert "victor_sdk" not in assistant
    assert "class SecurityPlugin(VictorPlugin):" in package_init
    assert "context.register_vertical(SecurityAssistant)" in package_init
    assert "async def on_activate_async" in package_init
    assert "@register_vertical(" in assistant
    assert "def get_tool_requirements" in assistant
    assert "CapabilityRequirement(" in assistant
    assert "def get_stages" in assistant
    assert "def test_get_definition()" in generated_test
