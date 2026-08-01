from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


_MODULES = [
    "victor_dataanalysis/assistant.py",
    "victor_dataanalysis/plugin.py",
    "victor_dataanalysis/protocols.py",
    "victor_dataanalysis/prompts.py",
    "victor_dataanalysis/safety.py",
    "victor_dataanalysis/safety_enhanced.py",
]

_BANNED_IMPORTS = (
    "victor.core.verticals.protocols",
    "victor.core.verticals.registration",
    "victor.core.verticals.base",
)


def test_contract_boundary_modules_avoid_core_vertical_protocol_imports() -> None:
    for module in _MODULES:
        source = (_REPO_ROOT / module).read_text(encoding="utf-8")
        for banned in _BANNED_IMPORTS:
            assert banned not in source, f"{module} still imports {banned}"


_CONTRACT_MODULES = [
    "victor_dataanalysis/assistant.py",
    "victor_dataanalysis/plugin.py",
    "victor_dataanalysis/protocols.py",
    "victor_dataanalysis/prompts.py",
    "victor_dataanalysis/safety.py",
]

_LEGACY_CONTRACT_IMPORTS = (
    "from victor" "_sdk import",
    "from victor" "_sdk.verticals import",
    "from victor" "_sdk.verticals.protocols import",
)


def test_public_contract_modules_avoid_legacy_import_namespace() -> None:
    for module in _CONTRACT_MODULES:
        source = (_REPO_ROOT / module).read_text(encoding="utf-8")
        for banned in _LEGACY_CONTRACT_IMPORTS:
            assert banned not in source, f"{module} still imports {banned}"


def test_plugin_module_avoids_runtime_imports() -> None:
    """The plugin entry point must stay importable with only victor-contracts."""
    source = (_REPO_ROOT / "victor_dataanalysis/plugin.py").read_text(encoding="utf-8")

    assert "from victor." not in source
    assert "import victor." not in source
