"""Sandhi floor-pin declarations, locks, and runtime messaging stay aligned.

The source declares a FLOOR in ``pyproject.toml`` (``sandhi-gateway>=0.26.0``):
sandhi ships maintenance/minor releases on its own train and the consumer
picks them up without a Victor edit. Deployment locks stay EXACT (the
``check_dependency_locks`` guard requires it): each lock pins one resolved
version that satisfies the declared floor. Runtime error text names the
floor, docs reference the floor, and the installed binding must satisfy it.
"""

from __future__ import annotations

import importlib.metadata
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

FLOOR_PREFIX = "sandhi-gateway>="
LOCK_PREFIX = "sandhi-gateway=="

LOCK_FILES = (
    "requirements.txt",
    "requirements/api/requirements.txt",
    "requirements/embeddings-cpu/requirements.txt",
)


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _declared_floor() -> str:
    return next(
        requirement
        for requirement in _pyproject()["project"]["dependencies"]
        if requirement.startswith(FLOOR_PREFIX)
    )


def _declared_floor_version() -> tuple[int, int, int]:
    match = re.fullmatch(r"sandhi-gateway>=(\d+)\.(\d+)\.(\d+)", _declared_floor())
    assert match is not None, _declared_floor()
    return tuple(int(part) for part in match.groups())


def test_pyproject_declares_a_floor_without_an_exact_pin():
    dependencies = _pyproject()["project"]["dependencies"]
    floors = [d for d in dependencies if d.startswith(FLOOR_PREFIX)]
    exact = [d for d in dependencies if d.startswith(LOCK_PREFIX)]
    assert len(floors) == 1, floors
    assert not exact, "pyproject must not exact-pin sandhi-gateway"


def test_all_dependency_locks_pin_one_version_satisfying_the_floor():
    floor = _declared_floor_version()
    pins = set()
    for relative_path in LOCK_FILES:
        locked = [
            line.strip()
            for line in (ROOT / relative_path).read_text(encoding="utf-8").splitlines()
            if line.startswith(LOCK_PREFIX)
        ]
        assert len(locked) == 1, (relative_path, locked)
        pin = locked[0]
        assert ">=" not in pin, f"{relative_path}: locks stay exact, got {pin}"
        pins.add(pin)
        version = tuple(int(part) for part in pin.removeprefix(LOCK_PREFIX).split("."))
        assert version >= floor, (relative_path, pin, _declared_floor())
    assert len(pins) == 1, pins


def test_canonical_gateway_qualification_tracks_current_floor():
    document = (ROOT / "docs/architecture/sandhi-subscription-gateway.md").read_text(
        encoding="utf-8"
    )
    assert (
        "Victor 0.35.0 source requires the published `sandhi-gateway>=0.26.0` wheel."
    ) in document


def test_installed_binding_satisfies_source_floor():
    floor = _declared_floor_version()
    installed = importlib.metadata.version("sandhi-gateway")
    installed_parts = tuple(int(part) for part in installed.split(".")[:3])
    assert installed_parts >= floor, (
        f"installed sandhi-gateway {installed} does not satisfy the declared floor "
        f"{_declared_floor()}"
    )


def test_runtime_installation_error_names_the_current_floor():
    floor = _declared_floor().removeprefix(FLOOR_PREFIX)
    expected_messages = {
        "victor/providers/sandhi_transport.py": (
            f"sandhi-gateway >= {floor} is required for provider transport"
        ),
        "victor/providers/openai_compat_model_policy.py": (
            f"sandhi-gateway >= {floor} is required for provider policy"
        ),
    }
    for relative_path, message in expected_messages.items():
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert message in source, relative_path


def test_current_documentation_uses_the_source_floor():
    expected = _declared_floor()
    for relative_path in (
        "docs/index.md",
        "docs/tech-stack.md",
        "docs/roadmap.md",
        "docs/architecture.md",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert expected in text, relative_path
