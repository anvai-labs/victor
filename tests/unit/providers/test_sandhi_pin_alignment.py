"""Sandhi floor-pin declarations and runtime messaging stay aligned.

The source declares a FLOOR (`sandhi-gateway>=0.26.0`), not an exact pin:
sandhi ships maintenance/minor releases on its own train and the consumer
picks them up without a Victor edit. Every declaration site (pyproject, the
requirement locks, runtime error text, docs) must state the same floor, and
the installed binding must satisfy it.
"""

from __future__ import annotations

import importlib.metadata
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

PIN_PREFIX = "sandhi-gateway>="


def _declared_floor() -> str:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return next(
        requirement
        for requirement in project["project"]["dependencies"]
        if requirement.startswith(PIN_PREFIX)
    )


def test_all_dependency_locks_use_the_same_published_sandhi_floor():
    expected = _declared_floor()
    floors = {expected}
    for relative_path in (
        "requirements.txt",
        "requirements/api/requirements.txt",
        "requirements/embeddings-cpu/requirements.txt",
    ):
        locked = {
            line.strip()
            for line in (ROOT / relative_path).read_text(encoding="utf-8").splitlines()
            if line.startswith(PIN_PREFIX)
        }
        floors.update(locked)
    # No exact pin may coexist with the floor anywhere.
    for relative_path in (
        "pyproject.toml",
        "requirements.txt",
        "requirements/api/requirements.txt",
        "requirements/embeddings-cpu/requirements.txt",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "sandhi-gateway==" not in text, relative_path
    assert floors == {expected}


def test_canonical_gateway_qualification_tracks_current_floor():
    document = (ROOT / "docs/architecture/sandhi-subscription-gateway.md").read_text(
        encoding="utf-8"
    )
    assert (
        "Victor 0.33.2 source requires the published `sandhi-gateway>=0.26.0` wheel."
    ) in document


def test_installed_binding_satisfies_source_floor():
    floor = _declared_floor().removeprefix(PIN_PREFIX)
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", floor)
    assert match is not None, floor
    installed = importlib.metadata.version("sandhi-gateway")
    installed_parts = tuple(int(part) for part in installed.split(".")[:3])
    assert installed_parts >= tuple(
        int(part) for part in match.groups()
    ), f"installed sandhi-gateway {installed} does not satisfy the declared floor {floor}"


def test_runtime_installation_error_names_the_current_floor():
    floor = _declared_floor().removeprefix(PIN_PREFIX)
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
