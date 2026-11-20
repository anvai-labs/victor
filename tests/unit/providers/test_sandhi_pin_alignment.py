"""Current Sandhi pin declarations and runtime messaging stay aligned."""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_all_dependency_locks_use_the_same_published_sandhi_pin():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = next(
        requirement
        for requirement in project["project"]["dependencies"]
        if requirement.startswith("sandhi-gateway==")
    )
    pins = {
        requirement
        for requirement in project["project"]["dependencies"]
        if requirement.startswith("sandhi-gateway==")
    }

    for relative_path in (
        "requirements.txt",
        "requirements/api/requirements.txt",
        "requirements/embeddings-cpu/requirements.txt",
    ):
        locked = {
            line.strip()
            for line in (ROOT / relative_path).read_text(encoding="utf-8").splitlines()
            if line.startswith("sandhi-gateway==")
        }
        pins.update(locked)

    assert pins == {expected}


def test_canonical_gateway_qualification_tracks_current_pin():
    document = (ROOT / "docs/architecture/sandhi-subscription-gateway.md").read_text(
        encoding="utf-8"
    )
    assert (
        "Victor's consumer candidate pins the prepared Sandhi release\n" "`sandhi-gateway==0.23.0`."
    ) in document


def test_installed_binding_matches_source_pin():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = next(
        requirement.removeprefix("sandhi-gateway==")
        for requirement in project["project"]["dependencies"]
        if requirement.startswith("sandhi-gateway==")
    )
    assert importlib.metadata.version("sandhi-gateway") == expected


def test_runtime_installation_error_names_the_current_pin():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = next(
        requirement.removeprefix("sandhi-gateway==")
        for requirement in project["project"]["dependencies"]
        if requirement.startswith("sandhi-gateway==")
    )
    expected_messages = {
        "victor/providers/sandhi_transport.py": (
            f"sandhi-gateway {expected} is required for provider transport"
        ),
        "victor/providers/openai_compat_model_policy.py": (
            f"sandhi-gateway {expected} is required for provider policy"
        ),
    }
    for relative_path, message in expected_messages.items():
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert message in source, relative_path


def test_current_documentation_uses_the_source_pin():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = next(
        requirement
        for requirement in project["project"]["dependencies"]
        if requirement.startswith("sandhi-gateway==")
    )
    for relative_path in (
        "docs/index.md",
        "docs/tech-stack.md",
        "docs/roadmap.md",
        "docs/architecture.md",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert expected in text, relative_path
