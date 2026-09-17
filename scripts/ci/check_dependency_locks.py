#!/usr/bin/env python3
"""Validate deployment locks against project metadata and shared constraints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


@dataclass(frozen=True)
class LockShape:
    path: Path
    extras: tuple[str, ...] = ()


LOCK_SHAPES = (
    LockShape(Path("requirements.txt")),
    LockShape(Path("requirements/api/requirements.txt"), ("api",)),
    LockShape(Path("requirements/embeddings-cpu/requirements.txt"), ("embeddings",)),
)


def _read_requirements(path: Path) -> dict[str, Requirement]:
    requirements: dict[str, Requirement] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        requirement = Requirement(line)
        requirements[canonicalize_name(requirement.name)] = requirement
    return requirements


def _pinned_version(requirement: Requirement) -> Version | None:
    pins = [specifier for specifier in requirement.specifier if specifier.operator == "=="]
    if len(pins) != 1 or "*" in pins[0].version:
        return None
    return Version(pins[0].version)


def _declared_requirements(
    pyproject: dict[str, object], extras: tuple[str, ...]
) -> list[Requirement]:
    project = pyproject["project"]
    assert isinstance(project, dict)
    raw_requirements = list(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {})
    assert isinstance(optional, dict)
    for extra in extras:
        raw_requirements.extend(optional[extra])
    return [Requirement(raw) for raw in raw_requirements]


def check_locks(root: Path) -> list[str]:
    """Return human-readable lock consistency failures."""

    constraints = _read_requirements(root / "constraints.txt")
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    failures: list[str] = []

    for shape in LOCK_SHAPES:
        lock = _read_requirements(root / shape.path)
        for name, requirement in lock.items():
            version = _pinned_version(requirement)
            if version is None:
                failures.append(f"{shape.path}: {requirement} is not exactly pinned")
                continue
            constraint = constraints.get(name)
            if constraint is not None and version not in constraint.specifier:
                failures.append(
                    f"{shape.path}: {name}=={version} violates constraint {constraint.specifier}"
                )

        for declared in _declared_requirements(pyproject, shape.extras):
            if declared.marker is not None and not declared.marker.evaluate(
                {"python_version": "3.12"}
            ):
                continue
            name = canonicalize_name(declared.name)
            locked = lock.get(name)
            if locked is None:
                failures.append(f"{shape.path}: missing declared dependency {name}")
                continue
            version = _pinned_version(locked)
            if version is not None and version not in declared.specifier:
                failures.append(
                    f"{shape.path}: {name}=={version} violates project requirement "
                    f"{declared.specifier}"
                )

    return failures


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    failures = check_locks(root)
    if failures:
        print("Dependency lock consistency check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Dependency lock consistency check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
