#!/usr/bin/env python3
"""Resolve publication destinations and checksums before any release side effect."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tomllib


def release_plan(event: str, ref: str, test_only: bool, version: str) -> dict[str, str]:
    """Admit a matching version tag or an explicitly isolated TestPyPI rehearsal."""
    if (
        event not in {"push", "workflow_dispatch"}
        or type(test_only) is not bool
        or not re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", version)
    ):
        raise ValueError("invalid release event, test-only flag, or version")
    rehearsal = event == "workflow_dispatch" and test_only
    if not rehearsal and ref != f"refs/tags/v{version}":
        raise ValueError("publication requires a tag matching the package version")
    prerelease = bool(re.search(r"(?:a|b|rc)\d+$", version))
    return {
        "version": version,
        "production": str(not rehearsal and not prerelease).lower(),
        "github_release": str(not rehearsal).lower(),
        "testpypi": str(rehearsal or prerelease).lower(),
        "prerelease": str(prerelease).lower(),
    }


def ruff_requirement(project: Path) -> str:
    """Read the sole CI Ruff pin without installing the complete CI environment."""
    data = tomllib.loads(project.read_text())
    requirements = data["project"]["optional-dependencies"]["ci"]
    matches: list[str] = [
        value for value in requirements if isinstance(value, str) and re.match(r"^ruff\b", value)
    ]
    if len(matches) != 1 or not re.fullmatch(r"ruff==\d+\.\d+\.\d+", matches[0]):
        raise ValueError("project CI requirements must contain exactly one pinned Ruff version")
    return matches[0]


def write_checksums(artifacts: Path, output: Path) -> None:
    """Hash every downloaded release artifact under its flat public asset name."""
    files = sorted(path for path in artifacts.rglob("*") if path.is_file())
    if not files:
        raise ValueError("release artifact inventory is empty")
    groups = {path.relative_to(artifacts).parts[0] for path in files}
    if (
        not {"python-package", "vscode-extension", "sbom"} <= groups
        or not any(group.startswith("native-wheels-") for group in groups)
        or not any(group.startswith("binary-") for group in groups)
    ):
        raise ValueError("release inventory is missing a required artifact category")
    names: set[str] = set()
    records = []
    for path in files:
        if path.is_symlink() or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", path.name):
            raise ValueError("release artifact must be a regular file with a safe asset name")
        if path.name in names:
            raise ValueError(f"duplicate release asset name: {path.name}")
        names.add(path.name)
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        records.append(f"{digest}  {path.name}\n")
    output.write_text("".join(records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--root", type=Path, default=Path("."))
    ruff = commands.add_parser("ruff")
    ruff.add_argument("--project", type=Path, default=Path("pyproject.toml"))
    checksums = commands.add_parser("checksums")
    checksums.add_argument("artifacts", type=Path)
    checksums.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "ruff":
        print(ruff_requirement(args.project))
    elif args.command == "checksums":
        write_checksums(args.artifacts, args.output)
    else:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        test_only = event.get("inputs", {}).get("publish_testpypi", False)
        # The dispatch webhook serializes its Boolean input as a string.
        if test_only in ("true", "false"):
            test_only = test_only == "true"
        version = (args.root / "VERSION").read_text().strip()
        project = tomllib.loads((args.root / "pyproject.toml").read_text())
        if project["project"]["version"] != version:
            raise ValueError("VERSION and package metadata differ")
        outputs = release_plan(
            os.environ["GITHUB_EVENT_NAME"], os.environ["GITHUB_REF"], test_only, version
        )
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as destination:
            destination.writelines(f"{key}={value}\n" for key, value in outputs.items())


if __name__ == "__main__":
    main()
