#!/usr/bin/env python3
"""Resolve publication destinations and checksums before any release side effect."""

from __future__ import annotations

import argparse
import ast
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import tomllib
import zipfile


def release_plan(
    event: str, ref: str, test_only: bool, version: str, *, tag_prefix: str = "v"
) -> dict[str, str]:
    """Admit a matching version tag or an explicitly isolated TestPyPI rehearsal."""
    if (
        event not in {"push", "workflow_dispatch"}
        or type(test_only) is not bool
        or not re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", version)
    ):
        raise ValueError("invalid release event, test-only flag, or version")
    rehearsal = event == "workflow_dispatch" and test_only
    if tag_prefix not in {"v", "victor-codegraph-v"}:
        raise ValueError("unsupported release tag prefix")
    if not rehearsal and ref != f"refs/tags/{tag_prefix}{version}":
        raise ValueError("publication requires a tag matching the package version")
    prerelease = bool(re.search(r"(?:a|b|rc)\d+$", version))
    return {
        "version": version,
        "production": str(not rehearsal and not prerelease).lower(),
        "github_release": str(not rehearsal).lower(),
        "testpypi": str(rehearsal or prerelease).lower(),
        "prerelease": str(prerelease).lower(),
    }


def codegraph_requirement(root: Path) -> str:
    """Require one exact graph-extra pin matching the independent package/runtime version."""
    package = root / "victor-codegraph"
    project = tomllib.loads((package / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if project["name"] != "victor-codegraph" or not re.fullmatch(
        r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", version
    ):
        raise ValueError("invalid CodeGraph package identity")
    module = ast.parse((package / "victor_codegraph/__init__.py").read_text())
    runtime_versions = [
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        )
    ]
    if runtime_versions != [version]:
        raise ValueError("CodeGraph runtime version and package metadata differ")
    consumer = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    matches = [
        requirement
        for requirement in consumer.get("optional-dependencies", {}).get("graph", [])
        if re.match(r"(?i)^victor[-_.]codegraph\b", requirement)
    ]
    expected = f"victor-codegraph[treesitter]=={version}"
    if matches != [expected]:
        raise ValueError(f"Victor graph extra must contain the exact pin {expected}")
    return expected


def validate_package_artifacts(
    artifacts: Path, name: str, version: str, *, codegraph_pin: str | None = None
) -> None:
    """Validate the exact wheel/sdist inventory and their embedded distribution metadata."""
    normalized = name.replace("-", "_")
    expected = {f"{normalized}-{version}-py3-none-any.whl", f"{normalized}-{version}.tar.gz"}
    files = list(artifacts.iterdir())
    if {path.name for path in files} != expected or any(
        path.is_symlink() or not path.is_file() for path in files
    ):
        raise ValueError("release requires exactly the expected wheel and sdist artifacts")
    for path in files:
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as wheel:
                key = f"{normalized}-{version}.dist-info/METADATA"
                if [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")] != [key]:
                    raise ValueError("invalid wheel metadata inventory")
                data = wheel.read(key)
        else:
            with tarfile.open(path, "r:gz") as sdist:
                key = f"{normalized}-{version}/PKG-INFO"
                members = [m for m in sdist.getmembers() if m.name == key]
                if len(members) != 1 or not members[0].isfile():
                    raise ValueError("invalid sdist metadata inventory")
                source = sdist.extractfile(members[0])
                if source is None:
                    raise ValueError("missing sdist metadata")
                with source:
                    data = source.read()
        metadata = BytesParser().parsebytes(data)
        if metadata.get_all("Name") != [name] or metadata.get_all("Version") != [version]:
            raise ValueError("artifact metadata differs from the declared package/version")
        if codegraph_pin is not None:
            pins = [
                re.sub(r"\s+", "", requirement).replace("'", '"')
                for requirement in metadata.get_all("Requires-Dist", [])
                if re.match(r"(?i)^victor[-_.]codegraph\b", requirement)
            ]
            if pins != [f'{codegraph_pin};extra=="graph"']:
                raise ValueError(
                    "artifact metadata must retain the exact CodeGraph graph-extra pin"
                )


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
    plan.add_argument("--package", choices=["victor-ai", "victor-codegraph"], default="victor-ai")
    codegraph = commands.add_parser("codegraph-requirement")
    codegraph.add_argument("--root", type=Path, default=Path("."))
    artifacts = commands.add_parser("package-artifacts")
    artifacts.add_argument("artifacts", type=Path)
    artifacts.add_argument("--root", type=Path, default=Path("."))
    artifacts.add_argument(
        "--package", choices=["victor-ai", "victor-codegraph"], default="victor-codegraph"
    )
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
    elif args.command == "codegraph-requirement":
        print(codegraph_requirement(args.root))
    elif args.command == "package-artifacts":
        requirement = codegraph_requirement(args.root)
        if args.package == "victor-ai":
            version = (args.root / "VERSION").read_text().strip()
            project = tomllib.loads((args.root / "pyproject.toml").read_text())
            if project["project"]["version"] != version:
                raise ValueError("VERSION and package metadata differ")
            validate_package_artifacts(
                args.artifacts, "victor-ai", version, codegraph_pin=requirement
            )
        else:
            validate_package_artifacts(
                args.artifacts, "victor-codegraph", requirement.split("==")[1]
            )
    else:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        test_only = event.get("inputs", {}).get("publish_testpypi", False)
        # The dispatch webhook serializes its Boolean input as a string.
        if test_only in ("true", "false"):
            test_only = test_only == "true"
        prefix = "v"
        if args.package == "victor-codegraph":
            version = codegraph_requirement(args.root).split("==")[1]
            prefix = "victor-codegraph-v"
        else:
            version = (args.root / "VERSION").read_text().strip()
            project = tomllib.loads((args.root / "pyproject.toml").read_text())
            if project["project"]["version"] != version:
                raise ValueError("VERSION and package metadata differ")
        outputs = release_plan(
            os.environ["GITHUB_EVENT_NAME"],
            os.environ["GITHUB_REF"],
            test_only,
            version,
            tag_prefix=prefix,
        )
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as destination:
            destination.writelines(f"{key}={value}\n" for key, value in outputs.items())


if __name__ == "__main__":
    main()
