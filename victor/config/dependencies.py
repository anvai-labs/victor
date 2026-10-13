"""Offline checks for installed packages drifting from Victor's core requirements."""

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
import shlex
import sys
import tomllib

from packaging.requirements import Requirement


@dataclass(frozen=True)
class DependencyIssue:
    message: str
    suggestion: str


def _requirements() -> tuple[list[str], Path | None]:
    # An editable install's dist-info can outlive a git pull. Read the project
    # owning this module, never the user's current working directory.
    root = Path(__file__).resolve().parents[2]
    manifest = root / "pyproject.toml"
    if manifest.is_file():
        project = tomllib.loads(manifest.read_text(encoding="utf-8")).get("project", {})
        if project.get("name") == "victor-ai":
            return project.get("dependencies", []), root
    return list(metadata.requires("victor-ai") or []), None


def check_dependencies() -> list[DependencyIssue]:
    """Report missing/incompatible core requirements without network or mutation.

    Optional extras are not inferred from installed packages: their absence is
    valid in a bare install. This checks versions, not wheel ABI compatibility.
    """
    requirements, source = _requirements()
    if source is not None:
        command = shlex.join(
            ["uv", "pip", "install", "--python", sys.executable, "-e", str(source)]
        )
    elif (Path(sys.prefix) / "uv-receipt.toml").is_file():
        command = "uv tool upgrade victor-ai"
    else:
        command = shlex.join(
            ["uv", "pip", "install", "--python", sys.executable, "--upgrade", "victor-ai"]
        )
    issues = []
    for raw in requirements:
        required = Requirement(raw)
        if required.marker is not None and not required.marker.evaluate({"extra": ""}):
            continue
        try:
            installed = metadata.version(required.name)
        except metadata.PackageNotFoundError:
            installed = None
        if installed is None or not required.specifier.contains(installed, prereleases=True):
            actual = f"installed {installed}" if installed is not None else "not installed"
            issues.append(
                DependencyIssue(
                    f"{required.name}{required.specifier} required; {actual}",
                    f"Run: {command}",
                )
            )
    return issues
