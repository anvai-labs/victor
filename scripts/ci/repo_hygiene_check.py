#!/usr/bin/env python3
"""Fail fast on foundational repo drift in first-party operational surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import importlib
import re
import sys

import yaml

BANNED_REPO_URLS = ("https://github.com/vijay-singh/codingagent",)

REPO_URL_SCAN_GLOBS = (
    "README.md",
    ".github/workflows/*.yml",
    "victor/**/victor-vertical.toml",
)

MARKDOWN_SCAN_GLOBS = (
    "README.md",
    "docs/**/*.md",
    "feps/**/*.md",
)

PRIMARY_VERTICAL_CONTRACT_FILES = (
    Path("pyproject.toml"),
    Path("scripts/scaffold_vertical.py"),
    Path("docs/guides/vertical-quickstart.md"),
    Path("docs/guides/FRAMEWORK_CAPABILITIES.md"),
    Path("docs/guides/EDGE_MODEL.md"),
    Path("docs/development/extending/verticals.md"),
    Path("docs/reference/verticals/index.md"),
    Path("docs/verticals/api_reference.md"),
    Path("docs/verticals/best_practices.md"),
    Path("docs/verticals/migration_guide.md"),
    Path("docs/demos/10-minute-vertical-demo.md"),
    Path("victor-contracts/README.md"),
    Path("victor-contracts/VERTICAL_DEVELOPMENT.md"),
    Path("victor-contracts/examples/minimal_vertical/README.md"),
    Path("victor-contracts/examples/minimal_vertical/pyproject.toml"),
    Path("examples/external_vertical/README.md"),
    Path("examples/external_vertical/pyproject.toml"),
)

STALE_VERTICAL_CONTRACT_PATTERNS = {
    '[project.entry-points."victor.verticals"]': (
        "primary vertical contract docs/examples must use victor.plugins, not victor.verticals"
    ),
    '[project.entry-points."victor.plugins.': (
        "primary vertical contract docs/examples must use the canonical victor.plugins"
        " group, not nested victor.plugins.* groups"
    ),
    '[project.entry-points."victor.sdk.protocols"]': (
        "primary vertical contract docs/examples must use victor.extension.protocols,"
        " not the legacy victor.sdk.protocols group"
    ),
    '[project.entry-points."victor.sdk.capabilities"]': (
        "primary vertical contract docs/examples must use victor.extension.capabilities,"
        " not the legacy victor.sdk.capabilities group"
    ),
    '[project.entry-points."victor.sdk.validators"]': (
        "primary vertical contract docs/examples must use victor.extension.validators,"
        " not the legacy victor.sdk.validators group"
    ),
    "SDK"
    "-first": (
        "primary vertical contract docs/examples must use contract-first wording,"
        " not legacy wording"
    ),
    "SDK"
    "-only": (
        "primary vertical contract docs/examples must use contract-only wording,"
        " not legacy wording"
    ),
    "from victor"
    "_sdk import": (
        "primary vertical contract docs/examples must use the victor_contracts import namespace,"
        " not the removed legacy namespace"
    ),
    "from victor"
    "_sdk.": (
        "primary vertical contract docs/examples must use the victor_contracts import namespace,"
        " not the removed legacy namespace"
    ),
    "`victor"
    "_sdk.": (
        "primary vertical contract docs/examples must point authors at victor_contracts,"
        " not the removed legacy namespace"
    ),
    "Must inherit from victor.core.verticals.VerticalBase": (
        "primary vertical contract docs/examples must point authors at victor_contracts.VerticalBase"
    ),
    "from victor.core.verticals import VerticalBase": (
        "primary vertical contract docs/examples must not import victor.core.verticals.VerticalBase"
    ),
    "from victor.verticals import VerticalBase": (
        "primary vertical contract docs/examples must point authors at victor_contracts.VerticalBase,"
        " not victor.verticals.VerticalBase"
    ),
    "from victor.framework.vertical_base import": (
        "primary vertical contract docs/examples must point authors at victor_contracts.VerticalBase,"
        " not victor.framework.vertical_base"
    ),
    "from victor.framework.extensions import VerticalBase": (
        "primary vertical contract docs/examples must point authors at victor_contracts.VerticalBase,"
        " not victor.framework.extensions.VerticalBase"
    ),
    "from victor.framework.extensions import StageDefinition": (
        "primary vertical contract docs/examples must point authors at victor_contracts.StageDefinition,"
        " not victor.framework.extensions.StageDefinition"
    ),
    "from victor.framework.extensions import VerticalConfig": (
        "primary vertical contract docs/examples must point authors at victor_contracts.VerticalConfig,"
        " not victor.framework.extensions.VerticalConfig"
    ),
    "from victor.core.verticals.registration import register_vertical": (
        "primary vertical contract docs/examples must import register_vertical from victor_contracts,"
        " not victor.core.verticals.registration"
    ),
    "victor.core.verticals.registration.register_vertical": (
        "primary vertical contract docs/examples must reference victor_contracts.register_vertical,"
        " not victor.core.verticals.registration.register_vertical"
    ),
    "from victor.verticals.base import StageDefinition": (
        "primary vertical contract docs/examples must point authors at victor_contracts.StageDefinition"
    ),
    "from victor.verticals.base import VerticalConfig": (
        "primary vertical contract docs/examples must point authors at victor_contracts.VerticalConfig"
    ),
    "VerticalRegistry.register(": (
        "primary vertical contract docs/examples must register external verticals through"
        " VictorPlugin/context.register_vertical(), not VerticalRegistry.register()"
    ),
    'get_entry_point("victor.verticals"': (
        "primary vertical contract docs/examples must use victor.plugins, not victor.verticals,"
        " for entry-point lookups"
    ),
    'get_entry_point_group("victor.verticals"': (
        "primary vertical contract docs/examples must use victor.plugins, not victor.verticals,"
        " for entry-point lookups"
    ),
}

ARCHIVED_DOC_BANNERS = {
    # Known archived planning docs that must advertise their non-canonical status.
    # docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md was removed during doc consolidation.
}

LEGACY_PATHS_THAT_MUST_STAY_REMOVED = {
    Path("victor/agent/protocols.py"): (
        "legacy monolithic protocol module reintroduced; keep victor/agent/protocols/ as the"
        " canonical source"
    ),
}

ROADMAP_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]*ROADMAP\.md(?:#[^)]+)?)\)")

# Canonical pointer docs: files whose whole job is to route readers to other
# docs. Every relative markdown/asset link inside them must resolve (TD-18).
# Root ARCHITECTURE.md / roadmap.md redirect stubs were removed during doc
# consolidation; docs/architecture.md and docs/roadmap.md are now the single sources.
CANONICAL_POINTER_DOCS = (
    Path("docs/roadmap.md"),
    Path("docs/architecture/adr/README.md"),
)

RELATIVE_DOC_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#\s]+\.(?:md|mmd|svg|png))(?:#[^)]*)?\)")
TARGET_HEADER_RE = re.compile(r"^[A-Za-z0-9_.-]+:\s*(?:#.*)?$")


@dataclass(frozen=True)
class HygieneFinding:
    """A single repo hygiene violation."""

    path: Path
    message: str


@dataclass(frozen=True)
class TextRequirement:
    """A required text fragment plus the error to report if it is missing."""

    needle: str
    missing_message: str


@dataclass(frozen=True)
class DeprecationContract:
    """Declarative deprecation contract for a public compatibility surface."""

    label: str
    runtime_files: tuple[Path, ...]
    activation_needles: tuple[str, ...]
    runtime_requirements: tuple[TextRequirement, ...]
    inventory_path: Path
    inventory_requirements: tuple[TextRequirement, ...]
    changelog_path: Path
    changelog_requirements: tuple[TextRequirement, ...]
    migration_path: Path
    migration_requirements: tuple[TextRequirement, ...]


TEAM_NODE_DEPRECATION_CONTRACT = DeprecationContract(
    label="TeamNode*",
    runtime_files=(
        Path("victor/framework/workflows/nodes.py"),
        Path("victor/workflows/definition.py"),
        Path("victor/workflows/__init__.py"),
        Path("victor/workflows/executors/__init__.py"),
        Path("victor/workflows/executors/team.py"),
    ),
    activation_needles=(
        "TeamNode",
        "TeamNodeConfig",
        "TeamNodeWorkflow",
        "TeamNodeExecutor",
    ),
    runtime_requirements=(
        TextRequirement(
            needle="v0.9.0",
            missing_message="deprecation warning must publish the target removal version",
        ),
        TextRequirement(
            needle="2027-03-31",
            missing_message="deprecation warning must publish the target removal date",
        ),
        TextRequirement(
            needle="docs/architecture/migration.md",
            missing_message="deprecation warning must point at the migration guide",
        ),
    ),
    inventory_path=Path("docs/development/deprecation-inventory-2026-03-03.md"),
    inventory_requirements=(
        TextRequirement(
            needle="TeamNode*",
            missing_message="deprecation inventory entry is missing 'TeamNode*'",
        ),
        TextRequirement(
            needle="v0.9.0",
            missing_message="deprecation inventory entry is missing the target removal version",
        ),
        TextRequirement(
            needle="2027-03-31",
            missing_message="deprecation inventory entry is missing the target removal date",
        ),
        TextRequirement(
            needle="TeamStep*",
            missing_message="deprecation inventory entry is missing the replacement path",
        ),
    ),
    changelog_path=Path("CHANGELOG.md"),
    changelog_requirements=(
        TextRequirement(
            needle="## [Unreleased]",
            missing_message="deprecation changelog entry is missing the Unreleased section",
        ),
        TextRequirement(
            needle="TeamNode*",
            missing_message="deprecation changelog entry is missing the shim family name",
        ),
        TextRequirement(
            needle="To be removed in:",
            missing_message="deprecation changelog entry is missing the removal-version line",
        ),
        TextRequirement(
            needle="v0.9.0",
            missing_message="deprecation changelog entry is missing the target removal version",
        ),
        TextRequirement(
            needle="Target removal date:",
            missing_message="deprecation changelog entry is missing the removal-date line",
        ),
        TextRequirement(
            needle="2027-03-31",
            missing_message="deprecation changelog entry is missing the target removal date",
        ),
        TextRequirement(
            needle="Replacement:",
            missing_message="deprecation changelog entry is missing the replacement line",
        ),
        TextRequirement(
            needle="TeamStep*",
            missing_message="deprecation changelog entry is missing the replacement path",
        ),
        TextRequirement(
            needle="Compatibility shim status:",
            missing_message="deprecation changelog entry is missing shim-status guidance",
        ),
    ),
    migration_path=Path("docs/architecture/migration.md"),
    migration_requirements=(
        TextRequirement(
            needle="TeamNode*",
            missing_message="migration guidance is missing the shim family name",
        ),
        TextRequirement(
            needle="v0.9.0",
            missing_message="migration guidance is missing the target removal version",
        ),
        TextRequirement(
            needle="2027-03-31",
            missing_message="migration guidance is missing the target removal date",
        ),
        TextRequirement(
            needle="TeamStep*",
            missing_message="migration guidance is missing the replacement path",
        ),
    ),
)

WORKFLOW_GRAPH_ALIAS_DEPRECATION_CONTRACT = DeprecationContract(
    label="WorkflowGraph alias from victor.workflows.graph",
    runtime_files=(Path("victor/workflows/graph.py"),),
    activation_needles=("WorkflowGraph from victor.workflows.graph is deprecated",),
    runtime_requirements=(
        TextRequirement(
            needle="v0.8.0",
            missing_message="deprecation warning must publish the target removal version",
        ),
        TextRequirement(
            needle="2026-12-31",
            missing_message="deprecation warning must publish the target removal date",
        ),
        TextRequirement(
            needle="docs/architecture/migration.md",
            missing_message="deprecation warning must point at the migration guide",
        ),
        TextRequirement(
            needle="BasicWorkflowGraph",
            missing_message="deprecation warning must publish the simple-container replacement",
        ),
        TextRequirement(
            needle="victor.workflows.graph_dsl.WorkflowGraph",
            missing_message="deprecation warning must publish the typed-DSL replacement",
        ),
    ),
    inventory_path=Path("docs/development/deprecation-inventory-2026-03-03.md"),
    inventory_requirements=(
        TextRequirement(
            needle="`WorkflowGraph` alias from `victor.workflows.graph`",
            missing_message="deprecation inventory entry is missing the shim family name",
        ),
        TextRequirement(
            needle="v0.8.0",
            missing_message="deprecation inventory entry is missing the target removal version",
        ),
        TextRequirement(
            needle="2026-12-31",
            missing_message="deprecation inventory entry is missing the target removal date",
        ),
        TextRequirement(
            needle="BasicWorkflowGraph",
            missing_message="deprecation inventory entry is missing the simple-container replacement",
        ),
        TextRequirement(
            needle="victor.workflows.graph_dsl.WorkflowGraph",
            missing_message="deprecation inventory entry is missing the typed-DSL replacement",
        ),
    ),
    changelog_path=Path("CHANGELOG.md"),
    changelog_requirements=(
        TextRequirement(
            needle="## [Unreleased]",
            missing_message="deprecation changelog entry is missing the Unreleased section",
        ),
        TextRequirement(
            needle="`WorkflowGraph` alias from `victor.workflows.graph`",
            missing_message="deprecation changelog entry is missing the shim family name",
        ),
        TextRequirement(
            needle="To be removed in:",
            missing_message="deprecation changelog entry is missing the removal-version line",
        ),
        TextRequirement(
            needle="v0.8.0",
            missing_message="deprecation changelog entry is missing the target removal version",
        ),
        TextRequirement(
            needle="Target removal date:",
            missing_message="deprecation changelog entry is missing the removal-date line",
        ),
        TextRequirement(
            needle="2026-12-31",
            missing_message="deprecation changelog entry is missing the target removal date",
        ),
        TextRequirement(
            needle="Replacement:",
            missing_message="deprecation changelog entry is missing the replacement line",
        ),
        TextRequirement(
            needle="BasicWorkflowGraph",
            missing_message="deprecation changelog entry is missing the simple-container replacement",
        ),
        TextRequirement(
            needle="victor.workflows.graph_dsl.WorkflowGraph",
            missing_message="deprecation changelog entry is missing the typed-DSL replacement",
        ),
        TextRequirement(
            needle="Compatibility shim status:",
            missing_message="deprecation changelog entry is missing shim-status guidance",
        ),
    ),
    migration_path=Path("docs/architecture/migration.md"),
    migration_requirements=(
        TextRequirement(
            needle="`WorkflowGraph` import from `victor.workflows.graph`",
            missing_message="migration guidance is missing the shim family name",
        ),
        TextRequirement(
            needle="v0.8.0",
            missing_message="migration guidance is missing the target removal version",
        ),
        TextRequirement(
            needle="2026-12-31",
            missing_message="migration guidance is missing the target removal date",
        ),
        TextRequirement(
            needle="BasicWorkflowGraph",
            missing_message="migration guidance is missing the simple-container replacement",
        ),
        TextRequirement(
            needle="victor.workflows.graph_dsl.WorkflowGraph",
            missing_message="migration guidance is missing the typed-DSL replacement",
        ),
    ),
)

FRAMEWORK_SHIM_DEPRECATION_CONTRACT = DeprecationContract(
    label="FrameworkShim compatibility surface",
    runtime_files=(
        Path("victor/framework/shim.py"),
        Path("victor/framework/__init__.py"),
    ),
    activation_needles=(
        "FrameworkShim is deprecated",
        "victor.framework.FrameworkShim is deprecated",
    ),
    runtime_requirements=(
        TextRequirement(
            needle="v1.0.0",
            missing_message="deprecation warning must publish the target removal version",
        ),
        TextRequirement(
            needle="2027-06-30",
            missing_message="deprecation warning must publish the target removal date",
        ),
        TextRequirement(
            needle="docs/architecture/migration.md",
            missing_message="deprecation warning must point at the migration guide",
        ),
        TextRequirement(
            needle="Agent.create()",
            missing_message="deprecation warning must publish the public replacement path",
        ),
    ),
    inventory_path=Path("docs/development/deprecation-inventory-2026-03-03.md"),
    inventory_requirements=(
        TextRequirement(
            needle="`FrameworkShim` compatibility surface",
            missing_message="deprecation inventory entry is missing the shim family name",
        ),
        TextRequirement(
            needle="v1.0.0",
            missing_message="deprecation inventory entry is missing the target removal version",
        ),
        TextRequirement(
            needle="2027-06-30",
            missing_message="deprecation inventory entry is missing the target removal date",
        ),
        TextRequirement(
            needle="Agent.create()",
            missing_message="deprecation inventory entry is missing the public replacement path",
        ),
        TextRequirement(
            needle="AgentFactory",
            missing_message="deprecation inventory entry is missing the internal replacement path",
        ),
    ),
    changelog_path=Path("CHANGELOG.md"),
    changelog_requirements=(
        TextRequirement(
            needle="## [Unreleased]",
            missing_message="deprecation changelog entry is missing the Unreleased section",
        ),
        TextRequirement(
            needle="`FrameworkShim` compatibility surface",
            missing_message="deprecation changelog entry is missing the shim family name",
        ),
        TextRequirement(
            needle="To be removed in:",
            missing_message="deprecation changelog entry is missing the removal-version line",
        ),
        TextRequirement(
            needle="v1.0.0",
            missing_message="deprecation changelog entry is missing the target removal version",
        ),
        TextRequirement(
            needle="Target removal date:",
            missing_message="deprecation changelog entry is missing the removal-date line",
        ),
        TextRequirement(
            needle="2027-06-30",
            missing_message="deprecation changelog entry is missing the target removal date",
        ),
        TextRequirement(
            needle="Replacement:",
            missing_message="deprecation changelog entry is missing the replacement line",
        ),
        TextRequirement(
            needle="Agent.create()",
            missing_message="deprecation changelog entry is missing the public replacement path",
        ),
        TextRequirement(
            needle="AgentFactory",
            missing_message="deprecation changelog entry is missing the internal replacement path",
        ),
        TextRequirement(
            needle="Compatibility shim status:",
            missing_message="deprecation changelog entry is missing shim-status guidance",
        ),
    ),
    migration_path=Path("docs/architecture/migration.md"),
    migration_requirements=(
        TextRequirement(
            needle="`FrameworkShim`",
            missing_message="migration guidance is missing the shim family name",
        ),
        TextRequirement(
            needle="v1.0.0",
            missing_message="migration guidance is missing the target removal version",
        ),
        TextRequirement(
            needle="2027-06-30",
            missing_message="migration guidance is missing the target removal date",
        ),
        TextRequirement(
            needle="Agent.create()",
            missing_message="migration guidance is missing the public replacement path",
        ),
        TextRequirement(
            needle="AgentFactory",
            missing_message="migration guidance is missing the internal replacement path",
        ),
    ),
)

PUBLIC_SHIM_DEPRECATION_CONTRACTS = (
    FRAMEWORK_SHIM_DEPRECATION_CONTRACT,
    TEAM_NODE_DEPRECATION_CONTRACT,
    WORKFLOW_GRAPH_ALIAS_DEPRECATION_CONTRACT,
)


def _load_toml_module():
    """Load a TOML parser compatible with Python 3.10+."""
    try:
        return importlib.import_module("tomllib")
    except ModuleNotFoundError:
        return importlib.import_module("tomli")


tomllib = _load_toml_module()


def _iter_unique_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(files)


def _relative(path: Path, root: Path) -> Path:
    return path.relative_to(root)


def _text_contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _check_text_requirements(
    *,
    findings: list[HygieneFinding],
    path: Path,
    text: str,
    contract: DeprecationContract,
    requirements: tuple[TextRequirement, ...],
) -> None:
    for requirement in requirements:
        if requirement.needle not in text:
            findings.append(HygieneFinding(path, f"{contract.label} {requirement.missing_message}"))


def _workflow_on_config(data: object) -> object | None:
    if not isinstance(data, dict):
        return None
    if "on" in data:
        return data["on"]
    if True in data:
        return data[True]
    return None


def check_workflow_yaml(root: Path) -> list[HygieneFinding]:
    """Ensure workflow files parse and expose a real trigger configuration."""
    findings: list[HygieneFinding] = []
    workflows_dir = root / ".github" / "workflows"
    for path in sorted(workflows_dir.glob("*.yml")):
        rel_path = _relative(path, root)
        try:
            loaded = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            findings.append(HygieneFinding(rel_path, f"workflow YAML failed to parse: {exc}"))
            continue

        if not isinstance(loaded, dict):
            findings.append(HygieneFinding(rel_path, "workflow must parse to a mapping"))
            continue

        name = loaded.get("name")
        if not isinstance(name, str) or not name.strip():
            findings.append(
                HygieneFinding(rel_path, "workflow is missing a non-empty top-level name")
            )

        on_config = _workflow_on_config(loaded)
        if on_config in (None, "", []):
            findings.append(
                HygieneFinding(
                    rel_path, "workflow is missing a non-empty top-level trigger (`on`)"
                ),
            )

    return findings


def check_nightly_workflow_contract(root: Path) -> list[HygieneFinding]:
    """Keep the develop nightly aligned with its required local dependencies."""
    rel_path = Path(".github/workflows/ci-develop-nightly.yml")
    path = root / rel_path
    if not path.is_file():
        return []

    try:
        loaded = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        # ``check_workflow_yaml`` reports the parse failure with better context.
        return []
    if not isinstance(loaded, dict):
        return []

    jobs = loaded.get("jobs")
    if not isinstance(jobs, dict):
        return []

    def step_run(job_name: str, step_name: str) -> str:
        job = jobs.get(job_name)
        if not isinstance(job, dict):
            return ""
        steps = job.get("steps")
        if not isinstance(steps, list):
            return ""
        for step in steps:
            if isinstance(step, dict) and step.get("name") == step_name:
                run = step.get("run")
                return run if isinstance(run, str) else ""
        return ""

    findings: list[HygieneFinding] = []
    install_run = step_run("full-suite", "Install dependencies")
    if "./victor-codegraph" not in install_run:
        findings.append(
            HygieneFinding(
                rel_path,
                "nightly dependency setup must install the in-repo victor-codegraph package",
            )
        )

    notify_run = step_run("report", "Notify on failure (tracking issue)")
    normalized_notify = notify_run.replace("\\\n", " ")
    for command in ("gh issue list", "gh issue comment", "gh issue create"):
        command_line = next(
            (line for line in normalized_notify.splitlines() if command in line),
            "",
        )
        if command_line and '--repo "$GITHUB_REPOSITORY"' not in command_line:
            findings.append(
                HygieneFinding(
                    rel_path,
                    f"nightly notification command `{command}` must pass an explicit repository",
                )
            )

    return findings


def check_banned_repo_urls(root: Path) -> list[HygieneFinding]:
    """Keep first-party operational files pointed at the canonical repo."""
    findings: list[HygieneFinding] = []
    for path in _iter_unique_files(root, REPO_URL_SCAN_GLOBS):
        text = path.read_text()
        for banned_url in BANNED_REPO_URLS:
            if banned_url in text:
                findings.append(
                    HygieneFinding(
                        _relative(path, root),
                        f"contains banned repo URL: {banned_url}",
                    )
                )
    return findings


def check_uppercase_roadmap_links(root: Path) -> list[HygieneFinding]:
    """Reject markdown links that point to the non-canonical uppercase roadmap path."""
    findings: list[HygieneFinding] = []
    for path in _iter_unique_files(root, MARKDOWN_SCAN_GLOBS):
        text = path.read_text()
        for match in ROADMAP_LINK_RE.finditer(text):
            findings.append(
                HygieneFinding(
                    _relative(path, root),
                    f"contains non-canonical roadmap link target: {match.group(1)}",
                )
            )
    return findings


def check_archived_doc_banners(root: Path) -> list[HygieneFinding]:
    """Ensure known archived planning docs advertise their non-canonical status."""
    findings: list[HygieneFinding] = []
    for rel_path, banner_text in ARCHIVED_DOC_BANNERS.items():
        path = root / rel_path
        if not path.is_file():
            findings.append(HygieneFinding(rel_path, "archived document is missing"))
            continue
        text = path.read_text()
        if banner_text not in text[:500]:
            findings.append(
                HygieneFinding(
                    rel_path,
                    f"archived document is missing the required banner text: {banner_text!r}",
                )
            )
    return findings


def check_release_tag_routing(root: Path) -> list[HygieneFinding]:
    """Each release-*.yml tag must trigger exactly one release workflow.

    The victor-ai release (release.yml) fires on ``v*``, which also matches sibling tags
    like ``victor-codegraph-v*`` and ``verticals-v*`` — so every sibling pattern must be
    excluded in release.yml, or pushing that tag spuriously runs (and fails) the victor-ai
    release. This has bitten twice; the guard makes a new release-*.yml that forgets the
    exclusion fail CI instead.
    """
    import fnmatch

    findings: list[HygieneFinding] = []
    wf_dir = root / ".github" / "workflows"

    def tag_patterns(path: Path) -> list[str]:
        try:
            data = yaml.safe_load(path.read_text())
        except Exception:
            return []
        push = (data or {}).get(True, {})  # PyYAML parses `on:` as the bool key True
        push = push.get("push", {}) if isinstance(push, dict) else {}
        tags = push.get("tags", []) if isinstance(push, dict) else []
        return [str(t) for t in tags] if isinstance(tags, list) else []

    def triggers(tag: str, patterns: list[str]) -> bool:
        pos = [p for p in patterns if not p.startswith("!")]
        neg = [p[1:] for p in patterns if p.startswith("!")]
        return any(fnmatch.fnmatch(tag, p) for p in pos) and not any(
            fnmatch.fnmatch(tag, n) for n in neg
        )

    release_wfs = {p.name: tag_patterns(p) for p in sorted(wf_dir.glob("release*.yml"))}
    # A representative concrete tag for each sibling's positive pattern (`*` → a version).
    for wf_name, patterns in release_wfs.items():
        if wf_name == "release.yml":
            continue
        for pattern in [p for p in patterns if not p.startswith("!")]:
            sample = pattern.replace("*", "0.0.0")
            firing = [w for w, pats in release_wfs.items() if triggers(sample, pats)]
            if firing != [wf_name]:
                findings.append(
                    HygieneFinding(
                        Path(".github/workflows/release.yml"),
                        f"tag '{sample}' (from {wf_name}) triggers {firing}, not just "
                        f"[{wf_name}] — add '!{pattern}' to release.yml's tag exclusions.",
                    )
                )
    return findings


def check_canonical_doc_pointers(root: Path) -> list[HygieneFinding]:
    """Ensure canonical pointer docs exist and their relative links resolve (TD-18).

    docs/roadmap.md was referenced by six documents while existing only as an
    untracked local file; this check makes that class of drift fail CI.
    """
    findings: list[HygieneFinding] = []
    for rel_path in CANONICAL_POINTER_DOCS:
        path = root / rel_path
        if not path.is_file():
            findings.append(HygieneFinding(rel_path, "canonical doc is missing"))
            continue
        text = path.read_text()
        for match in RELATIVE_DOC_LINK_RE.finditer(text):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:", "/")):
                continue
            if not (path.parent / target).resolve().exists():
                findings.append(
                    HygieneFinding(
                        rel_path,
                        f"broken canonical link target: {target}",
                    )
                )
    return findings


def check_removed_legacy_paths(root: Path) -> list[HygieneFinding]:
    """Prevent removed duplicate-source files from quietly returning."""
    findings: list[HygieneFinding] = []
    for rel_path, message in LEGACY_PATHS_THAT_MUST_STAY_REMOVED.items():
        if (root / rel_path).exists():
            findings.append(HygieneFinding(rel_path, message))
    return findings


def _extract_make_target(lines: list[str], target_name: str) -> list[str]:
    start_index: int | None = None
    for index, line in enumerate(lines):
        if line.startswith(f"{target_name}:"):
            start_index = index + 1
            break
    if start_index is None:
        return []

    block: list[str] = []
    for line in lines[start_index:]:
        if TARGET_HEADER_RE.match(line):
            break
        block.append(line)
    return block


def check_makefile_lint_gate(root: Path) -> list[HygieneFinding]:
    """Keep the local lint target aligned with the current mypy gate."""
    findings: list[HygieneFinding] = []
    path = root / "Makefile"
    lines = path.read_text().splitlines()
    lint_block = _extract_make_target(lines, "lint")
    rel_path = _relative(path, root)

    if not lint_block:
        return [HygieneFinding(rel_path, "Makefile is missing the lint target block")]

    mypy_lines = [line.strip() for line in lint_block if "mypy " in line]
    if not mypy_lines:
        findings.append(HygieneFinding(rel_path, "lint target is missing a mypy command"))
        return findings

    if not any("mypy victor" in line for line in mypy_lines):
        findings.append(HygieneFinding(rel_path, "lint target must run `mypy victor`"))

    if any("|| true" in line for line in mypy_lines):
        findings.append(
            HygieneFinding(rel_path, "lint target must not suppress mypy failure with `|| true`")
        )

    return findings


def check_vertical_extra_metadata(root: Path) -> list[HygieneFinding]:
    """Reject stale no-op metadata for extracted vertical package extras."""

    path = root / "pyproject.toml"
    if not path.is_file():
        return []
    rel_path = _relative(path, root)
    try:
        data = tomllib.loads(path.read_text())
    except Exception as exc:
        return [HygieneFinding(rel_path, f"pyproject.toml failed to parse: {exc}")]

    project = data.get("project", {})
    optional_deps = project.get("optional-dependencies", {})
    if not isinstance(optional_deps, dict):
        return [HygieneFinding(rel_path, "project.optional-dependencies must be a mapping")]
    project_name = str(project.get("name", "")).strip().lower()

    findings: list[HygieneFinding] = []
    for extra_name in ("coding", "research", "devops", "verticals"):
        values = optional_deps.get(extra_name)
        if not isinstance(values, list) or not values:
            findings.append(
                HygieneFinding(
                    rel_path,
                    f"extracted vertical extra '{extra_name}' must map to real package dependencies",
                )
            )

    for legacy_name in ("rag", "dataanalysis"):
        values = optional_deps.get(legacy_name)
        if isinstance(values, list) and not values:
            findings.append(
                HygieneFinding(
                    rel_path,
                    f"legacy no-op vertical extra '{legacy_name}' must not be present",
                )
            )

    if project_name:
        self_reference_prefix = f"{project_name}["
        for extra_name in ("dev", "ci", "verticals"):
            values = optional_deps.get(extra_name)
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, str):
                    continue
                if value.lower().startswith(self_reference_prefix):
                    findings.append(
                        HygieneFinding(
                            rel_path,
                            f"extra '{extra_name}' must not self-reference {project_name}[...]"
                            " because nested extras do not resolve reliably in CI",
                        )
                    )
                    break

    return findings


def check_primary_vertical_contract_docs(root: Path) -> list[HygieneFinding]:
    """Keep the primary authoring surfaces aligned with the plugin-first SDK contract."""

    findings: list[HygieneFinding] = []
    for rel_path in PRIMARY_VERTICAL_CONTRACT_FILES:
        path = root / rel_path
        if not path.is_file():
            continue

        text = path.read_text(encoding="utf-8")
        for needle, message in STALE_VERTICAL_CONTRACT_PATTERNS.items():
            if needle in text:
                findings.append(HygieneFinding(rel_path, message))

    return findings


def _workflow_has_blocking_trivy_step(path: Path) -> bool:
    try:
        loaded = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return False

    if not isinstance(loaded, dict):
        return False

    jobs = loaded.get("jobs")
    if not isinstance(jobs, dict):
        return False

    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            uses = step.get("uses")
            if not isinstance(uses, str) or "aquasecurity/trivy-action" not in uses:
                continue
            config = step.get("with")
            if not isinstance(config, dict):
                continue
            severity = str(config.get("severity", ""))
            exit_code = str(config.get("exit-code", ""))
            ignore_unfixed = str(config.get("ignore-unfixed", "")).lower()
            continue_on_error = bool(step.get("continue-on-error", False))
            if (
                "CRITICAL" in severity
                and exit_code == "1"
                and ignore_unfixed == "true"
                and not continue_on_error
            ):
                return True
    return False


def _workflow_has_blocking_pip_audit_step(path: Path) -> bool:
    try:
        loaded = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return False

    if not isinstance(loaded, dict):
        return False

    jobs = loaded.get("jobs")
    if not isinstance(jobs, dict):
        return False

    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            uses = step.get("uses")
            run = step.get("run")
            has_pip_audit_action = isinstance(uses, str) and "pypa/gh-action-pip-audit" in uses
            has_pip_audit_command = isinstance(run, str) and "pip-audit" in run
            if not has_pip_audit_action and not has_pip_audit_command:
                continue
            if bool(step.get("continue-on-error", False)):
                continue
            if isinstance(run, str) and "|| true" in run:
                continue
            return True
    return False


def _workflow_has_blocking_bandit_high_step(path: Path) -> bool:
    try:
        loaded = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return False

    if not isinstance(loaded, dict):
        return False

    jobs = loaded.get("jobs")
    if not isinstance(jobs, dict):
        return False

    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            run = step.get("run")
            if not isinstance(run, str):
                continue
            if "bandit -r victor/" not in run:
                continue
            if "--severity-level high" not in run or "--confidence-level high" not in run:
                continue
            if bool(step.get("continue-on-error", False)):
                continue
            return True
    return False


def check_security_baseline(root: Path) -> list[HygieneFinding]:
    """Ensure the documented security baseline matches enforced workflow reality."""
    findings: list[HygieneFinding] = []

    blocking_sources = [
        root / ".github" / "workflows" / "ci-fast.yml",
        root / ".github" / "workflows" / "security.yml",
    ]
    if not any(
        _workflow_has_blocking_trivy_step(path) for path in blocking_sources if path.is_file()
    ):
        findings.append(
            HygieneFinding(
                Path(".github/workflows"),
                "missing a blocking Trivy step for CRITICAL findings (`exit-code: 1` without continue-on-error)",
            )
        )

    security_workflow = root / ".github" / "workflows" / "security.yml"
    if security_workflow.is_file():
        if not _workflow_has_blocking_pip_audit_step(security_workflow):
            findings.append(
                HygieneFinding(
                    Path(".github/workflows/security.yml"),
                    "missing a blocking pip-audit step in the security baseline",
                )
            )
        if not _workflow_has_blocking_bandit_high_step(security_workflow):
            findings.append(
                HygieneFinding(
                    Path(".github/workflows/security.yml"),
                    "missing a blocking Bandit step for HIGH severity + HIGH confidence findings",
                )
            )

    security_doc = root / "SECURITY.md"
    if security_doc.is_file():
        text = security_doc.read_text()
        if "## Current CI Enforcement Baseline" not in text:
            findings.append(
                HygieneFinding(
                    Path("SECURITY.md"),
                    "missing the Current CI Enforcement Baseline section",
                ),
            )
        if "**Blocking today**" not in text:
            findings.append(
                HygieneFinding(
                    Path("SECURITY.md"),
                    "missing the Blocking today security baseline summary",
                ),
            )
        if "**Advisory today**" not in text:
            findings.append(
                HygieneFinding(
                    Path("SECURITY.md"),
                    "missing the Advisory today security baseline summary",
                ),
            )
        if "### Current Thresholds" not in text:
            findings.append(
                HygieneFinding(Path("SECURITY.md"), "missing the Current Thresholds section"),
            )
        if "Trivy filesystem scan" not in text:
            findings.append(
                HygieneFinding(
                    Path("SECURITY.md"),
                    "missing Trivy filesystem scan threshold documentation",
                ),
            )
        if "| Dependency audit | Blocking |" not in text:
            findings.append(
                HygieneFinding(
                    Path("SECURITY.md"),
                    "missing blocking dependency-audit threshold documentation",
                ),
            )
        if "| Bandit (SAST) | Blocking |" not in text:
            findings.append(
                HygieneFinding(
                    Path("SECURITY.md"),
                    "missing blocking Bandit threshold documentation",
                ),
            )
    else:
        findings.append(HygieneFinding(Path("SECURITY.md"), "security policy document is missing"))

    return findings


def check_deprecation_contract(root: Path, contract: DeprecationContract) -> list[HygieneFinding]:
    """Validate one public deprecation contract across code and docs.

    If the compatibility surface still exists in runtime files, its warning
    text, inventory entry, changelog note, and migration guide must remain
    synchronized.
    """

    findings: list[HygieneFinding] = []
    alias_runtime_files: list[tuple[Path, str]] = []

    for rel_path in contract.runtime_files:
        path = root / rel_path
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if not _text_contains_any(text, contract.activation_needles):
            continue
        alias_runtime_files.append((rel_path, text))
        _check_text_requirements(
            findings=findings,
            path=rel_path,
            text=text,
            contract=contract,
            requirements=contract.runtime_requirements,
        )

    if not alias_runtime_files:
        return findings

    inventory_path = root / contract.inventory_path
    if not inventory_path.is_file():
        findings.append(
            HygieneFinding(
                contract.inventory_path,
                f"{contract.label} deprecation inventory entry is missing",
            )
        )
    else:
        inventory_text = inventory_path.read_text(encoding="utf-8")
        _check_text_requirements(
            findings=findings,
            path=contract.inventory_path,
            text=inventory_text,
            contract=contract,
            requirements=contract.inventory_requirements,
        )

    changelog_path = root / contract.changelog_path
    if not changelog_path.is_file():
        findings.append(
            HygieneFinding(
                contract.changelog_path,
                f"{contract.label} deprecation changelog entry is missing",
            )
        )
    else:
        changelog_text = changelog_path.read_text(encoding="utf-8")
        _check_text_requirements(
            findings=findings,
            path=contract.changelog_path,
            text=changelog_text,
            contract=contract,
            requirements=contract.changelog_requirements,
        )

    migration_path = root / contract.migration_path
    if not migration_path.is_file():
        findings.append(
            HygieneFinding(
                contract.migration_path,
                f"{contract.label} migration guide entry is missing",
            )
        )
    else:
        migration_text = migration_path.read_text(encoding="utf-8")
        _check_text_requirements(
            findings=findings,
            path=contract.migration_path,
            text=migration_text,
            contract=contract,
            requirements=contract.migration_requirements,
        )

    return findings


def check_public_shim_deprecation_contracts(root: Path) -> list[HygieneFinding]:
    """Validate all registered public compatibility-shim deprecation contracts."""

    findings: list[HygieneFinding] = []
    for contract in PUBLIC_SHIM_DEPRECATION_CONTRACTS:
        findings.extend(check_deprecation_contract(root, contract))
    return findings


def run_checks(root: Path) -> list[HygieneFinding]:
    """Run all foundational repo hygiene checks."""
    findings: list[HygieneFinding] = []
    findings.extend(check_workflow_yaml(root))
    findings.extend(check_nightly_workflow_contract(root))
    findings.extend(check_banned_repo_urls(root))
    findings.extend(check_uppercase_roadmap_links(root))
    findings.extend(check_archived_doc_banners(root))
    findings.extend(check_canonical_doc_pointers(root))
    findings.extend(check_release_tag_routing(root))
    findings.extend(check_removed_legacy_paths(root))
    findings.extend(check_makefile_lint_gate(root))
    findings.extend(check_vertical_extra_metadata(root))
    findings.extend(check_primary_vertical_contract_docs(root))
    findings.extend(check_security_baseline(root))
    findings.extend(check_public_shim_deprecation_contracts(root))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root to scan")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    findings = run_checks(root)
    if not findings:
        print("Repo hygiene checks passed")
        return 0

    print("Repo hygiene checks failed:")
    for finding in findings:
        print(f"- {finding.path}: {finding.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
