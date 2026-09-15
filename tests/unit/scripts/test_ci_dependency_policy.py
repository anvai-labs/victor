# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression guards for dependency installation in CI workflows."""

from pathlib import Path
import shlex

import yaml

ROOT = Path(__file__).resolve().parents[3]
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
RUNNER_PREFLIGHT = "scripts/ci/runner_preflight.py"
REQUIRED_PREFLIGHT_OPTIONS = {
    "--require-os": "ubuntu",
    "--min-os-version": "24.04",
    "--min-glibc": "2.38",
}


def _ci_config_paths(root: Path = ROOT) -> list[Path]:
    """Return supported workflow and composite-action YAML files."""
    workflow_dir = root / ".github" / "workflows"
    action_dir = root / ".github" / "actions"
    paths = [*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")]
    paths.extend(action_dir.rglob("action.yml"))
    paths.extend(action_dir.rglob("action.yaml"))
    return sorted(set(paths))


def _run_steps(document: dict):
    """Yield named shell steps from a workflow or composite action."""
    if "jobs" in document:
        for job_name, job in document["jobs"].items():
            for step in job.get("steps", []):
                if "run" in step:
                    yield f"{job_name}: {step.get('name', '<unnamed>')}", step["run"]
        return

    for step in document.get("runs", {}).get("steps", []):
        if "run" in step:
            yield step.get("name", "<unnamed>"), step["run"]


def _cpu_torch_precedes_dev_installs(run: str) -> bool:
    """Return whether every active dev install follows a CPU Torch install."""
    commands = [
        line.strip()
        for line in run.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    dev_indexes = [
        index
        for index, command in enumerate(commands)
        if "pip install" in command and "-e" in command and ".[dev]" in command
    ]
    if not dev_indexes:
        return True

    cpu_indexes = [
        index
        for index, command in enumerate(commands)
        if "pip install" in command and "torch" in command and CPU_INDEX in command
    ]
    return bool(cpu_indexes) and max(cpu_indexes) < min(dev_indexes)


def test_policy_detector_rejects_missing_or_late_cpu_selection():
    """The guard recognizes ordering errors and common command spellings."""
    assert not _cpu_torch_precedes_dev_installs('pip install -e ".[dev]"')
    assert not _cpu_torch_precedes_dev_installs(
        'pip install -e ".[dev]"\npip install torch --index-url ' + CPU_INDEX
    )
    assert _cpu_torch_precedes_dev_installs(
        "python -m pip install torch --index-url "
        + CPU_INDEX
        + "\npython -m pip install -e '.[dev]'"
    )


def test_config_discovery_covers_yaml_and_nested_actions(tmp_path):
    """Supported extensions and nested composite actions cannot evade the guard."""
    expected = {
        tmp_path / ".github" / "workflows" / "one.yml",
        tmp_path / ".github" / "workflows" / "two.yaml",
        tmp_path / ".github" / "actions" / "nested" / "deep" / "action.yml",
        tmp_path / ".github" / "actions" / "other" / "action.yaml",
    }
    for path in expected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("name: test\n")

    assert set(_ci_config_paths(tmp_path)) == expected


def test_dev_installs_preselect_cpu_only_torch():
    """Linux CI must not resolve the dev extra to CUDA-enabled Torch wheels."""
    offenders = []
    checked = 0
    for path in _ci_config_paths():
        document = yaml.safe_load(path.read_text())
        for step_name, run in _run_steps(document):
            if ".[dev]" not in run or "pip install" not in run:
                continue
            checked += 1
            if not _cpu_torch_precedes_dev_installs(run):
                offenders.append(f"{path.relative_to(ROOT)} ({step_name})")

    assert checked > 0, "No editable dev installs found; update this policy guard"
    assert (
        not offenders
    ), "Dev installs must preselect CPU-only Torch before resolving .[dev]: " + ", ".join(offenders)


def _is_runner_preflight(run: str) -> bool:
    """Recognize the exact preflight command and its required ABI constraints."""
    try:
        arguments = shlex.split(run)
    except ValueError:
        return False
    if arguments[:2] != ["python3", RUNNER_PREFLIGHT]:
        return False
    if len(arguments) != 2 + 2 * len(REQUIRED_PREFLIGHT_OPTIONS):
        return False
    supplied_options = dict(zip(arguments[2::2], arguments[3::2]))
    return supplied_options == REQUIRED_PREFLIGHT_OPTIONS


def _preflight_precedes_python_setup(job: dict) -> bool:
    """Require a complete preflight before the first setup-python action."""
    steps = job.get("steps", [])
    preflight_indexes = [
        index for index, step in enumerate(steps) if _is_runner_preflight(step.get("run", ""))
    ]
    setup_indexes = [
        index
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("actions/setup-python@")
    ]
    return bool(preflight_indexes and setup_indexes) and min(preflight_indexes) < min(setup_indexes)


def test_runner_preflight_policy_rejects_noop_missing_flags_and_late_steps():
    """Mentions, weakened commands and post-setup checks cannot satisfy the guard."""
    setup = {"uses": "actions/setup-python@pinned"}
    valid = {
        "run": "python3 scripts/ci/runner_preflight.py "
        "--min-glibc 2.38 --require-os ubuntu --min-os-version 24.04"
    }
    assert not _preflight_precedes_python_setup(
        {"steps": [{"run": "echo scripts/ci/runner_preflight.py"}, setup]}
    )
    assert not _preflight_precedes_python_setup(
        {"steps": [{"run": "python3 scripts/ci/runner_preflight.py --min-glibc 2.38"}, setup]}
    )
    assert not _preflight_precedes_python_setup(
        {
            "steps": [
                {
                    "run": "python3 scripts/ci/runner_preflight.py "
                    "--require-os ubuntu --min-os-version 24.04 "
                    "--min-glibc 2.38 || true"
                },
                setup,
            ]
        }
    )
    assert not _preflight_precedes_python_setup({"steps": [setup, valid]})
    assert _preflight_precedes_python_setup({"steps": [valid, setup]})


def test_python_312_native_jobs_use_a_declared_host_abi():
    """Known ABI-sensitive jobs cannot drift back onto an ambiguous host label."""
    expected_jobs = {
        ".github/workflows/ci-fast.yml": {"rust-packages"},
        ".github/workflows/docs.yml": {"build", "deploy"},
        ".github/workflows/verticals.yml": {"boundaries", "test", "lint"},
    }
    missing_preflight = []
    wrong_runner = []

    for relative_path, job_names in expected_jobs.items():
        document = yaml.safe_load((ROOT / relative_path).read_text())
        for job_name in job_names:
            job = document["jobs"][job_name]
            if job["runs-on"] != "ubuntu-24.04":
                wrong_runner.append(f"{relative_path}:{job_name}")
            if job_name == "deploy":
                continue
            if not _preflight_precedes_python_setup(job):
                missing_preflight.append(f"{relative_path}:{job_name}")

    assert not wrong_runner, "ABI-sensitive jobs must use ubuntu-24.04: " + ", ".join(wrong_runner)
    assert not missing_preflight, "ABI-sensitive jobs need a host preflight: " + ", ".join(
        missing_preflight
    )


def _find_text_token(value, expected, path=""):
    """Return YAML paths whose string contains ``expected``."""
    matches = []
    if isinstance(value, dict):
        for key, child in value.items():
            matches.extend(_find_text_token(child, expected, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            matches.extend(_find_text_token(child, expected, f"{path}[{index}]"))
    elif isinstance(value, str) and expected in value:
        matches.append(path.lstrip("."))
    return matches


def test_runner_label_policy_finds_values_hidden_in_expressions():
    document = {"jobs": {"test": {"runs-on": "${{ 'ubuntu-latest' }}"}}}

    assert _find_text_token(document, "ubuntu-latest") == ["jobs.test.runs-on"]


def test_workflows_do_not_use_ambiguous_ubuntu_latest_label():
    """Self-hosted labels cannot impersonate Victor's hosted Linux baseline."""
    offenders = []
    for path in _ci_config_paths():
        if path.parent.name != "workflows":
            continue
        document = yaml.safe_load(path.read_text())
        for yaml_path in _find_text_token(document, "ubuntu-latest"):
            offenders.append(f"{path.relative_to(ROOT)}:{yaml_path}")

    assert (
        not offenders
    ), "Use an explicit hosted image or self-hosted capability labels: " + ", ".join(offenders)
