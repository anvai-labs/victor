# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression guards for dependency installation in CI workflows."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


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
