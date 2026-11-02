"""Promotion gates bind the complete battery to the same PR execution."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
PROMOTION_FILES = {
    "promotion-tests": "ci-test.yml",
    "promotion-integration": "ci-integration.yml",
    "promotion-build": "build.yml",
    "promotion-security": "security.yml",
    "promotion-validation": "validation.yml",
    "promotion-performance": "performance-tests.yml",
    "promotion-contracts": "ci-contracts.yml",
    "promotion-codegraph": "ci-codegraph.yml",
    "promotion-verticals": "verticals.yml",
    "promotion-vertical-packages": "vertical-validation.yml",
}


def workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


@pytest.mark.parametrize(
    "job_id,filename,names",
    [
        ("promotion-tests", "ci-test.yml", {"HF_TOKEN", "CODECOV_TOKEN"}),
        ("promotion-security", "security.yml", {"SEMGREP_APP_TOKEN"}),
    ],
)
def test_reusable_promotion_only_receives_declared_optional_credentials(job_id, filename, names):
    call = workflow("ci-fast.yml")["jobs"][job_id]
    assert call["secrets"] == {name: "${{ secrets." + name + " }}" for name in names}
    called = workflow(filename)
    trigger = called.get("on", called.get(True))
    declared = trigger["workflow_call"]["secrets"]
    assert set(declared) == names
    assert all(config["required"] is False for config in declared.values())


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_required_context_transitively_owns_every_promotion_surface():
    jobs = workflow("ci-fast.yml")["jobs"]
    aggregate = jobs["ci-success"]
    assert aggregate["name"] == (
        "${{ github.event_name == 'pull_request' && 'CI Success' || 'Non-PR CI Success' }}"
    )
    assert set(PROMOTION_FILES) <= set(aggregate["needs"])
    for job_id, filename in PROMOTION_FILES.items():
        call = jobs[job_id]
        assert call["uses"] == f"./.github/workflows/{filename}"
        assert (
            call["if"] == "${{ github.event_name == 'pull_request' && github.base_ref == 'main' }}"
        )
        assert not call.get("continue-on-error")
        called = workflow(filename)
        trigger = called.get("on", called.get(True))
        assert "workflow_call" in trigger
        pull_request = trigger.get("pull_request")
        if pull_request is not None:
            assert "main" not in pull_request.get("branches", [])
            assert "main" in pull_request.get("branches-ignore", []) or "branches" in pull_request
        group = called.get("concurrency", {}).get("group", "")
        assert "github.workflow" not in group
        if group:
            assert filename.removesuffix(".yml") in group


def test_push_baseline_does_not_trust_promotion_or_unrelated_history():
    quick = workflow("ci-fast.yml")["jobs"]["quick-tests"]
    step = next(s for s in quick["steps"] if s.get("name") == "Run unit tests for changed files")
    assert "event=push" in step["run"]
    assert "changed_test_base.py" in step["run"]


@pytest.fixture
def gate():
    return load_helper("gate_results")


@pytest.fixture
def base_selector():
    return load_helper("changed_test_base")


def needs_for(gate, *, promotion):
    needs = {job: {"result": "success"} for job in gate.CORE_JOBS}
    needs.update(
        {job: {"result": "success" if promotion else "skipped"} for job in gate.PROMOTION_JOBS}
    )
    if promotion:
        for job in gate.MAIN_DELEGATED_JOBS:
            needs[job]["result"] = "skipped"
        needs["quick-tests"]["result"] = "skipped"
    if "windows-source-changes" in needs:
        needs["windows-source-changes"]["outputs"] = {"required": "true"}
    return needs


def test_windows_qualification_is_required_by_the_aggregate():
    jobs = workflow("ci-fast.yml")["jobs"]
    assert {"windows-source-changes", "windows-source"} <= set(jobs["ci-success"]["needs"])
    called = jobs["windows-source"]
    assert called["uses"] == "./.github/workflows/ci-graph-windows.yml"
    assert called["needs"] == ["windows-source-changes"]
    assert called["if"] == "${{ needs.windows-source-changes.outputs.required == 'true' }}"
    trigger = workflow("ci-graph-windows.yml")
    trigger = trigger.get("on", trigger.get(True))
    assert set(trigger) == {"workflow_call", "workflow_dispatch"}


@pytest.mark.parametrize("flag", [None, "", "unknown", "TRUE", False])
def test_windows_missing_or_unknown_source_flag_fails_closed(gate, flag):
    needs = needs_for(gate, promotion=False)
    needs["windows-source-changes"] = {"result": "success", "outputs": {"required": flag}}
    needs["windows-source"] = {"result": "skipped"}
    with pytest.raises(ValueError, match="Windows source qualification flag"):
        gate.validate_results(needs, event="pull_request", base="develop")


@pytest.mark.parametrize("result", ["skipped", "failure", "cancelled", None, "unknown"])
def test_windows_required_matrix_cannot_be_skipped_or_fail(gate, result):
    needs = needs_for(gate, promotion=False)
    needs["windows-source"] = {"result": result}
    with pytest.raises(ValueError, match="windows-source"):
        gate.validate_results(needs, event="pull_request", base="develop")


def test_windows_skip_requires_an_explicit_successful_negative_selector(gate):
    needs = needs_for(gate, promotion=False)
    needs["windows-source-changes"]["outputs"] = {"required": "false"}
    needs["windows-source"] = {"result": "skipped"}
    gate.validate_results(needs, event="pull_request", base="develop")
    needs["windows-source-changes"]["result"] = "failure"
    with pytest.raises(ValueError, match="windows-source-changes"):
        gate.validate_results(needs, event="pull_request", base="develop")


def test_main_requires_full_battery_while_develop_stays_light(gate):
    gate.validate_results(needs_for(gate, promotion=True), event="pull_request", base="main")
    gate.validate_results(needs_for(gate, promotion=False), event="pull_request", base="develop")


@pytest.mark.parametrize(
    "result", ["failure", "cancelled", "skipped", "neutral", "", None, "unknown"]
)
@pytest.mark.parametrize("job", list(PROMOTION_FILES))
def test_main_rejects_every_nonpassing_promotion_surface(gate, job, result):
    needs = needs_for(gate, promotion=True)
    needs[job]["result"] = result
    with pytest.raises(ValueError, match=job):
        gate.validate_results(needs, event="pull_request", base="main")


def test_missing_dependency_and_unexpected_jobs_are_rejected(gate):
    needs = needs_for(gate, promotion=True)
    del needs["promotion-integration"]
    with pytest.raises(ValueError, match="promotion-integration"):
        gate.validate_results(needs, event="pull_request", base="main")
    needs = needs_for(gate, promotion=True)
    needs["unreviewed-job"] = {"result": "success"}
    with pytest.raises(ValueError, match="unreviewed-job"):
        gate.validate_results(needs, event="pull_request", base="main")


def test_normal_core_jobs_cannot_be_skipped_or_cancelled(gate):
    for result in ("skipped", "cancelled", "failure", None):
        needs = needs_for(gate, promotion=True)
        needs["format"]["result"] = result
        with pytest.raises(ValueError, match="format"):
            gate.validate_results(needs, event="pull_request", base="main")
    needs = needs_for(gate, promotion=False)
    needs["dependency-audit"]["result"] = "skipped"
    with pytest.raises(ValueError, match="dependency-audit"):
        gate.validate_results(needs, event="pull_request", base="develop")


def test_real_workflow_needs_and_helper_policy_are_complete(gate):
    jobs = workflow("ci-fast.yml")["jobs"]
    expected = set(gate.CORE_JOBS) | set(gate.PROMOTION_JOBS)
    assert set(jobs) - {"ci-success", "promotion-success"} == expected
    assert set(jobs["ci-success"]["needs"]) == expected
    assert jobs["ci-success"]["if"] == "always()"
    step = jobs["ci-success"]["steps"][-1]
    assert step["env"]["NEEDS_JSON"] == "${{ toJSON(needs) }}"
    assert "gate_results.py" in step["run"]


def test_security_is_not_scanned_twice_or_artifact_names_collided_on_main(gate):
    jobs = workflow("ci-fast.yml")["jobs"]
    for job in gate.MAIN_DELEGATED_JOBS:
        assert (
            jobs[job]["if"]
            == "${{ github.event_name != 'pull_request' || github.base_ref != 'main' }}"
        )
    assert jobs["promotion-security"]["permissions"] == {
        "contents": "read",
        "security-events": "write",
    }


def run_fixture(
    sha,
    *,
    event="push",
    number=1,
    branch="develop",
    repository="anvai-labs/victor",
    conclusion="success",
):
    return {
        "head_sha": sha,
        "event": event,
        "run_number": number,
        "head_branch": branch,
        "head_repository": {"full_name": repository},
        "conclusion": conclusion,
        "status": "completed",
    }


def choose(selector, runs, ancestors, *, before="b" * 40):
    return selector.choose_base(
        runs,
        head_sha="d" * 40,
        branch="develop",
        repository="anvai-labs/victor",
        push_before=before,
        is_ancestor=lambda sha: sha in ancestors,
    )


def test_successful_promotion_cannot_bury_failed_push_changes(base_selector):
    old_green, skipped_promotion = "a" * 40, "c" * 40
    runs = [run_fixture(skipped_promotion, event="pull_request", number=3), run_fixture(old_green)]
    assert choose(base_selector, runs, {old_green, skipped_promotion}) == old_green


def test_baseline_rejects_foreign_repo_branch_current_head_and_nonancestor(base_selector):
    good = "a" * 40
    runs = [
        run_fixture("f" * 40, number=8),
        run_fixture("d" * 40, number=7),
        run_fixture("c" * 40, repository="fork/victor", number=6),
        run_fixture("b" * 40, branch="main", number=5),
        run_fixture("e" * 40, event="workflow_dispatch", number=4),
        run_fixture(good, number=3),
    ]
    assert choose(base_selector, runs, {good, "b" * 40, "c" * 40, "d" * 40, "e" * 40}) == good


def test_baseline_uses_newest_eligible_ancestor_not_api_order(base_selector):
    older, latest = "a" * 40, "c" * 40
    assert (
        choose(base_selector, [run_fixture(older), run_fixture(latest, number=2)], {older, latest})
        == latest
    )


def test_base_fallbacks_cover_whole_push_and_fail_without_any_history(base_selector):
    assert choose(base_selector, [], {"b" * 40}) == "b" * 40
    assert choose(base_selector, [], {"HEAD~1"}, before="0" * 40) == "HEAD~1"
    with pytest.raises(ValueError, match="baseline"):
        choose(base_selector, [], set(), before="0" * 40)


def test_malformed_or_nonpassing_run_does_not_establish_baseline(base_selector):
    old_green = "a" * 40
    bad = run_fixture("c" * 40, number=9, conclusion="failure")
    malformed = deepcopy(bad)
    malformed["head_sha"] = "HEAD; arbitrary text"
    assert (
        choose(base_selector, [bad, malformed, run_fixture(old_green)], {old_green, "c" * 40})
        == old_green
    )


@pytest.mark.parametrize(
    "event,ref",
    [
        ("push", "refs/heads/develop"),
        ("push", "refs/heads/main"),
        ("workflow_dispatch", "refs/heads/develop"),
    ],
)
def test_push_and_manual_keep_the_existing_event_specific_skip_rules(gate, event, ref):
    needs = needs_for(gate, promotion=False)
    needs["lint-title"]["result"] = "skipped"
    if event == "push" and ref == "refs/heads/main":
        needs["quick-tests"]["result"] = "skipped"
    gate.validate_results(needs, event=event, base="", ref=ref)


def test_unknown_event_and_malformed_results_fail_closed(gate):
    with pytest.raises(ValueError, match="Unknown CI event"):
        gate.validate_results(needs_for(gate, promotion=False), event="unknown", base="")
    needs = needs_for(gate, promotion=True)
    needs["format"]["result"] = {}
    with pytest.raises(ValueError, match="format"):
        gate.validate_results(needs, event="pull_request", base="main")


def test_main_push_did_not_run_quick_tests_and_cannot_be_a_quick_baseline(base_selector):
    old_green, before = "a" * 40, "b" * 40
    assert (
        base_selector.choose_base(
            [run_fixture(old_green, branch="main")],
            head_sha="d" * 40,
            branch="main",
            repository="anvai-labs/victor",
            push_before=before,
            is_ancestor=lambda sha: sha in {old_green, before},
        )
        == before
    )


def test_baseline_cli_exercises_real_git_ancestry(tmp_path):
    import json
    import subprocess
    import sys

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    commits = []
    for index in range(3):
        git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-qm",
            f"fixture {index}",
        )
        commits.append(git("rev-parse", "HEAD"))
    payload = tmp_path / "runs.json"
    payload.write_text(
        json.dumps(
            {
                "workflow_runs": [
                    run_fixture(commits[1], event="pull_request", number=2),
                    run_fixture(commits[0], number=1),
                ]
            }
        )
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/ci/changed_test_base.py"),
        "--runs",
        str(payload),
        "--head",
        commits[2],
        "--branch",
        "develop",
        "--repository",
        "anvai-labs/victor",
        "--before",
        commits[1],
    ]
    result = subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == commits[0]
    payload.write_text('{"workflow_runs": "malformed"}')
    result = subprocess.run(command, cwd=tmp_path, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "no run list" in result.stderr


def test_promotion_contract_is_always_checked_not_only_for_changed_tests():
    job = workflow("ci-fast.yml")["jobs"]["hotspot-size-guard"]
    run = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Run guard + hygiene tests (always, not changed-files)"
    )
    assert "tests/unit/scripts/test_promotion_gates.py" in run


def test_unique_promotion_context_has_one_authoritative_writer_and_no_false_skip():
    jobs = workflow("ci-fast.yml")["jobs"]
    final = jobs["promotion-success"]
    predicate = "github.event_name == 'pull_request' && github.base_ref == 'main'"
    assert (
        final["name"]
        == "${{ " + predicate + " && 'Promotion CI Success' || 'Promotion Not Applicable' }}"
    )
    assert final["if"] == "${{ always() && " + predicate + " }}"
    assert final["needs"] == ["ci-success"]
    assert not final.get("continue-on-error")
    # No alternate workflow may emit the unique gate, including on metadata
    # edits. Retargets lack this check and must reopen to trigger the full run.
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        if path.name != "ci-fast.yml":
            assert "Promotion CI Success" not in path.read_text()


@pytest.mark.parametrize(
    "result,expected",
    [
        ("success", 0),
        ("failure", 1),
        ("cancelled", 1),
        ("skipped", 1),
        ("neutral", 1),
        ("unknown", 1),
        ("", 1),
    ],
)
def test_promotion_final_verdict_executes_fail_closed(result, expected):
    import os
    import subprocess

    step = workflow("ci-fast.yml")["jobs"]["promotion-success"]["steps"][0]
    assert step["env"]["CI_SUCCESS_RESULT"] == "${{ needs.ci-success.result }}"
    observed = subprocess.run(
        ["bash", "-c", step["run"]],
        env={**os.environ, "CI_SUCCESS_RESULT": result},
        check=False,
        capture_output=True,
        text=True,
    )
    assert observed.returncode == expected


def test_gate_cli_enforces_the_same_json_contract(gate):
    import json
    import subprocess
    import sys

    script = str(ROOT / "scripts/ci/gate_results.py")
    needs = needs_for(gate, promotion=True)
    for bad in (False, True):
        needs["promotion-integration"]["result"] = "failure" if bad else "success"
        result = subprocess.run(
            [
                sys.executable,
                script,
                "--needs",
                json.dumps(needs),
                "--event",
                "pull_request",
                "--base",
                "main",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == (1 if bad else 0)
        if bad:
            assert "promotion-integration" in result.stderr


DISCOVERY_STEPS = (
    ("validation.yml", "fep-validation", "Find modified FEP files"),
    ("validation.yml", "vertical-validation", "Find modified vertical directories"),
    ("vertical-validation.yml", "validate-verticals", "Find modified vertical directories"),
)


@pytest.mark.parametrize("filename,job_id,step_name", DISCOVERY_STEPS)
@pytest.mark.parametrize("git_status", [0, 128])
def test_validation_distinguishes_known_empty_diff_from_unknown_history(
    tmp_path, filename, job_id, step_name, git_status
):
    import os
    import subprocess

    step = next(
        s for s in workflow(filename)["jobs"][job_id]["steps"] if s.get("name") == step_name
    )
    script = f"git() {{ return {git_status}; }}\n" + step["run"]
    output = tmp_path / "output"
    observed = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        cwd=ROOT,
        env={**os.environ, "BASE_BRANCH": "missing", "GITHUB_OUTPUT": str(output)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert (observed.returncode == 0) is (git_status == 0)
    if git_status:
        assert (
            not output.exists()
        ), "unknown history must not publish an empty validation target list"
    else:
        assert output.read_text().strip().endswith("=")


def test_fep_discovery_cannot_turn_a_grep_error_into_no_changes(tmp_path):
    import os
    import subprocess

    steps = workflow("validation.yml")["jobs"]["fep-validation"]["steps"]
    step = next(s for s in steps if s.get("name") == "Find modified FEP files")
    script = "git() { return 0; }\ngrep() { return 2; }\n" + step["run"]
    observed = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        cwd=ROOT,
        env={**os.environ, "BASE_BRANCH": "main", "GITHUB_OUTPUT": str(tmp_path / "output")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert observed.returncode != 0
