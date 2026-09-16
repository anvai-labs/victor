"""Tests for the local adversarial-review pre-push gate."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml

_SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "adversarial_review.py"
_SPEC = importlib.util.spec_from_file_location("adversarial_review", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
review_gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(review_gate)

ReviewGateError = review_gate.ReviewGateError
record_review = review_gate.record_review
verify_review = review_gate.verify_review
commit_from_environment = review_gate._commit_from_environment


def test_only_attestation_hook_runs_at_pre_push():
    config_path = Path(__file__).parents[3] / ".pre-commit-config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    pre_push_hooks = [
        hook["id"]
        for repository in config["repos"]
        for hook in repository["hooks"]
        if "pre-push" in hook.get("stages", [])
    ]

    assert pre_push_hooks == ["adversarial-review"]


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "review-gate@example.invalid")
    _git(repo, "config", "user.name", "Review Gate Test")
    (repo / "tracked.txt").write_text("first\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "test: initial commit")
    return repo


def _record(repository: Path) -> Path:
    return record_review(
        repository,
        commit_ref="HEAD",
        base_ref="HEAD",
        reviewer="independent-session",
        summary="No open findings after negative-path review",
    )


def test_missing_attestation_fails_closed(repository: Path):
    with pytest.raises(ReviewGateError, match="No clean adversarial review"):
        verify_review(repository, "HEAD")


def test_exact_reviewed_commit_passes(repository: Path):
    path = _record(repository)

    payload = verify_review(repository, "HEAD")

    assert path.name == f"{_git(repository, 'rev-parse', 'HEAD')}.json"
    assert payload["verdict"] == "clean"
    assert payload["reviewer"] == "independent-session"


def test_unwritable_attestation_location_reports_gate_error(
    repository: Path, monkeypatch: pytest.MonkeyPatch
):
    blocked_parent = repository / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(
        review_gate,
        "_attestation_path",
        lambda _repo, commit_sha: blocked_parent / f"{commit_sha}.json",
    )

    with pytest.raises(ReviewGateError, match="Could not write local review attestation"):
        _record(repository)


def test_follow_up_commit_invalidates_review(repository: Path):
    _record(repository)
    (repository / "tracked.txt").write_text("second\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "test: change reviewed content")

    with pytest.raises(ReviewGateError, match="No clean adversarial review"):
        verify_review(repository, "HEAD")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("verdict", "changes_requested"),
        ("commit_sha", "0" * 40),
        ("tree_sha", "0" * 40),
        ("base_sha", "not-a-sha"),
        ("reviewer", ""),
    ],
)
def test_tampered_attestation_fails(repository: Path, field: str, value: str):
    path = _record(repository)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReviewGateError):
        verify_review(repository, "HEAD")


def test_pre_push_environment_uses_exact_local_sha(
    repository: Path, monkeypatch: pytest.MonkeyPatch
):
    head = _git(repository, "rev-parse", "HEAD")
    monkeypatch.setenv("PRE_COMMIT_REMOTE_BRANCH", "refs/heads/feature/reviewed")
    monkeypatch.setenv("PRE_COMMIT_TO_REF", head)

    assert commit_from_environment(repository) == head


@pytest.mark.parametrize(
    ("remote_branch", "to_ref"),
    [
        ("refs/tags/v1.0.0", "1" * 40),
        ("refs/heads/feature/deleted", "0" * 40),
    ],
)
def test_pre_push_environment_skips_non_branch_commits(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    remote_branch: str,
    to_ref: str,
):
    monkeypatch.setenv("PRE_COMMIT_REMOTE_BRANCH", remote_branch)
    monkeypatch.setenv("PRE_COMMIT_TO_REF", to_ref)

    assert commit_from_environment(repository) is None
