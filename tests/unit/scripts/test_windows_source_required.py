"""Windows qualification selection is explicit and fails closed on invalid diffs."""

import importlib.util
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def selector():
    path = Path(__file__).resolve().parents[3] / "scripts/ci/windows_source_required.py"
    spec = importlib.util.spec_from_file_location("windows_source_required", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name",
    [
        "victor/core/graph_rag/source_reader.py",
        "victor/core/graph_rag/windows_source.py",
        "tests/unit/core/graph_rag/test_windows_source.py",
        "scripts/ci/gate_results.py",
        ".github/workflows/ci-fast.yml",
        "pyproject.toml",
        "victor-contracts/pyproject.toml",
        "scripts/ci/windows-source-coverage.ini",
    ],
)
def test_source_and_gate_changes_require_windows(selector, name):
    assert selector.paths_require_windows([name])


def test_unrelated_docs_skip_only_with_a_known_diff(selector):
    assert not selector.paths_require_windows(["docs/guides/chat.md"])
    assert not selector.paths_require_windows([])


@pytest.mark.parametrize(
    "event,base", [("pull_request", ""), ("push", "unknown"), ("schedule", "")]
)
def test_unknown_or_missing_diff_context_fails_closed(selector, event, base):
    with pytest.raises(ValueError):
        selector.required(event=event, base=base)


def test_manual_and_initial_push_require_qualification(selector):
    assert selector.required(event="workflow_dispatch", base="")
    assert selector.required(event="push", base="0" * 40)


def test_git_diff_failure_never_becomes_a_skip(selector, tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        selector.required(event="pull_request", base="1" * 40, cwd=tmp_path)


def test_real_rename_and_source_deletion_remain_selected(selector, tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    folder = tmp_path / "victor/core/graph_rag"
    folder.mkdir(parents=True)
    (folder / "source.py").write_text("source")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        check=True,
    )
    base = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    (folder / "source.py").rename(tmp_path / "moved.txt")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "rename",
        ],
        check=True,
    )
    assert selector.required(event="pull_request", base=base, cwd=tmp_path)


def test_nonlinear_push_source_deletion_is_not_hidden_by_merge_base(selector, tmp_path):
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(tmp_path), *arguments], text=True).strip()

    def commit():
        git("add", "-A")
        git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        )

    git("init", "-q")
    (tmp_path / "docs.txt").write_text("baseline")
    commit()
    ancestor = git("rev-parse", "HEAD")
    folder = tmp_path / "victor/core/graph_rag"
    folder.mkdir(parents=True)
    (folder / "windows_source.py").write_text("source")
    commit()
    before = git("rev-parse", "HEAD")
    git("checkout", "-qb", "alternate", ancestor)
    (tmp_path / "docs.txt").write_text("documentation only on the alternate branch")
    commit()
    assert selector.required(event="push", base=before, cwd=tmp_path)
