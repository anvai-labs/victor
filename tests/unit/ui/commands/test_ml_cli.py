# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""``victor ml`` CLI — train / validate / mine."""

import json

import pytest
from typer.testing import CliRunner

pytest.importorskip("sklearn")  # train/validate need the [ml] extra

from victor.ui.commands.ml import ml_app  # noqa: E402

runner = CliRunner()

_PASS = [
    "all parser tests pass and the fix is verified",
    "the patch is complete and tests are green",
    "verified the handler fix works correctly",
    "pytest passes and the issue is resolved",
]
_FAIL = [
    "I think the bug is probably fixed now",
    "maybe this addresses the issue hopefully",
    "attempted a fix but not sure it works",
    "the change might resolve it i believe",
]


def _separable_outcomes():
    return {
        "task_completion": [(t, "pass") for t in _PASS] * 20 + [(t, "fail") for t in _FAIL] * 20
    }


def _write_manifest(path, n=12):
    recs = []
    for i in range(n):
        reward = "pass" if i < n // 2 else "fail"
        text = "all tests pass verified" if reward == "pass" else "maybe probably not sure"
        recs.append(
            {
                "session_id": f"s{i}",
                "task_id": f"t{i}",
                "reward": reward,
                "decisions": [{"type": "task_completion", "input": {"msg": text}}],
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in recs))
    return path


def test_help_lists_commands():
    result = runner.invoke(ml_app, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.stdout
    assert "validate" in result.stdout
    assert "mine" in result.stdout


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #
def test_validate_no_data_exits_1(monkeypatch):
    monkeypatch.setattr("victor.ml.parity_gate.load_outcome_samples", lambda dt=None: {})
    result = runner.invoke(ml_app, ["validate"])
    assert result.exit_code == 1
    assert "no reward-labeled" in result.stdout


def test_validate_ships_on_separable(monkeypatch):
    monkeypatch.setattr(
        "victor.ml.parity_gate.load_outcome_samples",
        lambda dt=None: _separable_outcomes(),
    )
    result = runner.invoke(ml_app, ["validate", "--min-samples", "5", "--min-coverage", "0.5"])
    assert result.exit_code == 0, result.stdout
    assert "SHIP" in result.stdout


def test_validate_does_not_ship_when_poor(monkeypatch):
    # All one label -> no diversity -> exit 1 (no verdict), not a ship.
    monkeypatch.setattr(
        "victor.ml.parity_gate.load_outcome_samples",
        lambda dt=None: {"task_completion": [("only pass text", "pass")] * 40},
    )
    result = runner.invoke(ml_app, ["validate"])
    assert result.exit_code == 1
    assert "diversity" in result.stdout


# --------------------------------------------------------------------------- #
# mine
# --------------------------------------------------------------------------- #
def test_mine_writes_rows(tmp_path):
    manifest = _write_manifest(tmp_path / "manifest.jsonl", n=12)
    out = tmp_path / "rows.jsonl"
    result = runner.invoke(ml_app, ["mine", str(manifest), "-o", str(out)])
    assert result.exit_code == 0, result.stdout
    assert "Mined 12 rows" in result.stdout
    assert out.exists()
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 12
    assert all("features" in r for r in rows)


def test_mine_empty_manifest_exits_1(tmp_path):
    manifest = tmp_path / "empty.jsonl"
    manifest.write_text("")
    result = runner.invoke(ml_app, ["mine", str(manifest)])
    assert result.exit_code == 1


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def test_train_from_manifest_writes_artifact(tmp_path):
    manifest = _write_manifest(tmp_path / "manifest.jsonl", n=40)
    out = tmp_path / "model.npz"
    result = runner.invoke(ml_app, ["train", "--from-manifest", str(manifest), "-o", str(out)])
    assert result.exit_code == 0, result.stdout
    assert "Trained artifact" in result.stdout
    assert out.exists()
    # The artifact is loadable.
    from victor.ml.model import EdgeClassifierModel

    model = EdgeClassifierModel.load(str(out))
    assert "task_completion" in model.heads


def test_train_no_diversity_exits_2(tmp_path):
    # Manifest with only one reward bucket -> can't train.
    manifest = tmp_path / "manifest.jsonl"
    recs = [
        {
            "session_id": f"s{i}",
            "task_id": f"t{i}",
            "reward": "pass",
            "decisions": [{"type": "task_completion", "input": {"msg": "pass text"}}],
        }
        for i in range(20)
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in recs))
    out = tmp_path / "model.npz"
    result = runner.invoke(ml_app, ["train", "--from-manifest", str(manifest), "-o", str(out)])
    assert result.exit_code == 2
    assert not out.exists()
