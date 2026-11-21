# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``victor evaluation field-data summary`` CLI — the ADR-036 evidence table."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from victor.evaluation.field_data import FieldDataRecorder
from victor.ui.commands.field_data import field_data_app

runner = CliRunner()


def _seed_store(tmp_path: Path) -> None:
    recorder = FieldDataRecorder(settings=None, store_dir=tmp_path, enabled=True)
    for i, (status, auditor) in enumerate(
        [("completed", True), ("completed", True), ("cancelled", False), ("error", True)]
    ):
        sid = f"session-{i}"
        recorder.record_session_start(
            session_id=sid,
            mode="auto",
            provider="ollama",
            model="m:1b",
            flags={"per_turn_auditor": auditor, "codebase_intent_indexing_enabled": False},
        )
        recorder.record_session_end(
            sid,
            completion_status=status,
            turn_count=2 + i,
            tool_call_count=i,
        )


# A single-command Typer app collapses to the command itself when invoked
# directly; nested under `victor evaluation` the field-data group keeps the
# `summary` subcommand (pinned by the mount test below).
def test_summary_renders_the_evidence_table(tmp_path: Path) -> None:
    _seed_store(tmp_path)
    result = runner.invoke(field_data_app, ["--store-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "per_turn_auditor" in result.output
    assert "codebase_intent_indexing_enabled" in result.output
    assert "completed" in result.output
    assert "| on |" in result.output and "| off |" in result.output


def test_summary_json_output_is_machine_readable(tmp_path: Path) -> None:
    _seed_store(tmp_path)
    result = runner.invoke(field_data_app, ["--store-dir", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["session_count"] == 4
    assert payload["status_counts"]["completed"] == 2


def test_summary_on_empty_store_exits_clean(tmp_path: Path) -> None:
    result = runner.invoke(field_data_app, ["--store-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "no field data" in result.output.lower()


def test_field_data_is_mounted_under_the_evaluation_verb_group() -> None:
    """Placement (not existence): `victor evaluation field-data summary` resolves."""
    import typer.main

    from victor.ui.cli import app

    click_app = typer.main.get_command(app)
    evaluation = click_app.commands.get("evaluation")
    assert evaluation is not None, "victor evaluation verb group missing"
    field_data = evaluation.commands.get("field-data")
    assert field_data is not None, "victor evaluation field-data group missing"
    assert "summary" in field_data.commands
