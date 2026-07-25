# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""The eval corpus mixes fixtures, no-op runs, and real evidence.

On a live machine 88% of ``~/.victor/evaluations`` is test fixtures and a
further 12% recorded no tool call at all, leaving ~1% that reflects an agent
actually working. Counting them together is what made the prompt-evolution
audit's numbers hard to read. Classification must not mistake one for another,
and archiving must stay reversible.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "prune_eval_corpus.py"
_spec = importlib.util.spec_from_file_location("prune_eval_corpus", _MODULE_PATH)
prune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prune)


def _write(directory: Path, name: str, payload) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload))
    return path


class TestClassify:
    def test_real_model_with_tool_calls_is_signal(self):
        kind, _ = prune.classify({"model": "glm-5.2", "tasks": [{"task_id": "t", "tool_calls": 7}]})
        assert kind == prune.SIGNAL

    @pytest.mark.parametrize("model", ["test", "test-model", "", None])
    def test_fixture_models_are_synthetic(self, model):
        kind, _ = prune.classify({"model": model, "tasks": [{"task_id": "t", "tool_calls": 9}]})
        assert kind == prune.SYNTHETIC

    def test_model_from_config_is_honoured(self):
        kind, _ = prune.classify(
            {"config": {"model": "test-model"}, "tasks": [{"task_id": "t", "tool_calls": 3}]}
        )
        assert kind == prune.SYNTHETIC

    def test_real_model_without_any_tool_call_is_inert(self):
        kind, reason = prune.classify(
            {"model": "glm-5.2", "tasks": [{"task_id": "a", "tool_calls": 0}]}
        )
        assert kind == prune.INERT
        assert "none with a tool call" in reason

    def test_one_active_task_among_many_still_counts_as_signal(self):
        kind, _ = prune.classify(
            {
                "model": "glm-5.2",
                "tasks": [{"task_id": "a", "tool_calls": 0}, {"task_id": "b", "tool_calls": 4}],
            }
        )
        assert kind == prune.SIGNAL

    def test_session_shaped_artifact_with_inline_task_is_read(self):
        # Session-truth artifacts carry the task inline instead of a list.
        kind, _ = prune.classify({"model": "glm-5.2", "task_id": "t", "tool_calls": 5})
        assert kind == prune.SIGNAL

    def test_no_tasks_is_inert(self):
        assert prune.classify({"model": "glm-5.2"})[0] == prune.INERT

    def test_unreadable_payload(self):
        assert prune.classify(None)[0] == prune.UNREADABLE


class TestArchiveRoundTrip:
    def _corpus(self, tmp_path):
        _write(tmp_path, "eval_a.json", {"model": "test", "tasks": [{"tool_calls": 1}]})
        _write(tmp_path, "eval_b.json", {"model": "glm-5.2", "tasks": [{"tool_calls": 0}]})
        _write(tmp_path, "eval_c.json", {"model": "glm-5.2", "tasks": [{"tool_calls": 6}]})
        return tmp_path

    def _args(self, tmp_path, **kw):
        defaults = {"dir": tmp_path, "apply": False, "include_unreadable": False}
        defaults.update(kw)
        return type("Args", (), defaults)()

    def test_dry_run_moves_nothing(self, tmp_path, capsys):
        self._corpus(tmp_path)
        prune.cmd_archive(self._args(tmp_path))
        assert {p.name for p in tmp_path.glob("eval_*.json")} == {
            "eval_a.json",
            "eval_b.json",
            "eval_c.json",
        }
        assert "Dry run" in capsys.readouterr().out

    def test_apply_archives_noise_and_keeps_signal(self, tmp_path):
        self._corpus(tmp_path)
        prune.cmd_archive(self._args(tmp_path, apply=True))
        assert {p.name for p in tmp_path.glob("eval_*.json")} == {"eval_c.json"}
        assert (tmp_path / "archive" / prune.SYNTHETIC / "eval_a.json").exists()
        assert (tmp_path / "archive" / prune.INERT / "eval_b.json").exists()

    def test_nothing_is_deleted(self, tmp_path):
        self._corpus(tmp_path)
        prune.cmd_archive(self._args(tmp_path, apply=True))
        assert len(list((tmp_path / "archive").glob("*/eval_*.json"))) == 2

    def test_undo_restores_everything(self, tmp_path):
        self._corpus(tmp_path)
        prune.cmd_archive(self._args(tmp_path, apply=True))
        prune.cmd_undo(self._args(tmp_path, apply=True))
        assert {p.name for p in tmp_path.glob("eval_*.json")} == {
            "eval_a.json",
            "eval_b.json",
            "eval_c.json",
        }

    def test_archived_files_are_not_reclassified_on_a_second_pass(self, tmp_path):
        self._corpus(tmp_path)
        prune.cmd_archive(self._args(tmp_path, apply=True))
        prune.cmd_archive(self._args(tmp_path, apply=True))
        assert len(list((tmp_path / "archive").glob("*/eval_*.json"))) == 2

    def test_report_counts_joinable_tasks(self, tmp_path, capsys):
        _write(
            tmp_path,
            "eval_c.json",
            {
                "model": "glm-5.2",
                "tasks": [{"tool_calls": 6, "session_id": "s1"}, {"tool_calls": 2}],
            },
        )
        prune.cmd_report(self._args(tmp_path))
        out = capsys.readouterr().out
        assert "Tasks in signal artifacts: 2" in out
        assert "joinable to a trace): 1" in out
