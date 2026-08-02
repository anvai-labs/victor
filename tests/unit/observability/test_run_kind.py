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

"""Run kind is recorded by whoever starts the run, never inferred afterwards.

Consumers need to tell a benchmark run from a delegate worker from a person
typing. That used to be inferred from prompt text, and the inference was wrong:
the turn-budget notice ("WARNING: N turns remaining out of 10") comes from the
shared agentic loop, so delegate work counted as benchmark runs and overstated
the eval share of trace evidence roughly twofold. No prompt string can fix that
— the prompt genuinely is shared.
"""

import asyncio
import json

import pytest

from victor.observability.run_kind import (
    RUN_KIND_ENV,
    RunKind,
    current_run_kind,
    run_kind_scope,
    tagged_run,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(RUN_KIND_ENV, raising=False)
    monkeypatch.delenv("VICTOR_HEADLESS_MODE", raising=False)


class TestCurrentRunKind:
    def test_defaults_to_interactive(self):
        assert current_run_kind() == RunKind.INTERACTIVE

    def test_headless_flag_is_honoured(self, monkeypatch):
        monkeypatch.setenv("VICTOR_HEADLESS_MODE", "true")
        assert current_run_kind() == RunKind.HEADLESS

    def test_env_seed_lets_a_subprocess_runner_tag_its_children(self, monkeypatch):
        monkeypatch.setenv(RUN_KIND_ENV, "evaluation")
        assert current_run_kind() == RunKind.EVALUATION

    def test_env_seed_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv(RUN_KIND_ENV, "  DELEGATE  ")
        assert current_run_kind() == RunKind.DELEGATE

    def test_unrecognized_env_value_is_ignored(self, monkeypatch):
        monkeypatch.setenv(RUN_KIND_ENV, "banana")
        assert current_run_kind() == RunKind.INTERACTIVE


class TestRunKindScope:
    def test_scope_applies_and_restores(self):
        assert current_run_kind() == RunKind.INTERACTIVE
        with run_kind_scope(RunKind.EVALUATION):
            assert current_run_kind() == RunKind.EVALUATION
        assert current_run_kind() == RunKind.INTERACTIVE

    def test_innermost_scope_wins_and_the_outer_resumes(self):
        # A delegate spawned inside an evaluation is delegate work while it runs.
        with run_kind_scope(RunKind.EVALUATION):
            with run_kind_scope(RunKind.DELEGATE):
                assert current_run_kind() == RunKind.DELEGATE
            assert current_run_kind() == RunKind.EVALUATION

    def test_restores_on_exception(self):
        with pytest.raises(RuntimeError):
            with run_kind_scope(RunKind.EVALUATION):
                raise RuntimeError("boom")
        assert current_run_kind() == RunKind.INTERACTIVE

    def test_scope_overrides_the_env_seed(self, monkeypatch):
        monkeypatch.setenv(RUN_KIND_ENV, "headless")
        with run_kind_scope(RunKind.EVALUATION):
            assert current_run_kind() == RunKind.EVALUATION

    def test_unknown_kind_is_rejected_loudly(self):
        with pytest.raises(ValueError, match="Unknown run kind"):
            with run_kind_scope("benchmarkish"):
                pass

    def test_concurrent_tasks_do_not_clobber_each_other(self):
        # A ContextVar, not a global: two coroutines interleaving must each keep
        # their own kind, or a delegate would retag the evaluation around it.
        seen = {}

        async def worker(name, kind, delay):
            with run_kind_scope(kind):
                await asyncio.sleep(delay)
                seen[name] = current_run_kind()

        async def main():
            await asyncio.gather(
                worker("a", RunKind.EVALUATION, 0.02),
                worker("b", RunKind.DELEGATE, 0.01),
            )

        asyncio.run(main())
        assert seen == {"a": RunKind.EVALUATION, "b": RunKind.DELEGATE}


class TestTaggedRun:
    def test_decorator_tags_the_whole_call(self):
        @tagged_run(RunKind.DELEGATE)
        async def work():
            return current_run_kind()

        assert asyncio.run(work()) == RunKind.DELEGATE
        assert current_run_kind() == RunKind.INTERACTIVE

    def test_decorator_restores_on_exception(self):
        @tagged_run(RunKind.DELEGATE)
        async def work():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            asyncio.run(work())
        assert current_run_kind() == RunKind.INTERACTIVE

    def test_decorator_preserves_identity(self):
        @tagged_run(RunKind.DELEGATE)
        async def work():
            """Docstring survives."""

        assert work.__name__ == "work"
        assert work.__doc__ == "Docstring survives."


class TestEmission:
    """The tag has to reach the log line, beside session_id."""

    def _logger(self, tmp_path):
        from victor.observability.analytics.enhanced_logger import EnhancedUsageLogger

        return EnhancedUsageLogger(log_file=tmp_path / "usage.jsonl", enabled=True)

    def _events(self, tmp_path):
        return [
            json.loads(line)
            for line in (tmp_path / "usage.jsonl").read_text().splitlines()
            if line.strip()
        ]

    def test_event_carries_the_current_run_kind(self, tmp_path):
        logger = self._logger(tmp_path)
        with run_kind_scope(RunKind.EVALUATION):
            logger.log_event("tool_result", {"tool_name": "read", "success": True})
        event = self._events(tmp_path)[0]
        assert event["run_kind"] == RunKind.EVALUATION

    def test_run_kind_sits_beside_session_id_not_inside_data(self, tmp_path):
        logger = self._logger(tmp_path)
        logger.log_event("user_prompt", {"content": "hi"})
        event = self._events(tmp_path)[0]
        assert "run_kind" in event
        assert "run_kind" not in event["data"]

    def test_untagged_process_still_emits_a_kind(self, tmp_path):
        logger = self._logger(tmp_path)
        logger.log_event("user_prompt", {"content": "hi"})
        assert self._events(tmp_path)[0]["run_kind"] == RunKind.INTERACTIVE

    def test_kind_changes_with_the_scope_across_events(self, tmp_path):
        logger = self._logger(tmp_path)
        logger.log_event("user_prompt", {"content": "before"})
        with run_kind_scope(RunKind.DELEGATE):
            logger.log_event("tool_result", {"success": True})
        logger.log_event("user_prompt", {"content": "after"})
        assert [e["run_kind"] for e in self._events(tmp_path)] == [
            RunKind.INTERACTIVE,
            RunKind.DELEGATE,
            RunKind.INTERACTIVE,
        ]
