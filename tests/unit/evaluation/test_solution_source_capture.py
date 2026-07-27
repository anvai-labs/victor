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

"""Code-generation benchmarks need source, not a patch.

MBPP and HumanEval execute ``agent_output + test_code`` as one Python file. The
benchmark callback handed them a git diff, so line 3 of solution.py read
``@@ -0,0 +1,27 @@`` and every task died on SyntaxError before a test ran. MBPP
scored 0 on all 135 real tasks it has ever been given, across eight runs, for
this one reason.

SWE-bench is the opposite — it applies the diff to a fresh clone — so the split
is by what the runner does with the string, and only the two benchmarks whose
runners were read and confirmed are opted in.
"""

from types import SimpleNamespace

import pytest

from victor.evaluation.agent_adapter import VictorAgentAdapter

SOLUTION = "def missing_number(nums):\n    return sum(range(len(nums) + 1)) - sum(nums)\n"
A_DIFF = "diff --git a/x.py b/x.py\n@@ -0,0 +1,27 @@\n+def missing_number(nums):\n"


def adapter_with(edits, working_dir=None):
    """An adapter carrying tracked edits, without running an agent."""
    adapter = VictorAgentAdapter.__new__(VictorAgentAdapter)
    adapter._file_edits = [
        SimpleNamespace(path=path, after_content=content) for path, content in edits
    ]
    adapter.config = SimpleNamespace(working_dir=working_dir)
    return adapter


class TestCapturingSource:
    def test_it_returns_the_written_python(self):
        adapter = adapter_with([("missing_number.py", SOLUTION)])
        assert adapter._capture_solution_source() == SOLUTION

    def test_the_captured_source_is_executable(self):
        """The whole point: the runner will exec this concatenated with tests."""
        adapter = adapter_with([("missing_number.py", SOLUTION)])
        namespace: dict = {}
        exec(
            adapter._capture_solution_source() + "\nassert missing_number([0,1,3]) == 2\n",
            namespace,
        )

    def test_a_diff_would_not_be_executable(self):
        """Documents the failure this replaces."""
        with pytest.raises(SyntaxError):
            compile(A_DIFF, "solution.py", "exec")

    def test_test_files_are_excluded(self):
        """The runner appends the benchmark's own tests; ours would shadow them."""
        adapter = adapter_with(
            [("solution.py", SOLUTION), ("test_solution.py", "def test_x():\n    assert False\n")]
        )
        captured = adapter._capture_solution_source()
        assert "missing_number" in captured
        assert "assert False" not in captured

    def test_non_python_files_are_ignored(self):
        adapter = adapter_with([("notes.md", "# hello"), ("solution.py", SOLUTION)])
        assert adapter._capture_solution_source() == SOLUTION

    def test_the_last_write_to_a_path_wins(self):
        """An agent that revises a file means the revision."""
        adapter = adapter_with([("s.py", "def f():\n    return 1\n"), ("s.py", SOLUTION)])
        assert adapter._capture_solution_source() == SOLUTION

    def test_multiple_files_are_concatenated(self):
        adapter = adapter_with(
            [("a.py", "def a():\n    return 1\n"), ("b.py", "def b():\n    return 2\n")]
        )
        captured = adapter._capture_solution_source()
        assert "def a()" in captured and "def b()" in captured

    def test_no_edits_yields_empty_so_the_caller_can_fall_back(self):
        assert adapter_with([])._capture_solution_source() == ""

    def test_it_reads_from_the_workspace_when_content_was_not_tracked(self, tmp_path):
        """after_content is best-effort; the workspace still exists at capture time."""
        (tmp_path / "solution.py").write_text(SOLUTION)
        adapter = adapter_with([("solution.py", "")], working_dir=tmp_path)
        assert adapter._capture_solution_source() == SOLUTION


class TestWhichBenchmarksGetSource:
    def test_only_the_confirmed_concatenating_runners_opt_in(self):
        from victor.ui.commands.benchmark import _SOURCE_BENCHMARKS

        assert _SOURCE_BENCHMARKS == {"mbpp", "human_eval"}

    def test_swe_bench_is_not_opted_in(self):
        """It applies the diff to a fresh clone; source would be wrong for it."""
        from victor.evaluation.protocol import BenchmarkType
        from victor.ui.commands.benchmark import _SOURCE_BENCHMARKS

        assert BenchmarkType.SWE_BENCH.value not in _SOURCE_BENCHMARKS

    def test_the_slugs_match_the_real_enum(self):
        """A renamed enum value must not silently disable the fix."""
        from victor.evaluation.protocol import BenchmarkType
        from victor.ui.commands.benchmark import _SOURCE_BENCHMARKS

        known = {b.value for b in BenchmarkType}
        assert _SOURCE_BENCHMARKS <= known
