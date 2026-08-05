# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Benchmark target resolution — graceful degradation when victor-coding is absent."""

import pytest
import typer

from victor.ui.commands import benchmark as benchmark_cmd


@pytest.mark.parametrize("name", ["swe-bench", "swe-bench-lite"])
def test_resolve_benchmark_target_missing_victor_coding_exits_cleanly(monkeypatch, name):
    """SWE-bench/HumanEval/MBPP/browser runners were extracted to victor-coding and
    soft-load to None when that optional package is missing. The CLI must exit(1) with
    an install hint instead of crashing with NoneType (direct symbol) or TypeError
    (lambda-wrapped variant)."""
    import victor.evaluation.benchmarks as benchmarks_mod

    monkeypatch.setattr(benchmarks_mod, "SWEBenchRunner", None, raising=False)

    with pytest.raises(typer.Exit) as exc_info:
        benchmark_cmd._resolve_benchmark_target(name, None)

    assert exc_info.value.exit_code == 1
