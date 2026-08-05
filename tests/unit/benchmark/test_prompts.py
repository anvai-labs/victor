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

"""Tests for benchmark prompt contributions."""

from victor.agent.prompt_section_texts import (
    GROUNDING_RULES as CANONICAL_GROUNDING_RULES,
    GROUNDING_RULES_EXTENDED as CANONICAL_GROUNDING_RULES_EXTENDED,
)
from victor.benchmark.prompts import (
    BENCHMARK_EVALUATION_EXTENDED_CONTEXT,
    BENCHMARK_EVALUATION_GROUNDING_CONTEXT,
    BENCHMARK_GROUNDING_EXTENDED,
    BENCHMARK_GROUNDING_RULES,
    BenchmarkPromptContributor,
)


def test_benchmark_grounding_rules_compose_canonical_baseline() -> None:
    assert BENCHMARK_GROUNDING_RULES.startswith(CANONICAL_GROUNDING_RULES)
    assert BENCHMARK_EVALUATION_GROUNDING_CONTEXT in BENCHMARK_GROUNDING_RULES


def test_benchmark_extended_grounding_rules_compose_canonical_baseline() -> None:
    assert BENCHMARK_GROUNDING_EXTENDED.startswith(CANONICAL_GROUNDING_RULES_EXTENDED)
    assert BENCHMARK_EVALUATION_EXTENDED_CONTEXT in BENCHMARK_GROUNDING_EXTENDED
    assert "POOR EVALUATION SCORES" in BENCHMARK_GROUNDING_EXTENDED


def test_benchmark_prompt_contributor_uses_minimal_grounding_by_default() -> None:
    contributor = BenchmarkPromptContributor()

    assert contributor.get_grounding_rules() == BENCHMARK_GROUNDING_RULES


def test_benchmark_prompt_contributor_can_use_extended_grounding() -> None:
    contributor = BenchmarkPromptContributor(use_extended_grounding=True)

    assert contributor.get_grounding_rules() == BENCHMARK_GROUNDING_EXTENDED
