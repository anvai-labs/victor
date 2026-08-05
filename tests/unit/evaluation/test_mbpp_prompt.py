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

"""The agent cannot match a function name it cannot see.

MBPP's task ``text`` is prose; the function name lives only in ``test_list``,
which the runner appends *after* the agent has written its solution. Asking the
model to infer names like ``remove_Occ`` or ``multiples_of_num`` from prose is
benchmark noise, not a prompt-quality signal: an n=60 develop run produced 38
``NameError`` failures (identifier mismatch), and the "match test identifiers"
prompt candidate was net-negative precisely because the identifiers are hidden
from the agent at solution time.

The harness — not the prompt — must reveal the expected name (HumanEval hands
the agent the full signature). Then residual failures become logic/spec issues
that prompt evolution can actually act on, instead of name-inference luck.
"""

from __future__ import annotations

from victor.evaluation.benchmarks.swe_bench import MBPPRunner


class TestMBPPPromptRevealsTheFunctionName:
    def _runner(self) -> MBPPRunner:
        return MBPPRunner()

    def _item(self, *, text: str = "an mbpp task", test_list: list[str] | None = None) -> dict:
        return {"text": text, "test_list": test_list or []}

    def test_expected_function_name_is_parsed_from_the_first_assertion(self):
        item = self._item(test_list=['assert remove_Occ("hello world", 2, "l") == "heo world"'])
        assert self._runner()._expected_function_name(item) == "remove_Occ"

    def test_build_prompt_tells_the_agent_the_exact_function_name(self):
        item = self._item(
            text="Remove the nth occurrence of a character.",
            test_list=['assert remove_Occ("hello", 2, "l") == "heo"'],
        )
        prompt = self._runner()._build_prompt(item)

        assert "remove_Occ" in prompt, "the unguessable name must be visible to the agent"

    def test_build_prompt_still_carries_the_task_text(self):
        item = self._item(text="A distinctive description.", test_list=["assert foo() == 1"])

        assert "A distinctive description." in self._runner()._build_prompt(item)

    def test_no_test_list_means_no_hint_not_a_crash(self):
        prompt = self._runner()._build_prompt(self._item(text="no tests here"))

        assert "no tests here" in prompt
        assert "exact name" not in prompt

    def test_unparseable_tests_mean_no_hint(self):
        item = self._item(text="obscure task", test_list=["this is not a callable assertion"])

        prompt = self._runner()._build_prompt(item)

        assert "obscure task" in prompt
        assert "exact name" not in prompt
