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

"""Served prompt identities must survive from the runtime into the artifact.

The producer half of eval attribution: the adapter samples the runtime's
per-turn served identities to build a per-task record, and the harness unions
those across tasks into ``observed_prompt_identities`` on the saved result.
"""

from types import SimpleNamespace

from victor.agent.services.runtime_intelligence import PromptOptimizationIdentity
from victor.evaluation.harness import _collect_observed_prompt_identities


def _identity(section, provider="zai", candidate_hash="h1"):
    return PromptOptimizationIdentity(
        provider=provider,
        prompt_candidate_hash=candidate_hash,
        section_name=section,
        prompt_section_name=section,
        strategy_name="gepa",
        source="thompson",
    )


class TestAdapterSampling:
    """A task spans many turns; identity evidence must not be last-turn-only.

    ``_last_served_prompt_identities`` is overwritten each turn, so the adapter
    samples it on every tool call and once more at payload time, and unions the
    result. The accumulator lives here rather than on the shared runtime service:
    it is an evaluation concern, and ``runtime_intelligence`` is a decomposed
    hotspot under a line-count ratchet.
    """

    def _adapter(self):
        from victor.evaluation.agent_adapter import VictorAgentAdapter

        adapter = VictorAgentAdapter.__new__(VictorAgentAdapter)
        adapter._served_prompt_identities = {}
        adapter.orchestrator = SimpleNamespace(
            runtime_intelligence=SimpleNamespace(_last_served_prompt_identities=[])
        )
        adapter.config = SimpleNamespace(prompt_binding=None)
        return adapter

    def _serve(self, adapter, *identities):
        adapter.orchestrator.runtime_intelligence._last_served_prompt_identities = list(identities)
        adapter._sample_served_prompt_identities()

    def test_accumulates_across_turns(self):
        adapter = self._adapter()
        self._serve(adapter, _identity("A"))
        self._serve(adapter, _identity("B"))
        sections = {i["section_name"] for i in adapter.get_served_prompt_identities()}
        assert sections == {"A", "B"}

    def test_repeated_turns_do_not_duplicate(self):
        adapter = self._adapter()
        for _ in range(5):
            self._serve(adapter, _identity("A"))
        assert len(adapter.get_served_prompt_identities()) == 1

    def test_distinct_candidates_for_one_section_are_both_kept(self):
        adapter = self._adapter()
        self._serve(adapter, _identity("A", candidate_hash="h1"))
        self._serve(adapter, _identity("A", candidate_hash="h2"))
        assert len(adapter.get_served_prompt_identities()) == 2

    def test_identities_without_a_hash_are_ignored(self):
        adapter = self._adapter()
        self._serve(adapter, _identity("A", candidate_hash=None))
        assert adapter.get_served_prompt_identities() == []

    def test_absent_runtime_intelligence_is_tolerated(self):
        adapter = self._adapter()
        adapter.orchestrator = SimpleNamespace()
        assert adapter.get_served_prompt_identities() == []

    def test_falls_back_to_the_explicit_binding(self):
        adapter = self._adapter()
        adapter.config = SimpleNamespace(
            prompt_binding=SimpleNamespace(
                provider="zai",
                prompt_candidate_hash="bound1",
                section_name="A",
            )
        )
        served = adapter.get_served_prompt_identities()
        assert served[0]["prompt_candidate_hash"] == "bound1"
        assert served[0]["source"] == "explicit_binding"

    def test_observed_identities_win_over_the_binding(self):
        adapter = self._adapter()
        adapter.config = SimpleNamespace(
            prompt_binding=SimpleNamespace(
                provider="zai", prompt_candidate_hash="bound1", section_name="A"
            )
        )
        self._serve(adapter, _identity("A", candidate_hash="served1"))
        served = adapter.get_served_prompt_identities()
        assert [s["prompt_candidate_hash"] for s in served] == ["served1"]

    def test_metadata_shape_matches_the_artifact_schema(self):
        adapter = self._adapter()
        self._serve(adapter, _identity("A"))
        entry = adapter.get_served_prompt_identities()[0]
        assert entry["prompt_candidate_hash"] == "h1"
        assert entry["section_name"] == "A"
        assert entry["prompt_section_name"] == "A"
        assert entry["provider"] == "zai"


class TestHarnessAggregation:
    def _result(self, *per_task_identities):
        tasks = [
            SimpleNamespace(metadata={"prompt_identities": list(identities)})
            for identities in per_task_identities
        ]
        return SimpleNamespace(task_results=tasks)

    def test_unions_across_tasks(self):
        result = self._result(
            [{"prompt_candidate_hash": "h1", "section_name": "A", "provider": "zai"}],
            [{"prompt_candidate_hash": "h2", "section_name": "B", "provider": "zai"}],
        )
        observed = _collect_observed_prompt_identities(result)
        assert {o["prompt_candidate_hash"] for o in observed} == {"h1", "h2"}

    def test_same_identity_across_tasks_appears_once(self):
        entry = {"prompt_candidate_hash": "h1", "section_name": "A", "provider": "zai"}
        result = self._result([entry], [dict(entry)], [dict(entry)])
        assert len(_collect_observed_prompt_identities(result)) == 1

    def test_entries_without_a_hash_are_dropped(self):
        result = self._result(
            [{"prompt_candidate_hash": None, "section_name": "A"}, "not-a-dict"],
        )
        assert _collect_observed_prompt_identities(result) == []

    def test_tasks_without_metadata_are_tolerated(self):
        result = SimpleNamespace(
            task_results=[SimpleNamespace(metadata=None), SimpleNamespace(metadata={})]
        )
        assert _collect_observed_prompt_identities(result) == []

    def test_empty_run_yields_empty_list(self):
        assert _collect_observed_prompt_identities(SimpleNamespace(task_results=[])) == []
