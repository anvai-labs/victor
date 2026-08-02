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

"""Evaluation artifacts must name the candidates that produced their outcomes.

`seed_from_evaluations()` turns per-task pass/fail into Pareto instance scores,
but only for artifacts that record *which* candidate was under test. Identity
used to be stamped solely by an explicit
``victor benchmark --prompt-candidate-hash`` A/B, so ordinary benchmark runs —
the overwhelming majority — produced ground truth that no candidate could be
credited with, and `agent_prompt_pareto_instance` stayed empty across thousands
of scored tasks.

Artifacts now also carry ``observed_prompt_identities``: what the runtime
actually served during the run.
"""

from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner


def _artifact(**overrides):
    payload = {
        "config": {"benchmark": "swe_bench", "model": "glm-5.2", "provider": "zai"},
        "tasks": [
            {"task_id": "astropy__astropy-1", "status": "passed"},
            {"task_id": "django__django-2", "status": "failed"},
        ],
    }
    payload.update(overrides)
    return payload


class TestArtifactIdentities:
    def test_explicit_binding_wins(self):
        payload = _artifact(
            prompt_candidate_hash="abc123",
            section_name="ASI_TOOL_EFFECTIVENESS_GUIDANCE",
            provider="zai",
            model="glm-5.2",
            observed_prompt_identities=[
                {
                    "prompt_candidate_hash": "ignored",
                    "section_name": "COMPLETION_GUIDANCE",
                    "provider": "zai",
                }
            ],
        )
        assert PromptOptimizerLearner._artifact_identities(payload) == [
            ("abc123", "ASI_TOOL_EFFECTIVENESS_GUIDANCE", "zai", "glm-5.2")
        ]

    def test_observed_identities_are_used_when_unbound(self):
        payload = _artifact(
            model="glm-5.2",
            observed_prompt_identities=[
                {
                    "prompt_candidate_hash": "deadbeef1234",
                    "prompt_section_name": "COMPLETION_GUIDANCE",
                    "provider": "moonshot",
                }
            ],
        )
        assert PromptOptimizerLearner._artifact_identities(payload) == [
            ("deadbeef1234", "COMPLETION_GUIDANCE", "moonshot", "glm-5.2")
        ]

    def test_multiple_sections_served_in_one_run(self):
        payload = _artifact(
            model="glm-5.2",
            observed_prompt_identities=[
                {"prompt_candidate_hash": "h1", "section_name": "A", "provider": "zai"},
                {"prompt_candidate_hash": "h2", "section_name": "B", "provider": "zai"},
            ],
        )
        identities = PromptOptimizerLearner._artifact_identities(payload)
        assert {(h, s) for h, s, _, _ in identities} == {("h1", "A"), ("h2", "B")}

    def test_duplicate_identities_collapse(self):
        entry = {"prompt_candidate_hash": "h1", "section_name": "A", "provider": "zai"}
        payload = _artifact(observed_prompt_identities=[entry, dict(entry)])
        assert len(PromptOptimizerLearner._artifact_identities(payload)) == 1

    def test_incomplete_entries_are_dropped(self):
        payload = _artifact(
            observed_prompt_identities=[
                {"prompt_candidate_hash": "h1"},  # no section
                {"section_name": "A"},  # no hash
                {"prompt_candidate_hash": None, "section_name": None},
                "not-a-dict",
            ]
        )
        assert PromptOptimizerLearner._artifact_identities(payload) == []

    def test_unstamped_artifact_yields_nothing(self):
        # The state of every ordinary benchmark run before this change.
        assert PromptOptimizerLearner._artifact_identities(_artifact()) == []

    def test_provider_falls_back_to_artifact_scope(self):
        payload = _artifact(
            provider="moonshot",
            model="kimi-k3",
            observed_prompt_identities=[
                {"prompt_candidate_hash": "h1", "section_name": "A"},
            ],
        )
        assert PromptOptimizerLearner._artifact_identities(payload) == [
            ("h1", "A", "moonshot", "kimi-k3")
        ]


class TestGroundTruthScoring:
    """The score must be the harness verdict, not a tool-tidiness proxy."""

    def test_pass_fail_maps_to_one_and_zero(self):
        scores = dict(
            PromptOptimizerLearner._artifact_instance_scores(_artifact(), model="glm-5.2")
        )
        assert scores["astropy__astropy-1::glm-5.2"] == 1.0
        assert scores["django__django-2::glm-5.2"] == 0.0
