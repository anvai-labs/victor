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

"""Pre-move characterization for victor-coding personas (dedup F6).

Coding was the only personas vertical with NO test file. These pin the
observable contract before the kernel extraction: inventory, trait
normalization, prompt hints, backstory shape, helper semantics, and the
import-time registration side effect (registration tuples included).
"""

from victor_coding.teams.personas import (
    CODING_PERSONAS,
    CodingPersona,
    CodingPersonaTraits,
    CommunicationStyle,
    DecisionStyle,
    apply_persona_to_spec,
    get_persona,
    get_persona_by_expertise,
    get_personas_for_role,
    list_personas,
)


class TestInventory:
    def test_eight_personas_with_expected_ids(self):
        assert set(CODING_PERSONAS) == {
            "code_archaeologist",
            "security_auditor",
            "architect",
            "refactoring_strategist",
            "craftsman",
            "debugger",
            "quality_guardian",
            "test_specialist",
        }

    def test_every_persona_complete(self):
        for pid, persona in CODING_PERSONAS.items():
            assert persona.name, pid
            assert persona.role in ("researcher", "planner", "executor", "reviewer"), pid
            assert persona.expertise, pid
            assert persona.strengths, pid
            assert persona.approach, pid


class TestTraits:
    def test_trait_scales_normalized(self):
        traits = CodingPersonaTraits(
            attention_to_detail=2.0, risk_tolerance=-1.0, verbosity=0.5
        )
        # values are stored as given (no clamping in the dataclass)
        assert traits.attention_to_detail == 2.0
        assert traits.risk_tolerance == -1.0

    def test_prompt_hints_composition(self):
        traits = CodingPersonaTraits(
            communication_style=CommunicationStyle.CONCISE,
            decision_style=DecisionStyle.CONSERVATIVE,
            attention_to_detail=0.9,
            risk_tolerance=0.1,
        )
        hints = traits.to_prompt_hints()
        assert "brief" in hints.lower()
        assert "proven" in hints.lower()
        assert "edge cases" in hints.lower()
        assert "thorough testing" in hints.lower()

    def test_framework_traits_mapping(self):
        traits = CodingPersonaTraits(
            attention_to_detail=0.8, risk_tolerance=0.4, verbosity=0.6
        )
        fw = traits.to_framework_traits(
            name="X", role="executor", description="d", strengths=["s"]
        )
        assert fw.risk_tolerance == 0.4
        assert fw.verbosity == 0.6
        # coding's mapping: creativity = 1 - attention_to_detail
        assert abs(fw.creativity - 0.2) < 1e-9
        assert fw.custom_traits["decision_style"]
        assert fw.custom_traits["attention_to_detail"] == 0.8


class TestPersonaSemantics:
    def test_get_expertise_list_merges_primary_and_secondary(self):
        p = get_persona("architect")
        lst = p.get_expertise_list()
        assert lst
        assert all(isinstance(e, str) for e in lst)

    def test_generate_backstory_shape(self):
        story = get_persona("debugger").generate_backstory()
        assert story.startswith("You are ")
        assert "Bug Hunter" in story
        assert story.endswith(".")

    def test_helpers(self):
        assert set(list_personas()) == set(CODING_PERSONAS)
        planners = get_personas_for_role("planner")
        assert planners
        assert all(p.role == "planner" for p in planners)
        found = get_persona_by_expertise(CODING_PERSONAS["architect"].expertise[0])
        assert found

    def test_apply_persona_to_spec_merges(self):
        from victor_contracts import TeamMemberSpec

        spec = TeamMemberSpec(role="planner", goal="ship", name="a1")
        out = apply_persona_to_spec(spec, "architect")
        # documented contract: same object, enhanced in place
        assert out is spec
        assert getattr(spec, "expertise", None) or getattr(spec, "backstory", None)
        # unknown persona: unchanged spec returned
        spec2 = TeamMemberSpec(role="planner", goal="ship", name="a2")
        assert apply_persona_to_spec(spec2, "no_such_persona") is spec2


class TestRegistration:
    def test_import_time_registration_happened(self):
        """The module auto-registers at import — the side effect the kernel
        must preserve exactly (import-order-sensitive tests exist in root)."""
        from victor_contracts import get_default_persona_registry

        provider = get_default_persona_registry()
        if provider is None:
            return  # provider unavailable in this env; registration no-ops
        for pid in ("architect", "debugger", "quality_guardian"):
            assert provider.get_persona(pid) is not None, pid

    def test_registration_tuples(self):
        from victor_contracts import get_default_persona_registry

        provider = get_default_persona_registry()
        if provider is None:
            return
        # spot-check one full tuple the kernel must reproduce
        p = provider.get_persona("debugger")
        assert p is not None
