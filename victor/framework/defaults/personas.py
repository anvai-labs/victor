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

"""Generic persona helper functions for verticals.

Eliminates the 5 identical helper functions (``get_persona``,
``get_personas_for_role``, ``get_persona_by_expertise``,
``apply_persona_to_spec``, ``list_personas``) duplicated across the
coding, devops, research, and data-analysis verticals.

Usage::

    from victor.framework.defaults import PersonaHelpers

    helpers = PersonaHelpers(CODING_PERSONAS)
    persona = helpers.get_persona("architect")
    roles   = helpers.get_personas_for_role("reviewer")
"""

from __future__ import annotations

from typing import Any, Dict, Generic, List, Optional, TypeVar

P = TypeVar("P")
"""TypeVar for persona objects (e.g. ``CodingPersona``, ``DevOpsPersona``)."""


class PersonaHelpers(Generic[P]):
    """Generic persona helper functions.

    Wraps a name-to-persona dictionary and provides the 5 standard
    lookup/mutation methods that every vertical needs.

    Type parameter ``P`` is the concrete persona dataclass used by the
    vertical (must have ``role``, ``expertise``, ``secondary_expertise``,
    ``traits``, and ``generate_backstory()``).
    """

    def __init__(self, personas: Dict[str, P]) -> None:
        self._personas = personas

    def get_persona(self, name: str) -> Optional[P]:
        """Get a persona by name.

        Args:
            name: Persona name (e.g. ``'code_archaeologist'``).

        Returns:
            Persona object if found, ``None`` otherwise.
        """
        return self._personas.get(name)

    def get_personas_for_role(self, role: str) -> List[P]:
        """Get all personas matching a specific role.

        Args:
            role: Role name (``researcher``, ``planner``, ``executor``,
                ``reviewer``).

        Returns:
            List of personas whose ``.role`` matches *role*.
        """
        return [p for p in self._personas.values() if getattr(p, "role", None) == role]

    def get_persona_by_expertise(self, expertise: Any) -> List[P]:
        """Get personas that have a specific expertise.

        Checks both primary ``.expertise`` and ``.secondary_expertise``.

        Args:
            expertise: Expertise category to search for.

        Returns:
            List of personas with that expertise.
        """
        results: List[P] = []
        for p in self._personas.values():
            primary = getattr(p, "expertise", [])
            secondary = getattr(p, "secondary_expertise", [])
            if expertise in primary or expertise in secondary:
                results.append(p)
        return results

    def apply_persona_to_spec(self, spec: Any, persona_name: str) -> Any:
        """Apply persona attributes to a TeamMemberSpec.

        Enhances the spec with the persona's expertise, personality traits,
        and generated backstory.  Follows the exact pattern from the coding
        vertical's ``apply_persona_to_spec``.

        Args:
            spec: ``TeamMemberSpec`` (or compatible) to enhance.
            persona_name: Name of persona to apply.

        Returns:
            The same *spec* object, modified in place.
        """
        persona = self.get_persona(persona_name)
        if persona is None:
            return spec

        # Merge expertise
        expertise_list = getattr(persona, "get_expertise_list", None)
        persona_expertise = expertise_list() if callable(expertise_list) else []
        if not getattr(spec, "expertise", None):
            spec.expertise = persona_expertise
        else:
            existing = set(spec.expertise)
            for e in persona_expertise:
                if e not in existing:
                    spec.expertise.append(e)

        # Generate backstory
        traits = getattr(persona, "traits", None)
        generate_backstory = getattr(persona, "generate_backstory", None)
        if not getattr(spec, "backstory", None):
            if callable(generate_backstory):
                spec.backstory = generate_backstory()
        else:
            if traits is not None:
                to_prompt_hints = getattr(traits, "to_prompt_hints", None)
                if callable(to_prompt_hints):
                    trait_hints = to_prompt_hints()
                    if trait_hints:
                        spec.backstory = f"{spec.backstory} {trait_hints}"

        # Set personality from traits
        if not getattr(spec, "personality", None) and traits is not None:
            comm = getattr(traits, "communication_style", None)
            dec = getattr(traits, "decision_style", None)
            if comm is not None and dec is not None:
                comm_val = comm.value if hasattr(comm, "value") else str(comm)
                dec_val = dec.value if hasattr(dec, "value") else str(dec)
                spec.personality = f"{comm_val} and {dec_val}"

        return spec

    def list_personas(self) -> List[str]:
        """List all available persona names.

        Returns:
            List of persona name strings.
        """
        return list(self._personas.keys())


__all__ = [
    "PersonaHelpers",
]
