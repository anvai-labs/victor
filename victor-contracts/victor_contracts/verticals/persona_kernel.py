"""Persona kernel shared by every vertical's ``teams/personas.py``.

Each vertical's personas module repeats the same machinery around its own
data: the persona dataclass (identical methods), lookup helpers, and a
framework registration pass with a per-vertical policy. The machinery lives
here once; a vertical keeps only its enums, trait tables, persona table,
and a declarative :class:`RegistrationPolicy`.

Design notes (from the four-vertical characterization):
- ``to_prompt_hints`` and ``to_framework_traits`` differ ONLY in their trait
  field sets, hint tables, scale thresholds, and the framework mapping — all
  expressed declaratively via :class:`TraitSpec` (the ALGORITHM is identical).
- The four registration variants (category-by-id vs by-role, tag policies,
  preferred-tools policy) collapse into :class:`RegistrationPolicy` fields.
- Personas reach the host through the vertical's ``get_personas()`` hook, which
  the host registers on activation. :func:`register_vertical_personas` remains
  as an explicit-provider translator for hosts that register personas
  themselves. Its default-registry path (``provider=None``) and the
  ``provider_getter`` / ``auto_register`` policy fields are deprecated since
  victor-contracts 0.14.0 and removable no earlier than 0.15.0 (FEP-0042).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScaleRule:
    """A numeric-threshold prompt hint."""

    field: str
    low_below: float
    low_hint: str
    high_above: float
    high_hint: str


@dataclass(frozen=True)
class TraitSpec:
    """Declarative schema for a vertical's persona traits.

    Args:
        communication_enum / decision_enum: the vertical's enums (values are
            shared across verticals by convention).
        style_hints: communication-style value -> hint.
        decision_hints: decision-style value -> hint.
        scale_rules: numeric threshold hints over trait fields.
        defaults: trait field -> default (``communication_style`` and
            ``decision_style`` keyed by enum member NAME).
        framework_mapping: given the traits dict, returns
            ``(extra_kwargs, custom_traits)`` merged into the framework
            result — every vertical's mapping (creativity source field,
            custom_traits contents) is data.
    """

    communication_enum: type
    decision_enum: type
    style_hints: Dict[Any, str]
    decision_hints: Dict[Any, str]
    scale_rules: Tuple[ScaleRule, ...] = ()
    defaults: Dict[str, Any] = field(default_factory=dict)
    framework_mapping: Optional[
        Callable[[Dict[str, Any]], Tuple[Dict[str, Any], Dict[str, Any]]]
    ] = None


class VerticalTraits:
    """Trait bag driven by a :class:`TraitSpec` (replaces each vertical's
    hand-written traits dataclass, keeping the same public methods)."""

    spec: TraitSpec

    def __init__(self, **overrides: Any) -> None:
        base = dict(self.spec.defaults)

        # enum defaults arrive as member names
        def _resolve(enum_cls: Any, raw: str) -> Any:
            # Enum classes are subscriptable by member name; the smoke-test
            # fakes are not — fall back to getattr lookup.
            try:
                return enum_cls[raw]
            except (TypeError, KeyError):
                return getattr(enum_cls, raw, raw)

        if isinstance(base.get("communication_style"), str):
            base["communication_style"] = _resolve(
                self.spec.communication_enum, base["communication_style"]
            )
        if isinstance(base.get("decision_style"), str):
            base["decision_style"] = _resolve(self.spec.decision_enum, base["decision_style"])
        for k, v in overrides.items():
            if k not in base and k not in ("communication_style", "decision_style"):
                raise TypeError(f"unknown trait {k!r} for {type(self).__name__}")
            base[k] = v
        self.__dict__.update(base)

    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __getattr__(self, name: str) -> Any:
        # Trait fields are schema-defined (TraitSpec.defaults) — attribute
        # access is dynamic by design; __init__ populates __dict__.
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(name) from None

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"{type(self).__name__}({self.__dict__})"

    def to_prompt_hints(self) -> str:
        """The shared hint algorithm: style lookup → decision lookup → scale rules."""
        hints: List[str] = []
        hints.append(self.spec.style_hints.get(self.communication_style, ""))
        hint = self.spec.decision_hints.get(self.decision_style)
        if hint:
            hints.append(hint)
        for rule in self.spec.scale_rules:
            value = self.__dict__.get(rule.field)
            if value is None:
                continue
            if value > rule.high_above:
                hints.append(rule.high_hint)
            elif value < rule.low_below:
                hints.append(rule.low_hint)
        return " ".join(h for h in hints if h)

    def to_framework_traits(
        self,
        name: str,
        role: str,
        description: str,
        strengths: Optional[List[str]] = None,
        preferred_tools: Optional[List[str]] = None,
        framework_traits_cls: Any = None,
        framework_style_cls: Any = None,
        expertise_level: Any = None,
    ) -> Any:
        """Build a framework PersonaTraits via the spec's declarative mapping."""
        from victor_contracts import ExpertiseLevel

        if framework_traits_cls is None:
            from victor_contracts import PersonaTraits as framework_traits_cls
        assert self.spec.framework_mapping is not None
        extra, custom = self.spec.framework_mapping(self.__dict__)
        return framework_traits_cls(
            name=name,
            role=role,
            description=description,
            communication_style=(
                self.communication_style.to_framework_style()
                if hasattr(self.communication_style, "to_framework_style")
                else self.communication_style
            ),
            expertise_level=expertise_level or ExpertiseLevel.EXPERT,
            strengths=strengths or [],
            preferred_tools=preferred_tools or [],
            **extra,
            custom_traits=custom,
        )


@dataclass
class VerticalPersona:
    """Persona record with the invariant methods (replaces each vertical's
    hand-written persona dataclass; ``traits`` is a ``VerticalTraits``)."""

    name: str
    role: str
    expertise: List[Any]
    secondary_expertise: List[Any] = field(default_factory=list)
    traits: Any = None
    strengths: List[str] = field(default_factory=list)
    approach: str = ""
    communication_patterns: List[str] = field(default_factory=list)
    working_style: str = ""

    def get_expertise_list(self) -> List[str]:
        """Primary + secondary expertise as string values."""
        return [e.value for e in self.expertise + self.secondary_expertise]

    def generate_backstory(self) -> str:
        """Multi-sentence backstory from the persona attributes."""
        parts = [f"You are {self.name}, a skilled {self.role}."]
        if self.expertise:
            primary = ", ".join(e.value.replace("_", " ") for e in self.expertise[:3])
            parts.append(f"Your expertise lies in {primary}.")
        if self.strengths:
            strengths_text = "; ".join(self.strengths[:3])
            parts.append(f"Your key strengths: {strengths_text}.")
        if self.approach:
            parts.append(self.approach)
        if self.working_style:
            parts.append(self.working_style)
        trait_hints = self.traits.to_prompt_hints()
        if trait_hints:
            parts.append(trait_hints)
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary representation (mirrors the hand-written originals)."""
        return {
            "name": self.name,
            "role": self.role,
            "expertise": self.get_expertise_list(),
            "strengths": self.strengths,
            "approach": self.approach,
            "communication_style": self.traits.communication_style.value,
            "decision_style": self.traits.decision_style.value,
            "backstory": self.generate_backstory(),
        }


def make_persona_helpers(
    personas: Dict[str, VerticalPersona],
) -> Tuple[
    Callable[[str], Optional[VerticalPersona]],
    Callable[[str], List[VerticalPersona]],
    Callable[[Any], List[VerticalPersona]],
    Callable[[Any, str], Any],
    Callable[[], List[str]],
]:
    """Build the five lookup helpers over a vertical's persona table."""

    def get_persona(name: str) -> Optional[VerticalPersona]:
        return personas.get(name)

    def get_personas_for_role(role: str) -> List[VerticalPersona]:
        return [p for p in personas.values() if p.role == role]

    def get_persona_by_expertise(expertise: Any) -> List[VerticalPersona]:
        return [
            p
            for p in personas.values()
            if expertise in p.expertise or expertise in p.secondary_expertise
        ]

    def apply_persona_to_spec(spec: Any, persona_name: str) -> Any:
        persona = personas.get(persona_name)
        if persona is None:
            return spec
        if not spec.expertise:
            spec.expertise = persona.get_expertise_list()
        else:
            existing = set(spec.expertise)
            for e in persona.get_expertise_list():
                if e not in existing:
                    spec.expertise.append(e)
        if not spec.backstory:
            spec.backstory = persona.generate_backstory()
        else:
            trait_hints = persona.traits.to_prompt_hints()
            if trait_hints:
                spec.backstory = f"{spec.backstory} {trait_hints}"
        if not spec.personality:
            spec.personality = (
                f"{persona.traits.communication_style.value} and "
                f"{persona.traits.decision_style.value}"
            )
        return spec

    def list_personas() -> List[str]:
        return list(personas.keys())

    return (
        get_persona,
        get_personas_for_role,
        get_persona_by_expertise,
        apply_persona_to_spec,
        list_personas,
    )


@dataclass(frozen=True)
class RegistrationPolicy:
    """Declarative form of the four registration variants.

    - ``categories``: persona-id -> category (the coding/devops style), OR
      ``category_by_role``: role -> category (research/dataanalysis style).
    - ``tags``: explicit persona-id -> tags; ``tag_policy`` otherwise:
      "expertise_role" | "top3_expertise_role_vertical" | "expertise".
    - ``preferred_tools``: "expertise" (coding) or none.
    - ``description``: "name_role" (default) — all four use this form.
    - ``provider_getter`` / ``auto_register``: deprecated since
      victor-contracts 0.14.0 (removable no earlier than 0.15.0). Nothing reads
      them; setting either to a non-default value emits ``DeprecationWarning``.
    """

    vertical: str
    categories: Optional[Dict[str, str]] = None
    category_by_role: Optional[Dict[str, str]] = None
    tags: Optional[Dict[str, List[str]]] = None
    tag_policy: str = "expertise_role"
    preferred_tools: str = "expertise"
    provider_getter: str = "default"
    auto_register: bool = True
    version: str = "1.0.0"
    author: str = "victor"

    def __post_init__(self) -> None:
        for name, default in (("provider_getter", "default"), ("auto_register", True)):
            if getattr(self, name) != default:
                # stacklevel 3: __post_init__ <- generated __init__ <- caller.
                warnings.warn(
                    f"RegistrationPolicy.{name} is deprecated since victor-contracts "
                    "0.14.0 and will be removed no earlier than 0.15.0: it is never "
                    "read. Expose personas through the vertical's get_personas() hook "
                    "(the host registers them on activation) and drop this argument.",
                    DeprecationWarning,
                    stacklevel=3,
                )


def register_vertical_personas(
    personas: Dict[str, VerticalPersona],
    policy: RegistrationPolicy,
    provider: Any = None,
) -> Optional[int]:
    """Register a vertical's personas with ``provider`` per the policy.

    Returns the number of personas registered, or ``None`` when no provider is
    available.

    .. deprecated:: 0.14.0
        Calling without ``provider`` (falling back to the process default
        persona registry) is deprecated and removable no earlier than 0.15.0,
        when ``provider`` becomes required. Verticals expose personas through
        their ``get_personas()`` hook instead; hosts pass ``provider=``.
    """
    if provider is None:
        from victor_contracts.registries import _current_default_persona_registry

        warnings.warn(
            "register_vertical_personas() without provider= is deprecated since "
            "victor-contracts 0.14.0 and will be removed no earlier than 0.15.0 (provider "
            "becomes required): no Victor host installs a default persona registry by "
            "default, so this path normally registers nothing. Expose personas through "
            "the vertical's get_personas() hook (the host registers them on activation); "
            "a host that "
            "registers personas itself should pass provider= explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        provider = _current_default_persona_registry()
    if provider is None:
        return None

    def category_for(pid: str, persona: VerticalPersona) -> str:
        if policy.categories and pid in policy.categories:
            return policy.categories[pid]
        if policy.category_by_role and persona.role in policy.category_by_role:
            return policy.category_by_role[persona.role]
        return "other"

    def tags_for(pid: str, persona: VerticalPersona) -> List[str]:
        if policy.tags and pid in policy.tags:
            return policy.tags[pid]
        expertise = [e.value for e in persona.expertise[:3]]
        if policy.tag_policy == "top3_expertise_role_vertical":
            return expertise + [persona.role, policy.vertical]
        if policy.tag_policy == "expertise":
            return expertise
        return expertise + [persona.role]

    count = 0
    for pid, persona in personas.items():
        fw = persona.traits.to_framework_traits(
            name=persona.name,
            role=persona.role,
            description=persona.approach,
            strengths=persona.strengths,
            preferred_tools=(
                persona.get_expertise_list() if policy.preferred_tools == "expertise" else []
            ),
        )
        provider.register_persona(
            name=pid,
            version=policy.version,
            persona=fw,
            category=category_for(pid, persona),
            description=f"{persona.name} - {persona.role}",
            tags=tags_for(pid, persona),
            author=policy.author,
            vertical=policy.vertical,
        )
        count += 1
    return count
