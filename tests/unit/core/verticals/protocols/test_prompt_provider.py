"""victor.core.verticals.protocols.prompt_provider re-exports the SDK protocol.

Round-5 dedup (PR-3): the submodule no longer re-declares
PromptContributorProtocol; the submodule, package and SDK names are one
object, so ``isinstance`` checks and inherited protocol defaults agree across
paths. The core-only pieces (``PromptSectionContribution``,
``collect_prompt_section_contributions``, ``TaskTypeHint`` re-export) stay.
"""

from __future__ import annotations

import importlib
from typing import Dict

MODULE = "victor.core.verticals.protocols.prompt_provider"
SDK = "victor_contracts.verticals.protocols.promoted"
PACKAGE = "victor.core.verticals.protocols"


def test_prompt_contributor_protocol_is_the_sdk_object() -> None:
    core = importlib.import_module(MODULE).PromptContributorProtocol
    assert core is importlib.import_module(SDK).PromptContributorProtocol
    assert core is importlib.import_module(PACKAGE).PromptContributorProtocol


def test_focused_prompt_extensions_use_the_same_object() -> None:
    from victor.core.verticals.protocols.focused.prompt_extensions import (
        PromptContributorProtocol as Focused,
    )
    from victor.core.verticals.protocols.prompt_provider import PromptContributorProtocol

    assert Focused is PromptContributorProtocol


def test_tasktypehint_stays_the_core_dataclass() -> None:
    from victor.core.vertical_types import TaskTypeHint as CoreHint
    from victor.core.verticals.protocols.prompt_provider import TaskTypeHint

    assert TaskTypeHint is CoreHint


def test_public_names_unchanged() -> None:
    assert sorted(importlib.import_module(MODULE).__all__) == [
        "PromptContributorProtocol",
        "PromptSectionContribution",
        "TaskTypeHint",
        "collect_prompt_section_contributions",
    ]


def test_runtime_checkable_matching_and_defaults_preserved() -> None:
    from victor.core.vertical_types import TaskTypeHint
    from victor.core.verticals.protocols.prompt_provider import PromptContributorProtocol

    class _Contributor:
        def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
            return {"edit": TaskTypeHint(task_type="edit", hint="Read first.")}

        def get_system_prompt_section(self) -> str:
            return "Read before editing."

        def get_grounding_rules(self) -> str:
            return ""

        def get_priority(self) -> int:
            return 25

    assert isinstance(_Contributor(), PromptContributorProtocol)
    assert not isinstance(object(), PromptContributorProtocol)

    # Subclassing the protocol inherits the SDK default implementations
    # (same bodies the local fork used to provide).
    class _Sub(PromptContributorProtocol):
        def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
            return {}

    sub = _Sub()
    assert sub.get_system_prompt_section() == ""
    assert sub.get_grounding_rules() == ""
    assert sub.get_priority() == 50


def test_core_only_helpers_unchanged() -> None:
    from victor.core.verticals.protocols.prompt_provider import (
        PromptSectionContribution,
        collect_prompt_section_contributions,
    )

    class _Contributor:
        def get_system_prompt_section(self) -> str:
            return "  Run tests after edits.  "

        def get_priority(self) -> int:
            return 10

    contributions = collect_prompt_section_contributions(_Contributor())
    assert len(contributions) == 1
    assert isinstance(contributions[0], PromptSectionContribution)
    assert contributions[0].name == "VERTICAL_CONTRIBUTOR"
    assert contributions[0].aliases == {"vertical_contributor"}
    assert contributions[0].text == "Run tests after edits."
    assert contributions[0].priority == 10
    assert collect_prompt_section_contributions(object()) == []
