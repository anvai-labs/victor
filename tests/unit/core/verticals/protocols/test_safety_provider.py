"""victor.core.verticals.protocols.safety_provider re-exports the SDK protocol.

Round-5 dedup (PR-3): the submodule no longer re-declares
SafetyExtensionProtocol; the submodule, package and SDK names are one object,
so ``isinstance`` checks and inherited protocol defaults agree across paths.
``SafetyPattern`` stays a core dataclass re-export (the SDK spelling is
``SafetyPatternData``).
"""

from __future__ import annotations

import importlib
from typing import Dict, List

MODULE = "victor.core.verticals.protocols.safety_provider"
SDK = "victor_contracts.verticals.protocols.promoted"
PACKAGE = "victor.core.verticals.protocols"


def test_safety_extension_protocol_is_the_sdk_object() -> None:
    core = importlib.import_module(MODULE).SafetyExtensionProtocol
    assert core is importlib.import_module(SDK).SafetyExtensionProtocol
    assert core is importlib.import_module(PACKAGE).SafetyExtensionProtocol


def test_focused_safety_extensions_use_the_same_object() -> None:
    from victor.core.verticals.protocols.focused.safety_extensions import (
        SafetyExtensionProtocol as Focused,
    )
    from victor.core.verticals.protocols.safety_provider import SafetyExtensionProtocol

    assert Focused is SafetyExtensionProtocol


def test_contrib_base_extension_subclasses_the_same_object() -> None:
    from victor.contrib.safety.base_extension import BaseSafetyExtension
    from victor.core.verticals.protocols.safety_provider import SafetyExtensionProtocol

    assert issubclass(BaseSafetyExtension, SafetyExtensionProtocol)


def test_safetypattern_stays_the_core_dataclass() -> None:
    from victor.core.verticals.protocols.safety_provider import SafetyPattern
    from victor.security.safety.types import SafetyPattern as CorePattern

    assert SafetyPattern is CorePattern


def test_public_names_unchanged() -> None:
    assert sorted(importlib.import_module(MODULE).__all__) == [
        "SafetyExtensionProtocol",
        "SafetyPattern",
    ]


def test_runtime_checkable_matching_and_defaults_preserved() -> None:
    from victor.core.verticals.protocols.safety_provider import SafetyExtensionProtocol
    from victor.security.safety.types import SafetyPattern

    class _Extension:
        def get_bash_patterns(self) -> List[SafetyPattern]:
            return [SafetyPattern(pattern=r"rm\s+-rf", description="Recursive delete")]

        def get_file_patterns(self) -> List[SafetyPattern]:
            return []

        def get_tool_restrictions(self) -> Dict[str, List[str]]:
            return {"bash": ["*.env"]}

        def get_category(self) -> str:
            return "coding"

    assert isinstance(_Extension(), SafetyExtensionProtocol)
    assert not isinstance(object(), SafetyExtensionProtocol)

    # Subclassing the protocol inherits the SDK default implementations
    # (same bodies the local fork used to provide).
    class _Sub(SafetyExtensionProtocol):
        def get_bash_patterns(self) -> List[SafetyPattern]:
            return []

    sub = _Sub()
    assert sub.get_file_patterns() == []
    assert sub.get_tool_restrictions() == {}
    assert sub.get_category() == "custom"
