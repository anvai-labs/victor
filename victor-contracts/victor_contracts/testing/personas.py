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

"""Persona-registration test helpers for vertical test suites (FEP-0042).

Personas reach the host through the vertical's ``get_personas()`` hook on
activation; a vertical must not register them at import. These helpers pin
both halves of that contract::

    from victor_contracts.testing.personas import (
        assert_explicit_persona_registration,
        assert_import_registers_no_personas,
    )

    def test_no_import_time_registration():
        assert_import_registers_no_personas("my_vertical.teams.personas")

    def test_explicit_provider_registration():
        from my_vertical.teams.personas import MY_PERSONAS, _register_personas

        assert_explicit_persona_registration(_register_personas, MY_PERSONAS, "my_vertical")
"""

from __future__ import annotations

import subprocess  # nosec B404 - runs the current interpreter on a fixed script
import sys
import textwrap
import warnings
from typing import Any, Callable, Dict, List, Mapping

__all__ = [
    "RecordingPersonaRegistry",
    "assert_explicit_persona_registration",
    "assert_import_registers_no_personas",
]

_SDK_DEPRECATION = r".*(victor.contracts|register_vertical_personas|RegistrationPolicy)"

_IMPORT_PROBE = textwrap.dedent("""
    import importlib
    import sys
    import warnings

    warnings.filterwarnings(
        "error",
        message=r".*(victor.contracts|register_vertical_personas|RegistrationPolicy)",
        category=DeprecationWarning,
    )
    from victor_contracts import registries

    calls = []

    class Recorder:
        def register_persona(self, **kwargs):
            calls.append(kwargs)

    # Installed directly (not via the deprecated setter) so a module that still
    # registers through the default-registry path is caught even if it does not warn.
    registries._default_persona_registry = Recorder()
    for module in sys.argv[1:]:
        importlib.import_module(module)
    if calls:
        raise SystemExit(f"import-time persona registration: {[c['name'] for c in calls]}")
    """)


class RecordingPersonaRegistry:
    """``PersonaRegistryProtocol`` double that records each registration's kwargs."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def register_persona(self, **kwargs: Any) -> None:
        """Record one ``register_persona`` call."""
        self.calls.append(kwargs)


def assert_import_registers_no_personas(*modules: str, timeout: float = 120.0) -> None:
    """Assert that importing ``modules`` in a fresh interpreter registers no personas.

    The probe runs with ``DeprecationWarning`` as an error and a recording default
    persona registry installed, so it fails if an import calls the deprecated
    default-registry path or writes to the default registry.

    Args:
        modules: Fully qualified module names, imported in order.
        timeout: Seconds before the probe is abandoned.

    Raises:
        AssertionError: If the probe exits non-zero; the message carries its stderr.
    """
    if not modules:
        raise ValueError("pass at least one module name")
    result = subprocess.run(  # nosec B603 - fixed argv: this interpreter + module names
        [sys.executable, "-c", _IMPORT_PROBE, *modules],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"importing {', '.join(modules)} failed the persona-registration probe:\n"
            f"{result.stderr.strip()}"
        )


def assert_explicit_persona_registration(
    register: Callable[[Any], Any],
    personas: Mapping[str, Any],
    vertical: str,
) -> List[Dict[str, Any]]:
    """Assert ``register(provider)`` registers every persona for ``vertical`` silently.

    Args:
        register: The vertical's explicit-provider helper.
        personas: The vertical's persona table (``{id: persona}``).
        vertical: The expected ``vertical=`` value on every registration.

    Returns:
        The recorded ``register_persona`` kwargs, for further assertions.

    Raises:
        AssertionError: If the names or vertical differ, or a ``DeprecationWarning`` fires.
    """
    recorder = RecordingPersonaRegistry()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=_SDK_DEPRECATION, category=DeprecationWarning)
        register(recorder)
    names = {call["name"] for call in recorder.calls}
    if names != set(personas):
        raise AssertionError(f"registered {sorted(names)}, expected {sorted(personas)}")
    verticals = {call.get("vertical") for call in recorder.calls}
    if verticals != {vertical}:
        raise AssertionError(f"registered for verticals {verticals}, expected {{{vertical!r}}}")
    return recorder.calls
