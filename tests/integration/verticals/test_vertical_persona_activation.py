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

"""Every first-party vertical's personas reach the flat PersonaRegistry on activation.

Personas are delivered only through the activation hook (FEP-0042): the host
calls each vertical's ``get_personas()`` and registers the result. Lives under
tests/integration/verticals/ because it needs the verticals installed.
"""

import importlib
import warnings
from unittest.mock import MagicMock

import pytest

from victor.core.verticals.adapters import ensure_runtime_vertical

PERSONA_VERTICALS = [
    ("victor_coding", "CodingAssistant", "coding"),
    ("victor_devops", "DevOpsAssistant", "devops"),
    ("victor_research", "ResearchAssistant", "research"),
    ("victor_dataanalysis", "DataAnalysisAssistant", "dataanalysis"),
]


@pytest.fixture(autouse=True)
def _fresh_persona_registry():
    from victor.framework.persona_registry import reset_persona_registry

    reset_persona_registry()
    yield
    reset_persona_registry()


def _runtime_vertical(module: str, cls: str):
    # ci-fast runs changed test files without the verticals installed; the
    # verticals-integration job installs (and verifies) every first-party vertical.
    package = pytest.importorskip(module)
    return ensure_runtime_vertical(getattr(package, cls))


@pytest.mark.parametrize("module,cls,_name", PERSONA_VERTICALS)
def test_vertical_exposes_personas_through_activation_hook(module, cls, _name) -> None:
    personas = _runtime_vertical(module, cls).get_personas()
    assert personas, f"{cls}.get_personas() returned no personas"


@pytest.mark.parametrize("module,cls,name", PERSONA_VERTICALS)
def test_activation_registers_personas_in_flat_registry(module, cls, name) -> None:
    from victor.framework.framework_integration_registry_service import (
        FrameworkIntegrationRegistryService,
    )
    from victor.framework.persona_registry import get_persona_registry
    from victor.framework.step_handlers import FrameworkStepHandler

    vertical = _runtime_vertical(module, cls)
    expected = set(vertical.get_personas())
    service = FrameworkIntegrationRegistryService()
    orchestrator = MagicMock()
    result = MagicMock()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "victor.framework.step_handlers.resolve_framework_integration_registry_service",
            lambda _orchestrator: service,
        )
        FrameworkStepHandler().apply_personas(orchestrator, vertical, MagicMock(), result)

    result.add_warning.assert_not_called()
    registered = set(get_persona_registry().list_personas(vertical=vertical.name))
    assert registered == {f"{vertical.name}:{pid}" for pid in expected}
    assert vertical.name == name


@pytest.mark.parametrize("module,_cls,_name", PERSONA_VERTICALS)
def test_teams_package_import_is_warning_free(module, _cls, _name) -> None:
    pytest.importorskip(module)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        importlib.import_module(f"{module}.teams")
