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

"""victor_sdk is a backward-compat shim for the renamed victor_contracts package."""

import sys
import warnings


def test_victor_sdk_reexports_with_deprecation():
    import victor  # noqa: F401 — triggers the contracts path bootstrap

    # Force a fresh import so the module-level deprecation warning fires.
    sys.modules.pop("victor_sdk", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import victor_sdk
        from victor_sdk import PluginContext, VictorPlugin

    assert PluginContext is not None
    assert VictorPlugin is not None
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_victor_sdk_getattr_delegates_to_contracts():
    import victor  # noqa: F401
    import victor_contracts
    import victor_sdk

    # Any public name not eagerly re-exported still resolves via __getattr__.
    assert victor_sdk.VerticalBase is victor_contracts.VerticalBase


def test_victor_sdk_submodule_imports_resolve_to_contracts():
    """Submodule imports (e.g. victor_sdk.capability_runtime) alias the same module."""
    import victor  # noqa: F401
    import importlib

    sdk_mod = importlib.import_module("victor_sdk.capability_runtime")
    contracts_mod = importlib.import_module("victor_contracts.capability_runtime")
    # Aliased to the *same* module object — no duplicate classes across old/new names.
    assert sdk_mod is contracts_mod
