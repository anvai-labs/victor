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

"""Tests for vertical import path resolution helpers."""

from __future__ import annotations

import warnings
from types import ModuleType
from unittest.mock import patch

from victor.core.verticals.import_resolver import (
    _WARNED_LEGACY_RESOLUTIONS,
    import_module_with_fallback,
    module_import_candidates,
    normalize_vertical_name,
    vertical_module_candidates,
    vertical_runtime_module_candidates,
)


def test_vertical_module_candidates_prefers_external_package() -> None:
    """External package path should be first, with legacy/contrib fallbacks."""
    candidates = vertical_module_candidates("research", "escape_hatches")

    assert candidates[0] == "victor_research.escape_hatches"
    assert "victor.research.escape_hatches" in candidates
    assert "victor.verticals.contrib.research.escape_hatches" in candidates


def test_vertical_module_candidates_handles_dataanalysis_alias() -> None:
    """Historical data-analysis spellings should map to victor_dataanalysis."""
    candidates = vertical_module_candidates("data-analysis", "capabilities")
    assert candidates[0] == "victor_dataanalysis.capabilities"


def test_vertical_runtime_module_candidates_prefer_runtime_modules() -> None:
    """Runtime-owned module families should prefer the runtime package layout."""
    candidates = vertical_runtime_module_candidates("research", "capabilities")

    assert candidates[:6] == [
        "victor_research.runtime.capabilities",
        "victor_research.capabilities",
        "victor.research.runtime.capabilities",
        "victor.research.capabilities",
        "victor.verticals.contrib.research.runtime.capabilities",
        "victor.verticals.contrib.research.capabilities",
    ]


def test_vertical_runtime_module_candidates_keep_definition_modules_rooted() -> None:
    """Definition-layer modules should keep the root-only lookup order."""
    assert vertical_runtime_module_candidates("research", "prompts") == vertical_module_candidates(
        "research",
        "prompts",
    )


def test_normalize_vertical_name_maps_data_analysis_aliases() -> None:
    """Data analysis aliases should normalize to canonical dataanalysis."""
    assert normalize_vertical_name("data-analysis") == "dataanalysis"
    assert normalize_vertical_name("data_analysis") == "dataanalysis"
    assert normalize_vertical_name("dataanalysis") == "dataanalysis"


def test_module_import_candidates_for_legacy_path() -> None:
    """Legacy victor.<vertical> paths should expand to external-first candidates."""
    candidates = module_import_candidates("victor.research.escape_hatches")
    assert candidates[:3] == [
        "victor_research.escape_hatches",
        "victor.research.escape_hatches",
        "victor.verticals.contrib.research.escape_hatches",
    ]


def test_module_import_candidates_for_external_path() -> None:
    """External paths should retain themselves first with backward fallbacks."""
    candidates = module_import_candidates("victor_research.escape_hatches")
    assert candidates[:3] == [
        "victor_research.escape_hatches",
        "victor.research.escape_hatches",
        "victor.verticals.contrib.research.escape_hatches",
    ]


def test_module_import_candidates_normalize_data_analysis_legacy_path() -> None:
    """Legacy data_analysis path should normalize to dataanalysis fallback modules."""
    candidates = module_import_candidates("victor.data_analysis.capabilities")
    assert candidates[:3] == [
        "victor_dataanalysis.capabilities",
        "victor.dataanalysis.capabilities",
        "victor.verticals.contrib.dataanalysis.capabilities",
    ]


def test_import_module_with_fallback_returns_first_importable_candidate(caplog) -> None:
    """Importer should skip missing candidates and return the first importable one."""
    legacy_module = ModuleType("victor.research.escape_hatches")
    _WARNED_LEGACY_RESOLUTIONS.clear()

    def _fake_import(module_name: str):
        if module_name == "victor_research.escape_hatches":
            raise ImportError("external package missing")
        if module_name == "victor.research.escape_hatches":
            return legacy_module
        raise ImportError("missing")

    with patch("importlib.import_module", side_effect=_fake_import):
        module, resolved = import_module_with_fallback("victor.research.escape_hatches")

    assert module is legacy_module
    assert resolved == "victor.research.escape_hatches"
    assert any("deprecated legacy package path" in r.message for r in caplog.records)


def test_import_module_with_fallback_warns_for_contrib_resolution(caplog) -> None:
    """Importer should log warning when the contrib fallback is the resolved module."""
    contrib_module = ModuleType("victor.verticals.contrib.research.escape_hatches")
    _WARNED_LEGACY_RESOLUTIONS.clear()

    def _fake_import(module_name: str):
        if module_name == "victor.verticals.contrib.research.escape_hatches":
            return contrib_module
        raise ImportError("missing")

    with patch("importlib.import_module", side_effect=_fake_import):
        module, resolved = import_module_with_fallback("victor.research.escape_hatches")

    assert module is contrib_module
    assert resolved == "victor.verticals.contrib.research.escape_hatches"
    assert any("deprecated contrib fallback" in r.message for r in caplog.records)


def test_import_module_with_fallback_does_not_warn_for_external_resolution() -> None:
    """Canonical external package resolution should stay warning-free."""
    external_module = ModuleType("victor_research.escape_hatches")
    _WARNED_LEGACY_RESOLUTIONS.clear()

    with patch("importlib.import_module", return_value=external_module):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            module, resolved = import_module_with_fallback("victor.research.escape_hatches")

    assert module is external_module
    assert resolved == "victor_research.escape_hatches"
    assert not recorded


def test_import_module_with_fallback_warns_once_per_resolved_legacy_module(
    caplog,
) -> None:
    """Repeated legacy resolution should not spam log warnings."""
    legacy_module = ModuleType("victor.research.escape_hatches")
    _WARNED_LEGACY_RESOLUTIONS.clear()

    def _fake_import(module_name: str):
        if module_name == "victor_research.escape_hatches":
            raise ImportError("external package missing")
        if module_name == "victor.research.escape_hatches":
            return legacy_module
        raise ImportError("missing")

    with patch("importlib.import_module", side_effect=_fake_import):
        import_module_with_fallback("victor.research.escape_hatches")
        import_module_with_fallback("victor.research.escape_hatches")

    legacy_warnings = [r for r in caplog.records if "deprecated" in r.message]
    assert len(legacy_warnings) == 1


def test_import_module_with_fallback_returns_none_when_all_candidates_fail() -> None:
    """Importer should return (None, None) when no candidate is importable."""
    with patch("importlib.import_module", side_effect=ImportError("missing")):
        module, resolved = import_module_with_fallback("victor.research.escape_hatches")

    assert module is None
    assert resolved is None
