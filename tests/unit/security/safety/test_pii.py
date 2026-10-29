# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""The runtime PII module re-exports the SDK definitions (one set of types)."""

import pytest

import victor.security.safety.pii as runtime_pii
import victor_contracts.safety.pii as sdk_pii
from victor.core.severity import SeverityLevel


@pytest.mark.parametrize("name", runtime_pii.__all__)
def test_every_runtime_name_is_the_sdk_object(name):
    assert getattr(runtime_pii, name) is getattr(sdk_pii, name)


def test_runtime_exports_match_sdk_exports():
    assert set(runtime_pii.__all__) == set(sdk_pii.__all__)


@pytest.mark.parametrize("severity", list(runtime_pii.PIISeverity))
def test_severity_maps_onto_unified_scale(severity):
    assert severity.to_severity() is SeverityLevel[severity.name]


def test_scanner_detects_and_redacts():
    matches = runtime_pii.detect_pii_in_content("ssn 123-45-6789 and jane.doe@example.com")
    found = {m.pii_type for m in matches}
    assert {runtime_pii.PIIType.SSN, runtime_pii.PIIType.EMAIL} <= found
    assert all("***" in m.matched_text for m in matches if len(m.matched_text) > 8)


def test_clean_content_has_no_pii():
    assert not runtime_pii.has_pii("the build passed on all shards")
    assert runtime_pii.detect_pii_columns(["score", "duration_ms"]) == []
