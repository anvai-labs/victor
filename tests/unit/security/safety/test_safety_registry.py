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

"""Tests for victor.security.safety.registry module."""

import pytest

from victor.security.safety.registry import (
    ISafetyScanner,
    SafetyRegistry,
)


class MockScanner:
    """Mock scanner for testing."""

    def __init__(self, findings: list[str] | None = None):
        self.findings = findings or []
        self.scan_called = False

    def scan(self, content: str) -> list[str]:
        self.scan_called = True
        return self.findings


class TestISafetyScanner:
    """Tests for ISafetyScanner Protocol."""

    def test_scanner_protocol_method(self):
        """ISafetyScanner should define scan method."""
        # MockScanner should be compatible with ISafetyScanner
        scanner = MockScanner(["finding1"])
        result = scanner.scan("test content")
        assert result == ["finding1"]


class TestSafetyRegistry:
    """Tests for SafetyRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return SafetyRegistry()

    def test_register_adds_scanner(self, registry):
        """register should add scanner to registry."""
        scanner = MockScanner()
        registry.register("secrets", scanner)
        # Should be able to retrieve it
        retrieved = registry.get_scanner("secrets")
        assert retrieved is scanner

    def test_register_multiple_scanners(self, registry):
        """register should allow multiple domain scanners."""
        secrets_scanner = MockScanner(["secret_found"])
        pii_scanner = MockScanner(["pii_found"])

        registry.register("secrets", secrets_scanner)
        registry.register("pii", pii_scanner)

        assert registry.get_scanner("secrets") is secrets_scanner
        assert registry.get_scanner("pii") is pii_scanner

    def test_get_scanner_returns_registered_scanner(self, registry):
        """get_scanner should return registered scanner."""
        scanner = MockScanner()
        registry.register("code_patterns", scanner)

        retrieved = registry.get_scanner("code_patterns")
        assert retrieved is scanner

    def test_get_scanner_returns_none_for_unknown(self, registry):
        """get_scanner should return None for unknown domain."""
        result = registry.get_scanner("unknown_domain")
        assert result is None

    def test_scan_all_runs_all_scanners(self, registry):
        """scan_all should run all registered scanners."""
        scanner1 = MockScanner()
        scanner2 = MockScanner()

        registry.register("secrets", scanner1)
        registry.register("pii", scanner2)

        registry.scan_all("test content")

        assert scanner1.scan_called is True
        assert scanner2.scan_called is True

    def test_scan_all_aggregates_findings(self, registry):
        """scan_all should aggregate findings from multiple scanners."""
        secrets_scanner = MockScanner(["AWS key detected", "Password found"])
        pii_scanner = MockScanner(["Email address found"])
        code_scanner = MockScanner(["Dangerous git command"])

        registry.register("secrets", secrets_scanner)
        registry.register("pii", pii_scanner)
        registry.register("code_patterns", code_scanner)

        findings = registry.scan_all("test content with various issues")

        assert len(findings) == 4
        assert "AWS key detected" in findings
        assert "Password found" in findings
        assert "Email address found" in findings
        assert "Dangerous git command" in findings

    def test_scan_all_empty_with_no_scanners(self, registry):
        """scan_all should return empty list when no scanners registered."""
        findings = registry.scan_all("some content")
        assert findings == []

    def test_scan_all_empty_with_no_findings(self, registry):
        """scan_all should return empty list when no findings."""
        scanner1 = MockScanner([])
        scanner2 = MockScanner([])

        registry.register("secrets", scanner1)
        registry.register("pii", scanner2)

        findings = registry.scan_all("clean content")
        assert findings == []

    def test_register_overwrites_existing(self, registry):
        """register should overwrite existing scanner for same domain."""
        scanner1 = MockScanner(["old_finding"])
        scanner2 = MockScanner(["new_finding"])

        registry.register("secrets", scanner1)
        registry.register("secrets", scanner2)

        retrieved = registry.get_scanner("secrets")
        assert retrieved is scanner2

    def test_list_domains(self, registry):
        """list_domains should return all registered domain names."""
        registry.register("secrets", MockScanner())
        registry.register("pii", MockScanner())
        registry.register("infrastructure", MockScanner())

        domains = registry.list_domains()
        assert set(domains) == {"secrets", "pii", "infrastructure"}

    def test_unregister_removes_scanner(self, registry):
        """unregister should remove scanner from registry."""
        scanner = MockScanner()
        registry.register("secrets", scanner)
        registry.unregister("secrets")

        assert registry.get_scanner("secrets") is None

    def test_unregister_nonexistent_is_safe(self, registry):
        """unregister should not raise for nonexistent domain."""
        # Should not raise
        registry.unregister("nonexistent")
