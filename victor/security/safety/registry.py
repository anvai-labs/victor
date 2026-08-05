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

"""Unified safety scanner registry for cross-vertical safety scanning.

This module provides a central registry for safety scanners across all domains
(secrets, PII, code patterns, infrastructure), enabling:
- Pluggable scanner registration per domain
- Unified scan_all for comprehensive safety checks
- Extension without modification (OCP principle)

Example usage:
    from victor.security.safety.registry import SafetyRegistry, ISafetyScanner
    from victor.security.safety.secrets import SecretScanner
    from victor.security.safety.pii import PIIScanner

    # Create registry
    registry = SafetyRegistry()

    # Register domain-specific scanners
    registry.register("secrets", SecretScanner())
    registry.register("pii", PIIScanner())

    # Scan content across all domains
    findings = registry.scan_all(code_content)
    for finding in findings:
        print(f"Issue: {finding}")

    # Get specific scanner
    secret_scanner = registry.get_scanner("secrets")
    if secret_scanner:
        results = secret_scanner.scan(content)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ISafetyScanner(Protocol):
    """Protocol interface for safety scanners.

    All safety scanners must implement this interface to be registered
    with the SafetyRegistry. The scan method takes content and returns
    a list of findings (strings describing issues found).

    Example implementation:
        class CustomScanner:
            def scan(self, content: str) -> List[str]:
                findings = []
                if "dangerous_pattern" in content:
                    findings.append("Dangerous pattern detected")
                return findings
    """

    def scan(self, content: str) -> List[str]:
        """Scan content for safety issues.

        Args:
            content: The text content to scan

        Returns:
            List of finding descriptions (empty if no issues found)
        """
        ...


class SafetyRegistry:
    """Central registry for safety scanners across all domains.

    Provides a unified interface for registering and invoking safety
    scanners across different domains (secrets, PII, code patterns,
    infrastructure, etc.).

    The registry follows the Open-Closed Principle (OCP), allowing
    new scanner types to be added without modifying existing code.

    Attributes:
        _scanners: Internal dictionary mapping domain names to scanners

    Example:
        registry = SafetyRegistry()
        registry.register("secrets", MySecretScanner())
        registry.register("pii", MyPIIScanner())

        # Get all findings
        findings = registry.scan_all(code)

        # Get specific scanner
        scanner = registry.get_scanner("secrets")
    """

    def __init__(self):
        """Initialize the registry with an empty scanner dictionary."""
        self._scanners: Dict[str, ISafetyScanner] = {}

    def register(self, domain: str, scanner: ISafetyScanner) -> None:
        """Register a scanner for a specific domain.

        If a scanner is already registered for the domain, it will be
        replaced with the new scanner.

        Args:
            domain: The safety domain name (e.g., "secrets", "pii", "code_patterns")
            scanner: A scanner instance implementing ISafetyScanner protocol
        """
        self._scanners[domain] = scanner

    def unregister(self, domain: str) -> None:
        """Remove a scanner from the registry.

        This is a safe operation - if the domain doesn't exist,
        no error is raised.

        Args:
            domain: The safety domain name to unregister
        """
        self._scanners.pop(domain, None)

    def get_scanner(self, domain: str) -> Optional[ISafetyScanner]:
        """Get a registered scanner by domain name.

        Args:
            domain: The safety domain name

        Returns:
            The registered scanner, or None if not found
        """
        return self._scanners.get(domain)

    def scan_all(self, content: str) -> List[str]:
        """Run all registered scanners and aggregate findings.

        Iterates through all registered scanners, runs each one on the
        provided content, and aggregates all findings into a single list.

        Args:
            content: The text content to scan

        Returns:
            List of all findings from all scanners
        """
        all_findings: List[str] = []

        for scanner in self._scanners.values():
            findings = scanner.scan(content)
            all_findings.extend(findings)

        return all_findings

    def list_domains(self) -> List[str]:
        """List all registered domain names.

        Returns:
            List of domain names that have registered scanners
        """
        return list(self._scanners.keys())

    def has_scanner(self, domain: str) -> bool:
        """Check if a scanner is registered for a domain.

        Args:
            domain: The safety domain name

        Returns:
            True if a scanner is registered, False otherwise
        """
        return domain in self._scanners

    def scanner_count(self) -> int:
        """Get the number of registered scanners.

        Returns:
            Count of registered scanners
        """
        return len(self._scanners)


__all__ = [
    "ISafetyScanner",
    "SafetyRegistry",
]
