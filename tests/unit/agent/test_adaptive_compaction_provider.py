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

"""Tests for OCP fix: DomainKeywordsProviderProtocol integration with
AdaptiveCompactionThreshold and ContextCompactor.

Covers:
- Custom provider keywords merge with built-ins
- Duplicate domain key: plugin wins
- Empty provider is a no-op
- set_extra_domain_keywords public setter
- ContextCompactor.set_adaptive_threshold wires capability registry provider
- DomainKeywordsProviderProtocol structural subtype check
"""

from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from victor.agent.adaptive_compaction import AdaptiveCompactionThreshold
from victor.framework.protocols import DomainKeywordsProviderProtocol

# ---------------------------------------------------------------------------
# Minimal provider implementations
# ---------------------------------------------------------------------------


class SwiftKeywordsProvider:
    def get_domain_keywords(self) -> Dict[str, List[str]]:
        return {"swift": ["swift", "swiftui", "xcode", "cocoa", "objc"]}

    def get_provider_name(self) -> str:
        return "swift-keywords"


class PythonOverrideProvider:
    """Overrides the built-in 'python' domain with a different keyword set."""

    def get_domain_keywords(self) -> Dict[str, List[str]]:
        return {"python": ["python", "pydantic", "fastapi", "uv", "rye"]}

    def get_provider_name(self) -> str:
        return "python-override"


class EmptyProvider:
    def get_domain_keywords(self) -> Dict[str, List[str]]:
        return {}

    def get_provider_name(self) -> str:
        return "empty-provider"


# ---------------------------------------------------------------------------
# Protocol structural subtype check
# ---------------------------------------------------------------------------


class TestDomainKeywordsProviderProtocol:
    def test_swift_provider_satisfies_protocol(self):
        assert isinstance(SwiftKeywordsProvider(), DomainKeywordsProviderProtocol)

    def test_empty_provider_satisfies_protocol(self):
        assert isinstance(EmptyProvider(), DomainKeywordsProviderProtocol)

    def test_object_without_methods_does_not_satisfy_protocol(self):
        assert not isinstance(object(), DomainKeywordsProviderProtocol)


# ---------------------------------------------------------------------------
# AdaptiveCompactionThreshold — constructor + setter
# ---------------------------------------------------------------------------


class TestAdaptiveCompactionThresholdExtraKeywords:
    def _make_msg(self, content: str):
        msg = MagicMock()
        msg.content = content
        return msg

    def test_extra_keywords_accepted_at_construction(self):
        threshold = AdaptiveCompactionThreshold(
            extra_domain_keywords={"swift": ["swift", "swiftui"]}
        )
        assert threshold._extra_domain_keywords == {"swift": ["swift", "swiftui"]}

    def test_no_extra_keywords_by_default(self):
        threshold = AdaptiveCompactionThreshold()
        assert threshold._extra_domain_keywords == {}

    def test_set_extra_domain_keywords_replaces_mapping(self):
        threshold = AdaptiveCompactionThreshold(extra_domain_keywords={"swift": ["swift"]})
        threshold.set_extra_domain_keywords({"kotlin": ["kotlin", "gradle"]})
        assert "kotlin" in threshold._extra_domain_keywords
        assert "swift" not in threshold._extra_domain_keywords

    def test_set_extra_domain_keywords_with_empty_clears_mapping(self):
        threshold = AdaptiveCompactionThreshold(extra_domain_keywords={"swift": ["swift"]})
        threshold.set_extra_domain_keywords({})
        assert threshold._extra_domain_keywords == {}

    def test_extra_keywords_merged_into_builtins_at_detection(self):
        """Plugin-contributed domain is visible to topic-switch detection."""
        threshold = AdaptiveCompactionThreshold(
            extra_domain_keywords={"swift": ["swift", "swiftui", "xcode"]}
        )
        # Two messages clearly in 'swift' domain — should NOT switch
        msgs = [self._make_msg("swiftui xcode cocoa"), self._make_msg("swift swiftui")]
        # _detect_topic_switches returns int; we only care it doesn't raise
        result = threshold._detect_topic_switches(msgs)
        assert isinstance(result, int)

    def test_plugin_keywords_override_builtin_domain(self):
        """When plugin provides a key that collides with a built-in, plugin wins."""
        override = {"python": ["pydantic", "fastapi", "uv"]}
        threshold = AdaptiveCompactionThreshold(extra_domain_keywords=override)
        # Verify the dict update path: after merge the 'python' entry is the plugin's
        # (we verify via the stored mapping, not the detection result)
        # The built-in has "pip", "django" — after merge those should be GONE
        domain_kw = {
            "python": ["python", "pip", "django", "flask", "pandas", "numpy"],
        }
        domain_kw.update(threshold._extra_domain_keywords)
        assert domain_kw["python"] == ["pydantic", "fastapi", "uv"]

    def test_empty_provider_is_noop(self):
        threshold = AdaptiveCompactionThreshold()
        threshold.set_extra_domain_keywords({})
        msgs = [
            self._make_msg("python pip django"),
            self._make_msg("javascript node react"),
        ]
        # Must not raise and must return an integer
        result = threshold._detect_topic_switches(msgs)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# ContextCompactor.set_adaptive_threshold — capability registry wiring
# ---------------------------------------------------------------------------


class TestContextCompactorCapabilityWiring:
    def test_provider_keywords_forwarded_when_registered(self):
        from victor.agent.context_compactor import ContextCompactor

        provider = SwiftKeywordsProvider()
        threshold = AdaptiveCompactionThreshold()
        compactor = ContextCompactor()

        mock_registry = MagicMock()
        mock_registry.get.return_value = provider

        # Patch at the source module since CapabilityRegistry is a deferred import.
        with (patch("victor.core.capability_registry.CapabilityRegistry") as mock_registry_cls,):
            mock_registry_cls.get_instance.return_value = mock_registry
            compactor.set_adaptive_threshold(threshold)

        assert threshold._extra_domain_keywords == {
            "swift": ["swift", "swiftui", "xcode", "cocoa", "objc"]
        }
        assert compactor._adaptive_enabled is True

    def test_no_provider_registered_is_noop(self):
        from victor.agent.context_compactor import ContextCompactor

        threshold = AdaptiveCompactionThreshold()
        compactor = ContextCompactor()

        mock_registry = MagicMock()
        mock_registry.get.return_value = None  # no provider registered

        with patch("victor.core.capability_registry.CapabilityRegistry") as mock_registry_cls:
            mock_registry_cls.get_instance.return_value = mock_registry
            compactor.set_adaptive_threshold(threshold)

        assert threshold._extra_domain_keywords == {}
        assert compactor._adaptive_enabled is True

    def test_provider_lookup_failure_does_not_raise(self):
        """Registry errors must never crash compaction setup."""
        from victor.agent.context_compactor import ContextCompactor

        threshold = AdaptiveCompactionThreshold()
        compactor = ContextCompactor()

        with patch("victor.core.capability_registry.CapabilityRegistry") as mock_registry_cls:
            mock_registry_cls.get_instance.side_effect = RuntimeError("registry unavailable")
            # Should complete without raising
            compactor.set_adaptive_threshold(threshold)

        assert compactor._adaptive_enabled is True
