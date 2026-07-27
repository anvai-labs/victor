# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""G5 (sandhi typed-integration gap ledger): hand-list ↔ descriptor drift guard.

Victor's transport routing rests on hand-maintained lists
(``VICTOR_NATIVE_ONLY_PROVIDER_ALIASES``, ``_SANDHI_VARIANTS``,
``usage_parsing.SANDHI_SLUGS``) that must stay in lockstep with what Sandhi's
typed catalog actually knows. ``resolve_transport_class`` is billing-critical
and fail-closed, so rather than rewiring it onto live descriptor lookups, this
guard turns list maintenance into a CI-checked invariant — drift in EITHER
direction fails a test that names the exact migration to make.

Skipped when the sandhi-gateway binding is not installed (CI installs it via
the install-sandhi action).
"""

from __future__ import annotations

import json

import pytest

sg = pytest.importorskip("sandhi_gateway")

from victor.providers.sandhi_transport import (  # noqa: E402
    VICTOR_NATIVE_ONLY_PROVIDER_ALIASES,
)
from victor.providers.usage_parsing import SANDHI_SLUGS  # noqa: E402

# Local OpenAI-compatible servers ride Sandhi's default openai dispatch with a
# custom base_url and deliberately have NO own-slug descriptor. If Sandhi ever
# grows explicit descriptors for these, this guard fires so Victor adopts them
# deliberately instead of drifting.
LOCAL_OPENAI_COMPAT_NO_DESCRIPTOR = frozenset({"vllm", "lmstudio", "llamacpp"})

# Registry primaries that are not LLM chat providers routed through the
# transport resolver at all.
NON_TRANSPORT_PRIMARIES = frozenset({"cache"})


def _descriptor(slug: str):
    try:
        return json.loads(sg.provider_descriptor_json(slug))
    except Exception:
        return None


def _registry_primaries() -> set[str]:
    from victor.providers import registry as registry_module

    # Ensure defaults are registered, then enumerate unique primary names.
    registry_module.ProviderRegistry.list_providers()
    specs = registry_module._lazy_provider_specs
    return {spec.primary_name for spec in specs.values()}


class TestDescriptorDriftGuard:
    def test_sandhi_routed_primaries_have_descriptors(self):
        """Every Victor provider that must resolve to a Sandhi transport is
        known to Sandhi's typed catalog."""
        missing = []
        for primary in sorted(_registry_primaries()):
            if primary in VICTOR_NATIVE_ONLY_PROVIDER_ALIASES:
                continue
            if primary in LOCAL_OPENAI_COMPAT_NO_DESCRIPTOR:
                continue
            if primary in NON_TRANSPORT_PRIMARIES:
                continue
            if _descriptor(primary) is None:
                missing.append(primary)

        assert not missing, (
            f"Sandhi's typed catalog has no descriptor for {missing}, but these "
            "providers are not in VICTOR_NATIVE_ONLY_PROVIDER_ALIASES — either "
            "add the provider to Sandhi's catalog, or classify it native-only "
            "explicitly in victor/providers/sandhi_transport.py."
        )

    def test_native_only_aliases_are_unknown_to_sandhi(self):
        """If Sandhi learns a provider Victor still routes natively, migrate it."""
        learned = [
            alias
            for alias in sorted(VICTOR_NATIVE_ONLY_PROVIDER_ALIASES)
            if _descriptor(alias) is not None
        ]
        assert not learned, (
            f"Sandhi now has typed descriptors for {learned}, which Victor still "
            "classifies native-only. Migrate them to Sandhi transport (add a "
            "_SANDHI_VARIANTS entry / policy shell) and remove them from "
            "VICTOR_NATIVE_ONLY_PROVIDER_ALIASES."
        )

    def test_local_openai_compat_exceptions_stay_descriptorless(self):
        """The local trio deliberately rides the default openai dispatch."""
        adopted = [
            alias
            for alias in sorted(LOCAL_OPENAI_COMPAT_NO_DESCRIPTOR)
            if _descriptor(alias) is not None
        ]
        assert not adopted, (
            f"Sandhi now has explicit descriptors for local providers {adopted}. "
            "Adopt them deliberately (route by slug instead of the openai "
            "default dispatch) and update this exception set."
        )

    def test_usage_parsing_slugs_align_with_descriptors(self):
        """Per-slug usage semantics must reference slugs Sandhi can meter,
        except bedrock (parser exists, transport pending — sandhi lib.rs:72)."""
        unknown = [
            slug for slug in sorted(SANDHI_SLUGS) if slug != "bedrock" and _descriptor(slug) is None
        ]
        assert not unknown, (
            f"usage_parsing.SANDHI_SLUGS references {unknown} with no Sandhi "
            "descriptor — the per-slug usage branch is unreachable via the "
            "typed transport."
        )

    def test_descriptor_families_match_victor_routing_shape(self):
        """Core typed providers land on the endpoint family Victor codes for."""
        expectations = {
            "anthropic": "anthropic_messages",
            "openai": "openai_chat_completions",
            "google": "gemini_generate_content",
            "ollama": "ollama_chat",
        }
        mismatches = {}
        for slug, family in expectations.items():
            descriptor = _descriptor(slug)
            actual = (descriptor or {}).get("endpoint_family")
            if actual != family:
                mismatches[slug] = actual
        assert not mismatches, (
            f"Endpoint family drift vs Victor's codecs: {mismatches}. Update the "
            "Sandhi variant codecs (sandhi_transport.py) to match."
        )
