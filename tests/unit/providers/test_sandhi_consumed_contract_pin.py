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

"""G9 (sandhi typed-integration gap ledger): consumed wire-contract pin.

Sandhi's additive-only-within-v1 policy is documented, not machine-enforced
(``ChatRequestV1::validate`` checks only the major version). This test pins the
**surface Victor's transport actually consumes** against the schemas the
installed binding serves: a field REMOVAL or RENAME that would break
``sandhi_transport.py``/``usage_parsing.py`` fails Victor CI with a named
contract diff, while additive evolution passes untouched — deliberately NOT a
hash pin, which would fire on every legal addition.

Skipped when the sandhi-gateway binding is not installed.
"""

from __future__ import annotations

import json

import pytest

sg = pytest.importorskip("sandhi_gateway")


def _schema(name: str) -> dict:
    return json.loads(sg.chat_contract_schema_json(name))


def _missing(schema: dict, expected: set[str]) -> set[str]:
    return expected - set((schema.get("properties") or {}).keys())


class TestConsumedContractPin:
    def test_wire_contract_major_version(self):
        assert sg.wire_contract_version() == "1", (
            "Sandhi bumped the wire-contract major version — coordinate the "
            "Victor transport migration before upgrading the binding."
        )

    def test_chat_request_carries_fields_victor_sends(self):
        missing = _missing(
            _schema("chat-request.v1"),
            {
                "schema_version",
                "model",
                "messages",
                "tools",
                "tool_choice",
                "temperature",
                "max_output_tokens",
                "extensions",
                "metadata",
            },
        )
        assert not missing, f"chat-request.v1 lost fields Victor sends: {missing}"

    def test_chat_response_carries_fields_victor_reads(self):
        missing = _missing(
            _schema("chat-response.v1"),
            {"schema_version", "id", "model", "output", "finish_reason", "usage", "extensions"},
        )
        assert not missing, f"chat-response.v1 lost fields Victor reads: {missing}"

    def test_stream_event_variants_victor_handles(self):
        schema = _schema("chat-stream-event.v1")
        variants = schema.get("oneOf") or schema.get("anyOf") or []
        tags = set()
        for variant in variants:
            event = (variant.get("properties") or {}).get("event", {})
            tags.update(event.get("enum") or [])
        consumed = {
            "response_start",
            "text_delta",
            "tool_call_start",
            "tool_call_arguments_delta",
            "tool_call_end",
            "usage",
            "finish",
            "error",
        }
        missing = consumed - tags
        assert not missing, (
            f"chat-stream-event.v1 lost event variants Victor's stream loop "
            f"handles: {missing} (served: {sorted(tags)})"
        )

    def test_usage_v2_carries_fields_victor_maps(self):
        """usage_dict_from_neutral + reasoning accounting read these."""
        missing = _missing(
            _schema("usage.v2"),
            {
                "tokens_in",
                "tokens_out",
                "cache_creation_tokens",
                "cache_read_tokens",
                "reasoning_tokens",
                "completeness",
                "attempts",
                "outcome",
                "upstream_request_id",
            },
        )
        assert not missing, f"usage.v2 lost fields Victor maps: {missing}"

    def test_provider_descriptor_carries_fields_victor_routes_on(self):
        """The G5 drift guard + G6 capability discovery read these."""
        missing = _missing(
            _schema("provider-descriptor.v1"),
            {"schema_version", "slug", "aliases", "endpoint_family", "capabilities", "models"},
        )
        assert not missing, f"provider-descriptor.v1 lost fields Victor routes on: {missing}"
