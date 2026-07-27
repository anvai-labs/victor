# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Argument-shape normalization at the tool-call parse boundary.

Regression coverage for session codingagent-363cca81 (2026-07-24): glm-5.2
emitted a ``write`` tool call whose ``arguments`` was a JSON list. The
untouched list crashed the dict-assuming status-message helper before
execution-time validation could feed a corrective error back to the model,
killing the entire streaming turn.

The parse boundary must guarantee ``arguments`` is a dict on every surviving
tool call: single-element ``[dict]`` is unwrapped (forgiveness), and any other
non-object shape is marked ``_invalid`` with a corrective ``_error`` so the
model can re-issue the call.
"""

from unittest import mock

from victor.agent.services.tool_service import ToolService, ToolServiceConfig


def _make_service() -> ToolService:
    service = ToolService(
        config=ToolServiceConfig(enable_caching=False),
        tool_selector=mock.Mock(),
        tool_executor=mock.Mock(),
        tool_registrar=mock.Mock(),
    )
    # Parse-boundary tests need name resolution only; treat every tool as enabled.
    service.is_tool_enabled = lambda name: True  # type: ignore[method-assign]
    service.resolve_tool_alias = lambda name: name  # type: ignore[method-assign]
    return service


def _parse(service: ToolService, tool_calls):
    parsed, _content = service.parse_and_validate_tool_calls(
        tool_calls, full_content="", tool_adapter=mock.Mock()
    )
    return parsed


class TestArgumentShapeNormalization:
    def test_dict_arguments_pass_through(self):
        service = _make_service()
        parsed = _parse(service, [{"name": "write", "arguments": {"path": "a.py"}}])
        assert parsed is not None
        assert parsed[0]["arguments"] == {"path": "a.py"}
        assert "_invalid" not in parsed[0]

    def test_single_element_list_of_dict_is_unwrapped(self):
        service = _make_service()
        parsed = _parse(
            service,
            [{"name": "write", "arguments": [{"path": "a.py", "content": "x"}]}],
        )
        assert parsed is not None
        assert parsed[0]["arguments"] == {"path": "a.py", "content": "x"}
        assert "_invalid" not in parsed[0]

    def test_multi_element_list_marked_invalid_with_corrective_error(self):
        service = _make_service()
        parsed = _parse(
            service,
            [{"name": "write", "arguments": [{"path": "a.py"}, {"path": "b.py"}]}],
        )
        assert parsed is not None
        tc = parsed[0]
        assert tc["_invalid"] is True
        assert "JSON object" in tc["_error"]
        assert "list" in tc["_error"]
        # Downstream dict access (status messages, telemetry) must be safe.
        assert tc["arguments"] == {}

    def test_scalar_arguments_marked_invalid(self):
        service = _make_service()
        for bad_args in (7, True, 3.5, ["a", "b"]):
            parsed = _parse(service, [{"name": "shell", "arguments": bad_args}])
            assert parsed is not None
            tc = parsed[0]
            assert tc["_invalid"] is True
            assert tc["arguments"] == {}

    def test_string_arguments_still_coerced_not_invalidated(self):
        """Existing str-coercion ladder is unchanged by the shape guard."""
        service = _make_service()
        parsed = _parse(service, [{"name": "write", "arguments": '{"path": "a.py"}'}])
        assert parsed is not None
        assert parsed[0]["arguments"] == {"path": "a.py"}
        assert "_invalid" not in parsed[0]

    def test_none_arguments_become_empty_dict(self):
        service = _make_service()
        parsed = _parse(service, [{"name": "read", "arguments": None}])
        assert parsed is not None
        assert parsed[0]["arguments"] == {}
        assert "_invalid" not in parsed[0]
