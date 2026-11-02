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

"""Tests for BedrockProvider — _build_converse_request helper."""

from unittest.mock import MagicMock

import pytest


class TestBedrockConverseRequest:
    """Test the shared _build_converse_request helper."""

    @pytest.fixture
    def provider(self):
        from victor.providers.bedrock_provider import BedrockProvider

        return BedrockProvider.__new__(BedrockProvider)

    def test_basic_request_params(self, provider):
        """Builds correct request params without system prompt or tools."""
        result = provider._build_converse_request(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            converse_messages=[{"role": "user", "content": [{"text": "hello"}]}],
            max_tokens=100,
            temperature=0.7,
            system_prompt=None,
            tools=None,
        )
        assert result["modelId"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert result["inferenceConfig"]["maxTokens"] == 100
        assert result["inferenceConfig"]["temperature"] == 0.7
        assert "system" not in result
        assert "toolConfig" not in result

    def test_request_with_system_prompt(self, provider):
        """Includes system prompt when provided."""
        result = provider._build_converse_request(
            model="test-model",
            converse_messages=[],
            max_tokens=50,
            temperature=0.5,
            system_prompt=[{"text": "You are helpful"}],
            tools=None,
        )
        assert result["system"] == [{"text": "You are helpful"}]

    def test_request_with_tools(self, provider):
        """Includes tool config when tools are provided."""
        tool = MagicMock()
        tool.name = "search"
        tool.description = "Search the web"
        tool.parameters = {"type": "object", "properties": {}}

        result = provider._build_converse_request(
            model="test-model",
            converse_messages=[],
            max_tokens=50,
            temperature=0.5,
            system_prompt=None,
            tools=[tool],
            log_label="converse_stream",
        )
        assert "toolConfig" in result
        assert len(result["toolConfig"]["tools"]) == 1
        assert result["toolConfig"]["tools"][0]["toolSpec"]["name"] == "search"
