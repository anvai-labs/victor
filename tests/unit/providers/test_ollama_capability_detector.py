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

"""Tests for Ollama capability detector module."""

from unittest.mock import MagicMock, patch
import httpx

from victor.providers.ollama_capability_detector import (
    TOOL_SUPPORT_PATTERNS,
    TOOL_FORMAT_PATTERNS,
    ModelToolSupport,
    OllamaCapabilityDetector,
    get_global_detector,
    get_model_tool_support,
    check_tool_support_sync,
)

# =============================================================================
# MODEL TOOL SUPPORT DATACLASS TESTS
# =============================================================================


class TestModelToolSupport:
    """Tests for ModelToolSupport dataclass."""

    def test_basic_creation(self):
        """Test basic dataclass creation."""
        support = ModelToolSupport(
            model="test-model",
            supports_tools=True,
            template_has_tools=True,
        )
        assert support.model == "test-model"
        assert support.supports_tools is True
        assert support.template_has_tools is True
        assert support.tool_response_format == "unknown"
        assert support.detection_method == "template"
        assert support.error is None

    def test_with_all_fields(self):
        """Test dataclass with all fields."""
        support = ModelToolSupport(
            model="qwen2.5-coder:14b",
            supports_tools=True,
            template_has_tools=True,
            tool_response_format="json",
            detection_method="template",
            error=None,
        )
        assert support.tool_response_format == "json"

    def test_with_error(self):
        """Test dataclass with error."""
        support = ModelToolSupport(
            model="missing-model",
            supports_tools=False,
            template_has_tools=False,
            error="Model not found",
        )
        assert support.error == "Model not found"


# =============================================================================
# PATTERN TESTS
# =============================================================================


class TestToolSupportPatterns:
    """Tests for tool support pattern detection."""

    def test_if_tools_pattern(self):
        """Test {{ if .Tools }} pattern."""
        import re

        template = "{{ if .Tools }}use tools{{ end }}"
        matched = any(re.search(p, template) for p in TOOL_SUPPORT_PATTERNS)
        assert matched is True

    def test_if_tools_with_dash_pattern(self):
        """Test {{- if .Tools }} pattern."""
        import re

        template = "{{- if .Tools }}use tools{{ end }}"
        matched = any(re.search(p, template) for p in TOOL_SUPPORT_PATTERNS)
        assert matched is True

    def test_if_or_system_tools_pattern(self):
        """Test {{ if or .System .Tools }} pattern."""
        import re

        template = "{{ if or .System .Tools }}content{{ end }}"
        matched = any(re.search(p, template) for p in TOOL_SUPPORT_PATTERNS)
        assert matched is True

    def test_range_tools_pattern(self):
        """Test {{ range .Tools }} pattern."""
        import re

        template = "{{ range .Tools }}tool: {{ . }}{{ end }}"
        matched = any(re.search(p, template) for p in TOOL_SUPPORT_PATTERNS)
        assert matched is True

    def test_range_dollar_tools_pattern(self):
        """Test {{ range $.Tools }} pattern."""
        import re

        template = "{{ range $.Tools }}tool{{ end }}"
        matched = any(re.search(p, template) for p in TOOL_SUPPORT_PATTERNS)
        assert matched is True

    def test_no_tools_pattern(self):
        """Test template without tools pattern."""
        import re

        template = "{{ .System }}{{ .Prompt }}"
        matched = any(re.search(p, template) for p in TOOL_SUPPORT_PATTERNS)
        assert matched is False


class TestToolFormatPatterns:
    """Tests for tool format pattern detection."""

    def test_xml_tool_call_pattern(self):
        """Test <tool_call> XML pattern."""
        import re

        template = "<tool_call>function</tool_call>"
        for pattern, format_name in TOOL_FORMAT_PATTERNS:
            if re.search(pattern, template):
                assert format_name == "xml"
                break

    def test_json_name_pattern(self):
        """Test JSON name pattern."""
        import re

        template = '{"name": "function", "parameters": {}}'
        format_found = None
        for pattern, format_name in TOOL_FORMAT_PATTERNS:
            if re.search(pattern, template):
                format_found = format_name
                break
        assert format_found == "json"


# =============================================================================
# OLLAMA CAPABILITY DETECTOR TESTS
# =============================================================================


class TestOllamaCapabilityDetector:
    """Tests for OllamaCapabilityDetector class."""

    def test_init_default(self):
        """Test default initialization."""
        detector = OllamaCapabilityDetector()
        assert detector.base_url == "http://localhost:11434"
        assert detector.timeout == 30
        assert len(detector._cache) == 0

    def test_init_with_custom_url(self):
        """Test initialization with custom URL."""
        detector = OllamaCapabilityDetector(base_url="http://192.168.1.100:11434", timeout=60)
        assert detector.base_url == "http://192.168.1.100:11434"
        assert detector.timeout == 60

    def test_init_strips_trailing_slash(self):
        """Test URL trailing slash is stripped."""
        detector = OllamaCapabilityDetector(base_url="http://localhost:11434/")
        assert detector.base_url == "http://localhost:11434"

    def test_detect_tool_support_empty_template(self):
        """Test detecting tool support in empty template."""
        detector = OllamaCapabilityDetector()
        result = detector.detect_tool_support("")

        assert result.supports_tools is False
        assert result.template_has_tools is False
        assert result.tool_response_format == "unknown"

    def test_detect_tool_support_with_tools(self):
        """Test detecting tool support in template with tools."""
        detector = OllamaCapabilityDetector()
        template = """
        {{ if .System }}{{ .System }}{{ end }}
        {{ if .Tools }}
        You have access to the following tools:
        {{ range .Tools }}{{ . }}{{ end }}
        {{ end }}
        {{ .Prompt }}
        """
        result = detector.detect_tool_support(template)

        assert result.supports_tools is True
        assert result.template_has_tools is True

    def test_detect_tool_support_without_tools(self):
        """Test detecting tool support in template without tools."""
        detector = OllamaCapabilityDetector()
        template = "{{ .System }}\n{{ .Prompt }}"
        result = detector.detect_tool_support(template)

        assert result.supports_tools is False
        assert result.template_has_tools is False

    def test_detect_tool_support_xml_format(self):
        """Test detecting XML tool format."""
        detector = OllamaCapabilityDetector()
        template = """
        {{ if .Tools }}
        <tool_call>{{ .ToolCall }}</tool_call>
        {{ end }}
        """
        result = detector.detect_tool_support(template)

        assert result.tool_response_format == "xml"

    def test_detect_tool_support_json_format(self):
        """Test detecting JSON tool format."""
        detector = OllamaCapabilityDetector()
        template = """
        {{ if .Tools }}
        {"name": "function", "parameters": {}}
        {{ end }}
        """
        result = detector.detect_tool_support(template)

        assert result.tool_response_format == "json"

    @patch.object(OllamaCapabilityDetector, "get_model_info")
    def test_get_tool_support_with_cache(self, mock_get_info):
        """Test get_tool_support uses cache."""
        mock_get_info.return_value = {"template": "{{ if .Tools }}{{ end }}"}

        detector = OllamaCapabilityDetector()

        # First call
        result1 = detector.get_tool_support("test-model")
        assert result1.supports_tools is True
        assert mock_get_info.call_count == 1

        # Second call should use cache
        result2 = detector.get_tool_support("test-model")
        assert result2.supports_tools is True
        assert mock_get_info.call_count == 1  # No additional call

    @patch.object(OllamaCapabilityDetector, "get_model_info")
    def test_get_tool_support_bypass_cache(self, mock_get_info):
        """Test get_tool_support with use_cache=False."""
        mock_get_info.return_value = {"template": "{{ if .Tools }}{{ end }}"}

        detector = OllamaCapabilityDetector()

        # First call
        detector.get_tool_support("test-model")

        # Second call bypassing cache
        detector.get_tool_support("test-model", use_cache=False)
        assert mock_get_info.call_count == 2

    @patch.object(OllamaCapabilityDetector, "get_model_info")
    def test_get_tool_support_error_handling(self, mock_get_info):
        """Test get_tool_support handles errors."""
        mock_get_info.return_value = None  # Simulate error

        detector = OllamaCapabilityDetector()
        result = detector.get_tool_support("missing-model")

        assert result.supports_tools is False
        assert result.error == "Failed to fetch model info"

    @patch.object(OllamaCapabilityDetector, "get_tool_support")
    def test_supports_tools_shorthand(self, mock_get_support):
        """Test supports_tools shorthand method."""
        mock_get_support.return_value = ModelToolSupport(
            model="test",
            supports_tools=True,
            template_has_tools=True,
        )

        detector = OllamaCapabilityDetector()
        result = detector.supports_tools("test-model")

        assert result is True

    @patch.object(OllamaCapabilityDetector, "get_tool_support")
    def test_get_tool_format(self, mock_get_support):
        """Test get_tool_format method."""
        mock_get_support.return_value = ModelToolSupport(
            model="test",
            supports_tools=True,
            template_has_tools=True,
            tool_response_format="json",
        )

        detector = OllamaCapabilityDetector()
        result = detector.get_tool_format("test-model")

        assert result == "json"

    def test_clear_cache(self):
        """Test clear_cache method."""
        detector = OllamaCapabilityDetector()
        detector._cache["test-model"] = ModelToolSupport(
            model="test-model",
            supports_tools=True,
            template_has_tools=True,
        )

        assert len(detector._cache) == 1

        detector.clear_cache()

        assert len(detector._cache) == 0


class TestOllamaCapabilityDetectorHTTP:
    """Tests for HTTP-related functionality."""

    @patch("httpx.Client.post")
    def test_get_model_info_success(self, mock_post):
        """Test successful model info fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "template": "{{ .System }}{{ .Prompt }}",
            "parameters": {"num_ctx": 4096},
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        detector = OllamaCapabilityDetector()
        result = detector.get_model_info("test-model")

        assert result is not None
        assert "template" in result

    @patch("httpx.Client.post")
    def test_get_model_info_error(self, mock_post):
        """Test model info fetch error handling."""
        mock_post.side_effect = httpx.HTTPError("Connection refused")

        detector = OllamaCapabilityDetector()
        result = detector.get_model_info("test-model")

        assert result is None


# =============================================================================
# GLOBAL FUNCTION TESTS
# =============================================================================


class TestGlobalFunctions:
    """Tests for module-level convenience functions."""

    @patch("victor.providers.ollama_capability_detector._global_detector", None)
    def test_get_global_detector_creates_instance(self):
        """Test get_global_detector creates instance."""
        detector = get_global_detector()
        assert isinstance(detector, OllamaCapabilityDetector)

    @patch("victor.providers.ollama_capability_detector._global_detector", None)
    def test_get_global_detector_with_custom_url(self):
        """Test get_global_detector with custom URL."""
        detector = get_global_detector("http://custom:11434")
        assert detector.base_url == "http://custom:11434"

    @patch.object(OllamaCapabilityDetector, "get_tool_support")
    def test_get_model_tool_support(self, mock_get_support):
        """Test get_model_tool_support convenience function."""
        mock_get_support.return_value = ModelToolSupport(
            model="test-model",
            supports_tools=True,
            template_has_tools=True,
        )

        result = get_model_tool_support("http://localhost:11434", "test-model")

        assert isinstance(result, ModelToolSupport)

    @patch.object(OllamaCapabilityDetector, "get_tool_support")
    def test_check_tool_support_sync(self, mock_get_support):
        """Test check_tool_support_sync convenience function."""
        mock_get_support.return_value = ModelToolSupport(
            model="test-model",
            supports_tools=True,
            template_has_tools=True,
        )

        result = check_tool_support_sync("http://localhost:11434", "test-model")

        assert result is True


# =============================================================================
# INTEGRATION-LIKE TESTS
# =============================================================================


class TestDetectorIntegration:
    """Integration-style tests for detector."""

    def test_qwen_template_detection(self):
        """Test detection with Qwen-style template."""
        detector = OllamaCapabilityDetector()
        # Simplified Qwen template pattern
        template = """
        {{- if .System }}{{ .System }}{{ end }}
        {{- if .Tools }}
        You have access to the following tools:
        {{ range .Tools }}
        - {{ .Function.Name }}: {{ .Function.Description }}
        {{ end }}
        {{ end }}
        {{ .Prompt }}
        """
        result = detector.detect_tool_support(template)

        assert result.supports_tools is True
        assert result.template_has_tools is True

    def test_llama_template_detection(self):
        """Test detection with Llama-style template."""
        detector = OllamaCapabilityDetector()
        # Llama 3.1 template pattern
        template = """
        <|begin_of_text|>
        {{- if or .System .Tools }}
        <|start_header_id|>system<|end_header_id|>
        {{ if .System }}{{ .System }}{{ end }}
        {{- if .Tools }}
        You have access to the following functions:
        {{ range $.Tools }}
        {{ json . }}
        {{ end }}
        {{ end }}
        <|eot_id|>
        {{ end }}
        """
        result = detector.detect_tool_support(template)

        assert result.supports_tools is True

    def test_basic_template_no_tools(self):
        """Test detection with basic template without tools."""
        detector = OllamaCapabilityDetector()
        template = """
        {{ if .System }}System: {{ .System }}{{ end }}
        User: {{ .Prompt }}
        Assistant:
        """
        result = detector.detect_tool_support(template)

        assert result.supports_tools is False
        assert result.template_has_tools is False


# =============================================================================
# CAPABILITIES API DETECTION TESTS
# =============================================================================


class TestCapabilitiesAPIDetection:
    """Tests for capabilities-based detection (Ollama 0.20+)."""

    @patch.object(OllamaCapabilityDetector, "get_model_info")
    def test_capabilities_api_tools_detected(self, mock_get_info):
        """Models with 'tools' in capabilities should be detected."""
        mock_get_info.return_value = {
            "capabilities": ["completion", "vision", "tools", "thinking"],
            "template": "{{ .Prompt }}",
        }
        detector = OllamaCapabilityDetector()
        result = detector.get_tool_support("gemma4:31b")

        assert result.supports_tools is True
        assert result.detection_method == "capabilities_api"
        assert result.model == "gemma4:31b"

    @patch.object(OllamaCapabilityDetector, "get_model_info")
    def test_capabilities_api_no_tools(self, mock_get_info):
        """Models without 'tools' in capabilities fall back to template."""
        mock_get_info.return_value = {
            "capabilities": ["completion"],
            "template": "{{ .Prompt }}",
        }
        detector = OllamaCapabilityDetector()
        result = detector.get_tool_support("basic-model:7b")

        assert result.supports_tools is False
        assert result.detection_method == "template"

    @patch.object(OllamaCapabilityDetector, "get_model_info")
    def test_capabilities_api_missing_field_falls_back(self, mock_get_info):
        """Old Ollama without capabilities field falls back to template."""
        mock_get_info.return_value = {
            "template": "{{ if .Tools }}tools{{ end }}{{ .Prompt }}",
        }
        detector = OllamaCapabilityDetector()
        result = detector.get_tool_support("qwen3:8b")

        assert result.supports_tools is True
        assert result.detection_method == "template"

    @patch.object(OllamaCapabilityDetector, "get_model_info")
    def test_capabilities_api_renderer_based_tools(self, mock_get_info):
        """Gemma4-style: RENDERER/PARSER, no template tools, but capabilities."""
        mock_get_info.return_value = {
            "capabilities": ["completion", "tools"],
            "template": "{{ .Prompt }}",
        }
        detector = OllamaCapabilityDetector()
        result = detector.get_tool_support("gemma4:9b")

        assert result.supports_tools is True
        assert result.detection_method == "capabilities_api"
