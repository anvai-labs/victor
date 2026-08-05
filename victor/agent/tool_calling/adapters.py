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

"""
Provider-specific tool calling adapters.

Each adapter implements the BaseToolCallingAdapter interface for its provider,
handling format conversion, parsing, and capability detection.

Capabilities are resolved from model_capabilities.yaml when available,
falling back to hardcoded defaults for robustness.
"""

from victor.core.json_utils import json_loads
from json import JSONDecodeError
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from victor.agent.tool_calling.base import (
    BaseToolCallingAdapter,
    FallbackParsingMixin,
    ToolCall,
    ToolCallingCapabilities,
    ToolCallFormat,
    ToolCallParseResult,
)
from victor.agent.tool_calling.capabilities import ModelCapabilityLoader
from victor.providers.base import ToolDefinition
from victor.tools.tool_names import get_canonical_name

logger = logging.getLogger(__name__)


# Capability loader instance (lazy-initialized)
class _CapabilityLoaderHolder:
    """Holder for capability loader singleton."""

    _instance: Optional[ModelCapabilityLoader] = None

    @classmethod
    def get(cls) -> ModelCapabilityLoader:
        """Get or create the capability loader singleton."""
        if cls._instance is None:
            cls._instance = ModelCapabilityLoader()
        return cls._instance


def _get_capability_loader() -> ModelCapabilityLoader:
    """Get or create the capability loader singleton."""
    return _CapabilityLoaderHolder.get()


class AnthropicToolCallingAdapter(BaseToolCallingAdapter):
    """Adapter for Anthropic Claude models.

    Anthropic uses a distinct tool format with input_schema instead of parameters.
    Tool calls are returned as content blocks with type="tool_use".
    """

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def get_capabilities(self) -> ToolCallingCapabilities:
        # Load from YAML config, use format hint for Anthropic
        loader = _get_capability_loader()
        caps = loader.get_capabilities("anthropic", self.model, ToolCallFormat.ANTHROPIC)

        # Anthropic always has native tool calls, override if YAML is missing
        if not caps.native_tool_calls:
            return ToolCallingCapabilities(
                native_tool_calls=True,
                streaming_tool_calls=True,
                parallel_tool_calls=True,
                tool_choice_param=True,
                json_fallback_parsing=False,
                xml_fallback_parsing=False,
                thinking_mode=False,
                requires_strict_prompting=False,
                tool_call_format=ToolCallFormat.ANTHROPIC,
                argument_format="json",
                recommended_max_tools=50,
                recommended_tool_budget=20,
            )
        return caps

    def convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert to Anthropic format (name, description, input_schema)."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        """Parse Anthropic tool calls.

        Anthropic returns tool calls in the raw_tool_calls list with format:
        {"id": "...", "name": "...", "arguments": {...}}
        """
        if not raw_tool_calls:
            return ToolCallParseResult(remaining_content=content)

        tool_calls = []
        warnings = []
        for tc in raw_tool_calls:
            name = tc.get("name", "")
            tc_id = tc.get("id")
            if not self.is_valid_tool_name(name):
                warnings.append(f"Skipped invalid tool name: {name}")
                # Keep with id so pipeline generates error response
                if tc_id:
                    tool_calls.append(ToolCall(name=name, arguments={}, id=tc_id))
                continue
            tool_calls.append(
                ToolCall(
                    name=name,
                    arguments=tc.get("arguments", {}),
                    id=tc_id,
                )
            )

        return ToolCallParseResult(
            tool_calls=tool_calls,
            remaining_content=content,
            parse_method="native",
            confidence=1.0,
            warnings=warnings,
        )


class OpenAIToolCallingAdapter(FallbackParsingMixin, BaseToolCallingAdapter):
    """Adapter for OpenAI GPT models and OpenAI-compatible providers.

    OpenAI uses function calling format with tool_calls containing
    function.name and function.arguments (as JSON string).

    This adapter now includes FallbackParsingMixin for handling cases where
    open-weight models on OpenAI-compatible providers (Cerebras, Groq, Together,
    etc.) output tool calls as text instead of structured JSON.

    Fallback parsing order:
    1. Native tool_calls from provider response
    2. JSON parsing from content
    3. XML parsing from content
    4. Python-style function call parsing from content
    """

    # Providers that sometimes need stronger tool calling hints
    _NEEDS_TOOL_HINTS = {"groqcloud", "fireworks", "together", "openrouter", "cerebras"}

    @property
    def provider_name(self) -> str:
        return "openai"

    def get_system_prompt_hints(self) -> str:
        """Get system prompt hints for OpenAI and OpenAI-compatible providers.

        For cloud OpenAI-compatible providers that sometimes ignore tool calls,
        provides stronger hints to encourage proper tool usage.
        """
        capabilities = self.get_capabilities()

        # Check if this is an OpenAI-compatible provider that needs hints
        # We detect this by checking the model name for common open-weight patterns
        model_lower = self.model.lower() if self.model else ""
        needs_hints = any(
            pattern in model_lower
            for pattern in ["llama", "qwen", "deepseek", "mistral", "gemma", "phi"]
        )

        if needs_hints:
            hints = [
                "IMPORTANT - TOOL USAGE REQUIRED:",
                "- You MUST use the provided tools to complete tasks.",
                "- DO NOT speculate or guess about file contents - READ them with tools.",
                "- When asked to read a file, call read() immediately.",
                "- When asked to run a command, call shell() immediately.",
                "- Only provide your answer AFTER gathering information with tools.",
                "- For multiline shell scripts, keep the tool call structured and use a heredoc inside cmd.",
                "",
                "TOOL CALL FORMAT:",
                "- When calling tools, use structured function calls in JSON format.",
                "- NEVER mix JSON tool calls with Python-style syntax in the same response.",
                '- Correct example: Call the \'read\' tool with arguments {"path": "file.py"}',
                "- Incorrect example: read(path='file.py')  # Do NOT use this format in JSON tool calls",
                "- If you need to explain tool usage, do so in natural language after the JSON tool call.",
            ]
            return "\n".join(hints)

        # For standard OpenAI models, use parallel hints if supported
        if capabilities.native_tool_calls and capabilities.parallel_tool_calls:
            return "Call MULTIPLE tools in parallel when operations are independent."

        return ""

    def get_capabilities(self) -> ToolCallingCapabilities:
        # Load from YAML config
        loader = _get_capability_loader()
        caps = loader.get_capabilities("openai", self.model, ToolCallFormat.OPENAI)

        # OpenAI always has native tool calls, override if YAML is missing
        if not caps.native_tool_calls:
            return ToolCallingCapabilities(
                native_tool_calls=True,
                streaming_tool_calls=True,
                parallel_tool_calls=True,
                tool_choice_param=True,
                json_fallback_parsing=True,  # Enable fallback for OpenAI-compat providers
                xml_fallback_parsing=True,
                thinking_mode=False,
                requires_strict_prompting=False,
                tool_call_format=ToolCallFormat.OPENAI,
                argument_format="json",
                recommended_max_tools=50,
                recommended_tool_budget=20,
            )
        return caps

    def convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert to OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        """Parse OpenAI tool calls with fallback parsing for OpenAI-compatible providers.

        OpenAI returns: {"id": "...", "name": "...", "arguments": "..."}
        where arguments is a JSON string that needs parsing.

        For OpenAI-compatible providers using open-weight models, falls back to
        content parsing when native tool calls are empty or invalid.
        """
        native_warnings: List[str] = []

        # Try native tool calls first
        if raw_tool_calls:
            tool_calls = []

            for tc in raw_tool_calls:
                name = tc.get("name", "")
                tc_id = tc.get("id")
                if not self.is_valid_tool_name(name):
                    native_warnings.append(f"Skipped invalid tool name: {name}")
                    # Keep with id so pipeline generates error response
                    # (OpenAI spec: every tool_calls[].id needs a role=tool)
                    if tc_id:
                        tool_calls.append(ToolCall(name=name, arguments={}, id=tc_id))
                    continue

                # Parse arguments (may be string or dict)
                args = tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json_loads(args)
                    except JSONDecodeError:
                        native_warnings.append(f"Failed to parse arguments for {name}")
                        args = {}

                tool_calls.append(ToolCall(name=name, arguments=args, id=tc_id))

            if tool_calls:
                return ToolCallParseResult(
                    tool_calls=tool_calls,
                    remaining_content=content,
                    parse_method="native",
                    confidence=1.0,
                    warnings=native_warnings,
                )

        # Fallback: Try parsing tool calls from content
        # This handles open-weight models that output tool calls as text
        if content:
            # Get valid tool names for Python call extraction
            valid_names = None
            if hasattr(self, "_valid_tool_names"):
                valid_names = self._valid_tool_names

            result = self.parse_from_content(
                content,
                validate_name_fn=self.is_valid_tool_name,
                valid_tool_names=valid_names,
            )
            if result.tool_calls:
                # Combine warnings from native parsing attempt
                all_warnings = native_warnings + result.warnings
                logger.debug(
                    f"OpenAI adapter: Extracted {len(result.tool_calls)} tool calls "
                    f"from content using {result.parse_method}"
                )
                return ToolCallParseResult(
                    tool_calls=result.tool_calls,
                    remaining_content=result.remaining_content,
                    parse_method=result.parse_method,
                    confidence=result.confidence,
                    warnings=all_warnings,
                )

        # Return with any native parsing warnings
        return ToolCallParseResult(remaining_content=content, warnings=native_warnings)


class OllamaToolCallingAdapter(FallbackParsingMixin, BaseToolCallingAdapter):
    """Adapter for Ollama local models.

    Ollama supports native tool calling for Llama 3.1+, Qwen 2.5+, Mistral, etc.
    Falls back to JSON/XML parsing from content for unsupported models.

    Uses FallbackParsingMixin for common parsing logic shared across adapters.

    IMPORTANT: Tool support is now detected dynamically by checking the model's
    template via Ollama's /api/show endpoint. This is more accurate than static
    lists because:
    1. Model templates can be customized to add/remove tool support
    2. Different quantizations of the same model may have different templates
    3. Custom models may have tool support not in our static list

    References:
    - https://docs.ollama.com/capabilities/tool-calling
    - https://ollama.com/search?c=tools
    """

    # Models with known good native tool calling support (fallback when dynamic detection fails)
    # Full list from https://ollama.com/search?c=tools (December 2025)
    NATIVE_TOOL_MODELS = frozenset(
        [
            # Llama family
            "llama3.1",
            "llama-3.1",
            "llama3.2",
            "llama-3.2",
            "llama3.3",
            "llama-3.3",
            "llama4",
            "llama-4",
            "llama3-groq-tool-use",
            # Qwen family
            "qwen2",
            "qwen-2",
            "qwen2.5",
            "qwen-2.5",
            "qwen3",
            "qwen-3",
            "qwq",
            # Mistral family
            "mistral",
            "mixtral",
            "mistral-small",
            "mistral-nemo",
            "mistral-large",
            "devstral",
            "magistral",
            # Command-R family
            "command-r",
            "command-r-plus",
            "command-r7b",
            "command-a",
            # Tool-specialized models
            "firefunction",
            "hermes",
            "hermes3",
            "functionary",
            "athene-v2",
            # DeepSeek (when using -tools variant)
            "deepseek",
            # Granite family (IBM)
            "granite3",
            "granite3.1",
            "granite3.2",
            "granite3.3",
            "granite4",
            # Other tool-capable models
            "gpt-oss",
            "nemotron",
            "nemotron-mini",
            "smollm2",
            "aya-expanse",
            "cogito",
            "phi4",
        ]
    )

    # Parameter aliases: maps model-specific parameter names to standard names
    # Format: {tool_name: {model_param: standard_param}}
    PARAMETER_ALIASES = {
        "read": {
            "line_start": "offset",  # gpt-oss uses line_start/line_end
            "line_end": "_line_end",  # Special handling needed
            "loc": "offset",  # browser.open style
            "num_lines": "limit",
        },
        "list_directory": {
            "dir": "path",
            "directory": "path",
        },
        "write": {
            "file": "path",
            "file_path": "path",
            "text": "content",
        },
        "edit": {
            "file": "path",
            "file_path": "path",
        },
        "shell": {
            "cmd": "command",
        },
        "search": {
            "q": "query",
            "term": "query",
        },
    }

    # Cache for dynamic tool support detection results
    _tool_support_cache: Dict[str, bool] = {}

    @property
    def provider_name(self) -> str:
        return "ollama"

    def _has_native_support(self) -> bool:
        """Check if current model has native tool calling.

        Uses dynamic detection via Ollama's /api/show endpoint first,
        falling back to static model list if detection fails.

        The detection result is cached per model to avoid repeated API calls.
        """
        # Check class-level cache first
        if self.model in OllamaToolCallingAdapter._tool_support_cache:
            return OllamaToolCallingAdapter._tool_support_cache[self.model]

        # Try dynamic detection via Ollama /api/show
        base_url = (
            self.config.get("base_url", "http://localhost:11434")
            if self.config
            else "http://localhost:11434"
        )

        try:
            from victor.providers.ollama_capability_detector import (
                OllamaCapabilityDetector,
            )

            detector = OllamaCapabilityDetector(base_url, timeout=5)
            support = detector.get_tool_support(self.model)

            if support.error is None:
                # Successfully detected from template
                logger.debug(
                    f"Dynamic tool support detection for {self.model}: "
                    f"supports_tools={support.supports_tools}, "
                    f"format={support.tool_response_format}"
                )
                OllamaToolCallingAdapter._tool_support_cache[self.model] = support.supports_tools
                return support.supports_tools
            else:
                logger.debug(f"Dynamic detection failed for {self.model}: {support.error}")
        except Exception as e:
            logger.debug(f"Failed to dynamically detect tool support for {self.model}: {e}")

        # Fallback to static list
        has_static_support = any(pattern in self.model_lower for pattern in self.NATIVE_TOOL_MODELS)
        OllamaToolCallingAdapter._tool_support_cache[self.model] = has_static_support
        return has_static_support

    def _has_thinking_mode(self) -> bool:
        """Check if model supports Qwen3 thinking mode."""
        return "qwen3" in self.model_lower or "qwen-3" in self.model_lower

    def get_capabilities(self) -> ToolCallingCapabilities:
        """Get tool calling capabilities for the current model.

        This method uses dynamic detection via Ollama's /api/show endpoint
        to determine if the model's template supports tools. The dynamic
        detection result takes precedence over YAML config because:
        1. Templates can be customized to add/remove tool support
        2. Different quantizations may have different templates
        3. Custom models may have tool support not in YAML config
        """
        # Determine format based on model detection (uses dynamic detection first)
        has_native = self._has_native_support()
        format_hint = ToolCallFormat.OLLAMA_NATIVE if has_native else ToolCallFormat.OLLAMA_JSON

        # Load from YAML config with model pattern matching
        loader = _get_capability_loader()
        caps = loader.get_capabilities("ollama", self.model, format_hint)

        # IMPORTANT: Override YAML native_tool_calls with dynamic detection result
        # Dynamic detection checks the actual template, which is more accurate
        if caps.native_tool_calls != has_native:
            logger.debug(
                f"Overriding YAML native_tool_calls={caps.native_tool_calls} "
                f"with dynamic detection result={has_native} for {self.model}"
            )
            caps = ToolCallingCapabilities(
                native_tool_calls=has_native,
                streaming_tool_calls=caps.streaming_tool_calls if has_native else False,
                parallel_tool_calls=caps.parallel_tool_calls if has_native else False,
                tool_choice_param=caps.tool_choice_param if has_native else False,
                json_fallback_parsing=True,  # Always enable fallback parsing
                xml_fallback_parsing=True,
                thinking_mode=caps.thinking_mode,
                requires_strict_prompting=not has_native,  # Strict prompting when no native support
                tool_call_format=format_hint,
                argument_format=caps.argument_format,
                recommended_max_tools=caps.recommended_max_tools if has_native else 10,
                recommended_tool_budget=(caps.recommended_tool_budget if has_native else 5),
            )

        # Apply model-specific overrides that YAML can't detect
        if self._has_thinking_mode() and not caps.thinking_mode:
            # Qwen3 thinking mode detected but not in YAML
            caps = ToolCallingCapabilities(
                native_tool_calls=caps.native_tool_calls,
                streaming_tool_calls=caps.streaming_tool_calls,
                parallel_tool_calls=caps.parallel_tool_calls,
                tool_choice_param=caps.tool_choice_param,
                json_fallback_parsing=caps.json_fallback_parsing,
                xml_fallback_parsing=caps.xml_fallback_parsing,
                thinking_mode=True,
                requires_strict_prompting=caps.requires_strict_prompting,
                tool_call_format=format_hint,
                argument_format=caps.argument_format,
                recommended_max_tools=caps.recommended_max_tools,
                recommended_tool_budget=caps.recommended_tool_budget,
            )

        return caps

    def convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert to Ollama format (OpenAI-compatible)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _parse_hybrid_xml_format(
        self,
        content: str,
        validate_name_fn: Optional[Callable[[str], bool]] = None,
    ) -> ToolCallParseResult:
        """Parse Ollama-specific hybrid XML format.

        Handles malformed patterns like <function=X>...</tool_call> that some
        Ollama models (especially Qwen3-coder) produce. This is specific to Ollama
        and not added to the base class to avoid regressions for other adapters.

        Args:
            content: Response content to parse
            validate_name_fn: Optional function to validate tool names

        Returns:
            ToolCallParseResult with parsed tool calls
        """
        import re
        import json

        # Hybrid pattern: <function=name> with </tool_call> closing
        # This is a common malformed pattern from Qwen3 models on Ollama
        hybrid_pattern = r"<function=([^>]+)>(.*?)</tool_call>"
        hybrid_matches = re.findall(hybrid_pattern, content, re.DOTALL | re.IGNORECASE)

        if not hybrid_matches:
            return ToolCallParseResult(remaining_content=content)

        tool_calls: List[ToolCall] = []
        warnings: List[str] = ["Used hybrid XML pattern recovery (Ollama-specific)"]

        for name, params_content in hybrid_matches:
            name = name.strip()

            # Validate name if validator provided
            if validate_name_fn and not validate_name_fn(name):
                warnings.append(f"Skipped invalid tool name: {name}")
                continue

            # Parse <parameter=X>value</parameter> tags
            param_pattern = r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>"
            param_matches = re.findall(param_pattern, params_content, re.DOTALL)
            args = {p.strip(): v.strip() for p, v in param_matches}

            tool_calls.append(ToolCall(name=name, arguments=args))

        if not tool_calls:
            return ToolCallParseResult(remaining_content=content)

        # Remove matched content
        remaining = re.sub(hybrid_pattern, "", content, flags=re.DOTALL | re.IGNORECASE)
        remaining = remaining.strip()

        return ToolCallParseResult(
            tool_calls=tool_calls,
            remaining_content=remaining,
            parse_method="ollama_hybrid_xml",
            confidence=0.65,  # Slightly lower confidence for recovered format
            warnings=warnings,
        )

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        """Parse Ollama tool calls with fallback support.

        Uses FallbackParsingMixin methods for common parsing logic.
        Includes Ollama-specific hybrid XML pattern recovery.
        """
        # Try native tool calls first (using mixin method)
        if raw_tool_calls:
            result = self.parse_native_tool_calls(raw_tool_calls, self.is_valid_tool_name)
            if result.tool_calls:
                # Preserve original content for reference
                return ToolCallParseResult(
                    tool_calls=result.tool_calls,
                    remaining_content=content,
                    parse_method=result.parse_method,
                    confidence=result.confidence,
                    warnings=result.warnings,
                )

        # Fallback: Try content parsing (using mixin method)
        if content:
            result = self.parse_from_content(content, self.is_valid_tool_name)
            if result.tool_calls:
                return result

            # Ollama-specific: Try hybrid XML pattern recovery
            # This handles <function=X>...</tool_call> malformed patterns
            hybrid_result = self._parse_hybrid_xml_format(content, self.is_valid_tool_name)
            if hybrid_result.tool_calls:
                return hybrid_result

        return ToolCallParseResult(remaining_content=content)

    def get_system_prompt_hints(self) -> str:
        """Get system prompt hints for Ollama models."""
        capabilities = self.get_capabilities()

        if capabilities.native_tool_calls:
            hints = [
                "TOOL USAGE:",
                "- Use ls and read to inspect code.",
                "- Call tools one at a time, waiting for results.",
                "- After 2-3 successful tool calls, provide your answer.",
                "- Do NOT make identical repeated tool calls.",
                "",
                "RESPONSE FORMAT:",
                "- Write your answer in plain, readable text.",
                "- Do NOT output raw JSON in your response.",
                "- Do NOT output XML tags or function call syntax.",
                "- Be concise and answer the question directly.",
            ]

            if capabilities.thinking_mode:
                hints.extend(
                    [
                        "",
                        "QWEN3 MODE:",
                        "- Use /no_think for simple questions.",
                        "- Provide direct answers without excessive reasoning.",
                    ]
                )

            return "\n".join(hints)
        else:
            return "\n".join(
                [
                    "CRITICAL TOOL RULES:",
                    "1. Call tools ONE AT A TIME. Never batch calls.",
                    "2. After reading 2-3 files, STOP and answer.",
                    "3. Do NOT repeat the same tool call.",
                    "4. Do NOT invent file contents.",
                    "",
                    "CRITICAL OUTPUT RULES:",
                    "1. Write your answer in plain English.",
                    '2. Do NOT output JSON objects like {"name": ...}.',
                    "3. Do NOT output XML tags like </function> or </parameter>.",
                    "4. Do NOT output function call syntax.",
                    "5. Keep your answer focused and concise.",
                ]
            )


class GoogleToolCallingAdapter(FallbackParsingMixin, BaseToolCallingAdapter):
    """Adapter for Google Gemini models.

    Google Gemini uses a different function calling format than OpenAI.
    Tool definitions use 'function_declarations' format.

    Gemini models may also output tool calls in text format like:
    - <execute_bash>command</execute_bash>
    - <tool_code>python_code</tool_code>
    - ```tool_code ... ```

    This adapter handles both native function calls and fallback text parsing.

    References:
    - https://ai.google.dev/gemini-api/docs/function-calling
    """

    # Patterns for text-based tool calls from Gemini
    EXECUTE_BASH_PATTERN = re.compile(r"<execute_bash>\s*(.*?)\s*</execute_bash>", re.DOTALL)
    TOOL_CODE_PATTERN = re.compile(r"<tool_code>\s*(.*?)\s*</tool_code>", re.DOTALL)
    CODE_BLOCK_PATTERN = re.compile(r"```tool_code\s*(.*?)\s*```", re.DOTALL)
    # Pattern for <ctrl42>call: `tool_name` followed by ```json {...}```
    CTRL42_CALL_PATTERN = re.compile(
        r"<ctrl42>call:\s*`?(\w+)`?\s*```json\s*(.*?)\s*```", re.DOTALL
    )
    # Pattern for simpler call: `tool_name` followed by json block
    SIMPLE_CALL_PATTERN = re.compile(
        r"(?:call|Call):\s*`?(\w+)`?\s*```json\s*(.*?)\s*```", re.DOTALL
    )
    # Pattern for Python-style function call: tool_name("value") or tool_name(path="value")
    # Handles both positional and keyword arguments
    PYTHON_CALL_PATTERN = re.compile(
        r"(?:^|\n)\s*(?:print\s*\(\s*)?(\w+)\s*\(\s*"
        r"(?:(?:path|command|file_path|query)\s*=\s*)?"  # Optional keyword
        r"['\"]([^'\"]+)['\"]\s*"
        r"\)?",
        re.MULTILINE,
    )
    # Pattern to detect hallucinated tool output (Gemini likes to fake results)
    TOOL_OUTPUT_HALLUCINATION = re.compile(
        r"<TOOL_OUTPUT>.*?</TOOL_OUTPUT>|```text\n.*?```", re.DOTALL
    )

    @property
    def provider_name(self) -> str:
        return "google"

    def get_capabilities(self) -> ToolCallingCapabilities:
        """Get capabilities for Google Gemini models."""
        loader = _get_capability_loader()
        caps = loader.get_capabilities("google", self.model, ToolCallFormat.GOOGLE)

        # Gemini always has native tool calls when tools are provided
        if not caps.native_tool_calls:
            return ToolCallingCapabilities(
                native_tool_calls=True,
                streaming_tool_calls=True,
                parallel_tool_calls=True,
                tool_choice_param=False,  # Gemini has different tool_config
                json_fallback_parsing=True,
                xml_fallback_parsing=True,  # For <execute_bash> style
                thinking_mode=False,
                requires_strict_prompting=False,
                tool_call_format=ToolCallFormat.GOOGLE,
                argument_format="json",
                recommended_max_tools=30,
                recommended_tool_budget=15,
            )
        return caps

    def convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert to Google Gemini function_declarations format.

        Google format requires:
        {
            "function_declarations": [
                {
                    "name": "tool_name",
                    "description": "description",
                    "parameters": {...}
                }
            ]
        }
        """
        function_declarations = []
        for tool in tools:
            func_decl = {
                "name": tool.name,
                "description": tool.description,
            }
            # Only add parameters if they exist and are not empty
            if tool.parameters and tool.parameters.get("properties"):
                func_decl["parameters"] = tool.parameters
            function_declarations.append(func_decl)

        return [{"function_declarations": function_declarations}]

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        """Parse Google Gemini tool calls.

        Handles:
        1. Native function calls from API response
        2. Text-based <execute_bash>...</execute_bash> format
        3. Text-based <tool_code>...</tool_code> format
        """
        warnings = []

        # Try native tool calls first (from function_call response)
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                name = tc.get("name", "")
                if not self.is_valid_tool_name(name):
                    warnings.append(f"Skipped invalid tool name: {name}")
                    continue

                # Google returns args as dict directly, not as JSON string
                args = tc.get("args", {}) or tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json_loads(args)
                    except JSONDecodeError:
                        warnings.append(f"Failed to parse arguments for {name}")
                        args = {}

                tool_calls.append(ToolCall(name=name, arguments=args, id=tc.get("id")))

            if tool_calls:
                return ToolCallParseResult(
                    tool_calls=tool_calls,
                    remaining_content=content,
                    parse_method="native",
                    confidence=1.0,
                    warnings=warnings,
                )

        # Fallback: Parse text-based tool calls from Gemini output
        if content:
            result = self._parse_gemini_text_tools(content)
            if result.tool_calls:
                return result

            # Try standard JSON fallback (using mixin)
            result = self.parse_from_content(content, self.is_valid_tool_name)
            if result.tool_calls:
                return result

        return ToolCallParseResult(remaining_content=content)

    def _parse_gemini_text_tools(self, content: str) -> ToolCallParseResult:
        """Parse Gemini's text-based tool call format.

        Handles patterns like:
        - <execute_bash>ls -la</execute_bash>
        - <tool_code>print(read("path"))</tool_code>
        - <ctrl42>call: `tool_name` ```json {...}```
        - call: `tool_name` ```json {...}```
        - print(ls(path="..."))
        """
        tool_calls = []
        warnings = []
        remaining = content

        # Parse <ctrl42>call: patterns first (most specific)
        for match in self.CTRL42_CALL_PATTERN.finditer(content):
            tool_name = match.group(1).strip()
            json_str = match.group(2).strip()
            if tool_name and self.is_valid_tool_name(tool_name):
                try:
                    args = json_loads(json_str) if json_str else {}
                    canonical_tool_name = get_canonical_name(tool_name)
                    tool_calls.append(
                        ToolCall(
                            name=canonical_tool_name,
                            arguments=self.normalize_arguments(args, canonical_tool_name),
                        )
                    )
                    remaining = remaining.replace(match.group(0), "")
                    logger.debug(f"Parsed ctrl42 call: {tool_name}({args})")
                except JSONDecodeError:
                    warnings.append(f"Failed to parse JSON for {tool_name}: {json_str[:50]}")

        # Parse simpler call: patterns
        for match in self.SIMPLE_CALL_PATTERN.finditer(content):
            tool_name = match.group(1).strip()
            json_str = match.group(2).strip()
            if tool_name and self.is_valid_tool_name(tool_name):
                try:
                    args = json_loads(json_str) if json_str else {}
                    canonical_tool_name = get_canonical_name(tool_name)
                    tool_calls.append(
                        ToolCall(
                            name=canonical_tool_name,
                            arguments=self.normalize_arguments(args, canonical_tool_name),
                        )
                    )
                    remaining = remaining.replace(match.group(0), "")
                    logger.debug(f"Parsed simple call: {tool_name}({args})")
                except JSONDecodeError:
                    warnings.append(f"Failed to parse JSON for {tool_name}")

        # Strip hallucinated tool output (Gemini likes to make up results)
        content_cleaned = self.TOOL_OUTPUT_HALLUCINATION.sub("", content)
        if content_cleaned != content:
            logger.warning("Stripped hallucinated tool output from Gemini response")
            remaining = content_cleaned

        # Parse Python-style function calls: tool_name("value") or tool_name(path="value")
        for match in self.PYTHON_CALL_PATTERN.finditer(content_cleaned):
            tool_name = match.group(1).strip()
            arg_value = match.group(2).strip()
            if tool_name and self.is_valid_tool_name(tool_name):
                canonical_tool_name = get_canonical_name(tool_name)
                # Map common tool argument names
                if canonical_tool_name in ("ls", "read"):
                    args = {"path": arg_value}
                elif canonical_tool_name == "shell":
                    args = {"cmd": arg_value}
                elif tool_name in ("code_search", "semantic_code_search"):
                    args = {"query": arg_value}
                else:
                    args = {"path": arg_value}  # Default
                tool_calls.append(
                    ToolCall(
                        name=canonical_tool_name,
                        arguments=self.normalize_arguments(args, canonical_tool_name),
                    )
                )
                remaining = remaining.replace(match.group(0), "")
                logger.debug(f"Parsed Python call: {tool_name}({args})")

        # Parse <execute_bash> patterns
        for match in self.EXECUTE_BASH_PATTERN.finditer(content):
            command = match.group(1).strip()
            if command:
                tool_calls.append(
                    ToolCall(
                        name="shell",
                        arguments={"cmd": command},
                    )
                )
                remaining = remaining.replace(match.group(0), "")

        # Parse <tool_code> patterns (Python code that reads files, etc.)
        for match in self.TOOL_CODE_PATTERN.finditer(content):
            code = match.group(1).strip()
            if code:
                # Try to extract read_file calls
                read_file_match = re.search(r'read(?:_file)?\s*\(\s*["\']([^"\']+)["\']\s*\)', code)
                if read_file_match:
                    path = read_file_match.group(1)
                    tool_calls.append(
                        ToolCall(
                            name="read",
                            arguments={"path": path},
                        )
                    )
                else:
                    # Generic tool code - might be execute_python
                    warnings.append(f"Unrecognized tool_code: {code[:50]}...")
                remaining = remaining.replace(match.group(0), "")

        # Parse ```tool_code blocks
        for match in self.CODE_BLOCK_PATTERN.finditer(content):
            code = match.group(1).strip()
            if code:
                read_file_match = re.search(r'read(?:_file)?\s*\(\s*["\']([^"\']+)["\']\s*\)', code)
                if read_file_match:
                    path = read_file_match.group(1)
                    tool_calls.append(
                        ToolCall(
                            name="read",
                            arguments={"path": path},
                        )
                    )
                remaining = remaining.replace(match.group(0), "")

        if tool_calls:
            return ToolCallParseResult(
                tool_calls=tool_calls,
                remaining_content=remaining.strip(),
                parse_method="gemini_text_fallback",
                confidence=0.8,
                warnings=warnings,
            )

        return ToolCallParseResult(remaining_content=content)

    def get_system_prompt_hints(self) -> str:
        """Get system prompt hints for Google Gemini."""
        return "\n".join(
            [
                "TOOL USAGE:",
                "- Use the available tools to gather information when needed.",
                "- Call MULTIPLE tools in parallel when operations are independent.",
                "  Example: Read multiple files simultaneously, search while listing.",
                "- After gathering sufficient information, provide your answer.",
                "- Do NOT repeat identical tool calls.",
                "",
                "RESPONSE FORMAT:",
                "- Provide your final answer in clear, readable text.",
                "- Be concise and directly answer the question.",
            ]
        )


class LMStudioToolCallingAdapter(FallbackParsingMixin, BaseToolCallingAdapter):
    """Dedicated adapter for LMStudio local models.

    LMStudio provides an OpenAI-compatible API but requires specialized handling:
    - Native tool support for "hammer badge" models (Qwen2.5, Llama3.1, Ministral)
    - [TOOL_REQUEST]...[END_TOOL_REQUEST] format for default mode
    - JSON fallback parsing for unsupported models

    Uses FallbackParsingMixin for common parsing logic (shared with OllamaToolCallingAdapter).

    References:
    - https://lmstudio.ai/docs/advanced/tool-use
    - https://lmstudio.ai/docs/api/openai-api
    """

    # Models with native LMStudio tool support (hammer badge)
    NATIVE_TOOL_MODELS = frozenset(
        [
            "qwen2.5",
            "qwen-2.5",
            "qwen3",
            "qwen-3",
            "llama-3.1",
            "llama3.1",
            "llama-3.2",
            "llama3.2",
            "llama-3.3",
            "llama3.3",
            "ministral",
            "mistral",
            "mixtral",
            "command-r",
            "hermes",
            "functionary",
        ]
    )

    @property
    def provider_name(self) -> str:
        return "lmstudio"

    def _has_native_support(self) -> bool:
        """Check if current model has native tool calling support."""
        return any(pattern in self.model_lower for pattern in self.NATIVE_TOOL_MODELS)

    def _has_thinking_mode(self) -> bool:
        """Check if model supports Qwen3 thinking mode."""
        return "qwen3" in self.model_lower or "qwen-3" in self.model_lower

    def get_capabilities(self) -> ToolCallingCapabilities:
        """Get capabilities for LMStudio models."""
        has_native = self._has_native_support()
        format_hint = (
            ToolCallFormat.LMSTUDIO_NATIVE if has_native else ToolCallFormat.LMSTUDIO_DEFAULT
        )

        # Load from YAML config with model pattern matching
        loader = _get_capability_loader()
        caps = loader.get_capabilities("lmstudio", self.model, format_hint)

        # Apply runtime model detection for thinking mode
        if self._has_thinking_mode() and not caps.thinking_mode:
            caps = ToolCallingCapabilities(
                native_tool_calls=caps.native_tool_calls,
                streaming_tool_calls=caps.streaming_tool_calls,
                parallel_tool_calls=caps.parallel_tool_calls,
                tool_choice_param=caps.tool_choice_param,
                json_fallback_parsing=True,
                xml_fallback_parsing=False,
                thinking_mode=True,
                requires_strict_prompting=caps.requires_strict_prompting,
                tool_call_format=format_hint,
                argument_format=caps.argument_format,
                recommended_max_tools=caps.recommended_max_tools,
                recommended_tool_budget=caps.recommended_tool_budget,
            )

        return caps

    def convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert to OpenAI format (LMStudio is OpenAI-compatible)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        """Parse LMStudio tool calls with multi-format support.

        Handles:
        1. Native tool calls from API response (hammer badge models)
        2. [TOOL_REQUEST]...[END_TOOL_REQUEST] format (default mode)
        3. JSON fallback from content (using FallbackParsingMixin)
        """

        # 1. Try native tool calls first (using mixin method)
        if raw_tool_calls:
            result = self.parse_native_tool_calls(raw_tool_calls, self.is_valid_tool_name)
            if result.tool_calls:
                return ToolCallParseResult(
                    tool_calls=result.tool_calls,
                    remaining_content=content,
                    parse_method=result.parse_method,
                    confidence=result.confidence,
                    warnings=result.warnings,
                )

        # 2. Try [TOOL_REQUEST] format (LMStudio default mode)
        if content:
            result = self._parse_tool_request_format(content)
            if result.tool_calls:
                return result

        # 3. JSON fallback from content (using mixin method)
        if content:
            result = self.parse_from_content(content, self.is_valid_tool_name)
            if result.tool_calls:
                return result

        return ToolCallParseResult(remaining_content=content)

    def _parse_tool_request_format(self, content: str) -> ToolCallParseResult:
        """Parse LMStudio [TOOL_REQUEST]...[END_TOOL_REQUEST] format.

        Example:
        [TOOL_REQUEST]{"name": "read_file", "arguments": {"path": "/etc/hosts"}}[END_TOOL_REQUEST]
        """
        pattern = r"\[TOOL_REQUEST\](.*?)\[END_TOOL_REQUEST\]"
        matches = re.findall(pattern, content, re.DOTALL)

        if not matches:
            return ToolCallParseResult(remaining_content=content)

        tool_calls = []
        warnings = []

        for match in matches:
            try:
                data = json_loads(match.strip())
                name = data.get("name", "")
                if self.is_valid_tool_name(name):
                    args = data.get("arguments") or data.get("parameters", {})
                    tool_calls.append(ToolCall(name=name, arguments=args))
                else:
                    warnings.append(f"Skipped invalid tool name: {name}")
            except JSONDecodeError:
                warnings.append(f"Failed to parse TOOL_REQUEST JSON: {match[:50]}...")

        remaining = re.sub(pattern, "", content, flags=re.DOTALL).strip()

        return ToolCallParseResult(
            tool_calls=tool_calls,
            remaining_content=remaining,
            parse_method="tool_request_format",
            confidence=0.85,
            warnings=warnings,
        )

    def get_system_prompt_hints(self) -> str:
        """Get system prompt hints for LMStudio models."""
        capabilities = self.get_capabilities()

        if capabilities.native_tool_calls:
            hints = [
                "TOOL USAGE:",
                "- Use the available tools to gather information.",
                "- Call tools one at a time and wait for results.",
                "- After 2-3 successful tool calls, provide your answer.",
                "- Do NOT repeat identical tool calls.",
                "",
                "RESPONSE FORMAT:",
                "- Provide your answer in plain, readable text.",
                "- Do NOT include JSON, XML, or tool syntax in your response.",
                "- Be direct and answer the user's question.",
            ]

            if capabilities.thinking_mode:
                hints.extend(
                    [
                        "",
                        "QWEN3 MODE:",
                        "- Use /no_think for simple questions.",
                        "- Provide direct answers without excessive reasoning.",
                    ]
                )

            return "\n".join(hints)
        else:
            return "\n".join(
                [
                    "CRITICAL RULES:",
                    "1. Call tools ONE AT A TIME. Wait for each result.",
                    "2. After reading 2-3 files, STOP and provide your answer.",
                    "3. Do NOT repeat the same tool call.",
                    "4. Do NOT invent or guess file contents.",
                    "",
                    "OUTPUT FORMAT:",
                    "1. Your answer must be in plain English text.",
                    "2. Do NOT output JSON objects in your response.",
                    "3. Do NOT output XML tags.",
                    "4. Do NOT output [TOOL_REQUEST] markers in your final answer.",
                ]
            )


class OpenAICompatToolCallingAdapter(BaseToolCallingAdapter):
    """Adapter for OpenAI-compatible local providers (vLLM).

    Note: LMStudio now uses the dedicated LMStudioToolCallingAdapter.
    This adapter is retained for vLLM compatibility.

    vLLM:
    - Requires --enable-auto-tool-choice and --tool-call-parser flags
    - Supports many model-specific parsers (hermes, mistral, llama3_json, etc.)

    References:
    - https://docs.vllm.ai/en/stable/features/tool_calling/
    """

    # Models with native LMStudio tool support
    LMSTUDIO_NATIVE_MODELS = frozenset(
        [
            "qwen2.5",
            "qwen-2.5",
            "qwen3",
            "qwen-3",
            "llama-3.1",
            "llama3.1",
            "llama-3.2",
            "llama3.2",
            "ministral",
            "mistral",
        ]
    )

    def __init__(
        self,
        model: str = "",
        config: Optional[Dict[str, Any]] = None,
        provider_variant: str = "lmstudio",
    ):
        """Initialize adapter.

        Args:
            model: Model name
            config: Configuration
            provider_variant: "lmstudio" or "vllm"
        """
        super().__init__(model, config)
        self.provider_variant = provider_variant

    @property
    def provider_name(self) -> str:
        return self.provider_variant

    def _has_native_support(self) -> bool:
        """Check if model has native tool support."""
        return any(pattern in self.model_lower for pattern in self.LMSTUDIO_NATIVE_MODELS)

    def get_capabilities(self) -> ToolCallingCapabilities:
        has_native = self._has_native_support()

        if self.provider_variant == "vllm":
            # vLLM: Load from YAML with model pattern matching
            loader = _get_capability_loader()
            caps = loader.get_capabilities("vllm", self.model, ToolCallFormat.VLLM)

            # vLLM always has native when properly configured
            if not caps.native_tool_calls:
                return ToolCallingCapabilities(
                    native_tool_calls=True,
                    streaming_tool_calls=True,
                    parallel_tool_calls=True,
                    tool_choice_param=True,
                    json_fallback_parsing=True,
                    xml_fallback_parsing=False,
                    thinking_mode=False,
                    requires_strict_prompting=False,
                    tool_call_format=ToolCallFormat.VLLM,
                    argument_format="json",
                    recommended_max_tools=30,
                    recommended_tool_budget=15,
                )
            return caps
        else:
            # LMStudio: Load from YAML with model pattern matching
            format_hint = (
                ToolCallFormat.LMSTUDIO_NATIVE if has_native else ToolCallFormat.LMSTUDIO_DEFAULT
            )
            loader = _get_capability_loader()
            caps = loader.get_capabilities("lmstudio", self.model, format_hint)

            # Apply runtime model detection for thinking mode
            if "qwen3" in self.model_lower and not caps.thinking_mode:
                caps = ToolCallingCapabilities(
                    native_tool_calls=caps.native_tool_calls,
                    streaming_tool_calls=caps.streaming_tool_calls,
                    parallel_tool_calls=caps.parallel_tool_calls,
                    tool_choice_param=caps.tool_choice_param,
                    json_fallback_parsing=caps.json_fallback_parsing,
                    xml_fallback_parsing=caps.xml_fallback_parsing,
                    thinking_mode=True,
                    requires_strict_prompting=caps.requires_strict_prompting,
                    tool_call_format=format_hint,
                    argument_format=caps.argument_format,
                    recommended_max_tools=caps.recommended_max_tools,
                    recommended_tool_budget=caps.recommended_tool_budget,
                )
            return caps

    def convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        """Parse tool calls from OpenAI-compatible response."""
        warnings = []

        # Try native tool calls
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                name = tc.get("name", "")
                tc_id = tc.get("id")
                if not self.is_valid_tool_name(name):
                    warnings.append(f"Skipped invalid tool name: {name}")
                    # Keep with id so pipeline generates error response
                    if tc_id:
                        tool_calls.append(ToolCall(name=name, arguments={}, id=tc_id))
                    continue

                args = tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json_loads(args)
                    except JSONDecodeError:
                        warnings.append(f"Failed to parse arguments for {name}")
                        args = {}

                tool_calls.append(ToolCall(name=name, arguments=args, id=tc_id))

            if tool_calls:
                return ToolCallParseResult(
                    tool_calls=tool_calls,
                    remaining_content=content,
                    parse_method="native",
                    confidence=1.0,
                    warnings=warnings,
                )

        # Fallback: Parse JSON from content
        if content:
            result = self._parse_json_from_content(content)
            if result.tool_calls:
                return result

            # LMStudio default format: [TOOL_REQUEST]...[END_TOOL_REQUEST]
            result = self._parse_tool_request_format(content)
            if result.tool_calls:
                return result

        return ToolCallParseResult(remaining_content=content)

    def _parse_json_from_content(self, content: str) -> ToolCallParseResult:
        """Parse JSON tool calls from content."""
        content = content.strip()
        if not content:
            return ToolCallParseResult()

        try:
            data = json_loads(content)
            if isinstance(data, dict) and "name" in data:
                name = data.get("name", "")
                if self.is_valid_tool_name(name):
                    args = data.get("arguments") or data.get("parameters", {})
                    return ToolCallParseResult(
                        tool_calls=[ToolCall(name=name, arguments=args)],
                        remaining_content="",
                        parse_method="json_fallback",
                        confidence=0.9,
                    )
        except JSONDecodeError:
            pass

        return ToolCallParseResult(remaining_content=content)

    def _parse_tool_request_format(self, content: str) -> ToolCallParseResult:
        """Parse LMStudio [TOOL_REQUEST] format."""
        pattern = r"\[TOOL_REQUEST\](.*?)\[END_TOOL_REQUEST\]"
        matches = re.findall(pattern, content, re.DOTALL)

        if not matches:
            return ToolCallParseResult(remaining_content=content)

        tool_calls = []
        warnings = []

        for match in matches:
            try:
                data = json_loads(match.strip())
                name = data.get("name", "")
                if self.is_valid_tool_name(name):
                    args = data.get("arguments") or data.get("parameters", {})
                    tool_calls.append(ToolCall(name=name, arguments=args))
            except JSONDecodeError:
                warnings.append("Failed to parse TOOL_REQUEST content")

        remaining = re.sub(pattern, "", content, flags=re.DOTALL).strip()

        return ToolCallParseResult(
            tool_calls=tool_calls,
            remaining_content=remaining,
            parse_method="tool_request_format",
            confidence=0.8,
            warnings=warnings,
        )

    def get_system_prompt_hints(self) -> str:
        """Get system prompt hints."""
        capabilities = self.get_capabilities()

        if self.provider_variant == "vllm":
            return "\n".join(
                [
                    "TOOL CALLING:",
                    "- Tools are called using the standard OpenAI function calling format.",
                    "- You may call tools when needed to gather information.",
                    "- After gathering sufficient information (2-4 tool calls), provide your answer.",
                    "- Do NOT repeat the same tool call with identical arguments.",
                    "",
                    "IMPORTANT:",
                    "- Do not output raw JSON tool calls in your text response.",
                    "- When you're done using tools, provide a human-readable answer.",
                ]
            )

        if capabilities.native_tool_calls:
            return "\n".join(
                [
                    "TOOL USAGE:",
                    "- Use tools via the OpenAI function calling format.",
                    "- Call tools one at a time and wait for results.",
                    "- After 2-3 tool calls, provide your answer.",
                    "- Do NOT repeat identical tool calls.",
                    "",
                    "RESPONSE FORMAT:",
                    "- Provide answers in plain, readable text.",
                    "- Do NOT include JSON, XML, or tool syntax in your response.",
                    "- Be direct and answer the user's question.",
                ]
            )
        else:
            return "\n".join(
                [
                    "CRITICAL RULES:",
                    "1. Call tools ONE AT A TIME. Wait for each result.",
                    "2. After reading 2-3 files, STOP and provide your answer.",
                    "3. Do NOT repeat the same tool call.",
                    "4. Do NOT invent or guess file contents.",
                    "",
                    "OUTPUT FORMAT:",
                    "1. Your answer must be in plain English text.",
                    "2. Do NOT output JSON objects in your response.",
                    "3. Do NOT output XML tags like </function> or <parameter>.",
                    "4. Do NOT output [TOOL_REQUEST] or similar markers.",
                ]
            )


class DeepSeekToolCallingAdapter(FallbackParsingMixin, BaseToolCallingAdapter):
    """Dedicated adapter for DeepSeek models.

    DeepSeek provides OpenAI-compatible API with model-specific behaviors:
    - deepseek-chat: Non-thinking mode with native function calling
    - deepseek-reasoner: Thinking mode with Chain of Thought (NO function calling)

    This adapter handles:
    - Model-specific capability detection
    - Reasoning content extraction for deepseek-reasoner
    - Appropriate system prompts for each model type
    - Fallback parsing for edge cases

    References:
    - https://api-docs.deepseek.com/guides/function_calling
    - https://api-docs.deepseek.com/guides/reasoning_model
    """

    @property
    def provider_name(self) -> str:
        return "deepseek"

    def _is_reasoner_model(self) -> bool:
        """Check if current model is deepseek-reasoner (thinking mode)."""
        return "reasoner" in self.model_lower or "r1" in self.model_lower

    def _has_native_support(self) -> bool:
        """Check if current model supports native tool calling.

        deepseek-reasoner does NOT support function calling.
        deepseek-chat and other variants do.
        """
        return not self._is_reasoner_model()

    def get_capabilities(self) -> ToolCallingCapabilities:
        """Get capabilities for DeepSeek models."""
        is_reasoner = self._is_reasoner_model()

        # Load from YAML config with model pattern matching
        loader = _get_capability_loader()
        caps = loader.get_capabilities("deepseek", self.model, ToolCallFormat.OPENAI)

        # Override based on model type - reasoner has no tool support
        if is_reasoner:
            return ToolCallingCapabilities(
                native_tool_calls=False,
                streaming_tool_calls=False,
                parallel_tool_calls=False,
                tool_choice_param=False,
                json_fallback_parsing=False,  # No tool parsing for reasoner
                xml_fallback_parsing=False,
                thinking_mode=True,
                requires_strict_prompting=False,
                tool_call_format=ToolCallFormat.OPENAI,
                argument_format="json",
                recommended_max_tools=0,  # No tools for reasoner
                recommended_tool_budget=0,
            )

        # deepseek-chat: Full native tool calling support
        if not caps.native_tool_calls:
            return ToolCallingCapabilities(
                native_tool_calls=True,
                streaming_tool_calls=True,
                parallel_tool_calls=True,
                tool_choice_param=True,
                json_fallback_parsing=True,
                xml_fallback_parsing=False,
                thinking_mode=False,
                requires_strict_prompting=False,
                tool_call_format=ToolCallFormat.OPENAI,
                argument_format="json",
                recommended_max_tools=50,
                recommended_tool_budget=20,
            )
        return caps

    def convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert to OpenAI function format (DeepSeek is OpenAI-compatible)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        """Parse DeepSeek tool calls.

        DeepSeek returns tool calls in OpenAI format:
        {"id": "...", "function": {"name": "...", "arguments": "..."}}
        where arguments is a JSON string.

        For deepseek-reasoner, no tool calls are expected.
        """
        warnings: List[str] = []

        # Reasoner model: no tool parsing
        if self._is_reasoner_model():
            return ToolCallParseResult(remaining_content=content)

        # Try native tool calls first
        if raw_tool_calls:
            tool_calls = []

            for tc in raw_tool_calls:
                tc_id = tc.get("id")
                # Handle OpenAI format with nested function
                if "function" in tc:
                    func = tc["function"]
                    name = func.get("name", "")
                    args = func.get("arguments", "{}")
                else:
                    # Direct format
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})

                if not self.is_valid_tool_name(name):
                    warnings.append(f"Skipped invalid tool name: {name}")
                    # Keep with id so pipeline generates error response
                    if tc_id:
                        tool_calls.append(ToolCall(name=name, arguments={}, id=tc_id))
                    continue

                # Parse arguments (may be string or dict)
                if isinstance(args, str):
                    try:
                        args = json_loads(args)
                    except JSONDecodeError:
                        warnings.append(f"Failed to parse arguments for {name}")
                        args = {}

                tool_calls.append(ToolCall(name=name, arguments=args, id=tc_id))

            if tool_calls:
                return ToolCallParseResult(
                    tool_calls=tool_calls,
                    remaining_content=content,
                    parse_method="native",
                    confidence=1.0,
                    warnings=warnings,
                )

        # Fallback: Try parsing from content (using mixin)
        if content:
            result = self.parse_from_content(content, self.is_valid_tool_name)
            if result.tool_calls:
                all_warnings = warnings + result.warnings
                return ToolCallParseResult(
                    tool_calls=result.tool_calls,
                    remaining_content=result.remaining_content,
                    parse_method=result.parse_method,
                    confidence=result.confidence,
                    warnings=all_warnings,
                )

        return ToolCallParseResult(remaining_content=content, warnings=warnings)

    def get_system_prompt_hints(self) -> str:
        """Get system prompt hints for DeepSeek models."""
        if self._is_reasoner_model():
            # Reasoner mode: No tools, emphasize reasoning
            return "\n".join(
                [
                    "DEEPSEEK REASONER MODE:",
                    "- You are in reasoning/thinking mode.",
                    "- Focus on step-by-step logical analysis.",
                    "- Tools are NOT available in this mode.",
                    "- Provide thorough explanations with your reasoning.",
                ]
            )

        # deepseek-chat: Normal tool calling
        return "\n".join(
            [
                "TOOL USAGE:",
                "- Use the available tools to gather information when needed.",
                "- Call MULTIPLE tools in parallel when operations are independent.",
                "- After gathering sufficient information (2-4 tool calls), provide your answer.",
                "- Do NOT repeat identical tool calls.",
                "",
                "RESPONSE FORMAT:",
                "- Provide your final answer in clear, readable text.",
                "- Be concise and directly answer the question.",
            ]
        )


class BedrockToolCallingAdapter(BaseToolCallingAdapter):
    """Adapter for AWS Bedrock models using the Converse API.

    Bedrock uses a distinct tool format with toolSpec/inputSchema.
    Tool calls are returned as toolUse blocks with toolUseId, name, input.

    Supports Claude, Llama, Mistral, and Cohere models on Bedrock.
    Amazon Titan models do NOT support tool calling.
    """

    # Models that support tool calling on Bedrock
    TOOL_SUPPORTED_PREFIXES = (
        "anthropic.claude",
        "meta.llama",
        "mistral.",
        "cohere.command",
    )

    # Models that do NOT support tool calling
    NO_TOOL_PREFIXES = ("amazon.titan",)

    @property
    def provider_name(self) -> str:
        return "bedrock"

    def _supports_tools(self) -> bool:
        """Check if the current model supports tools."""
        model_lower = self.model.lower()

        # Check explicit no-tool models
        for prefix in self.NO_TOOL_PREFIXES:
            if model_lower.startswith(prefix):
                return False

        # Check supported prefixes
        for prefix in self.TOOL_SUPPORTED_PREFIXES:
            if model_lower.startswith(prefix):
                return True

        # Default: assume no support for unknown models
        return False

    def get_capabilities(self) -> ToolCallingCapabilities:
        """Get capabilities for Bedrock models."""
        loader = _get_capability_loader()
        caps = loader.get_capabilities("bedrock", self.model, ToolCallFormat.BEDROCK)

        if caps.native_tool_calls:
            return caps

        # Determine capabilities based on model
        supports_tools = self._supports_tools()

        return ToolCallingCapabilities(
            native_tool_calls=supports_tools,
            streaming_tool_calls=supports_tools,
            parallel_tool_calls=supports_tools,
            tool_choice_param=supports_tools,
            json_fallback_parsing=False,
            xml_fallback_parsing=False,
            thinking_mode=False,
            requires_strict_prompting=False,
            tool_call_format=ToolCallFormat.BEDROCK,
            argument_format="json",
            recommended_max_tools=30,
            recommended_tool_budget=15,
        )

    def convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert to Bedrock Converse API format.

        Bedrock uses toolSpec with inputSchema.json for tool definitions.
        """
        if not self._supports_tools():
            return []

        return [
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {"json": tool.parameters},
                }
            }
            for tool in tools
        ]

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        """Parse Bedrock tool calls.

        Bedrock returns tool calls in the raw_tool_calls list with format:
        {"id": "...", "name": "...", "arguments": {...}}

        The arguments are already parsed (not JSON strings like OpenAI).
        """
        if not raw_tool_calls or not self._supports_tools():
            return ToolCallParseResult(remaining_content=content)

        tool_calls = []
        warnings: List[str] = []

        for tc in raw_tool_calls:
            name = tc.get("name", "")
            tc_id = tc.get("id")

            if not self.is_valid_tool_name(name):
                warnings.append(f"Skipped invalid tool name: {name}")
                # Keep with id so pipeline generates error response
                if tc_id:
                    tool_calls.append(ToolCall(name=name, arguments={}, id=tc_id))
                continue

            # Bedrock returns arguments already parsed as dict
            arguments = tc.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json_loads(arguments)
                except JSONDecodeError:
                    warnings.append(f"Failed to parse arguments for {name}")
                    arguments = {}

            tool_calls.append(
                ToolCall(
                    name=name,
                    arguments=arguments,
                    id=tc_id,
                )
            )

        return ToolCallParseResult(
            tool_calls=tool_calls,
            remaining_content=content,
            parse_method="native",
            confidence=1.0,
            warnings=warnings if warnings else None,
        )

    def get_system_prompt_hints(self) -> str:
        """Get system prompt hints for Bedrock models."""
        if not self._supports_tools():
            return ""

        return "\n".join(
            [
                "TOOL USAGE:",
                "- Use available tools to gather information when needed.",
                "- Tools return results in structured format.",
                "- Provide clear, direct answers based on tool results.",
            ]
        )


class AzureOpenAIToolCallingAdapter(FallbackParsingMixin, BaseToolCallingAdapter):
    """Adapter for Azure OpenAI Service models.

    Azure OpenAI uses the same format as OpenAI but has some model-specific
    considerations:
    - o1-preview and o1-mini do NOT support tool calling
    - Standard GPT models (gpt-4o, gpt-4-turbo, gpt-35-turbo) support tools
    - Microsoft Phi models on Azure AI also support tools

    Uses FallbackParsingMixin for robustness with edge cases.
    """

    # o1 models do NOT support tools
    NO_TOOL_MODELS = ("o1-preview", "o1-mini", "o1")

    @property
    def provider_name(self) -> str:
        return "azure"

    def _is_o1_model(self) -> bool:
        """Check if this is an o1 reasoning model (no tool support)."""
        model_lower = self.model.lower()
        return any(model_lower.startswith(prefix) for prefix in self.NO_TOOL_MODELS)

    def _supports_tools(self) -> bool:
        """Check if the current model supports tools."""
        return not self._is_o1_model()

    def get_capabilities(self) -> ToolCallingCapabilities:
        """Get capabilities for Azure OpenAI models."""
        loader = _get_capability_loader()
        caps = loader.get_capabilities("azure", self.model, ToolCallFormat.OPENAI)

        if caps.native_tool_calls and not self._is_o1_model():
            return caps

        # o1 models: no tools, thinking mode
        if self._is_o1_model():
            return ToolCallingCapabilities(
                native_tool_calls=False,
                streaming_tool_calls=False,
                parallel_tool_calls=False,
                tool_choice_param=False,
                json_fallback_parsing=False,
                xml_fallback_parsing=False,
                thinking_mode=True,
                requires_strict_prompting=False,
                tool_call_format=ToolCallFormat.NONE,
                argument_format="json",
                recommended_max_tools=0,
                recommended_tool_budget=0,
            )

        # Standard Azure OpenAI models (GPT-4o, GPT-4-turbo, etc.)
        return ToolCallingCapabilities(
            native_tool_calls=True,
            streaming_tool_calls=True,
            parallel_tool_calls=True,
            tool_choice_param=True,
            json_fallback_parsing=True,
            xml_fallback_parsing=False,
            thinking_mode=False,
            requires_strict_prompting=False,
            tool_call_format=ToolCallFormat.OPENAI,
            argument_format="json",
            recommended_max_tools=50,
            recommended_tool_budget=20,
        )

    def convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert to OpenAI function format (Azure uses same format)."""
        if not self._supports_tools():
            return []

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        """Parse Azure OpenAI tool calls (same as OpenAI format)."""
        if self._is_o1_model():
            # o1 models don't support tools
            return ToolCallParseResult(remaining_content=content)

        warnings: List[str] = []

        # Try native tool calls first
        if raw_tool_calls:
            tool_calls = []

            for tc in raw_tool_calls:
                tc_id = tc.get("id")
                # Handle OpenAI format with nested function
                if "function" in tc:
                    func = tc["function"]
                    name = func.get("name", "")
                    args = func.get("arguments", "{}")
                else:
                    # Direct format (already normalized)
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})

                if not self.is_valid_tool_name(name):
                    warnings.append(f"Skipped invalid tool name: {name}")
                    # Keep with id so pipeline generates error response
                    if tc_id:
                        tool_calls.append(ToolCall(name=name, arguments={}, id=tc_id))
                    continue

                # Parse arguments (may be string or dict)
                if isinstance(args, str):
                    try:
                        args = json_loads(args)
                    except JSONDecodeError:
                        warnings.append(f"Failed to parse arguments for {name}")
                        args = {}

                tool_calls.append(ToolCall(name=name, arguments=args, id=tc_id))

            if tool_calls:
                return ToolCallParseResult(
                    tool_calls=tool_calls,
                    remaining_content=content,
                    parse_method="native",
                    confidence=1.0,
                    warnings=warnings if warnings else None,
                )

        # Fallback: Try parsing from content (using mixin)
        if content:
            result = self.parse_from_content(content, self.is_valid_tool_name)
            if result.tool_calls:
                all_warnings = warnings + (result.warnings or [])
                return ToolCallParseResult(
                    tool_calls=result.tool_calls,
                    remaining_content=result.remaining_content,
                    parse_method=result.parse_method,
                    confidence=result.confidence,
                    warnings=all_warnings if all_warnings else None,
                )

        return ToolCallParseResult(
            remaining_content=content, warnings=warnings if warnings else None
        )

    def get_system_prompt_hints(self) -> str:
        """Get system prompt hints for Azure OpenAI models."""
        if self._is_o1_model():
            return "\n".join(
                [
                    "AZURE O1 REASONING MODE:",
                    "- You are in advanced reasoning mode.",
                    "- Focus on step-by-step logical analysis.",
                    "- Tools are NOT available in this mode.",
                    "- Provide thorough explanations with your reasoning.",
                ]
            )

        return "\n".join(
            [
                "TOOL USAGE:",
                "- Use available tools to gather information when needed.",
                "- Call MULTIPLE tools in parallel when operations are independent.",
                "- After gathering sufficient information, provide your answer.",
                "- Do NOT repeat identical tool calls.",
            ]
        )
