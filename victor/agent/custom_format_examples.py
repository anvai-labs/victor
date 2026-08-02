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

"""Custom formatting strategy examples.

This module demonstrates how to create and register custom formatting
strategies for domain-specific use cases.

Examples:
- Markdown formatting for documentation tools
- CSV formatting for data export tools
- YAML formatting for configuration tools
- Custom domain-specific formats

Usage:
    from victor.agent.custom_format_examples import (
        register_custom_strategies,
        MarkdownFormatStrategy,
    )

    # Register all custom strategies
    register_custom_strategies()

    # Use in provider
    from victor.providers.base import BaseProvider

    class CustomProvider(BaseProvider):
        def get_tool_output_format(self):
            from victor.agent.format_strategies import ToolOutputFormat
            return ToolOutputFormat(style="markdown")
"""

from __future__ import annotations

import csv
import io
from victor.core.json_utils import json_dumps
import logging
from typing import Any, Dict, List, Optional

from victor.agent.format_strategies import FormatStrategyFactory, ToolOutputFormat

logger = logging.getLogger(__name__)


# =============================================================================
# Markdown Format Strategy
# =============================================================================


class MarkdownFormatStrategy:
    """Markdown formatting strategy for documentation tools.

    Formats tool outputs as Markdown, ideal for:
    - Documentation generation
    - README creation
    - API documentation
    - Technical writing

    Example output:
        ```markdown
        # Tool: read

        **Path:** `example.py`

        ```python
        def hello():
            print("Hello, World!")
        ```
        ```
    """

    def format(
        self,
        tool_name: str,
        args: Dict[str, Any],
        output: Any,
        format_hint: Optional[str] = None,
    ) -> str:
        """Format tool output as Markdown."""
        lines = [
            f"# Tool: {tool_name}",
            "",
        ]

        # Add arguments
        if args:
            lines.append("## Arguments")
            lines.append("")
            for key, value in args.items():
                lines.append(f"- **{key}:** `{value}`")
            lines.append("")

        # Add output
        lines.append("## Output")
        lines.append("")

        # Format based on output type
        if isinstance(output, str):
            # Check if it's code
            if any(
                output.strip().startswith(prefix)
                for prefix in ["def ", "class ", "import ", "from "]
            ):
                lines.append("```python")
                lines.append(output)
                lines.append("```")
            else:
                lines.append(output)
        elif isinstance(output, dict):
            lines.append("```json")
            lines.append(json_dumps(output, indent=2, default=str))
            lines.append("```")
        elif isinstance(output, list):
            lines.append("```json")
            lines.append(json_dumps(output, indent=2, default=str))
            lines.append("```")
        else:
            lines.append(f"``{output}```")

        return "\n".join(lines)

    def estimate_tokens(self, content: str) -> int:
        """Estimate tokens (Markdown has some overhead)."""
        # Markdown formatting adds ~20% overhead
        base_tokens = len(content) // 4
        return int(base_tokens * 1.2)


# =============================================================================
# CSV Format Strategy
# =============================================================================


class CSVFormatStrategy:
    """CSV formatting strategy for tabular data tools.

    Formats tool outputs as CSV, ideal for:
    - Data export
    - Spreadsheet generation
    - Tabular search results
    - Analytics output

    Example output:
        ```csv
        name,age,city
        Alice,30,NYC
        Bob,25,SF
        ```
    """

    def format(
        self,
        tool_name: str,
        args: Dict[str, Any],
        output: Any,
        format_hint: Optional[str] = None,
    ) -> str:
        """Format tool output as CSV."""
        # Only format list of dicts as CSV
        if not isinstance(output, list) or not output:
            return str(output)

        if not all(isinstance(item, dict) for item in output):
            return str(output)

        # Extract headers from first item
        headers = list(output[0].keys())

        # Build CSV
        output_io = io.StringIO()
        writer = csv.DictWriter(output_io, fieldnames=headers)
        writer.writeheader()
        writer.writerows(output)

        return output_io.getvalue()

    def estimate_tokens(self, content: str) -> int:
        """Estimate tokens (CSV is very efficient)."""
        # CSV is ~40% more efficient than JSON
        base_tokens = len(content) // 4
        return int(base_tokens * 0.6)


# =============================================================================
# YAML Format Strategy
# =============================================================================


class YAMLFormatStrategy:
    """YAML formatting strategy for configuration tools.

    Formats tool outputs as YAML, ideal for:
    - Configuration files
    - Kubernetes manifests
    - CI/CD pipelines
    - Infrastructure as code

    Example output:
        ```yaml
        tool: read
        args:
          path: example.py
        output: |
          def hello():
              print("Hello, World!")
        ```
    """

    def format(
        self,
        tool_name: str,
        args: Dict[str, Any],
        output: Any,
        format_hint: Optional[str] = None,
    ) -> str:
        """Format tool output as YAML."""
        try:
            import yaml  # noqa: F401

            data = {
                "tool": tool_name,
                "args": args,
                "output": output,
            }

            return yaml.dump(data, default_flow_style=False, sort_keys=False)

        except ImportError:
            # Fallback to simple YAML-like format
            lines = [f"tool: {tool_name}"]

            if args:
                lines.append("args:")
                for key, value in args.items():
                    lines.append(f"  {key}: {value}")

            lines.append("output: |")
            if isinstance(output, str):
                for line in output.split("\n"):
                    lines.append(f"  {line}")
            else:
                lines.append(f"  {output}")

            return "\n".join(lines)

    def estimate_tokens(self, content: str) -> int:
        """Estimate tokens (YAML is similar to JSON)."""
        return len(content) // 4


# =============================================================================
# Code Block Format Strategy
# =============================================================================


class CodeBlockFormatStrategy:
    """Code block formatting strategy with syntax highlighting hints.

    Formats tool outputs as code blocks with language detection,
    ideal for:
    - Code search results
    - File reading tools
    - Code analysis tools
    - Diff/patch tools

    Example output:
        ```python
        # File: example.py
        def hello():
            print("Hello, World!")
        ```
    """

    def format(
        self,
        tool_name: str,
        args: Dict[str, Any],
        output: Any,
        format_hint: Optional[str] = None,
    ) -> str:
        """Format tool output as code block with language detection."""
        # Detect language from tool name or args
        language = self._detect_language(tool_name, args)

        lines = []

        # Add file path if available
        if "path" in args:
            lines.append(f"# File: {args['path']}")
            lines.append("")

        # Add code block
        lines.append(f"```{language}")
        if isinstance(output, str):
            lines.append(output)
        else:
            lines.append(str(output))
        lines.append("```")

        return "\n".join(lines)

    def _detect_language(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Detect programming language from context.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Language identifier for code block
        """
        # Check file extension
        if "path" in args:
            path = str(args["path"]).lower()
            ext_map = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".jsx": "jsx",
                ".tsx": "tsx",
                ".java": "java",
                ".cpp": "cpp",
                ".c": "c",
                ".cs": "csharp",
                ".go": "go",
                ".rs": "rust",
                ".rb": "ruby",
                ".php": "php",
                ".swift": "swift",
                ".kt": "kotlin",
                ".scala": "scala",
                ".sh": "bash",
                ".yaml": "yaml",
                ".yml": "yaml",
                ".json": "json",
                ".xml": "xml",
                ".html": "html",
                ".css": "css",
                ".sql": "sql",
                ".md": "markdown",
            }

            for ext, lang in ext_map.items():
                if path.endswith(ext):
                    return lang

        # Check tool name
        if "python" in tool_name.lower():
            return "python"
        if "javascript" in tool_name.lower() or "js" in tool_name.lower():
            return "javascript"

        # Default
        return ""

    def estimate_tokens(self, content: str) -> int:
        """Estimate tokens (code blocks are efficient)."""
        return len(content) // 4


# =============================================================================
# Registration Function
# =============================================================================


def register_custom_strategies() -> None:
    """Register all custom formatting strategies with the factory.

    Call this during application initialization to enable custom formats.

    Example:
        from victor.agent.custom_format_examples import register_custom_strategies

        # Register during app startup
        register_custom_strategies()

        # Now providers can use these formats
        class DocumentationProvider(BaseProvider):
            def get_tool_output_format(self):
                return ToolOutputFormat(style="markdown")
    """
    strategies = {
        "markdown": MarkdownFormatStrategy,
        "csv": CSVFormatStrategy,
        "yaml": YAMLFormatStrategy,
        "code": CodeBlockFormatStrategy,
    }

    for style, strategy_class in strategies.items():
        try:
            FormatStrategyFactory.register_strategy(style, strategy_class)
            logger.info(f"Registered custom format strategy: {style}")
        except Exception as e:
            logger.error(f"Failed to register {style} strategy: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================


def create_markdown_format() -> ToolOutputFormat:
    """Create a Markdown format specification.

    Returns:
        ToolOutputFormat configured for Markdown
    """
    from dataclasses import dataclass

    @dataclass
    class MarkdownFormatSpec:
        style: str = "markdown"

    return MarkdownFormatSpec()


def create_csv_format() -> ToolOutputFormat:
    """Create a CSV format specification.

    Returns:
        ToolOutputFormat configured for CSV
    """
    from dataclasses import dataclass

    @dataclass
    class CSVFormatSpec:
        style: str = "csv"

    return CSVFormatSpec()


def create_yaml_format() -> ToolOutputFormat:
    """Create a YAML format specification.

    Returns:
        ToolOutputFormat configured for YAML
    """
    from dataclasses import dataclass

    @dataclass
    class YAMLFormatSpec:
        style: str = "yaml"

    return YAMLFormatSpec()


def create_code_format() -> ToolOutputFormat:
    """Create a code block format specification.

    Returns:
        ToolOutputFormat configured for code blocks
    """
    from dataclasses import dataclass

    @dataclass
    class CodeFormatSpec:
        style: str = "code"

    return CodeFormatSpec()
