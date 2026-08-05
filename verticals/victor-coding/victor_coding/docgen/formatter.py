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

"""Documentation formatters for various output formats.

Supports Markdown, HTML, RST, and plain text output.
"""

import html
import logging
from abc import ABC, abstractmethod

from victor_coding.docgen.protocol import (
    ClassDoc,
    DocConfig,
    DocFormat,
    FunctionDoc,
    ModuleDoc,
)

logger = logging.getLogger(__name__)


class BaseFormatter(ABC):
    """Abstract base for documentation formatters.

    Implements Strategy pattern for different output formats.
    """

    @property
    @abstractmethod
    def format(self) -> DocFormat:
        """Get the output format."""
        pass

    @abstractmethod
    def format_module(self, module: ModuleDoc, config: DocConfig) -> str:
        """Format a module's documentation.

        Args:
            module: Module documentation
            config: Generation configuration

        Returns:
            Formatted documentation string
        """
        pass

    @abstractmethod
    def format_class(self, cls: ClassDoc, config: DocConfig) -> str:
        """Format a class's documentation."""
        pass

    @abstractmethod
    def format_function(self, func: FunctionDoc, config: DocConfig) -> str:
        """Format a function's documentation."""
        pass


class MarkdownFormatter(BaseFormatter):
    """Formats documentation as Markdown."""

    @property
    def format(self) -> DocFormat:
        return DocFormat.MARKDOWN

    def format_module(self, module: ModuleDoc, config: DocConfig) -> str:
        """Format module as Markdown."""
        lines = []

        # Title
        lines.append(f"# {module.name}")
        lines.append("")

        # Description
        if module.description:
            lines.append(module.description)
            lines.append("")

        if module.long_description:
            lines.append(module.long_description)
            lines.append("")

        # Source link
        if config.include_source_links:
            lines.append(f"**Source:** `{module.file_path}`")
            lines.append("")

        # Table of contents
        if config.include_toc:
            toc = self._generate_toc(module, config)
            if toc:
                lines.append("## Table of Contents")
                lines.append("")
                lines.extend(toc)
                lines.append("")

        # Constants
        if module.constants:
            lines.append("## Constants")
            lines.append("")
            for const in module.constants:
                lines.append(f"### `{const.name}`")
                if const.type_hint:
                    lines.append(f"**Type:** `{const.type_hint}`")
                if const.description:
                    lines.append(const.description)
                lines.append("")

        # Functions
        if module.functions:
            lines.append("## Functions")
            lines.append("")
            for func in module.functions:
                if func.name.startswith("_") and not config.include_private:
                    continue
                lines.append(self.format_function(func, config))
                lines.append("")

        # Classes
        if module.classes:
            lines.append("## Classes")
            lines.append("")
            for cls in module.classes:
                if cls.name.startswith("_") and not config.include_private:
                    continue
                lines.append(self.format_class(cls, config))
                lines.append("")

        # Examples
        if config.include_examples and module.examples:
            lines.append("## Examples")
            lines.append("")
            for example in module.examples:
                if example.description:
                    lines.append(example.description)
                lines.append("```python")
                lines.append(example.code)
                lines.append("```")
                if example.output:
                    lines.append(f"Output: `{example.output}`")
                lines.append("")

        # Notes
        if module.notes:
            lines.append("## Notes")
            lines.append("")
            for note in module.notes:
                lines.append(f"> {note}")
            lines.append("")

        return "\n".join(lines)

    def format_class(self, cls: ClassDoc, config: DocConfig) -> str:
        """Format class as Markdown."""
        lines = []

        # Class header
        lines.append(f"### `class {cls.name}`")
        lines.append("")

        # Inheritance
        if cls.bases:
            lines.append(f"**Bases:** {', '.join(f'`{b}`' for b in cls.bases)}")
            lines.append("")

        # Description
        if cls.description:
            lines.append(cls.description)
            lines.append("")

        if cls.long_description:
            lines.append(cls.long_description)
            lines.append("")

        # Deprecation warning
        if cls.deprecation:
            lines.append(f"> ⚠️ **Deprecated:** {cls.deprecation}")
            lines.append("")

        # Attributes
        if cls.attributes:
            lines.append("#### Attributes")
            lines.append("")
            lines.append("| Name | Type | Description |")
            lines.append("|------|------|-------------|")
            for attr in cls.attributes:
                type_str = f"`{attr.type_hint}`" if attr.type_hint else ""
                lines.append(f"| `{attr.name}` | {type_str} | {attr.description} |")
            lines.append("")

        # Constructor
        init_methods = [m for m in cls.methods if m.name == "__init__"]
        if init_methods:
            lines.append("#### Constructor")
            lines.append("")
            lines.append(self._format_method(init_methods[0], config))
            lines.append("")

        # Properties
        if cls.properties:
            lines.append("#### Properties")
            lines.append("")
            for prop in cls.properties:
                if prop.name.startswith("_") and not config.include_private:
                    continue
                lines.append(self._format_method(prop, config, is_property=True))
                lines.append("")

        # Methods
        public_methods = [
            m
            for m in cls.methods
            if m.name != "__init__" and (not m.name.startswith("_") or config.include_private)
        ]
        if public_methods:
            lines.append("#### Methods")
            lines.append("")
            for method in public_methods:
                lines.append(self._format_method(method, config))
                lines.append("")

        # Class methods
        if cls.class_methods:
            lines.append("#### Class Methods")
            lines.append("")
            for method in cls.class_methods:
                if method.name.startswith("_") and not config.include_private:
                    continue
                lines.append(self._format_method(method, config))
                lines.append("")

        # Static methods
        if cls.static_methods:
            lines.append("#### Static Methods")
            lines.append("")
            for method in cls.static_methods:
                if method.name.startswith("_") and not config.include_private:
                    continue
                lines.append(self._format_method(method, config))
                lines.append("")

        return "\n".join(lines)

    def format_function(self, func: FunctionDoc, config: DocConfig) -> str:
        """Format function as Markdown."""
        return self._format_method(func, config)

    def _format_method(
        self,
        func: FunctionDoc,
        config: DocConfig,
        is_property: bool = False,
    ) -> str:
        """Format a method/function."""
        lines = []

        # Signature
        if is_property:
            lines.append(f"##### `{func.name}` (property)")
        else:
            lines.append(f"##### `{func.signature}`")
        lines.append("")

        # Description
        if func.description:
            lines.append(func.description)
            lines.append("")

        # Deprecation
        if func.deprecation:
            lines.append(f"> ⚠️ **Deprecated:** {func.deprecation}")
            lines.append("")

        # Parameters
        if func.parameters and not is_property:
            params = [p for p in func.parameters if p.name not in ("self", "cls")]
            if params:
                lines.append("**Parameters:**")
                lines.append("")
                for param in params:
                    type_str = f" (`{param.type_hint}`)" if param.type_hint else ""
                    default_str = f" = `{param.default}`" if param.default else ""
                    lines.append(f"- `{param.name}`{type_str}{default_str}: {param.description}")
                lines.append("")

        # Returns
        if func.returns:
            lines.append("**Returns:**")
            type_str = f" `{func.returns.type_hint}`" if func.returns.type_hint else ""
            lines.append(f"{type_str}: {func.returns.description}")
            lines.append("")

        # Yields
        if func.yields:
            lines.append("**Yields:**")
            type_str = f" `{func.yields.type_hint}`" if func.yields.type_hint else ""
            lines.append(f"{type_str}: {func.yields.description}")
            lines.append("")

        # Raises
        if func.raises:
            lines.append("**Raises:**")
            lines.append("")
            for exc in func.raises:
                lines.append(f"- `{exc.exception_type}`: {exc.description}")
            lines.append("")

        # Examples
        if config.include_examples and func.examples:
            lines.append("**Examples:**")
            lines.append("")
            for example in func.examples:
                lines.append("```python")
                lines.append(example.code)
                lines.append("```")
                if example.output:
                    lines.append(f"Output: `{example.output}`")
            lines.append("")

        return "\n".join(lines)

    def _generate_toc(self, module: ModuleDoc, config: DocConfig) -> list[str]:
        """Generate table of contents."""
        toc = []

        if module.constants:
            toc.append("- [Constants](#constants)")

        if module.functions:
            toc.append("- [Functions](#functions)")
            for func in module.functions:
                if func.name.startswith("_") and not config.include_private:
                    continue
                anchor = func.name.lower().replace("_", "-")
                toc.append(f"  - [`{func.name}`](#{anchor})")

        if module.classes:
            toc.append("- [Classes](#classes)")
            for cls in module.classes:
                if cls.name.startswith("_") and not config.include_private:
                    continue
                anchor = f"class-{cls.name.lower()}"
                toc.append(f"  - [`{cls.name}`](#{anchor})")

        return toc


class HTMLFormatter(BaseFormatter):
    """Formats documentation as HTML."""

    @property
    def format(self) -> DocFormat:
        return DocFormat.HTML

    def format_module(self, module: ModuleDoc, config: DocConfig) -> str:
        """Format module as HTML."""
        lines = []

        # HTML header
        lines.append("<!DOCTYPE html>")
        lines.append("<html>")
        lines.append("<head>")
        lines.append(f"<title>{html.escape(module.name)} - Documentation</title>")
        lines.append("<style>")
        lines.append(self._get_css())
        lines.append("</style>")
        lines.append("</head>")
        lines.append("<body>")
        lines.append('<div class="container">')

        # Title
        lines.append(f"<h1>{html.escape(module.name)}</h1>")

        # Description
        if module.description:
            lines.append(f"<p class='description'>{html.escape(module.description)}</p>")

        if module.long_description:
            lines.append(f"<p>{html.escape(module.long_description)}</p>")

        # Source link
        if config.include_source_links:
            lines.append(
                f"<p class='source'>Source: <code>{html.escape(str(module.file_path))}</code></p>"
            )

        # Functions
        if module.functions:
            lines.append("<h2>Functions</h2>")
            for func in module.functions:
                if func.name.startswith("_") and not config.include_private:
                    continue
                lines.append(self.format_function(func, config))

        # Classes
        if module.classes:
            lines.append("<h2>Classes</h2>")
            for cls in module.classes:
                if cls.name.startswith("_") and not config.include_private:
                    continue
                lines.append(self.format_class(cls, config))

        lines.append("</div>")
        lines.append("</body>")
        lines.append("</html>")

        return "\n".join(lines)

    def format_class(self, cls: ClassDoc, config: DocConfig) -> str:
        """Format class as HTML."""
        lines = []

        lines.append(f'<div class="class" id="{html.escape(cls.name)}">')
        lines.append(f"<h3>class {html.escape(cls.name)}</h3>")

        if cls.bases:
            bases_str = ", ".join(html.escape(b) for b in cls.bases)
            lines.append(f"<p class='bases'>Inherits: {bases_str}</p>")

        if cls.description:
            lines.append(f"<p>{html.escape(cls.description)}</p>")

        # Methods
        if cls.methods:
            lines.append("<h4>Methods</h4>")
            for method in cls.methods:
                if method.name.startswith("_") and not config.include_private:
                    continue
                lines.append(self._format_method_html(method, config))

        lines.append("</div>")

        return "\n".join(lines)

    def format_function(self, func: FunctionDoc, config: DocConfig) -> str:
        """Format function as HTML."""
        return self._format_method_html(func, config)

    def _format_method_html(self, func: FunctionDoc, config: DocConfig) -> str:
        """Format method as HTML."""
        lines = []

        lines.append(f'<div class="function" id="{html.escape(func.name)}">')
        lines.append(f"<h4><code>{html.escape(func.signature)}</code></h4>")

        if func.description:
            lines.append(f"<p>{html.escape(func.description)}</p>")

        # Parameters
        if func.parameters:
            params = [p for p in func.parameters if p.name not in ("self", "cls")]
            if params:
                lines.append("<h5>Parameters</h5>")
                lines.append("<table>")
                lines.append("<tr><th>Name</th><th>Type</th><th>Description</th></tr>")
                for param in params:
                    type_str = html.escape(param.type_hint or "")
                    lines.append(
                        f"<tr><td><code>{html.escape(param.name)}</code></td>"
                        f"<td><code>{type_str}</code></td>"
                        f"<td>{html.escape(param.description)}</td></tr>"
                    )
                lines.append("</table>")

        # Returns
        if func.returns:
            lines.append("<h5>Returns</h5>")
            type_str = (
                f"<code>{html.escape(func.returns.type_hint or '')}</code>: "
                if func.returns.type_hint
                else ""
            )
            lines.append(f"<p>{type_str}{html.escape(func.returns.description)}</p>")

        lines.append("</div>")

        return "\n".join(lines)

    def _get_css(self) -> str:
        """Get CSS styles."""
        return """
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; }
            .container { max-width: 900px; margin: 0 auto; padding: 20px; }
            h1, h2, h3, h4 { color: #333; }
            code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: monospace; }
            pre { background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
            .function, .class { border-left: 3px solid #007acc; padding-left: 15px; margin: 20px 0; }
            .description { font-size: 1.1em; color: #555; }
            .source { color: #666; font-size: 0.9em; }
            table { border-collapse: collapse; width: 100%; margin: 10px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background: #f4f4f4; }
        """


class RSTFormatter(BaseFormatter):
    """Formats documentation as reStructuredText."""

    @property
    def format(self) -> DocFormat:
        return DocFormat.RST

    def format_module(self, module: ModuleDoc, config: DocConfig) -> str:
        """Format module as RST."""
        lines = []

        # Title
        title = module.name
        lines.append("=" * len(title))
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")

        if module.description:
            lines.append(module.description)
            lines.append("")

        # Functions
        if module.functions:
            lines.append("Functions")
            lines.append("-" * len("Functions"))
            lines.append("")
            for func in module.functions:
                if func.name.startswith("_") and not config.include_private:
                    continue
                lines.append(self.format_function(func, config))
                lines.append("")

        # Classes
        if module.classes:
            lines.append("Classes")
            lines.append("-" * len("Classes"))
            lines.append("")
            for cls in module.classes:
                if cls.name.startswith("_") and not config.include_private:
                    continue
                lines.append(self.format_class(cls, config))
                lines.append("")

        return "\n".join(lines)

    def format_class(self, cls: ClassDoc, config: DocConfig) -> str:
        """Format class as RST."""
        lines = []

        lines.append(f".. py:class:: {cls.name}")
        lines.append("")

        if cls.description:
            lines.append(f"   {cls.description}")
            lines.append("")

        for method in cls.methods:
            if method.name.startswith("_") and not config.include_private:
                continue
            method_lines = self._format_method_rst(method, config)
            for line in method_lines.split("\n"):
                lines.append(f"   {line}")
            lines.append("")

        return "\n".join(lines)

    def format_function(self, func: FunctionDoc, config: DocConfig) -> str:
        """Format function as RST."""
        return self._format_method_rst(func, config)

    def _format_method_rst(self, func: FunctionDoc, config: DocConfig) -> str:
        """Format method as RST."""
        lines = []

        lines.append(f".. py:function:: {func.signature}")
        lines.append("")

        if func.description:
            lines.append(f"   {func.description}")
            lines.append("")

        # Parameters
        for param in func.parameters:
            if param.name in ("self", "cls"):
                continue
            type_str = f" ({param.type_hint})" if param.type_hint else ""
            lines.append(f"   :param {param.name}: {param.description}{type_str}")

        # Returns
        if func.returns:
            type_str = f" ({func.returns.type_hint})" if func.returns.type_hint else ""
            lines.append(f"   :returns: {func.returns.description}{type_str}")

        # Raises
        for exc in func.raises:
            lines.append(f"   :raises {exc.exception_type}: {exc.description}")

        return "\n".join(lines)


# Formatter registry
FORMATTERS: dict[DocFormat, type[BaseFormatter]] = {
    DocFormat.MARKDOWN: MarkdownFormatter,
    DocFormat.HTML: HTMLFormatter,
    DocFormat.RST: RSTFormatter,
}


def get_formatter(format: DocFormat) -> BaseFormatter:
    """Get a formatter for the specified format.

    Args:
        format: Output format

    Returns:
        Formatter instance
    """
    formatter_class = FORMATTERS.get(format)
    if formatter_class:
        return formatter_class()
    return MarkdownFormatter()  # Default
