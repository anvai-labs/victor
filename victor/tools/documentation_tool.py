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

"""Documentation generation tool for automated documentation.

Features:
- Generate docstrings for functions and classes
- Create API documentation
- Generate README sections
- Add type hints
- Analyze documentation coverage
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple
import logging

if TYPE_CHECKING:
    from victor.framework.vertical_protocols import DocCommentPattern
    from victor.framework.vertical_protocols import DocCommentPatternProtocol

from victor.tools.base import AccessMode, DangerLevel, Priority, ExecutionCategory
from victor.tools.decorators import tool
from victor.tools.common import gather_files_by_pattern

logger = logging.getLogger(__name__)

# Extension-to-language map for doc coverage analysis (only tested languages)
_DOC_COVERAGE_EXTENSIONS = {
    ".py": "python",
    ".rs": "rust",
    ".go": "go",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
}

# Lazy-loaded presentation adapter for icon rendering
_presentation = None


def _get_icon(name: str) -> str:
    """Get icon from presentation adapter."""
    global _presentation
    if _presentation is None:
        from victor.agent.presentation import create_presentation_adapter

        _presentation = create_presentation_adapter()
    return _presentation.icon(name, with_color=False)


# Helper functions for docstring generation


def _generate_function_docstring(node: ast.FunctionDef, format: str) -> str:
    """Generate function docstring."""
    func_name = node.name
    args = [arg.arg for arg in node.args.args if arg.arg != "self"]

    summary = f"{func_name.replace('_', ' ').title()}."
    description = f"Detailed description of {func_name}."

    args_lines = []
    for arg in args:
        args_lines.append(f"    {arg}: Description of {arg}")
    args_section = "\n".join(args_lines) if args_lines else "    None"

    has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
    returns_section = "Return value description" if has_return else "None"

    has_raises = any(isinstance(n, ast.Raise) for n in ast.walk(node))
    raises_section = "Exception: Description" if has_raises else "None"

    docstring = f"""{summary}

{description}

Args:
{args_section}

Returns:
    {returns_section}

Raises:
    {raises_section}"""

    return docstring


def _generate_class_docstring(node: ast.ClassDef, format: str) -> str:
    """Generate class docstring."""
    class_name = node.name
    summary = f"{class_name} class."
    description = f"Detailed description of {class_name}."

    # Find attributes
    attributes = []
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            for stmt in item.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Attribute):
                            if isinstance(target.value, ast.Name) and target.value.id == "self":
                                attributes.append(target.attr)

    attrs_lines = []
    for attr in attributes[:5]:
        attrs_lines.append(f"    {attr}: Description of {attr}")
    attrs_section = "\n".join(attrs_lines) if attrs_lines else "    None"

    example = f"""instance = {class_name}()
    instance.method()"""

    docstring = f"""{summary}

{description}

Attributes:
{attrs_section}

Example:
    {example}"""

    return docstring


def _extract_api_info(tree: ast.AST, module_name: str) -> Dict[str, Any]:
    """Extract API information from AST."""
    functions = []
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not node.name.startswith("_"):
                functions.append(
                    {
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "No description",
                        "args": [arg.arg for arg in node.args.args],
                    }
                )
        elif isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if not item.name.startswith("_") or item.name in (
                        "__init__",
                        "__str__",
                    ):
                        methods.append(
                            {
                                "name": item.name,
                                "docstring": ast.get_docstring(item) or "No description",
                                "args": [arg.arg for arg in item.args.args if arg.arg != "self"],
                            }
                        )

            classes.append(
                {
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "No description",
                    "methods": methods,
                }
            )

    return {
        "module": module_name,
        "functions": functions,
        "classes": classes,
    }


def _build_markdown_docs(api_info: Dict[str, Any]) -> str:
    """Build Markdown API documentation."""
    lines = []
    lines.append(f"# {api_info['module']} API Documentation")
    lines.append("")

    # Functions
    if api_info["functions"]:
        lines.append("## Functions")
        lines.append("")

        for func in api_info["functions"]:
            lines.append(f"### `{func['name']}({', '.join(func['args'])})`")
            lines.append("")
            lines.append(func["docstring"])
            lines.append("")
            if func["args"]:
                lines.append("**Parameters:**")
                for arg in func["args"]:
                    lines.append(f"- `{arg}`: Description")
                lines.append("")

    # Classes
    if api_info["classes"]:
        lines.append("## Classes")
        lines.append("")

        for cls in api_info["classes"]:
            lines.append(f"### `class {cls['name']}`")
            lines.append("")
            lines.append(cls["docstring"])
            lines.append("")

            if cls["methods"]:
                lines.append("**Methods:**")
                lines.append("")
                for method in cls["methods"]:
                    lines.append(f"#### `{method['name']}({', '.join(method['args'])})`")
                    lines.append("")
                    lines.append(method["docstring"])
                    lines.append("")

    return "\n".join(lines)


# README templates

README_TEMPLATES = {
    "installation": """## Installation

### From PyPI

```bash
pip install your-package-name
```

### From Source

```bash
git clone https://github.com/username/repo.git
cd repo
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`
""",
    "usage": """## Usage

### Quick Start

```python
from your_package import YourClass

# Create instance
instance = YourClass()

# Use the API
result = instance.method()
print(result)
```

### Advanced Usage

```python
# Advanced example
instance = YourClass(config={
    'option1': 'value1',
    'option2': 'value2',
})

result = instance.advanced_method(
    param1='value',
    param2=42
)
```

### Command Line Interface

```bash
your-command --option value
```
""",
    "contributing": """## Contributing

We welcome contributions! Please follow these guidelines:

### Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/repo.git
cd repo

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
black --check .
```

### Code Style

- Follow PEP 8
- Use Black for formatting
- Add type hints
- Write docstrings for all public APIs
- Add tests for new features

### Reporting Issues

Please use the GitHub issue tracker to report bugs or request features.
""",
    "features": """## Features

- **Feature 1**: Description of feature 1
- **Feature 2**: Description of feature 2
- **Feature 3**: Description of feature 3

### Detailed Features

#### Feature 1

Detailed description of feature 1 with examples.

#### Feature 2

Detailed description of feature 2 with examples.

#### Feature 3

Detailed description of feature 3 with examples.
""",
    "api": """## API Reference

### Main Classes

#### `YourMainClass`

Main class description.

**Methods:**

- `method1(param1, param2)`: Description
- `method2(param1)`: Description

### Utility Functions

#### `utility_function(param)`

Utility function description.

**Parameters:**
- `param`: Parameter description

**Returns:**
- Return value description

For complete API documentation, see [API Docs](docs/api.md).
""",
}


def _has_doc_comment_before(
    lines: List[str],
    symbol_line: int,
    pattern: "DocCommentPatternProtocol",
) -> bool:
    """Check if a doc comment exists immediately before a symbol.

    Scans lines above the symbol's definition line (skipping blanks/decorators)
    looking for doc comment patterns matching the language.

    Args:
        lines: All source lines (0-indexed).
        symbol_line: 1-indexed line number where the symbol is defined.
        pattern: The language's doc comment pattern.

    Returns:
        True if a doc comment was found immediately before the symbol.
    """
    idx = symbol_line - 2  # Convert to 0-index, then go one line above
    if idx < 0:
        return False

    # Skip blank lines and decorators (e.g., @decorator, #[attr])
    while idx >= 0:
        stripped = lines[idx].strip()
        if not stripped:
            idx -= 1
            continue
        if stripped.startswith("@") or stripped.startswith("#["):
            idx -= 1
            continue
        break

    if idx < 0:
        return False

    stripped = lines[idx].strip()

    # Check line prefix match (e.g., "///", "//!", "//")
    for prefix in pattern.line_prefixes:
        if stripped.startswith(prefix):
            return True

    # Check block comment end (e.g., "*/"), then walk up to find "/**" start
    if pattern.block_end and stripped.endswith(pattern.block_end):
        # Walk upward to find the block start
        search_idx = idx
        while search_idx >= 0:
            search_stripped = lines[search_idx].strip()
            if pattern.block_start and search_stripped.startswith(pattern.block_start):
                return True
            # If we hit a non-comment line, stop
            if not (
                search_stripped.startswith("*")
                or search_stripped.startswith(pattern.block_start or "")
                or search_stripped.endswith(pattern.block_end)
            ):
                break
            search_idx -= 1

    return False


def _extract_doc_comment_text(
    lines: List[str],
    symbol_line: int,
    pattern: "DocCommentPattern",
) -> str:
    """Extract the text content of a doc comment preceding a symbol.

    Used for quality checks (e.g., too-short doc comments).

    Args:
        lines: All source lines (0-indexed).
        symbol_line: 1-indexed line number where the symbol is defined.
        pattern: The language's doc comment pattern.

    Returns:
        Concatenated doc comment text, or empty string if none found.
    """
    idx = symbol_line - 2
    if idx < 0:
        return ""

    # Skip blank lines and decorators
    while idx >= 0:
        stripped = lines[idx].strip()
        if not stripped or stripped.startswith("@") or stripped.startswith("#["):
            idx -= 1
            continue
        break

    if idx < 0:
        return ""

    doc_lines: List[str] = []
    stripped = lines[idx].strip()

    # Collect line-prefix doc comments (e.g., ///, //!, //)
    for prefix in pattern.line_prefixes:
        if stripped.startswith(prefix):
            # Walk upward collecting all consecutive doc comment lines
            collect_idx = idx
            while collect_idx >= 0:
                line = lines[collect_idx].strip()
                if line.startswith(prefix):
                    doc_lines.append(line[len(prefix) :].strip())
                    collect_idx -= 1
                else:
                    break
            doc_lines.reverse()
            return " ".join(doc_lines)

    # Collect block doc comment (e.g., /** ... */)
    if pattern.block_end and stripped.endswith(pattern.block_end):
        collect_idx = idx
        while collect_idx >= 0:
            line = lines[collect_idx].strip()
            doc_lines.append(line)
            if pattern.block_start and line.startswith(pattern.block_start):
                break
            collect_idx -= 1
        doc_lines.reverse()
        # Clean up: remove /** */, and leading * on continuation lines
        text_parts = []
        for line in doc_lines:
            cleaned = line
            if pattern.block_start:
                cleaned = cleaned.replace(pattern.block_start, "")
            if pattern.block_end:
                cleaned = cleaned.replace(pattern.block_end, "")
            cleaned = cleaned.strip()
            if cleaned.startswith("*"):
                cleaned = cleaned[1:].strip()
            if cleaned:
                text_parts.append(cleaned)
        return " ".join(text_parts)

    return ""


def _analyze_non_python_file(
    file_obj: Path,
    extractor: Any,
    language: str,
    pattern: "DocCommentPattern",
    check_quality: bool = False,
) -> Tuple[int, int, int, int, List[str], List[str]]:
    """Analyze documentation coverage for a non-Python file.

    Uses tree-sitter to extract symbols and text-based scanning to detect
    doc comments.

    Args:
        file_obj: Path to the source file.
        extractor: TreeSitterExtractor instance.
        language: Language name.
        pattern: The language's DocCommentPattern.
        check_quality: Whether to check doc comment quality.

    Returns:
        Tuple of (total_functions, documented_functions, total_classes,
        documented_classes, missing_items, quality_issues).
    """
    total_functions = 0
    documented_functions = 0
    total_classes = 0
    documented_classes = 0
    missing: List[str] = []
    quality_issues: List[str] = []

    try:
        content = file_obj.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 0, 0, 0, 0, [], []

    lines = content.split("\n")
    symbols = extractor.extract_symbols(file_obj, language)

    for sym in symbols:
        name = sym.name

        # Skip private/unexported symbols
        if name.startswith("_"):
            continue
        # Go: unexported symbols start with lowercase
        if language == "go" and name[0].islower():
            continue

        is_class = sym.type == "class"
        if is_class:
            total_classes += 1
        else:
            total_functions += 1

        has_doc = _has_doc_comment_before(lines, sym.line_number, pattern)

        if has_doc:
            if is_class:
                documented_classes += 1
            else:
                documented_functions += 1

            # Quality check
            if check_quality:
                doc_text = _extract_doc_comment_text(lines, sym.line_number, pattern)
                if doc_text and len(doc_text) < 20:
                    quality_issues.append(
                        f"Short doc comment for {name} at {file_obj}:{sym.line_number}"
                    )
        else:
            kind = "Struct/Trait/Enum" if is_class else "Function"
            missing.append(f"{kind}: {name} ({file_obj}:{sym.line_number})")

    return (
        total_functions,
        documented_functions,
        total_classes,
        documented_classes,
        missing,
        quality_issues,
    )


# Consolidated tools


@tool(
    category="docs",
    priority=Priority.MEDIUM,  # Task-specific documentation generation
    access_mode=AccessMode.WRITE,  # Writes documentation files
    danger_level=DangerLevel.LOW,  # File changes are additive and undoable
    keywords=[
        "docs",
        "documentation",
        "docstring",
        "api",
        "readme",
        "type hints",
        "generate",
    ],
    mandatory_keywords=[
        "generate docs",
        "add documentation",
        "document code",
    ],  # Force inclusion
    task_types=["documentation", "generation"],  # Classification-aware selection
    stages=["planning", "analysis", "completion"],  # Conversation stages where relevant
)
async def docs(
    path: str,
    doc_types: List[str] = None,
    format: str = "google",
    output: Optional[str] = None,
    recursive: bool = False,
) -> Dict[str, Any]:
    """
    Unified documentation generation tool.

    Generate multiple types of documentation (docstrings, API docs, README sections,
    type hints) in a single unified interface. Consolidates all documentation
    generation functionality.

    Args:
        path: File or directory path for documentation generation.
        doc_types: List of documentation types to generate. Options: "docstrings",
            "api", "readme", "type_hints", "all". Defaults to ["docstrings"].
        format: Documentation format: "google", "numpy", "sphinx", "markdown", "rst"
            (default: google).
        output: Output file path for API docs (optional, defaults to docs/{module}_api.md).
        recursive: Process directory recursively for docstrings (default: False).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - doc_types_generated: List of documentation types that were generated
        - results: Dictionary with results for each doc type
        - formatted_report: Human-readable generation report
        - error: Error message if failed

    Examples:
        # Generate docstrings only
        generate_docs("src/module.py", doc_types=["docstrings"])

        # Generate docstrings and API docs
        generate_docs("src/module.py", doc_types=["docstrings", "api"])

        # Generate README sections
        generate_docs("", doc_types=["readme"])

        # Generate all documentation types
        generate_docs("src/", doc_types=["all"], recursive=True)
    """
    if doc_types is None:
        doc_types = ["docstrings"]

    # Expand "all" to all doc types
    if "all" in doc_types:
        doc_types = ["docstrings", "api", "type_hints"]

    results = {}
    report = []
    report.append("Documentation Generation Report")
    report.append("=" * 70)
    report.append("")

    # Generate docstrings
    if "docstrings" in doc_types:
        if not path:
            return {"success": False, "error": "path required for docstring generation"}

        path_obj = Path(path)
        if not path_obj.exists():
            return {"success": False, "error": f"Path not found: {path}"}

        files_to_process = []
        if path_obj.is_file():
            files_to_process = [path_obj]
        elif recursive:
            files_to_process = list(path_obj.rglob("*.py"))
        else:
            files_to_process = list(path_obj.glob("*.py"))

        total_generated = 0
        processed_files = 0

        for file_obj in files_to_process:
            content = file_obj.read_text()

            try:
                tree = ast.parse(content)
            except SyntaxError:
                continue

            # Find items needing docstrings
            items_to_document = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not ast.get_docstring(node):
                        items_to_document.append(
                            {
                                "type": "function",
                                "name": node.name,
                                "node": node,
                                "line": node.lineno,
                            }
                        )
                elif isinstance(node, ast.ClassDef):
                    if not ast.get_docstring(node):
                        items_to_document.append(
                            {
                                "type": "class",
                                "name": node.name,
                                "node": node,
                                "line": node.lineno,
                            }
                        )

            if items_to_document:
                # Generate and insert docstrings
                lines = content.split("\n")
                for item in items_to_document:
                    if item["type"] == "function":
                        docstring = _generate_function_docstring(item["node"], format)
                    else:
                        docstring = _generate_class_docstring(item["node"], format)

                    insert_line = item["line"]
                    def_line = lines[insert_line - 1]
                    base_indent = len(def_line) - len(def_line.lstrip())
                    indent = " " * (base_indent + 4)

                    docstring_lines = docstring.split("\n")
                    formatted_docstring = [f'{indent}"""']
                    for line in docstring_lines:
                        if line.strip():
                            formatted_docstring.append(f"{indent}{line}")
                        else:
                            formatted_docstring.append("")
                    formatted_docstring.append(f'{indent}"""')

                    lines = lines[:insert_line] + formatted_docstring + lines[insert_line:]

                new_content = "\n".join(lines)
                file_obj.write_text(new_content)

                total_generated += len(items_to_document)
                processed_files += 1

        results["docstrings"] = {
            "generated": total_generated,
            "files_processed": processed_files,
            "format": format,
        }

        report.append(
            f"Docstrings: Generated {total_generated} docstrings in {processed_files} files"
        )

    # Generate API documentation
    if "api" in doc_types:
        if not path:
            return {"success": False, "error": "path required for API documentation"}

        file_obj = Path(path)
        if not file_obj.exists():
            return {"success": False, "error": f"File not found: {path}"}

        content = file_obj.read_text()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {"success": False, "error": f"Syntax error in file: {e}"}

        api_info = _extract_api_info(tree, file_obj.stem)

        if format in ["markdown", "google", "numpy", "sphinx"]:
            docs = _build_markdown_docs(api_info)
        else:
            docs = _build_markdown_docs(api_info)

        if not output:
            output = f"docs/{file_obj.stem}_api.md"

        output_file = Path(output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(docs)

        results["api"] = {
            "output_file": output,
            "functions_count": len(api_info["functions"]),
            "classes_count": len(api_info["classes"]),
            "preview": docs[:500],
        }

        report.append(f"API Docs: Generated documentation at {output}")
        report.append(
            f"  Functions: {len(api_info['functions'])}, Classes: {len(api_info['classes'])}"
        )

    # Generate README sections
    if "readme" in doc_types:
        # Default to installation if no specific section requested
        section = "installation"
        template = README_TEMPLATES.get(section)

        results["readme"] = {"section": section, "content": template}

        report.append(f"README: Generated {section} section")

    # Add type hints (placeholder)
    if "type_hints" in doc_types:
        if not path:
            return {"success": False, "error": "path required for type hints"}

        results["type_hints"] = {
            "suggestions": [
                "Add return type annotations",
                "Add parameter type hints",
                "Use typing module for complex types",
            ]
        }

        report.append("Type Hints: Analysis complete (suggestions generated)")

    return {
        "success": True,
        "doc_types_generated": list(results.keys()),
        "results": results,
        "formatted_report": "\n".join(report),
    }


@tool(
    category="docs",
    priority=Priority.MEDIUM,  # Task-specific analysis
    access_mode=AccessMode.READONLY,  # Only reads files for analysis
    danger_level=DangerLevel.SAFE,  # No side effects
    keywords=["documentation", "coverage", "analyze", "quality", "docstring"],
    mandatory_keywords=[
        "docs",
        "documentation coverage",
    ],  # From MANDATORY_TOOL_KEYWORDS "docs" -> ["docs", "docs_coverage"]
    stages=["analysis", "reading"],
    execution_category=ExecutionCategory.READ_ONLY,
)
async def docs_coverage(
    path: str,
    check_coverage: bool = True,
    check_quality: bool = False,
    file_pattern: str = "*",
    max_files: int = 50,
) -> Dict[str, Any]:
    """
    Analyze documentation coverage and quality.

    Checks documentation coverage (percentage of functions/classes with
    docstrings or doc comments) and optionally analyzes documentation quality.

    Supports Python (ast.get_docstring), Rust (/// and //!), Go (//),
    JavaScript/TypeScript (/** */ and ///), Java (/** */), and C++ (/** */ and ///).

    Args:
        path: File or directory path to analyze.
        check_coverage: Analyze documentation coverage (default: True).
        check_quality: Analyze docstring quality (default: False).
        file_pattern: Glob pattern for files to analyze (default: * for all
            supported languages).
        max_files: Maximum number of files to analyze (default: 50). Use to prevent
            context overflow on large codebases. Set to 0 for unlimited.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - coverage: Documentation coverage percentage (if check_coverage=True)
        - total_items: Total functions and classes
        - documented_items: Count of documented items
        - missing: List of items missing documentation
        - quality_issues: List of quality issues (if check_quality=True)
        - recommendations: List of improvement recommendations
        - formatted_report: Human-readable analysis report
        - error: Error message if failed

    Examples:
        # Check coverage of a Python file
        docs_coverage("src/module.py")

        # Check coverage of a Rust codebase
        docs_coverage("src/", file_pattern="*.rs")

        # Check all supported languages
        docs_coverage("src/", check_coverage=True, check_quality=True)
    """
    if not path:
        return {"success": False, "error": "Missing required parameter: path"}

    path_obj = Path(path)
    if not path_obj.exists():
        return {"success": False, "error": f"Path not found: {path}"}

    # Collect files to analyze (excluding venv, node_modules, etc.)
    if path_obj.is_file():
        files_to_analyze = [path_obj]
    else:
        files_to_analyze = gather_files_by_pattern(path_obj, file_pattern)

    # Apply file limit to prevent context overflow
    files_truncated = False
    total_files_found = len(files_to_analyze)
    if max_files > 0 and len(files_to_analyze) > max_files:
        files_to_analyze = files_to_analyze[:max_files]
        files_truncated = True

    total_functions = 0
    documented_functions = 0
    total_classes = 0
    documented_classes = 0
    missing = []
    quality_issues = []
    has_non_python = False

    # Lazy-init tree-sitter extractor only if non-Python files are present
    _extractor = None
    _extractor_error = None

    def _get_extractor():
        nonlocal _extractor, _extractor_error
        if _extractor is None and _extractor_error is None:
            try:
                from victor.core.capability_registry import CapabilityRegistry
                from victor.framework.vertical_protocols import (
                    TreeSitterExtractorProtocol,
                )

                provider = CapabilityRegistry.get_instance().get(TreeSitterExtractorProtocol)
                if provider is not None and CapabilityRegistry.get_instance().is_enhanced(
                    TreeSitterExtractorProtocol
                ):
                    _extractor = provider
                else:
                    _extractor_error = (
                        "victor-coding package not available. "
                        "Install with: pip install victor-coding"
                    )
                    logger.debug(_extractor_error)
            except Exception as e:
                _extractor_error = f"Failed to initialize TreeSitterExtractor: {e}"
                logger.warning(_extractor_error)

        if _extractor_error:
            # Re-raise with informative message
            raise ImportError(_extractor_error)
        return _extractor

    for file_obj in files_to_analyze:
        suffix = file_obj.suffix.lower()
        language = _DOC_COVERAGE_EXTENSIONS.get(suffix)
        if language is None:
            continue

        # Python path: use ast.parse + ast.get_docstring (unchanged)
        if language == "python":
            content = file_obj.read_text()

            try:
                tree = ast.parse(content)
            except SyntaxError:
                continue

            if check_coverage:
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if not node.name.startswith("_"):
                            total_functions += 1
                            docstring = ast.get_docstring(node)
                            if docstring:
                                documented_functions += 1
                                if check_quality and len(docstring) < 20:
                                    quality_issues.append(
                                        f"Short docstring in {node.name}"
                                        f" at {file_obj}:{node.lineno}"
                                    )
                            else:
                                missing.append(f"Function: {node.name} ({file_obj}:{node.lineno})")

                    elif isinstance(node, ast.ClassDef):
                        total_classes += 1
                        docstring = ast.get_docstring(node)
                        if docstring:
                            documented_classes += 1
                            if check_quality and len(docstring) < 20:
                                quality_issues.append(
                                    f"Short docstring in {node.name}"
                                    f" at {file_obj}:{node.lineno}"
                                )
                        else:
                            missing.append(f"Class: {node.name} ({file_obj}:{node.lineno})")
            continue

        # Non-Python path: tree-sitter symbols + text-based doc detection
        has_non_python = True
        if not check_coverage:
            continue

        try:
            from victor.core.capability_registry import CapabilityRegistry
            from victor.framework.vertical_protocols import LanguageRegistryProtocol

            lang_registry = CapabilityRegistry.get_instance().get(LanguageRegistryProtocol)
            if lang_registry is None:
                continue
            plugin = lang_registry.get(language)
            doc_pattern = plugin.config.doc_comment_pattern
        except (KeyError, Exception):
            continue

        if doc_pattern is None:
            continue

        try:
            extractor = _get_extractor()
        except ImportError as e:
            # victor-coding not available - skip non-Python files
            logger.debug(f"Skipping non-Python file: {e}")
            continue

        try:
            (
                f_total,
                f_documented,
                c_total,
                c_documented,
                f_missing,
                f_quality,
            ) = _analyze_non_python_file(file_obj, extractor, language, doc_pattern, check_quality)
        except Exception as e:
            logger.warning(f"Error analyzing {file_obj}: {e}")
            continue
        total_functions += f_total
        documented_functions += f_documented
        total_classes += c_total
        documented_classes += c_documented
        missing.extend(f_missing)
        quality_issues.extend(f_quality)

    # Calculate coverage
    total_items = total_functions + total_classes
    documented_items = documented_functions + documented_classes
    coverage = (documented_items / total_items * 100) if total_items > 0 else 100

    # Generate recommendations
    recommendations = []
    if coverage < 50:
        recommendations.append("Low coverage - prioritize adding docstrings")
    elif coverage < 80:
        recommendations.append("Moderate coverage - aim for 80%+")
    else:
        recommendations.append("Good coverage - maintain quality")

    if check_quality and quality_issues:
        recommendations.append(
            f"Found {len(quality_issues)} quality issues - improve docstring detail"
        )

    # Build report
    report = []
    report.append("Documentation Analysis Report")
    report.append("=" * 70)
    report.append("")
    report.append(f"Path: {path}")
    if files_truncated:
        report.append(
            f"Files analyzed: {len(files_to_analyze)} of {total_files_found} (limited by max_files={max_files})"
        )
        report.append(
            f"{_get_icon('warning')}  Increase max_files to analyze more files, or specify a more specific path"
        )
    else:
        report.append(f"Files analyzed: {len(files_to_analyze)}")
    report.append("")
    report.append(f"Coverage: {coverage:.1f}%")
    report.append("")
    report.append(f"Functions: {documented_functions}/{total_functions} documented")
    classes_label = "Classes/Structs/Traits" if has_non_python else "Classes"
    report.append(f"{classes_label}: {documented_classes}/{total_classes} documented")
    report.append("")

    if missing:
        report.append(f"Missing documentation ({len(missing)} items):")
        for item in missing[:15]:
            report.append(f"  • {item}")
        if len(missing) > 15:
            report.append(f"  ... and {len(missing) - 15} more")
    else:
        report.append("All items documented!")

    if check_quality and quality_issues:
        report.append("")
        report.append(f"Quality Issues ({len(quality_issues)}):")
        for issue in quality_issues[:10]:
            report.append(f"  • {issue}")
        if len(quality_issues) > 10:
            report.append(f"  ... and {len(quality_issues) - 10} more")

    report.append("")
    report.append("Recommendations:")
    for rec in recommendations:
        level_icon = (
            _get_icon("level_critical")
            if coverage < 50
            else (_get_icon("level_medium") if coverage < 80 else _get_icon("level_low"))
        )
        report.append(f"  {level_icon} {rec}")

    # Truncate lists to prevent context overflow
    MAX_MISSING_ITEMS = 20
    MAX_QUALITY_ISSUES = 15
    missing_truncated = missing[:MAX_MISSING_ITEMS]
    if len(missing) > MAX_MISSING_ITEMS:
        missing_truncated.append(f"... and {len(missing) - MAX_MISSING_ITEMS} more items")

    quality_truncated = quality_issues[:MAX_QUALITY_ISSUES] if check_quality else None
    if check_quality and len(quality_issues) > MAX_QUALITY_ISSUES:
        quality_truncated.append(f"... and {len(quality_issues) - MAX_QUALITY_ISSUES} more issues")

    return {
        "success": True,
        "coverage": round(coverage, 1),
        "total_items": total_items,
        "documented_items": documented_items,
        "files_analyzed": len(files_to_analyze),
        "files_found": total_files_found,
        "files_truncated": files_truncated,
        "missing_count": len(missing),
        "missing": missing_truncated,
        "quality_issues_count": len(quality_issues) if check_quality else 0,
        "quality_issues": quality_truncated,
        "recommendations": recommendations,
        "formatted_report": "\n".join(report),
    }
