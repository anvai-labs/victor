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

"""Automated documentation generation module.

This module provides automated documentation generation from
Python source code with support for multiple output formats.

Example usage:
    from victor_coding.docgen import get_docgen_manager, DocConfig, DocFormat
    from pathlib import Path

    # Get manager
    manager = get_docgen_manager()

    # Generate documentation for a file
    result = manager.generate_for_file(
        Path("my_module.py"),
        config=DocConfig(output_format=DocFormat.MARKDOWN),
    )

    # Generate documentation for a package
    result = manager.generate_for_package(
        Path("my_package/"),
        write_files=True,
    )

    # Preview without writing
    content = manager.preview_documentation(Path("my_module.py"))
    print(content)

    # Analyze documentation coverage
    coverage = manager.analyze_documentation_coverage(Path("src/"))
    for file, stats in coverage.items():
        print(f"{file}: {stats['coverage_percent']:.1f}% documented")
"""

from victor_coding.docgen.protocol import (
    Attribute,
    ClassDoc,
    DocConfig,
    DocFormat,
    DocGenResult,
    DocStyle,
    Example,
    FunctionDoc,
    GeneratedDoc,
    ModuleDoc,
    PackageDoc,
    Parameter,
    RaisedException,
    ReturnValue,
)
from victor_coding.docgen.parser import (
    BaseDocstringParser,
    CodeAnalyzer,
    GoogleDocstringParser,
    NumpyDocstringParser,
)
from victor_coding.docgen.formatter import (
    BaseFormatter,
    HTMLFormatter,
    MarkdownFormatter,
    RSTFormatter,
    get_formatter,
)
from victor_coding.docgen.manager import (
    DocGenManager,
    get_docgen_manager,
    reset_docgen_manager,
)

__all__ = [
    # Protocol types
    "Attribute",
    "ClassDoc",
    "DocConfig",
    "DocFormat",
    "DocGenResult",
    "DocStyle",
    "Example",
    "FunctionDoc",
    "GeneratedDoc",
    "ModuleDoc",
    "PackageDoc",
    "Parameter",
    "RaisedException",
    "ReturnValue",
    # Parsers
    "BaseDocstringParser",
    "CodeAnalyzer",
    "GoogleDocstringParser",
    "NumpyDocstringParser",
    # Formatters
    "BaseFormatter",
    "HTMLFormatter",
    "MarkdownFormatter",
    "RSTFormatter",
    "get_formatter",
    # Manager
    "DocGenManager",
    "get_docgen_manager",
    "reset_docgen_manager",
]
