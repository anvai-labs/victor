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

"""Documentation generation protocol types.

Defines data structures for automated documentation generation.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class DocFormat(Enum):
    """Output documentation formats."""

    MARKDOWN = "markdown"
    HTML = "html"
    RST = "rst"  # reStructuredText
    PLAINTEXT = "plaintext"


class DocStyle(Enum):
    """Docstring styles to parse."""

    GOOGLE = "google"
    NUMPY = "numpy"
    SPHINX = "sphinx"
    EPYTEXT = "epytext"
    AUTO = "auto"  # Auto-detect


@dataclass
class Parameter:
    """A function/method parameter."""

    name: str
    type_hint: Optional[str] = None
    description: str = ""
    default: Optional[str] = None
    is_optional: bool = False
    is_keyword_only: bool = False
    is_positional_only: bool = False


@dataclass
class ReturnValue:
    """A function return value."""

    type_hint: Optional[str] = None
    description: str = ""


@dataclass
class RaisedException:
    """An exception that can be raised."""

    exception_type: str
    description: str = ""


@dataclass
class Example:
    """A usage example."""

    code: str
    description: str = ""
    output: Optional[str] = None


@dataclass
class FunctionDoc:
    """Documentation for a function/method."""

    name: str
    signature: str
    description: str = ""
    long_description: str = ""
    parameters: list[Parameter] = field(default_factory=list)
    returns: Optional[ReturnValue] = None
    yields: Optional[ReturnValue] = None
    raises: list[RaisedException] = field(default_factory=list)
    examples: list[Example] = field(default_factory=list)
    deprecation: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    see_also: list[str] = field(default_factory=list)
    is_async: bool = False
    is_static: bool = False
    is_classmethod: bool = False
    is_property: bool = False
    decorators: list[str] = field(default_factory=list)
    source_line: int = 0


@dataclass
class Attribute:
    """A class attribute."""

    name: str
    type_hint: Optional[str] = None
    description: str = ""
    default: Optional[str] = None
    is_class_var: bool = False


@dataclass
class ClassDoc:
    """Documentation for a class."""

    name: str
    description: str = ""
    long_description: str = ""
    bases: list[str] = field(default_factory=list)
    attributes: list[Attribute] = field(default_factory=list)
    methods: list[FunctionDoc] = field(default_factory=list)
    class_methods: list[FunctionDoc] = field(default_factory=list)
    static_methods: list[FunctionDoc] = field(default_factory=list)
    properties: list[FunctionDoc] = field(default_factory=list)
    examples: list[Example] = field(default_factory=list)
    deprecation: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    see_also: list[str] = field(default_factory=list)
    source_line: int = 0


@dataclass
class ModuleDoc:
    """Documentation for a module."""

    name: str
    file_path: Path
    description: str = ""
    long_description: str = ""
    functions: list[FunctionDoc] = field(default_factory=list)
    classes: list[ClassDoc] = field(default_factory=list)
    constants: list[Attribute] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    all_exports: list[str] = field(default_factory=list)
    examples: list[Example] = field(default_factory=list)
    deprecation: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    version: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None


@dataclass
class PackageDoc:
    """Documentation for a package."""

    name: str
    path: Path
    description: str = ""
    modules: list[ModuleDoc] = field(default_factory=list)
    subpackages: list["PackageDoc"] = field(default_factory=list)
    readme: Optional[str] = None


@dataclass
class DocConfig:
    """Configuration for documentation generation."""

    output_format: DocFormat = DocFormat.MARKDOWN
    input_style: DocStyle = DocStyle.AUTO
    include_private: bool = False
    include_dunder: bool = False
    include_source_links: bool = True
    include_toc: bool = True
    include_index: bool = True
    include_examples: bool = True
    max_depth: int = 3
    output_dir: Optional[Path] = None
    base_url: str = ""
    template_dir: Optional[Path] = None


@dataclass
class GeneratedDoc:
    """A generated documentation file."""

    path: Path
    content: str
    format: DocFormat


@dataclass
class DocGenResult:
    """Result of documentation generation."""

    success: bool
    generated_files: list[GeneratedDoc] = field(default_factory=list)
    modules_documented: int = 0
    classes_documented: int = 0
    functions_documented: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
