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

"""Docstring parsers for various styles.

Supports Google, NumPy, Sphinx, and Epytext docstring formats.
"""

import ast
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from victor_coding.docgen.protocol import (
    Attribute,
    ClassDoc,
    DocStyle,
    Example,
    FunctionDoc,
    ModuleDoc,
    Parameter,
    RaisedException,
    ReturnValue,
)

logger = logging.getLogger(__name__)


class BaseDocstringParser(ABC):
    """Abstract base for docstring parsers.

    Implements Strategy pattern for different docstring styles.
    """

    @property
    @abstractmethod
    def style(self) -> DocStyle:
        """Get the docstring style this parser handles."""
        pass

    @abstractmethod
    def parse(self, docstring: str) -> dict:
        """Parse a docstring into components.

        Args:
            docstring: Raw docstring text

        Returns:
            Dictionary with parsed components
        """
        pass


class GoogleDocstringParser(BaseDocstringParser):
    """Parser for Google-style docstrings."""

    @property
    def style(self) -> DocStyle:
        return DocStyle.GOOGLE

    SECTION_REGEX = re.compile(
        r"^(\s*)(Args|Arguments|Parameters|Returns|Yields|Raises|"
        r"Attributes|Examples?|Notes?|References|See Also|Warnings?|"
        r"Todo|Deprecated):\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    PARAM_REGEX = re.compile(r"^\s*(\*{0,2}\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.*)$")

    def parse(self, docstring: str) -> dict:
        """Parse Google-style docstring."""
        if not docstring:
            return {}

        result = {
            "description": "",
            "long_description": "",
            "parameters": [],
            "returns": None,
            "yields": None,
            "raises": [],
            "attributes": [],
            "examples": [],
            "notes": [],
            "see_also": [],
            "deprecated": None,
        }

        # Split into sections
        sections = self._split_sections(docstring)

        # Parse each section
        for section_name, section_content in sections.items():
            if section_name == "_description":
                desc_parts = section_content.strip().split("\n\n", 1)
                result["description"] = desc_parts[0].strip()
                if len(desc_parts) > 1:
                    result["long_description"] = desc_parts[1].strip()

            elif section_name.lower() in ("args", "arguments", "parameters"):
                result["parameters"] = self._parse_parameters(section_content)

            elif section_name.lower() == "returns":
                result["returns"] = self._parse_return(section_content)

            elif section_name.lower() == "yields":
                result["yields"] = self._parse_return(section_content)

            elif section_name.lower() == "raises":
                result["raises"] = self._parse_raises(section_content)

            elif section_name.lower() == "attributes":
                result["attributes"] = self._parse_attributes(section_content)

            elif section_name.lower() in ("example", "examples"):
                result["examples"] = self._parse_examples(section_content)

            elif section_name.lower() in ("note", "notes"):
                result["notes"] = [section_content.strip()]

            elif section_name.lower() == "see also":
                result["see_also"] = [s.strip() for s in section_content.split("\n") if s.strip()]

            elif section_name.lower() == "deprecated":
                result["deprecated"] = section_content.strip()

        return result

    def _split_sections(self, docstring: str) -> dict[str, str]:
        """Split docstring into sections."""
        sections = {}
        current_section = "_description"
        current_content = []

        for line in docstring.split("\n"):
            match = self.SECTION_REGEX.match(line)
            if match:
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content)
                current_section = match.group(2)
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = "\n".join(current_content)

        return sections

    def _parse_parameters(self, content: str) -> list[Parameter]:
        """Parse parameters section."""
        parameters = []
        lines = content.strip().split("\n")
        current_param = None
        current_desc = []

        for line in lines:
            match = self.PARAM_REGEX.match(line)
            if match:
                # Save previous param
                if current_param:
                    current_param.description = " ".join(current_desc).strip()
                    parameters.append(current_param)

                name = match.group(1)
                type_hint = match.group(2)
                desc = match.group(3)

                current_param = Parameter(
                    name=name.lstrip("*"),
                    type_hint=type_hint,
                    description=desc,
                    is_optional="optional" in (type_hint or "").lower(),
                )
                current_desc = [desc] if desc else []
            elif current_param and line.strip():
                current_desc.append(line.strip())

        # Save last param
        if current_param:
            current_param.description = " ".join(current_desc).strip()
            parameters.append(current_param)

        return parameters

    def _parse_return(self, content: str) -> Optional[ReturnValue]:
        """Parse returns/yields section."""
        content = content.strip()
        if not content:
            return None

        # Try to parse type: description format
        match = re.match(r"^(\w+(?:\[.+\])?)\s*:\s*(.*)$", content, re.DOTALL)
        if match:
            return ReturnValue(
                type_hint=match.group(1),
                description=match.group(2).strip(),
            )

        return ReturnValue(description=content)

    def _parse_raises(self, content: str) -> list[RaisedException]:
        """Parse raises section."""
        raises = []
        lines = content.strip().split("\n")

        for line in lines:
            match = re.match(r"^\s*(\w+)\s*:\s*(.*)$", line)
            if match:
                raises.append(
                    RaisedException(
                        exception_type=match.group(1),
                        description=match.group(2).strip(),
                    )
                )

        return raises

    def _parse_attributes(self, content: str) -> list[Attribute]:
        """Parse attributes section."""
        attributes = []
        lines = content.strip().split("\n")

        for line in lines:
            match = self.PARAM_REGEX.match(line)
            if match:
                attributes.append(
                    Attribute(
                        name=match.group(1),
                        type_hint=match.group(2),
                        description=match.group(3),
                    )
                )

        return attributes

    def _parse_examples(self, content: str) -> list[Example]:
        """Parse examples section."""
        examples = []
        lines = content.strip().split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            if line.strip().startswith(">>>"):
                in_code = True
                code_lines.append(line.strip())
            elif in_code and line.strip().startswith("..."):
                code_lines.append(line.strip())
            elif in_code and not line.strip().startswith(">>>"):
                # This is output or end of example
                if code_lines:
                    examples.append(
                        Example(
                            code="\n".join(code_lines),
                            output=line.strip() if line.strip() else None,
                        )
                    )
                code_lines = []
                in_code = False

        if code_lines:
            examples.append(Example(code="\n".join(code_lines)))

        return examples


class NumpyDocstringParser(BaseDocstringParser):
    """Parser for NumPy-style docstrings."""

    @property
    def style(self) -> DocStyle:
        return DocStyle.NUMPY

    SECTION_REGEX = re.compile(
        r"^(\s*)(Parameters|Returns|Yields|Raises|See Also|Notes|"
        r"References|Examples|Attributes|Methods|Warnings)\s*\n\s*-+\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    def parse(self, docstring: str) -> dict:
        """Parse NumPy-style docstring."""
        if not docstring:
            return {}

        result = {
            "description": "",
            "long_description": "",
            "parameters": [],
            "returns": None,
            "yields": None,
            "raises": [],
            "attributes": [],
            "examples": [],
            "notes": [],
            "see_also": [],
        }

        # Split into sections
        sections = self._split_sections(docstring)

        for section_name, section_content in sections.items():
            if section_name == "_description":
                desc_parts = section_content.strip().split("\n\n", 1)
                result["description"] = desc_parts[0].strip()
                if len(desc_parts) > 1:
                    result["long_description"] = desc_parts[1].strip()

            elif section_name.lower() == "parameters":
                result["parameters"] = self._parse_numpy_params(section_content)

            elif section_name.lower() == "returns":
                result["returns"] = self._parse_numpy_return(section_content)

            elif section_name.lower() == "yields":
                result["yields"] = self._parse_numpy_return(section_content)

            elif section_name.lower() == "raises":
                result["raises"] = self._parse_numpy_raises(section_content)

            elif section_name.lower() == "examples":
                result["examples"] = self._parse_examples(section_content)

            elif section_name.lower() == "notes":
                result["notes"] = [section_content.strip()]

            elif section_name.lower() == "see also":
                result["see_also"] = [s.strip() for s in section_content.split("\n") if s.strip()]

        return result

    def _split_sections(self, docstring: str) -> dict[str, str]:
        """Split docstring into sections."""
        sections = {}
        lines = docstring.split("\n")
        current_section = "_description"
        current_content = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check for section header (name followed by dashes)
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if re.match(r"^\s*-+\s*$", next_line):
                    header_match = re.match(r"^\s*(\w+(?:\s+\w+)?)\s*$", line)
                    if header_match:
                        # Save previous section
                        if current_content:
                            sections[current_section] = "\n".join(current_content)
                        current_section = header_match.group(1)
                        current_content = []
                        i += 2  # Skip header and dashes
                        continue

            current_content.append(line)
            i += 1

        if current_content:
            sections[current_section] = "\n".join(current_content)

        return sections

    def _parse_numpy_params(self, content: str) -> list[Parameter]:
        """Parse NumPy-style parameters."""
        parameters = []
        lines = content.strip().split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # Parameter line: name : type
            match = re.match(r"^\s*(\w+)\s*:\s*(.+)$", line)
            if match:
                name = match.group(1)
                type_hint = match.group(2)

                # Get description from following indented lines
                desc_lines = []
                i += 1
                while i < len(lines) and lines[i].startswith("    "):
                    desc_lines.append(lines[i].strip())
                    i += 1

                parameters.append(
                    Parameter(
                        name=name,
                        type_hint=type_hint,
                        description=" ".join(desc_lines),
                    )
                )
            else:
                i += 1

        return parameters

    def _parse_numpy_return(self, content: str) -> Optional[ReturnValue]:
        """Parse NumPy-style return."""
        lines = content.strip().split("\n")
        if not lines:
            return None

        # First line might be type
        first_line = lines[0].strip()
        match = re.match(r"^(\w+(?:\[.+\])?)\s*$", first_line)

        if match:
            type_hint = match.group(1)
            description = " ".join(ln.strip() for ln in lines[1:])
            return ReturnValue(type_hint=type_hint, description=description)

        return ReturnValue(description=" ".join(ln.strip() for ln in lines))

    def _parse_numpy_raises(self, content: str) -> list[RaisedException]:
        """Parse NumPy-style raises."""
        raises = []
        lines = content.strip().split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]
            if re.match(r"^\w+$", line.strip()):
                exc_type = line.strip()
                desc_lines = []
                i += 1
                while i < len(lines) and lines[i].startswith("    "):
                    desc_lines.append(lines[i].strip())
                    i += 1
                raises.append(
                    RaisedException(
                        exception_type=exc_type,
                        description=" ".join(desc_lines),
                    )
                )
            else:
                i += 1

        return raises

    def _parse_examples(self, content: str) -> list[Example]:
        """Parse examples (same as Google style)."""
        examples = []
        code_lines = []

        for line in content.strip().split("\n"):
            if line.strip().startswith(">>>"):
                code_lines.append(line.strip())
            elif code_lines and not line.strip().startswith(">>>"):
                examples.append(
                    Example(
                        code="\n".join(code_lines),
                        output=line.strip() if line.strip() else None,
                    )
                )
                code_lines = []

        if code_lines:
            examples.append(Example(code="\n".join(code_lines)))

        return examples


class CodeAnalyzer:
    """Analyzes Python code to extract documentation."""

    def __init__(self, style: DocStyle = DocStyle.AUTO):
        """Initialize the analyzer.

        Args:
            style: Docstring style to parse
        """
        self.style = style
        self._parsers: dict[DocStyle, BaseDocstringParser] = {
            DocStyle.GOOGLE: GoogleDocstringParser(),
            DocStyle.NUMPY: NumpyDocstringParser(),
        }

    def analyze_file(self, file_path: Path) -> Optional[ModuleDoc]:
        """Analyze a Python file.

        Args:
            file_path: Path to the file

        Returns:
            ModuleDoc or None if parsing fails
        """
        try:
            source = file_path.read_text()
            return self.analyze_source(source, file_path)
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return None

    def analyze_source(self, source: str, file_path: Path) -> Optional[ModuleDoc]:
        """Analyze Python source code.

        Args:
            source: Python source code
            file_path: Path for reference

        Returns:
            ModuleDoc or None
        """
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return None

        # Detect docstring style if auto
        style = self._detect_style(source) if self.style == DocStyle.AUTO else self.style
        parser = self._parsers.get(style, self._parsers[DocStyle.GOOGLE])

        # Parse module docstring
        module_docstring = ast.get_docstring(tree)
        parsed_doc = parser.parse(module_docstring or "")

        module_doc = ModuleDoc(
            name=file_path.stem,
            file_path=file_path,
            description=parsed_doc.get("description", ""),
            long_description=parsed_doc.get("long_description", ""),
            examples=parsed_doc.get("examples", []),
            notes=parsed_doc.get("notes", []),
        )

        # Extract __all__ if present
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            module_doc.all_exports = [
                                elt.value
                                for elt in node.value.elts
                                if isinstance(elt, ast.Constant)
                            ]

        # Extract functions and classes
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_doc = self._analyze_function(node, parser)
                if func_doc:
                    module_doc.functions.append(func_doc)

            elif isinstance(node, ast.ClassDef):
                class_doc = self._analyze_class(node, parser)
                if class_doc:
                    module_doc.classes.append(class_doc)

            elif isinstance(node, ast.Assign):
                # Module-level constants
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        module_doc.constants.append(
                            Attribute(
                                name=target.id,
                                type_hint=self._get_type_annotation(node.value),
                                is_class_var=True,
                            )
                        )

        return module_doc

    def _analyze_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parser: BaseDocstringParser,
    ) -> Optional[FunctionDoc]:
        """Analyze a function definition."""
        docstring = ast.get_docstring(node)
        parsed = parser.parse(docstring or "")

        # Build signature
        signature = self._build_signature(node)

        # Extract parameters from AST
        parameters = self._extract_parameters(node)

        # Merge with docstring params
        docstring_params = {p.name: p for p in parsed.get("parameters", [])}
        for param in parameters:
            if param.name in docstring_params:
                doc_param = docstring_params[param.name]
                param.description = doc_param.description
                if not param.type_hint:
                    param.type_hint = doc_param.type_hint

        # Extract decorators
        decorators = []
        is_static = False
        is_classmethod = False
        is_property = False

        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
                if dec.id == "staticmethod":
                    is_static = True
                elif dec.id == "classmethod":
                    is_classmethod = True
                elif dec.id == "property":
                    is_property = True
            elif isinstance(dec, ast.Attribute):
                decorators.append(ast.unparse(dec))

        return FunctionDoc(
            name=node.name,
            signature=signature,
            description=parsed.get("description", ""),
            long_description=parsed.get("long_description", ""),
            parameters=parameters,
            returns=parsed.get("returns"),
            yields=parsed.get("yields"),
            raises=parsed.get("raises", []),
            examples=parsed.get("examples", []),
            notes=parsed.get("notes", []),
            see_also=parsed.get("see_also", []),
            deprecation=parsed.get("deprecated"),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_static=is_static,
            is_classmethod=is_classmethod,
            is_property=is_property,
            decorators=decorators,
            source_line=node.lineno,
        )

    def _analyze_class(
        self,
        node: ast.ClassDef,
        parser: BaseDocstringParser,
    ) -> Optional[ClassDoc]:
        """Analyze a class definition."""
        docstring = ast.get_docstring(node)
        parsed = parser.parse(docstring or "")

        # Extract bases
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(ast.unparse(base))

        class_doc = ClassDoc(
            name=node.name,
            description=parsed.get("description", ""),
            long_description=parsed.get("long_description", ""),
            bases=bases,
            attributes=parsed.get("attributes", []),
            examples=parsed.get("examples", []),
            notes=parsed.get("notes", []),
            see_also=parsed.get("see_also", []),
            deprecation=parsed.get("deprecated"),
            source_line=node.lineno,
        )

        # Analyze methods
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_doc = self._analyze_function(item, parser)
                if method_doc:
                    if method_doc.is_static:
                        class_doc.static_methods.append(method_doc)
                    elif method_doc.is_classmethod:
                        class_doc.class_methods.append(method_doc)
                    elif method_doc.is_property:
                        class_doc.properties.append(method_doc)
                    else:
                        class_doc.methods.append(method_doc)

        return class_doc

    def _build_signature(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> str:
        """Build function signature string."""
        args = []

        # Regular args
        for i, arg in enumerate(node.args.args):
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"

            # Check for default
            default_idx = i - (len(node.args.args) - len(node.args.defaults))
            if default_idx >= 0:
                default = node.args.defaults[default_idx]
                arg_str += f" = {ast.unparse(default)}"

            args.append(arg_str)

        # *args
        if node.args.vararg:
            arg_str = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                arg_str += f": {ast.unparse(node.args.vararg.annotation)}"
            args.append(arg_str)

        # **kwargs
        if node.args.kwarg:
            arg_str = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                arg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
            args.append(arg_str)

        # Return type
        return_type = ""
        if node.returns:
            return_type = f" -> {ast.unparse(node.returns)}"

        prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        return f"{prefix}def {node.name}({', '.join(args)}){return_type}"

    def _extract_parameters(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[Parameter]:
        """Extract parameters from function AST."""
        parameters = []

        for i, arg in enumerate(node.args.args):
            type_hint = None
            if arg.annotation:
                type_hint = ast.unparse(arg.annotation)

            default = None
            default_idx = i - (len(node.args.args) - len(node.args.defaults))
            if default_idx >= 0:
                default = ast.unparse(node.args.defaults[default_idx])

            parameters.append(
                Parameter(
                    name=arg.arg,
                    type_hint=type_hint,
                    default=default,
                    is_optional=default is not None,
                )
            )

        return parameters

    def _get_type_annotation(self, node: ast.AST) -> Optional[str]:
        """Get type annotation from a value node."""
        if isinstance(node, ast.Constant):
            return type(node.value).__name__
        elif isinstance(node, ast.List):
            return "list"
        elif isinstance(node, ast.Dict):
            return "dict"
        elif isinstance(node, ast.Set):
            return "set"
        return None

    def _detect_style(self, source: str) -> DocStyle:
        """Auto-detect docstring style."""
        # Look for style indicators
        if re.search(r"^\s*Args:\s*$", source, re.MULTILINE):
            return DocStyle.GOOGLE
        elif re.search(r"^\s*Parameters\s*\n\s*-+", source, re.MULTILINE):
            return DocStyle.NUMPY
        elif re.search(r":param \w+:", source):
            return DocStyle.SPHINX

        return DocStyle.GOOGLE  # Default
