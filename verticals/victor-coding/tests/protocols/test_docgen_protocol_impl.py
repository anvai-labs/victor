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

"""Tests for docgen protocol module - achieving 70%+ coverage."""

import pytest
from pathlib import Path

pytest.importorskip("victor_coding")

from victor_coding.docgen.protocol import (
    DocFormat,
    DocStyle,
    Parameter,
    ReturnValue,
    RaisedException,
    Example,
    FunctionDoc,
    Attribute,
    ClassDoc,
    ModuleDoc,
    PackageDoc,
    DocConfig,
    GeneratedDoc,
    DocGenResult,
)


class TestDocFormat:
    """Tests for DocFormat enum."""

    def test_markdown_value(self):
        """Test MARKDOWN value."""
        assert DocFormat.MARKDOWN.value == "markdown"

    def test_html_value(self):
        """Test HTML value."""
        assert DocFormat.HTML.value == "html"

    def test_rst_value(self):
        """Test RST value."""
        assert DocFormat.RST.value == "rst"

    def test_plaintext_value(self):
        """Test PLAINTEXT value."""
        assert DocFormat.PLAINTEXT.value == "plaintext"

    def test_all_formats_defined(self):
        """Test all expected formats are defined."""
        expected = {"markdown", "html", "rst", "plaintext"}
        actual = {f.value for f in DocFormat}
        assert actual == expected


class TestDocStyle:
    """Tests for DocStyle enum."""

    def test_google_value(self):
        """Test GOOGLE value."""
        assert DocStyle.GOOGLE.value == "google"

    def test_numpy_value(self):
        """Test NUMPY value."""
        assert DocStyle.NUMPY.value == "numpy"

    def test_sphinx_value(self):
        """Test SPHINX value."""
        assert DocStyle.SPHINX.value == "sphinx"

    def test_epytext_value(self):
        """Test EPYTEXT value."""
        assert DocStyle.EPYTEXT.value == "epytext"

    def test_auto_value(self):
        """Test AUTO value."""
        assert DocStyle.AUTO.value == "auto"

    def test_all_styles_defined(self):
        """Test all expected styles are defined."""
        expected = {"google", "numpy", "sphinx", "epytext", "auto"}
        actual = {s.value for s in DocStyle}
        assert actual == expected


class TestParameter:
    """Tests for Parameter dataclass."""

    def test_required_field_only(self):
        """Test Parameter with only required field."""
        param = Parameter(name="arg1")
        assert param.name == "arg1"
        assert param.type_hint is None
        assert param.description == ""
        assert param.default is None
        assert param.is_optional is False
        assert param.is_keyword_only is False
        assert param.is_positional_only is False

    def test_all_fields(self):
        """Test Parameter with all fields."""
        param = Parameter(
            name="items",
            type_hint="list[str]",
            description="List of items to process",
            default="[]",
            is_optional=True,
            is_keyword_only=True,
            is_positional_only=False,
        )
        assert param.name == "items"
        assert param.type_hint == "list[str]"
        assert param.description == "List of items to process"
        assert param.default == "[]"
        assert param.is_optional is True
        assert param.is_keyword_only is True
        assert param.is_positional_only is False

    def test_positional_only_parameter(self):
        """Test positional-only parameter."""
        param = Parameter(name="x", is_positional_only=True)
        assert param.is_positional_only is True
        assert param.is_keyword_only is False


class TestReturnValue:
    """Tests for ReturnValue dataclass."""

    def test_default_values(self):
        """Test ReturnValue with defaults."""
        ret = ReturnValue()
        assert ret.type_hint is None
        assert ret.description == ""

    def test_all_fields(self):
        """Test ReturnValue with all fields."""
        ret = ReturnValue(type_hint="dict[str, Any]", description="Parsed configuration dictionary")
        assert ret.type_hint == "dict[str, Any]"
        assert ret.description == "Parsed configuration dictionary"


class TestRaisedException:
    """Tests for RaisedException dataclass."""

    def test_required_field_only(self):
        """Test RaisedException with only required field."""
        exc = RaisedException(exception_type="ValueError")
        assert exc.exception_type == "ValueError"
        assert exc.description == ""

    def test_all_fields(self):
        """Test RaisedException with all fields."""
        exc = RaisedException(
            exception_type="FileNotFoundError",
            description="When the config file doesn't exist",
        )
        assert exc.exception_type == "FileNotFoundError"
        assert exc.description == "When the config file doesn't exist"


class TestExample:
    """Tests for Example dataclass."""

    def test_required_field_only(self):
        """Test Example with only required field."""
        ex = Example(code="print('hello')")
        assert ex.code == "print('hello')"
        assert ex.description == ""
        assert ex.output is None

    def test_all_fields(self):
        """Test Example with all fields."""
        ex = Example(code="result = add(1, 2)", description="Adding two numbers", output="3")
        assert ex.code == "result = add(1, 2)"
        assert ex.description == "Adding two numbers"
        assert ex.output == "3"


class TestFunctionDoc:
    """Tests for FunctionDoc dataclass."""

    def test_required_fields_only(self):
        """Test FunctionDoc with only required fields."""
        func = FunctionDoc(name="my_func", signature="my_func()")
        assert func.name == "my_func"
        assert func.signature == "my_func()"
        assert func.description == ""
        assert func.long_description == ""
        assert func.parameters == []
        assert func.returns is None
        assert func.yields is None
        assert func.raises == []
        assert func.examples == []
        assert func.deprecation is None
        assert func.notes == []
        assert func.see_also == []
        assert func.is_async is False
        assert func.is_static is False
        assert func.is_classmethod is False
        assert func.is_property is False
        assert func.decorators == []
        assert func.source_line == 0

    def test_all_fields(self):
        """Test FunctionDoc with all fields populated."""
        func = FunctionDoc(
            name="process_data",
            signature="process_data(items: list[str], **kwargs) -> dict",
            description="Process a list of items.",
            long_description="This function processes items and returns results.",
            parameters=[
                Parameter(name="items", type_hint="list[str]"),
                Parameter(name="kwargs", description="Additional options"),
            ],
            returns=ReturnValue(type_hint="dict", description="Processed results"),
            yields=None,
            raises=[RaisedException(exception_type="ValueError")],
            examples=[Example(code="process_data(['a', 'b'])")],
            deprecation="Use process_data_v2 instead",
            notes=["This is thread-safe"],
            see_also=["process_data_v2", "batch_process"],
            is_async=True,
            is_static=False,
            is_classmethod=False,
            is_property=False,
            decorators=["@asyncio.coroutine", "@deprecated"],
            source_line=42,
        )
        assert func.name == "process_data"
        assert len(func.parameters) == 2
        assert func.returns.type_hint == "dict"
        assert func.deprecation == "Use process_data_v2 instead"
        assert func.is_async is True
        assert func.source_line == 42
        assert len(func.decorators) == 2

    def test_generator_function(self):
        """Test FunctionDoc for generator function."""
        func = FunctionDoc(
            name="iter_items",
            signature="iter_items() -> Iterator[str]",
            yields=ReturnValue(type_hint="str", description="Next item in sequence"),
        )
        assert func.yields is not None
        assert func.yields.type_hint == "str"

    def test_static_method(self):
        """Test FunctionDoc for static method."""
        func = FunctionDoc(
            name="from_dict",
            signature="from_dict(data: dict) -> Self",
            is_static=True,
            decorators=["@staticmethod"],
        )
        assert func.is_static is True

    def test_classmethod(self):
        """Test FunctionDoc for classmethod."""
        func = FunctionDoc(
            name="create",
            signature="create(cls, name: str) -> Self",
            is_classmethod=True,
            decorators=["@classmethod"],
        )
        assert func.is_classmethod is True

    def test_property(self):
        """Test FunctionDoc for property."""
        func = FunctionDoc(
            name="value",
            signature="value(self) -> int",
            is_property=True,
            decorators=["@property"],
        )
        assert func.is_property is True


class TestAttribute:
    """Tests for Attribute dataclass."""

    def test_required_field_only(self):
        """Test Attribute with only required field."""
        attr = Attribute(name="count")
        assert attr.name == "count"
        assert attr.type_hint is None
        assert attr.description == ""
        assert attr.default is None
        assert attr.is_class_var is False

    def test_all_fields(self):
        """Test Attribute with all fields."""
        attr = Attribute(
            name="DEFAULT_TIMEOUT",
            type_hint="int",
            description="Default timeout in seconds",
            default="30",
            is_class_var=True,
        )
        assert attr.name == "DEFAULT_TIMEOUT"
        assert attr.type_hint == "int"
        assert attr.description == "Default timeout in seconds"
        assert attr.default == "30"
        assert attr.is_class_var is True


class TestClassDoc:
    """Tests for ClassDoc dataclass."""

    def test_required_field_only(self):
        """Test ClassDoc with only required field."""
        cls = ClassDoc(name="MyClass")
        assert cls.name == "MyClass"
        assert cls.description == ""
        assert cls.long_description == ""
        assert cls.bases == []
        assert cls.attributes == []
        assert cls.methods == []
        assert cls.class_methods == []
        assert cls.static_methods == []
        assert cls.properties == []
        assert cls.examples == []
        assert cls.deprecation is None
        assert cls.notes == []
        assert cls.see_also == []
        assert cls.source_line == 0

    def test_all_fields(self):
        """Test ClassDoc with all fields."""
        cls = ClassDoc(
            name="DataProcessor",
            description="Process data from various sources.",
            long_description="A comprehensive data processor.",
            bases=["BaseProcessor", "Serializable"],
            attributes=[
                Attribute(name="name", type_hint="str"),
                Attribute(name="config", type_hint="dict"),
            ],
            methods=[
                FunctionDoc(name="process", signature="process() -> None"),
            ],
            class_methods=[
                FunctionDoc(name="from_config", signature="from_config(cfg) -> Self"),
            ],
            static_methods=[
                FunctionDoc(name="validate", signature="validate(data) -> bool"),
            ],
            properties=[
                FunctionDoc(name="is_ready", signature="is_ready -> bool"),
            ],
            examples=[Example(code="processor = DataProcessor()")],
            deprecation="Use DataProcessorV2 instead",
            notes=["Thread-safe after initialization"],
            see_also=["DataProcessorV2"],
            source_line=100,
        )
        assert cls.name == "DataProcessor"
        assert len(cls.bases) == 2
        assert len(cls.attributes) == 2
        assert len(cls.methods) == 1
        assert len(cls.class_methods) == 1
        assert len(cls.static_methods) == 1
        assert len(cls.properties) == 1
        assert cls.deprecation is not None
        assert cls.source_line == 100


class TestModuleDoc:
    """Tests for ModuleDoc dataclass."""

    def test_required_fields_only(self):
        """Test ModuleDoc with only required fields."""
        mod = ModuleDoc(name="mymodule", file_path=Path("src/mymodule.py"))
        assert mod.name == "mymodule"
        assert mod.file_path == Path("src/mymodule.py")
        assert mod.description == ""
        assert mod.long_description == ""
        assert mod.functions == []
        assert mod.classes == []
        assert mod.constants == []
        assert mod.imports == []
        assert mod.all_exports == []
        assert mod.examples == []
        assert mod.deprecation is None
        assert mod.notes == []
        assert mod.version is None
        assert mod.author is None
        assert mod.license is None

    def test_all_fields(self):
        """Test ModuleDoc with all fields."""
        mod = ModuleDoc(
            name="utils",
            file_path=Path("victor/utils.py"),
            description="Utility functions for Victor.",
            long_description="Contains helper functions and utilities.",
            functions=[
                FunctionDoc(name="slugify", signature="slugify(s: str) -> str"),
                FunctionDoc(name="truncate", signature="truncate(s: str, n: int) -> str"),
            ],
            classes=[
                ClassDoc(name="Timer"),
            ],
            constants=[
                Attribute(name="VERSION", type_hint="str", default='"1.0.0"'),
            ],
            imports=["os", "sys", "pathlib.Path"],
            all_exports=["slugify", "truncate", "Timer"],
            examples=[Example(code="from utils import slugify")],
            deprecation=None,
            notes=["All functions are pure"],
            version="1.0.0",
            author="John Doe",
            license="Apache-2.0",
        )
        assert mod.name == "utils"
        assert len(mod.functions) == 2
        assert len(mod.classes) == 1
        assert len(mod.constants) == 1
        assert len(mod.imports) == 3
        assert len(mod.all_exports) == 3
        assert mod.version == "1.0.0"
        assert mod.author == "John Doe"
        assert mod.license == "Apache-2.0"


class TestPackageDoc:
    """Tests for PackageDoc dataclass."""

    def test_required_fields_only(self):
        """Test PackageDoc with only required fields."""
        pkg = PackageDoc(name="victor", path=Path("victor"))
        assert pkg.name == "victor"
        assert pkg.path == Path("victor")
        assert pkg.description == ""
        assert pkg.modules == []
        assert pkg.subpackages == []
        assert pkg.readme is None

    def test_all_fields(self):
        """Test PackageDoc with all fields."""
        pkg = PackageDoc(
            name="victor",
            path=Path("victor"),
            description="Victor AI coding assistant package.",
            modules=[
                ModuleDoc(name="__init__", file_path=Path("victor/__init__.py")),
                ModuleDoc(name="config", file_path=Path("victor/config.py")),
            ],
            subpackages=[
                PackageDoc(name="tools", path=Path("victor/tools")),
                PackageDoc(name="providers", path=Path("victor/providers")),
            ],
            readme="# Victor\n\nAn AI coding assistant.",
        )
        assert pkg.name == "victor"
        assert len(pkg.modules) == 2
        assert len(pkg.subpackages) == 2
        assert pkg.readme.startswith("# Victor")

    def test_nested_subpackages(self):
        """Test deeply nested subpackages."""
        leaf = PackageDoc(name="handlers", path=Path("victor/api/handlers"))
        api = PackageDoc(
            name="api",
            path=Path("victor/api"),
            subpackages=[leaf],
        )
        root = PackageDoc(
            name="victor",
            path=Path("victor"),
            subpackages=[api],
        )
        assert root.subpackages[0].name == "api"
        assert root.subpackages[0].subpackages[0].name == "handlers"


class TestDocConfig:
    """Tests for DocConfig dataclass."""

    def test_default_values(self):
        """Test DocConfig with default values."""
        config = DocConfig()
        assert config.output_format == DocFormat.MARKDOWN
        assert config.input_style == DocStyle.AUTO
        assert config.include_private is False
        assert config.include_dunder is False
        assert config.include_source_links is True
        assert config.include_toc is True
        assert config.include_index is True
        assert config.include_examples is True
        assert config.max_depth == 3
        assert config.output_dir is None
        assert config.base_url == ""
        assert config.template_dir is None

    def test_custom_values(self):
        """Test DocConfig with custom values."""
        config = DocConfig(
            output_format=DocFormat.HTML,
            input_style=DocStyle.GOOGLE,
            include_private=True,
            include_dunder=True,
            include_source_links=False,
            include_toc=False,
            include_index=False,
            include_examples=False,
            max_depth=5,
            output_dir=Path("docs/api"),
            base_url="https://example.com/api/",
            template_dir=Path("templates/doc"),
        )
        assert config.output_format == DocFormat.HTML
        assert config.input_style == DocStyle.GOOGLE
        assert config.include_private is True
        assert config.include_dunder is True
        assert config.include_source_links is False
        assert config.max_depth == 5
        assert config.output_dir == Path("docs/api")
        assert config.base_url == "https://example.com/api/"
        assert config.template_dir == Path("templates/doc")

    def test_rst_format(self):
        """Test DocConfig with RST format."""
        config = DocConfig(output_format=DocFormat.RST)
        assert config.output_format == DocFormat.RST

    def test_numpy_style(self):
        """Test DocConfig with NumPy style."""
        config = DocConfig(input_style=DocStyle.NUMPY)
        assert config.input_style == DocStyle.NUMPY


class TestGeneratedDoc:
    """Tests for GeneratedDoc dataclass."""

    def test_required_fields(self):
        """Test GeneratedDoc with all required fields."""
        doc = GeneratedDoc(
            path=Path("docs/api/utils.md"),
            content="# Utils\n\nUtility functions.",
            format=DocFormat.MARKDOWN,
        )
        assert doc.path == Path("docs/api/utils.md")
        assert doc.content.startswith("# Utils")
        assert doc.format == DocFormat.MARKDOWN

    def test_html_format(self):
        """Test GeneratedDoc with HTML format."""
        doc = GeneratedDoc(
            path=Path("docs/api/utils.html"),
            content="<html><body><h1>Utils</h1></body></html>",
            format=DocFormat.HTML,
        )
        assert doc.format == DocFormat.HTML
        assert doc.path.suffix == ".html"


class TestDocGenResult:
    """Tests for DocGenResult dataclass."""

    def test_success_minimal(self):
        """Test successful DocGenResult with minimal data."""
        result = DocGenResult(success=True)
        assert result.success is True
        assert result.generated_files == []
        assert result.modules_documented == 0
        assert result.classes_documented == 0
        assert result.functions_documented == 0
        assert result.errors == []
        assert result.warnings == []
        assert result.duration_ms == 0.0

    def test_failure_result(self):
        """Test failed DocGenResult."""
        result = DocGenResult(
            success=False,
            errors=["Failed to parse module: syntax error"],
            warnings=["Missing docstring in function 'foo'"],
        )
        assert result.success is False
        assert len(result.errors) == 1
        assert len(result.warnings) == 1

    def test_success_full(self):
        """Test successful DocGenResult with all counts."""
        result = DocGenResult(
            success=True,
            generated_files=[
                GeneratedDoc(
                    path=Path("docs/index.md"),
                    content="# API Reference",
                    format=DocFormat.MARKDOWN,
                ),
                GeneratedDoc(
                    path=Path("docs/utils.md"),
                    content="# Utils Module",
                    format=DocFormat.MARKDOWN,
                ),
            ],
            modules_documented=5,
            classes_documented=10,
            functions_documented=50,
            errors=[],
            warnings=["Function 'foo' has no return type hint"],
            duration_ms=1234.5,
        )
        assert result.success is True
        assert len(result.generated_files) == 2
        assert result.modules_documented == 5
        assert result.classes_documented == 10
        assert result.functions_documented == 50
        assert len(result.warnings) == 1
        assert result.duration_ms == 1234.5


class TestDataclassEquality:
    """Tests for dataclass equality comparison."""

    def test_parameter_equality(self):
        """Test Parameter equality."""
        p1 = Parameter(name="x", type_hint="int")
        p2 = Parameter(name="x", type_hint="int")
        p3 = Parameter(name="y", type_hint="int")
        assert p1 == p2
        assert p1 != p3

    def test_return_value_equality(self):
        """Test ReturnValue equality."""
        r1 = ReturnValue(type_hint="str", description="result")
        r2 = ReturnValue(type_hint="str", description="result")
        r3 = ReturnValue(type_hint="int", description="result")
        assert r1 == r2
        assert r1 != r3

    def test_function_doc_equality(self):
        """Test FunctionDoc equality."""
        f1 = FunctionDoc(name="foo", signature="foo()")
        f2 = FunctionDoc(name="foo", signature="foo()")
        f3 = FunctionDoc(name="bar", signature="bar()")
        assert f1 == f2
        assert f1 != f3

    def test_class_doc_equality(self):
        """Test ClassDoc equality."""
        c1 = ClassDoc(name="MyClass")
        c2 = ClassDoc(name="MyClass")
        c3 = ClassDoc(name="OtherClass")
        assert c1 == c2
        assert c1 != c3


class TestDataclassHashing:
    """Tests for dataclass behavior (unhashable by default with mutable fields)."""

    def test_doc_format_hashable(self):
        """Test DocFormat is hashable (enums are hashable)."""
        formats = {DocFormat.MARKDOWN, DocFormat.HTML}
        assert DocFormat.MARKDOWN in formats
        assert len(formats) == 2

    def test_doc_style_hashable(self):
        """Test DocStyle is hashable."""
        styles = {DocStyle.GOOGLE, DocStyle.NUMPY}
        assert DocStyle.GOOGLE in styles
        assert len(styles) == 2


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_function_doc_lists(self):
        """Test FunctionDoc with empty lists stays empty."""
        func = FunctionDoc(name="f", signature="f()")
        func.parameters.append(Parameter(name="x"))
        # Verify list modification works (default_factory creates new list)
        assert len(func.parameters) == 1

    def test_path_types(self):
        """Test Path handling in dataclasses."""
        mod = ModuleDoc(name="test", file_path=Path("/absolute/path/test.py"))
        assert mod.file_path.is_absolute()

        mod2 = ModuleDoc(name="test2", file_path=Path("relative/test.py"))
        assert not mod2.file_path.is_absolute()

    def test_unicode_in_descriptions(self):
        """Test unicode characters in descriptions."""
        param = Parameter(name="emoji", description="Supports unicode: emoji, CJK characters")
        assert "emoji" in param.description

    def test_multiline_description(self):
        """Test multiline descriptions."""
        func = FunctionDoc(
            name="complex_func",
            signature="complex_func()",
            description="Short description.",
            long_description="""This is a longer description
            that spans multiple lines
            and contains detailed information.""",
        )
        assert "\n" in func.long_description


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
