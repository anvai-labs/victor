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

"""Unit tests for language-specific edge detection handlers."""

import pytest
from pathlib import Path
from typing import Any

pytest.importorskip(
    "victor_coding.languages.registry",
    reason="victor_coding language plugins are optional in core CI",
)

from victor.core.graph_rag.language_handlers import (
    get_edge_handler,
    _AnalysisProviderEdgeHandler,
    _VictorCodingPluginAdapter,
    EdgeHandlerRegistry,
)

# The provider-backed handler is the preferred return type from
# get_edge_handler when the analysis provider is registered; the legacy
# victor_coding adapter is the fallback. Tests assert either is acceptable.
_AcceptableHandler = (_AnalysisProviderEdgeHandler, _VictorCodingPluginAdapter)


def _require_parser(module_name: str) -> None:
    """Skip a language-specific test when its tree-sitter parser wheel is absent."""
    pytest.importorskip(module_name, reason=f"{module_name} is optional in core CI")


def _parse_code(code: str, language_module: Any, language_attr: str = "language") -> Any:
    """Helper to parse code using tree-sitter with new API.

    Args:
        code: Source code to parse
        language_module: Tree-sitter language module
        language_attr: Attribute name for language function (default: "language")

    Returns:
        Parsed tree-sitter tree
    """
    import tree_sitter as ts

    lang_func = getattr(language_module, language_attr)
    lang_obj = lang_func()
    ts_language = ts.Language(lang_obj) if not isinstance(lang_obj, ts.Language) else lang_obj
    parser = ts.Parser(ts_language)
    return parser.parse(bytes(code, "utf-8"))


@pytest.mark.unit
class TestLanguageHandlerRegistry:
    """Test language handler registry."""

    def test_all_handlers_registered(self):
        """Test that all handlers are registered via victor_coding."""
        # victor_coding provides handlers for these languages
        expected = {
            "python",
            "py",
            "javascript",
            "js",
            "typescript",
            "ts",
            "tsx",
            "jsx",
            "go",
            "golang",
            "rust",
            "rs",
            "java",
            "cpp",
            "c++",
            "csharp",
            "cs",
            "kotlin",
            "kt",
            "swift",
            "c",
            "ruby",
            "rb",
            "php",
        }
        # Check that we can get handlers for all expected languages
        for lang in expected:
            handler = get_edge_handler(lang)
            assert handler is not None, f"No handler found for {lang}"
            assert isinstance(
                handler, _AcceptableHandler
            ), f"Handler for {lang} must be analysis-provider-backed or victor_coding adapter"

    def test_get_python_handler(self):
        """Test getting Python handler."""
        handler = get_edge_handler("python")
        assert handler is not None
        assert isinstance(handler, _AcceptableHandler)

    def test_get_javascript_handler(self):
        """Test getting JavaScript handler."""
        handler = get_edge_handler("javascript")
        assert handler is not None
        assert isinstance(handler, _AcceptableHandler)

    def test_get_typescript_handler(self):
        """Test getting TypeScript handler."""
        handler = get_edge_handler("typescript")
        assert handler is not None
        assert isinstance(handler, _AcceptableHandler)

    def test_get_go_handler(self):
        """Test getting Go handler."""
        handler = get_edge_handler("go")
        assert handler is not None
        assert isinstance(handler, _AcceptableHandler)

    def test_get_rust_handler(self):
        """Test getting Rust handler."""
        handler = get_edge_handler("rust")
        assert handler is not None
        assert isinstance(handler, _AcceptableHandler)

    def test_get_java_handler(self):
        """Test getting Java handler."""
        handler = get_edge_handler("java")
        assert handler is not None
        assert isinstance(handler, _AcceptableHandler)

    def test_get_cpp_handler(self):
        """Test getting C++ handler."""
        handler = get_edge_handler("cpp")
        assert handler is not None
        assert isinstance(handler, _AcceptableHandler)

    def test_get_csharp_handler(self):
        """Test getting C# handler."""
        handler = get_edge_handler("csharp")
        assert handler is not None
        assert isinstance(handler, _AcceptableHandler)

    def test_get_kotlin_handler(self):
        """Test getting Kotlin handler."""
        handler = get_edge_handler("kotlin")
        assert handler is not None
        assert isinstance(handler, _AcceptableHandler)

    def test_get_swift_handler(self):
        """Test getting Swift handler."""
        handler = get_edge_handler("swift")
        assert handler is not None
        assert isinstance(handler, _AcceptableHandler)

    def test_get_c_handler(self):
        """Test getting C handler."""
        handler = get_edge_handler("c")
        assert handler is not None
        assert isinstance(handler, _AcceptableHandler)

    def test_get_ruby_handler(self):
        """Test getting Ruby handler."""
        handler = get_edge_handler("ruby")
        assert handler is not None
        assert isinstance(handler, _AcceptableHandler)

    def test_get_php_handler(self):
        """Test getting PHP handler."""
        handler = get_edge_handler("php")
        assert handler is not None
        assert isinstance(handler, _AcceptableHandler)


@pytest.mark.unit
class TestTypeScriptEdgeHandler:
    """Test TypeScript edge detection handler."""

    @pytest.fixture
    def handler(self):
        _require_parser("tree_sitter_typescript")
        from victor_coding.languages.registry import get_language_registry
        from victor.core.graph_rag.language_handlers import _VictorCodingPluginAdapter

        # Ensure plugins are discovered
        registry = get_language_registry()
        if not registry.list_languages():
            registry.discover_plugins()

        plugin = registry.get("typescript")
        return _VictorCodingPluginAdapter(plugin)

    @pytest.mark.asyncio
    async def test_detect_simple_calls(self, handler, tmp_path):
        """Test detecting simple function calls in TypeScript."""
        import tree_sitter_typescript

        code = """
function foo(): string {
    return "foo";
}

function bar(): string {
    return foo();  // bar calls foo
}
"""
        test_file = tmp_path / "test.ts"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_typescript, "language_typescript")
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert ("bar", "foo") in call_edges

    @pytest.mark.asyncio
    async def test_detect_method_calls(self, handler, tmp_path):
        """Test detecting method calls in TypeScript."""
        import tree_sitter_typescript

        code = """
class Calculator {
    add(a: number, b: number): number {
        return a + b;
    }
}

function compute(calc: Calculator): number {
    return calc.add(1, 2);  // compute calls calc.add
}
"""
        test_file = tmp_path / "test.ts"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_typescript, "language_typescript")
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert any(caller == "compute" and "add" in callee for caller, callee in call_edges)


@pytest.mark.unit
class TestCppEdgeHandler:
    """Test C++ edge detection handler."""

    @pytest.fixture
    def handler(self):
        _require_parser("tree_sitter_cpp")
        from victor_coding.languages.registry import get_language_registry
        from victor.core.graph_rag.language_handlers import _VictorCodingPluginAdapter

        # Ensure plugins are discovered
        registry = get_language_registry()
        if not registry.list_languages():
            registry.discover_plugins()

        plugin = registry.get("cpp")
        return _VictorCodingPluginAdapter(plugin)

    @pytest.mark.asyncio
    async def test_detect_simple_calls(self, handler, tmp_path):
        """Test detecting simple function calls in C++."""
        import tree_sitter_cpp

        code = """
#include <string>

std::string foo() {
    return "foo";
}

std::string bar() {
    return foo();  // bar calls foo
}
"""
        test_file = tmp_path / "test.cpp"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_cpp)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert ("bar", "foo") in call_edges

    @pytest.mark.asyncio
    async def test_detect_method_calls(self, handler, tmp_path):
        """Test detecting method calls in C++."""
        import tree_sitter_cpp

        code = """
class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }
};

int compute(Calculator* calc) {
    return calc->add(1, 2);  // compute calls calc.add
}
"""
        test_file = tmp_path / "test.cpp"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_cpp)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert any(caller == "compute" and "add" in callee for caller, callee in call_edges)

    @pytest.mark.asyncio
    async def test_detect_namespace_calls(self, handler, tmp_path):
        """Test detecting namespace-qualified calls in C++."""
        import tree_sitter_cpp

        code = """
namespace std {
    void helper() {}
}

void caller() {
    std::helper();  // caller calls std::helper
}
"""
        test_file = tmp_path / "test.cpp"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_cpp)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert any(caller == "caller" and "helper" in callee for caller, callee in call_edges)


@pytest.mark.unit
class TestCEdgeHandler:
    """Test C edge detection handler."""

    @pytest.fixture
    def handler(self):
        _require_parser("tree_sitter_c")
        from victor_coding.languages.registry import get_language_registry
        from victor.core.graph_rag.language_handlers import _VictorCodingPluginAdapter

        # Ensure plugins are discovered
        registry = get_language_registry()
        if not registry.list_languages():
            registry.discover_plugins()

        plugin = registry.get("c")
        return _VictorCodingPluginAdapter(plugin)

    @pytest.mark.asyncio
    async def test_detect_simple_calls(self, handler, tmp_path):
        """Test detecting simple function calls in C."""
        import tree_sitter_c

        code = """
void foo() {
}

void bar() {
    foo();  // bar calls foo
}
"""
        test_file = tmp_path / "test.c"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_c)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert ("bar", "foo") in call_edges


@pytest.mark.unit
class TestCSharpEdgeHandler:
    """Test C# edge detection handler."""

    @pytest.fixture
    def handler(self):
        _require_parser("tree_sitter_c_sharp")
        from victor_coding.languages.registry import get_language_registry
        from victor.core.graph_rag.language_handlers import _VictorCodingPluginAdapter

        # Ensure plugins are discovered
        registry = get_language_registry()
        if not registry.list_languages():
            registry.discover_plugins()

        plugin = registry.get("csharp")
        return _VictorCodingPluginAdapter(plugin)

    @pytest.mark.asyncio
    async def test_detect_simple_calls(self, handler, tmp_path):
        """Test detecting simple method calls in C#."""
        import tree_sitter_c_sharp

        code = """
class Program {
    static void Foo() {
    }

    static void Bar() {
        Foo();  // Bar calls Foo
    }
}
"""
        test_file = tmp_path / "test.cs"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_c_sharp)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert ("Bar", "Foo") in call_edges

    @pytest.mark.asyncio
    async def test_detect_method_calls(self, handler, tmp_path):
        """Test detecting method calls in C#."""
        import tree_sitter_c_sharp

        code = """
class Calculator {
    public int Add(int a, int b) {
        return a + b;
    }
}

class Program {
    static int Compute(Calculator calc) {
        return calc.Add(1, 2);  // Compute calls calc.Add
    }
}
"""
        test_file = tmp_path / "test.cs"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_c_sharp)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert any(caller == "Compute" and "Add" in callee for caller, callee in call_edges)


@pytest.mark.unit
class TestKotlinEdgeHandler:
    """Test Kotlin edge detection handler."""

    @pytest.fixture
    def handler(self):
        _require_parser("tree_sitter_kotlin")
        from victor_coding.languages.registry import get_language_registry
        from victor.core.graph_rag.language_handlers import _VictorCodingPluginAdapter

        # Ensure plugins are discovered
        registry = get_language_registry()
        if not registry.list_languages():
            registry.discover_plugins()

        plugin = registry.get("kotlin")
        return _VictorCodingPluginAdapter(plugin)

    @pytest.mark.asyncio
    async def test_detect_simple_calls(self, handler, tmp_path):
        """Test detecting simple function calls in Kotlin."""
        import tree_sitter_kotlin

        code = """
fun foo() {
}

fun bar() {
    foo()  // bar calls foo
}
"""
        test_file = tmp_path / "test.kt"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_kotlin)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert ("bar", "foo") in call_edges

    @pytest.mark.asyncio
    async def test_detect_method_calls(self, handler, tmp_path):
        """Test detecting method calls in Kotlin."""
        import tree_sitter_kotlin

        code = """
class Calculator {
    fun add(a: Int, b: Int): Int {
        return a + b
    }
}

fun compute(calc: Calculator): Int {
    return calc.add(1, 2)  // compute calls calc.add
}
"""
        test_file = tmp_path / "test.kt"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_kotlin)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert any(caller == "compute" and "add" in callee for caller, callee in call_edges)


@pytest.mark.unit
class TestSwiftEdgeHandler:
    """Test Swift edge detection handler."""

    @pytest.fixture
    def handler(self):
        _require_parser("tree_sitter_swift")
        from victor_coding.languages.registry import get_language_registry
        from victor.core.graph_rag.language_handlers import _VictorCodingPluginAdapter

        # Ensure plugins are discovered
        registry = get_language_registry()
        if not registry.list_languages():
            registry.discover_plugins()

        plugin = registry.get("swift")
        return _VictorCodingPluginAdapter(plugin)

    @pytest.mark.asyncio
    async def test_detect_simple_calls(self, handler, tmp_path):
        """Test detecting simple function calls in Swift."""
        import tree_sitter_swift

        code = """
func foo() {
}

func bar() {
    foo()  // bar calls foo
}
"""
        test_file = tmp_path / "test.swift"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_swift)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert ("bar", "foo") in call_edges

    @pytest.mark.asyncio
    async def test_detect_method_calls(self, handler, tmp_path):
        """Test detecting method calls in Swift."""
        import tree_sitter_swift

        code = """
class Calculator {
    func add(_ a: Int, _ b: Int) -> Int {
        return a + b
    }
}

func compute(calc: Calculator) -> Int {
    return calc.add(1, 2)  // compute calls calc.add
}
"""
        test_file = tmp_path / "test.swift"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_swift)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert any(caller == "compute" and "add" in callee for caller, callee in call_edges)


@pytest.mark.unit
class TestRubyEdgeHandler:
    """Test Ruby edge detection handler."""

    @pytest.fixture
    def handler(self):
        _require_parser("tree_sitter_ruby")
        from victor_coding.languages.registry import get_language_registry
        from victor.core.graph_rag.language_handlers import _VictorCodingPluginAdapter

        # Ensure plugins are discovered
        registry = get_language_registry()
        if not registry.list_languages():
            registry.discover_plugins()

        plugin = registry.get("ruby")
        return _VictorCodingPluginAdapter(plugin)

    @pytest.mark.asyncio
    async def test_detect_simple_calls(self, handler, tmp_path):
        """Test detecting simple method calls in Ruby."""
        import tree_sitter_ruby

        code = """
def foo
end

def bar
  foo  # bar calls foo
end
"""
        test_file = tmp_path / "test.rb"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_ruby)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert ("bar", "foo") in call_edges

    @pytest.mark.asyncio
    async def test_detect_method_calls(self, handler, tmp_path):
        """Test detecting method calls with receiver in Ruby."""
        import tree_sitter_ruby

        code = """
class Calculator
  def add(a, b)
    a + b
  end
end

def compute(calc)
  calc.add(1, 2)  # compute calls calc.add
end
"""
        test_file = tmp_path / "test.rb"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_ruby)
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert any(caller == "compute" and "add" in callee for caller, callee in call_edges)


@pytest.mark.unit
class TestPhpEdgeHandler:
    """Test PHP edge detection handler."""

    @pytest.fixture
    def handler(self):
        _require_parser("tree_sitter_php")
        from victor_coding.languages.registry import get_language_registry
        from victor.core.graph_rag.language_handlers import _VictorCodingPluginAdapter

        # Ensure plugins are discovered
        registry = get_language_registry()
        if not registry.list_languages():
            registry.discover_plugins()

        plugin = registry.get("php")
        return _VictorCodingPluginAdapter(plugin)

    @pytest.mark.asyncio
    async def test_detect_simple_calls(self, handler, tmp_path):
        """Test detecting simple function calls in PHP."""
        import tree_sitter_php

        code = """
<?php
function foo() {
}

function bar() {
    foo();  // bar calls foo
}
"""
        test_file = tmp_path / "test.php"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_php, "language_php")
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert ("bar", "foo") in call_edges

    @pytest.mark.asyncio
    async def test_detect_method_calls(self, handler, tmp_path):
        """Test detecting method calls in PHP."""
        import tree_sitter_php

        code = """
<?php
class Calculator {
    public function add($a, $b) {
        return $a + $b;
    }
}

function compute($calc) {
    return $calc->add(1, 2);  // compute calls calc.add
}
"""
        test_file = tmp_path / "test.php"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_php, "language_php")
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert any(caller == "compute" and "add" in callee for caller, callee in call_edges)

    @pytest.mark.asyncio
    async def test_detect_static_calls(self, handler, tmp_path):
        """Test detecting static method calls in PHP."""
        import tree_sitter_php

        code = """
<?php
class Helper {
    public static function process() {
    }
}

function caller() {
    Helper::process();  // caller calls Helper.process
}
"""
        test_file = tmp_path / "test.php"
        test_file.write_text(code)

        tree = _parse_code(code, tree_sitter_php, "language_php")
        result = await handler.detect_calls_edges(tree, code, test_file)

        assert len(result.calls) > 0
        call_edges = [(c.caller_name, c.callee_name) for c in result.calls]
        assert any(caller == "caller" and "process" in callee for caller, callee in call_edges)


# ────────────────────────────────────────────────────────────────────────
# Analysis-provider-backed handler
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestAnalysisProviderEdgeHandler:
    """Verifies the new TreeSitterAnalysisProtocol-backed handler path."""

    @pytest.mark.asyncio
    async def test_handler_converts_provider_edge_dicts_to_call_and_relationship_edges(
        self, tmp_path
    ):
        from victor.core.graph_rag.language_handlers import (
            _AnalysisProviderEdgeHandler,
            CallEdge,
            RelationshipEdge,
        )

        class _FakeProvider:
            def __init__(self):
                self.calls = []

            def supports_language(self, language):
                return True

            def extract_edges(self, content, language, *, file_path):
                self.calls.append((file_path, language))
                return [
                    {
                        "source": "caller",
                        "target": "foo",
                        "edge_type": "CALLS",
                        "file_path": file_path,
                        "line_number": 3,
                        "is_method_call": False,
                        "receiver_type": None,
                    },
                    {
                        "source": "caller",
                        "target": "bar",
                        "edge_type": "CALLS",
                        "file_path": file_path,
                        "line_number": 4,
                        "is_method_call": True,
                        "receiver_type": "Foo",
                    },
                    # Non-CALLS structural edges — must be routed to
                    # result.relationships, not silently dropped (regression
                    # guard: TSA path used to lose these entirely).
                    {
                        "source": "Child",
                        "target": "Parent",
                        "edge_type": "INHERITS",
                        "file_path": file_path,
                        "line_number": 1,
                    },
                    {
                        "source": "Child",
                        "target": "Iface",
                        "edge_type": "IMPLEMENTS",
                        "file_path": file_path,
                        "line_number": 1,
                    },
                    {
                        "source": "Child",
                        "target": "Helper",
                        "edge_type": "COMPOSITION",
                        "file_path": file_path,
                        "line_number": 2,
                    },
                    # Unknown edge type stays dropped — we only promote
                    # the structural set we currently know about.
                    {
                        "source": "x",
                        "target": "y",
                        "edge_type": "MYSTERY",
                    },
                ]

        provider = _FakeProvider()
        handler = _AnalysisProviderEdgeHandler(provider, "python")
        result = await handler.detect_calls_edges(
            tree=None, source_code="def caller(): pass\n", file_path=tmp_path / "a.py"
        )

        assert len(result.calls) == 2
        assert all(isinstance(c, CallEdge) for c in result.calls)
        targets = {(c.callee_name, c.is_method_call, c.receiver_type) for c in result.calls}
        assert ("foo", False, None) in targets
        assert ("bar", True, "Foo") in targets
        assert provider.calls == [(str(tmp_path / "a.py"), "python")]

        rels = result.relationships
        assert len(rels) == 3
        assert all(isinstance(r, RelationshipEdge) for r in rels)
        rel_summary = {(r.source_name, r.target_name, r.edge_type) for r in rels}
        assert ("Child", "Parent", "INHERITS") in rel_summary
        assert ("Child", "Iface", "IMPLEMENTS") in rel_summary
        assert ("Child", "Helper", "COMPOSITION") in rel_summary

    def test_handler_advertises_supported_language_only(self):
        from victor.core.graph_rag.language_handlers import _AnalysisProviderEdgeHandler

        handler = _AnalysisProviderEdgeHandler(provider=object(), language="rust")
        assert handler.get_supported_languages() == ["rust"]
        assert handler.can_handle("rust") is True
        assert handler.can_handle("Rust") is True
        assert handler.can_handle("python") is False

    def test_get_edge_handler_prefers_analysis_provider(self, monkeypatch):
        """When the registry has an enhanced TreeSitterAnalysisProtocol provider,
        get_edge_handler returns the new adapter (not the legacy victor_coding one).
        """
        from victor.core import capability_registry as cap_registry_mod
        from victor.core.graph_rag.language_handlers import (
            _AnalysisProviderEdgeHandler,
            get_edge_handler,
        )

        class _StubProvider:
            def supports_language(self, language):
                return language == "python"

        class _FakeRegistry:
            def is_enhanced(self, protocol):
                return True

            def get(self, protocol):
                return _StubProvider()

        monkeypatch.setattr(
            cap_registry_mod.CapabilityRegistry,
            "get_instance",
            staticmethod(lambda: _FakeRegistry()),
        )

        handler = get_edge_handler("python")
        assert isinstance(handler, _AnalysisProviderEdgeHandler)

    def test_get_edge_handler_falls_back_when_provider_disabled(self, monkeypatch):
        """With only the null stub registered, get_edge_handler falls back to
        the victor_coding adapter so existing CI keeps working.
        """
        from victor.core import capability_registry as cap_registry_mod
        from victor.core.graph_rag.language_handlers import (
            _VictorCodingPluginAdapter,
            get_edge_handler,
        )

        class _FakeRegistry:
            def is_enhanced(self, protocol):
                return False

            def get(self, protocol):
                return None

        monkeypatch.setattr(
            cap_registry_mod.CapabilityRegistry,
            "get_instance",
            staticmethod(lambda: _FakeRegistry()),
        )

        handler = get_edge_handler("python")
        # Either the legacy adapter or None depending on whether victor_coding
        # is installed in this environment — both are acceptable fallbacks.
        assert handler is None or isinstance(handler, _VictorCodingPluginAdapter)
