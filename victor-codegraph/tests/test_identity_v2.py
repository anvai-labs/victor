from __future__ import annotations

import pytest

from victor_codegraph import parse, parse_path, to_proxima_records
from victor_codegraph.treesitter_parser import _get_parser


def _by_name(parsed, name):
    return next(s for s in parsed.symbols if s.simple_name == name)


def test_parser_identity_survives_line_shift_and_keeps_legacy_alias():
    base = parse("def f(x):\n    return x\n", file_path="pkg/m.py")
    moved = parse("\n\n\ndef f(x):\n    return x\n", file_path="pkg/m.py")
    a = _by_name(base, "f")
    b = _by_name(moved, "f")

    assert a.id == b.id
    assert a.identity_version == b.identity_version == "v2"
    assert a.legacy_id != b.legacy_id


def test_parse_path_identity_is_portable_across_checkout_roots(tmp_path):
    roots = [tmp_path / "one", tmp_path / "two"]
    parsed = []
    for root in roots:
        path = root / "pkg" / "m.py"
        path.parent.mkdir(parents=True)
        path.write_text("def f():\n    return 1\n")
        parsed.append(parse_path(path, repo_root=root))

    assert parsed[0] is not None and parsed[1] is not None
    assert parsed[0].file_path == parsed[1].file_path == "pkg/m.py"
    assert parsed[0].symbols[0].id == parsed[1].symbols[0].id


def test_path_segments_cannot_alias_dotted_file_names():
    nested = parse("def f(): pass\n", file_path="a/b.py").symbols[0]
    dotted = parse("def f(): pass\n", file_path="a.b.py").symbols[0]
    assert nested.fully_qualified_name == dotted.fully_qualified_name
    assert nested.id != dotted.id


def test_cpp_overloads_have_distinct_stable_ids():
    parsed = parse(
        "int f(int x) { return x; }\ndouble f(double x) { return x; }\n",
        language="cpp",
        file_path="f.cpp",
    )
    overloads = [s for s in parsed.symbols if s.simple_name == "f"]
    assert len(overloads) == 2
    assert len({s.id for s in overloads}) == 2


def test_anonymous_function_expressions_do_not_collide():
    parsed = parse(
        "const a = function() { return 1 }; const b = function() { return 2 };",
        language="javascript",
        file_path="f.js",
    )
    assert all(s.simple_name != "<anonymous>" for s in parsed.symbols)
    assert len({s.id for s in parsed.symbols}) == len(parsed.symbols)


def test_python_decorator_change_updates_content_version_and_snippet():
    a = parse("@first\ndef f():\n    return 1\n", file_path="m.py")
    b = parse("@second\ndef f():\n    return 1\n", file_path="m.py")
    sa, sb = _by_name(a, "f"), _by_name(b, "f")
    ra = to_proxima_records(a, "repo")[0]
    rb = to_proxima_records(b, "repo")[0]

    assert sa.id == sb.id
    assert sa.source_code.startswith("@first")
    assert sb.source_code.startswith("@second")
    assert ra["props"]["content_version"] != rb["props"]["content_version"]


def test_rust_inline_modules_preserve_scope_and_identity():
    source = "mod a { pub fn same() {} } mod b { pub fn same() {} }"
    parsed = parse(source, language="rust", file_path="src/lib.rs")
    moved = parse("\n" + source, language="rust", file_path="src/lib.rs")
    functions = [s for s in parsed.symbols if s.simple_name == "same"]
    assert {tuple(s.scope_chain) for s in functions} == {("a",), ("b",)}
    assert len({s.id for s in functions}) == 2
    assert not parsed.diagnostics
    assert {s.id for s in parsed.symbols} == {s.id for s in moved.symbols}
    modules = {s.id for s in parsed.symbols if s.symbol_type.name == "MODULE"}
    assert len(modules) == 2
    assert sum(r.from_symbol_id in modules for r in parsed.relations) == 2


@pytest.mark.parametrize(
    "source,name",
    [
        ("struct T; impl A for T { fn fmt() {} } impl B for T { fn fmt() {} }", "fmt"),
        ("#[cfg(unix)] fn main() {} #[cfg(not(unix))] fn main() {}", "main"),
    ],
)
def test_rust_trait_and_cfg_variants_have_distinct_stable_identity(source, name):
    parsed = parse(source, language="rust", file_path="src/lib.rs")
    moved = parse("\n\n" + source, language="rust", file_path="src/lib.rs")
    variants = [s for s in parsed.symbols if s.simple_name == name]
    assert len(variants) == len({s.id for s in variants}) == 2
    assert not parsed.diagnostics
    assert {s.id for s in parsed.symbols} == {s.id for s in moved.symbols}


@pytest.mark.parametrize(
    "source",
    [
        "function configure() { use({element(x) { return x; }}); use({element(x) { return x; }}); }",
        "function configure(){ xs.map(() => ({element(x){return x;}})); ys.map(() => ({element(x){return x;}})); }",
    ],
)
def test_object_literal_methods_have_structural_scopes(source):
    parsed = parse(source, file_path="a.js")
    moved = parse("// header\n\n" + source, file_path="a.js")
    methods = [s for s in parsed.symbols if s.simple_name == "element"]
    assert len(methods) == len({s.id for s in methods}) == 2
    assert not parsed.diagnostics
    assert {s.id for s in parsed.symbols} == {s.id for s in moved.symbols}


class TestAttributesContainerGrammar:
    """Dual-shape support: the 1.26+ grammar wraps item attributes in an
    ``attributes`` container child instead of source-level siblings."""

    def _cfg_collected(self, source: str) -> list[str]:
        parser = _get_parser("rust")
        tree = parser.parse(source.encode())
        from victor_codegraph.treesitter_parser import _rust_cfg_attributes

        items = [
            c
            for c in tree.root_node.named_children
            if c.type in ("function_item", "mod_item", "use_declaration")
        ]
        assert items, "no item parsed"
        return _rust_cfg_attributes(items[0], source.encode())

    def test_cfg_attribute_on_item_is_collected_from_container_child(self):
        # Under the 1.26+ grammar the attribute is a CHILD container of the
        # function; under older packs it is a prev sibling. Both must collect.
        assert self._cfg_collected("#[cfg(unix)]\nfn f() {}") == ["#[cfg(unix)]"]

    def test_cfg_attr_variant_collected_from_container_child(self):
        assert self._cfg_collected("#[cfg_attr(test, foo)]\nfn f() {}") == [
            "#[cfg_attr(test, foo)]"
        ]

    def test_plain_attribute_is_not_collected(self):
        assert self._cfg_collected("#[inline]\nfn f() {}") == []


class TestAttributeCollectionOrder:
    """Collection order must be nearest-to-item first on BOTH grammar shapes.

    The order feeds the signature prefix via reversed() in _mk; a forward-
    order container scan drifted stable_symbol_key for the same source
    across pack versions (stored OIDs stop joining after a re-index).
    """

    def test_multi_attribute_collection_is_reverse_source_order(self):
        source = "#[cfg(windows)]\n#[cfg_attr(x, y)]\nfn g() {}"
        parser = _get_parser("rust")
        tree = parser.parse(source.encode())
        from victor_codegraph.treesitter_parser import _rust_cfg_attributes

        fn = [c for c in tree.root_node.named_children if c.type == "function_item"][0]
        collected = _rust_cfg_attributes(fn, source.encode())
        assert collected == ["#[cfg_attr(x, y)]", "#[cfg(windows)]"], (
            "must be nearest-to-item first on both pack shapes; got "
            f"{collected} — a forward-order flip drifts stable_symbol_key "
            "across pack versions"
        )

    def test_signature_prefix_is_source_order_on_both_shapes(self):
        source = "#[cfg(windows)]\n#[cfg_attr(x, y)]\nfn g() {}"
        parser = _get_parser("rust")
        tree = parser.parse(source.encode())
        from victor_codegraph.treesitter_parser import _rust_cfg_attributes

        fn = [c for c in tree.root_node.named_children if c.type == "function_item"][0]
        attrs = _rust_cfg_attributes(fn, source.encode())
        signature = " ".join([*reversed(attrs), "()"])
        assert signature == "#[cfg(windows)] #[cfg_attr(x, y)] ()"


def _cpp_symbols(source, file_path="arrow/builder.cc"):
    parsed = parse(source, language="cpp", file_path=file_path)
    return parsed, [s for s in parsed.symbols if s.symbol_type.name != "MODULE"]


def _qualified(symbol):
    """The FQN below the file component (``a::B::f``)."""
    return symbol.fully_qualified_name.split("::", 1)[1]


def test_cpp_out_of_class_definitions_keep_their_qualifier_identity():
    # Two classes' FinishInternal definitions in one .cc (Arrow builder_adaptive.cc).
    source = (
        "Status AdaptiveIntBuilder::FinishInternal(std::shared_ptr<ArrayData>* out) {\n"
        "  return Status::OK();\n}\n"
        "Status AdaptiveUIntBuilder::FinishInternal(std::shared_ptr<ArrayData>* out) {\n"
        "  return Status::OK();\n}\n"
    )
    parsed, symbols = _cpp_symbols(source)
    finish = [s for s in symbols if s.simple_name == "FinishInternal"]
    assert [_qualified(s) for s in finish] == [
        "AdaptiveIntBuilder::FinishInternal",
        "AdaptiveUIntBuilder::FinishInternal",
    ]
    # Qualifiers are identity, not containment: the class lives in another file.
    assert [s.scope_chain for s in finish] == [[], []]
    assert len({s.id for s in finish}) == 2
    assert not [d for d in parsed.diagnostics if d.code == "identity_collision"]


def test_cpp_nested_qualifiers_templates_and_destructors():
    source = (
        "namespace a {\n"
        "void B::C::run() {}\n"
        "B::B() {}\n"
        "B::~B() {}\n"
        "template <typename T> void Box<T>::put(T v) {}\n"
        "}\n"
    )
    parsed, symbols = _cpp_symbols(source)
    assert sorted(_qualified(s) for s in symbols) == [
        "a::B::B",
        "a::B::C::run",
        "a::B::~B",
        "a::Box::put",
    ]
    assert len({s.id for s in symbols}) == len(symbols)
    assert not [d for d in parsed.diagnostics if d.code == "identity_collision"]


def test_cpp_in_class_constructor_and_destructor_do_not_collide():
    parsed, symbols = _cpp_symbols("class A {\n public:\n  A() {}\n  ~A() {}\n};\n", "a.h")
    members = sorted(s.simple_name for s in symbols if s.scope_chain == ["A"])
    assert members == ["A", "~A"]
    assert not [d for d in parsed.diagnostics if d.code == "identity_collision"]


def test_cpp_namespaces_qualify_identity_without_becoming_containers():
    source = (
        "namespace arrow {\nnamespace compute {\n"
        "class Kernel { public: int Run() { return 0; } };\n"
        "int Helper() { return 1; }\n"
        "}  // namespace compute\n"
        "int Helper() { return 2; }\n"
        "}  // namespace arrow\n"
        "namespace x::y { int Nested() { return 0; } }\n"
        "namespace {\nint Local() { return 3; }\n}\n"
    )
    parsed, symbols = _cpp_symbols(source)
    kernel = _by_name(parsed, "Kernel")
    run = _by_name(parsed, "Run")
    helpers = [s for s in symbols if s.simple_name == "Helper"]
    assert _qualified(kernel) == "arrow::compute::Kernel"
    assert _qualified(run) == "arrow::compute::Kernel::Run"
    assert sorted(_qualified(s) for s in helpers) == ["arrow::Helper", "arrow::compute::Helper"]
    assert _qualified(_by_name(parsed, "Nested")) == "x::y::Nested"
    assert _qualified(_by_name(parsed, "Local")) == "(anonymous)::Local"
    # Structural scope is unchanged: namespaced declarations stay top-level.
    assert kernel.scope_chain == [] and run.scope_chain == ["Kernel"]
    assert all(s.scope_chain == [] for s in helpers)
    assert run.symbol_type.name == "METHOD"
    assert all(s.symbol_type.name == "FUNCTION" for s in helpers)
    assert len({s.id for s in helpers}) == 2
    assert not {"arrow", "compute", "x", "y"} & {s.simple_name for s in symbols}
    contains = {
        (r.from_symbol_id, r.to_symbol_id)
        for r in parsed.relations
        if r.relation_type.name == "CONTAINS"
    }
    assert (kernel.id, run.id) in contains


def test_cpp_cv_and_ref_qualified_overloads_have_distinct_stable_ids():
    source = (
        "struct S {\n"
        "  int& Get() { return v; }\n"
        "  const int& Get() const { return v; }\n"
        "  int Take() && { return v; }\n"
        "  int Take() & { return v; }\n"
        "  int v;\n"
        "};\n"
    )
    parsed, symbols = _cpp_symbols(source, "s.h")
    gets = [s for s in symbols if s.simple_name == "Get"]
    takes = [s for s in symbols if s.simple_name == "Take"]
    assert len(gets) == 2 and len({s.id for s in gets}) == 2
    assert len(takes) == 2 and len({s.id for s in takes}) == 2
    assert not [d for d in parsed.diagnostics if d.code == "identity_collision"]


def test_cpp_qualified_identity_survives_line_shift():
    source = "namespace n {\nvoid K::f(int x) {}\n}\n"
    base = _by_name(parse(source, language="cpp", file_path="k.cc"), "f")
    moved = _by_name(parse("\n\n" + source, language="cpp", file_path="k.cc"), "f")
    assert base.id == moved.id
    assert _qualified(base) == "n::K::f"


def test_cpp_sfinae_template_overloads_have_distinct_stable_ids():
    source = (
        "struct V {\n"
        "  template <typename T>\n"
        "  enable_if_t<is_integer<T>::value, Status> Visit(const T& t) { return Ok(); }\n"
        "  template <typename T>\n"
        "  enable_if_t<is_floating<T>::value, Status> Visit(const T& t) { return Ok(); }\n"
        "  template <typename T, typename U>\n"
        "  Status Pair(const T& t) { return Ok(); }\n"
        "  template <typename T>\n"
        "  Status Pair(const T& t) { return Ok(); }\n"
        "};\n"
    )
    parsed, symbols = _cpp_symbols(source, "v.h")
    visits = [s for s in symbols if s.simple_name == "Visit"]
    pairs = [s for s in symbols if s.simple_name == "Pair"]
    assert len(visits) == 2 and len({s.id for s in visits}) == 2
    assert len(pairs) == 2 and len({s.id for s in pairs}) == 2
    assert not [d for d in parsed.diagnostics if d.code == "identity_collision"]
    # Whitespace inside the template head does not change identity.
    respaced = parse(
        source.replace("<typename T>", "<typename   T>"), language="cpp", file_path="v.h"
    )
    assert {s.id for s in respaced.symbols} == {s.id for s in parsed.symbols}


def test_cpp_signature_reads_as_a_declaration():
    parsed, symbols = _cpp_symbols(
        "template <typename T>\nT Clamp(T v) const noexcept { return v; }\n"
        "int Plain(int x) { return x; }\n",
        "c.h",
    )
    assert _by_name(parsed, "Clamp").signature == "template<typename T> T (T v) const noexcept"
    # Non-templated definitions keep the plain parameter-list signature.
    assert _by_name(parsed, "Plain").signature == "(int x)"


class TestModAttributeBoundary:
    """A module's path-attribute flag must come only from ITS OWN attributes.

    Walking through non-attribute siblings steals #[path]/#[cfg_attr] from a
    preceding item and silently breaks the module's bindings.
    """

    def _mods(self, source: str):
        from victor_codegraph.parser import parse

        return parse(source, language="rust").symbols

    @staticmethod
    def _module_meta(symbols, module_name):
        for symbol in symbols:
            if symbol.fully_qualified_name.endswith(f"::{module_name}"):
                return symbol.metadata
        raise AssertionError(f"module {module_name} not parsed")

    def test_path_attribute_does_not_leak_to_following_mod(self):
        symbols = self._mods('#[path = "custom.rs"]\nmod a;\nmod b;\n')
        assert self._module_meta(symbols, "a").get("rust_path_attribute") is True
        assert self._module_meta(symbols, "b").get("rust_path_attribute") is None, (
            "#[path] from the PRECEDING mod leaked into mod b — the unbounded "
            "sibling walk steals attributes from earlier items"
        )

    def test_cfg_attr_on_preceding_fn_does_not_leak_to_mod(self):
        symbols = self._mods('#[cfg_attr(feature = "x", inline)]\nfn helper() {}\nmod b;\n')
        assert (
            self._module_meta(symbols, "b").get("rust_path_attribute") is None
        ), "#[cfg_attr] from the PRECEDING fn leaked into mod b"

    def test_own_path_attribute_still_detected(self):
        symbols = self._mods('#[path = "other.rs"]\nmod dep;\n')
        assert self._module_meta(symbols, "dep").get("rust_path_attribute") is True

    def test_own_cfg_attr_still_detected(self):
        symbols = self._mods('#[cfg_attr(unix, path = "unix.rs")]\nmod dep;\n')
        assert self._module_meta(symbols, "dep").get("rust_path_attribute") is True
