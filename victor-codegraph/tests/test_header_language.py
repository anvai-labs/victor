"""`.h` headers resolve to C or C++ from their text, project context and a retry."""

import pytest

from victor_codegraph import parse
from victor_codegraph.languages import (
    classify_header,
    matches_languages,
    project_header_default,
    resolve_language,
)
from victor_codegraph.model import ParseStatus
from victor_codegraph.repository_index import parse_repo
from victor_codegraph.treesitter_parser import GrammarUnavailable

CPP_EXPORT_HEADER = """
#pragma once
#include <memory>
namespace arrow {
/// A class with an export macro, as Arrow writes them.
class ARROW_EXPORT Array {
 public:
  virtual ~Array() = default;
  std::shared_ptr<Array> Slice(int64_t offset) const;
};
}  // namespace arrow
"""

C_API_HEADER = """
#ifndef ABI_H
#define ABI_H
#ifdef __cplusplus
extern "C" {
#endif
struct ArrowArray {
  int64_t length;
  void (*release)(struct ArrowArray*);
};
static inline int arrow_copy(char *restrict dst, const char *restrict src, int new) {
  return new;
}
#ifdef __cplusplus
}
#endif
#endif
"""

CLEAN_CPP_HEADER = """
#pragma once
namespace arrow {
class Array {
 public:
  int64_t length() const;
};
}  // namespace arrow
"""

C_RESTRICT_HEADER = """
#ifndef COPY_H
#define COPY_H
static inline int arrow_copy(char *restrict dst, const char *restrict src, int new) {
  return new;
}
#endif
"""

NEUTRAL_HEADER = """
#ifndef POINT_H
#define POINT_H
struct point { int x; int y; };
int point_norm(const struct point *p);
#endif
"""


def _require_grammars():
    try:
        parse("int f(void);", language="c", file_path="probe.c")
        parse("int f();", language="cpp", file_path="probe.cpp")
    except GrammarUnavailable:  # pragma: no cover - environment without the grammar pack
        pytest.skip("C/C++ grammars not installed")


@pytest.mark.parametrize(
    "source,expected",
    [
        (CPP_EXPORT_HEADER, "cpp"),
        (C_API_HEADER, "c"),
        (NEUTRAL_HEADER, None),
        ("template <typename T> T id(T v) { return v; }", "cpp"),
        ("void f(int *this, int class);", "c"),
        ('// namespace std\n/* class X { */\nconst char *s = "a::b";\n', None),
    ],
)
def test_classify_header_reads_language_signals(source, expected):
    assert classify_header(source) == expected


def test_resolve_language_uses_content_then_header_default():
    assert resolve_language("a.h", NEUTRAL_HEADER) == "cpp"
    assert resolve_language("a.h", NEUTRAL_HEADER, header_default="c") == "c"
    assert resolve_language("a.h", C_API_HEADER) == "c"
    assert resolve_language("a.h", CPP_EXPORT_HEADER, header_default="c") == "cpp"
    assert resolve_language("a.h") == "cpp"
    # Unambiguous extensions are never reclassified by content.
    assert resolve_language("a.c", CPP_EXPORT_HEADER) == "c"
    assert resolve_language("a.hpp", C_API_HEADER) == "cpp"
    with pytest.raises(ValueError):
        resolve_language("a.h", NEUTRAL_HEADER, header_default="rust")


def test_project_header_default_prefers_c_only_for_pure_c_inventories():
    assert project_header_default(["a.c", "b.h", "c.py"]) == "c"
    assert project_header_default(["a.c", "b.cc", "c.h"]) == "cpp"
    assert project_header_default(["b.h"]) == "cpp"


def test_language_filter_keeps_headers_for_either_c_family_language():
    assert matches_languages("a.h", {"cpp"})
    assert matches_languages("a.h", {"c"})
    assert not matches_languages("a.h", {"python"})
    assert not matches_languages("a.c", {"cpp"})
    assert not matches_languages("a.unknown", None)


def test_cpp_header_parses_with_cpp_grammar():
    _require_grammars()
    parsed = parse(CLEAN_CPP_HEADER, file_path="cpp/src/arrow/array/array_base.h")
    assert parsed.language == "cpp"
    assert parsed.status is ParseStatus.SUCCESS
    assert "Array" in {symbol.simple_name for symbol in parsed.symbols}


def test_export_macro_header_is_cpp_and_keeps_its_class():
    _require_grammars()
    # Export macros still leave tree-sitter error nodes (a separate limitation);
    # the header must nevertheless be C++ so the class is extracted at all.
    parsed = parse(CPP_EXPORT_HEADER, file_path="cpp/src/arrow/array/array_base.h")
    assert parsed.language == "cpp"
    assert "Array" in {symbol.simple_name for symbol in parsed.symbols}
    as_c = parse(CPP_EXPORT_HEADER, language="c", file_path="array_base.h")
    assert "Array" not in {symbol.simple_name for symbol in as_c.symbols}


def test_c_header_keeps_c_grammar():
    _require_grammars()
    parsed = parse(C_RESTRICT_HEADER, file_path="cpp/src/arrow/c/copy.h")
    assert parsed.language == "c"
    assert parsed.status is ParseStatus.SUCCESS
    assert "arrow_copy" in {symbol.simple_name for symbol in parsed.symbols}


def test_misclassified_header_retries_the_other_grammar():
    _require_grammars()
    # The header has no signals, so the project default (C) picks the grammar;
    # the anonymous-namespace-free C++ reference syntax still fails under C.
    source = "struct S { int &ref; };\n"
    parsed = parse(source, file_path="s.h", header_default="c")
    assert parsed.language == "cpp"
    assert parsed.status is ParseStatus.SUCCESS
    assert any(d.code == "header_language_retry" for d in parsed.diagnostics)


def test_explicit_language_disables_header_resolution():
    _require_grammars()
    parsed = parse(CPP_EXPORT_HEADER, language="c", file_path="array_base.h")
    assert parsed.language == "c"
    assert parsed.status is ParseStatus.PARTIAL


def test_parse_repo_resolves_headers_with_project_context(tmp_path):
    _require_grammars()
    (tmp_path / "pure").mkdir()
    (tmp_path / "pure" / "point.c").write_text("int point_norm(void) { return 0; }\n")
    (tmp_path / "pure" / "point.h").write_text(NEUTRAL_HEADER)
    (tmp_path / "mixed").mkdir()
    (tmp_path / "mixed" / "array.cc").write_text("int run() { return 0; }\n")
    (tmp_path / "mixed" / "point.h").write_text(NEUTRAL_HEADER)
    (tmp_path / "mixed" / "array.h").write_text(CLEAN_CPP_HEADER)

    pure = parse_repo(tmp_path / "pure")
    assert pure.files["point.h"].language == "c"
    mixed = parse_repo(tmp_path / "mixed", languages=["cpp"])
    assert mixed.files["point.h"].language == "cpp"
    assert mixed.files["array.h"].language == "cpp"
    assert mixed.files["array.h"].status is ParseStatus.SUCCESS


# ── Review regressions: keywords that C and C++ share ─────────────────────────


@pytest.mark.parametrize(
    "source",
    [
        "int *f(void) { return nullptr; }\n",  # C23
        "constexpr int limit = 10;\n",  # C23
    ],
)
def test_keywords_shared_with_c23_are_evidence_for_neither_language(source):
    assert classify_header(source) is None
    assert resolve_language("a.h", source) == "cpp"  # the project default decides
    assert resolve_language("a.h", source, header_default="c") == "c"


def test_cpp_only_keywords_still_signal_cpp():
    assert classify_header("struct S { virtual ~S(); };\n") == "cpp"
    assert classify_header("void f() noexcept;\n") == "cpp"
    assert classify_header("int g() { auto x = static_cast<int>(1); return x; }\n") == "cpp"


def test_cpp_keywords_used_as_c_identifiers_signal_c():
    assert classify_header("struct opts { int explicit; int mutable; };\n") == "c"
    # ...and outweigh a shared-keyword hit in a mixed inventory.
    source = "struct opts { int mutable; };\nconstexpr int limit = 10;\n"
    assert classify_header(source) == "c"


def test_c23_header_follows_the_project_default_when_parsed():
    _require_grammars()
    source = "int *f(void) { return nullptr; }\n"
    assert parse(source, file_path="pure-c/expr.h", header_default="c").language == "c"
    assert parse(source, file_path="mixed/expr.h").language == "cpp"


def _without_grammar(monkeypatch, missing):
    """Make ``parse_treesitter`` report one C-family grammar as not installed."""
    import victor_codegraph.parser as parser_module

    real = parser_module.parse_treesitter

    def parse_treesitter(content, file_path, language, *, emit_references=False):
        if language == missing:
            raise GrammarUnavailable(f"{missing} grammar not installed")
        return real(content, file_path, language, emit_references=emit_references)

    monkeypatch.setattr(parser_module, "parse_treesitter", parse_treesitter)


def test_header_retries_the_other_grammar_when_the_resolved_one_is_missing(monkeypatch):
    _require_grammars()
    _without_grammar(monkeypatch, "c")
    # ``restrict`` says C; both grammars read the declaration itself cleanly.
    declaration = "int arrow_copy(char *restrict dst, const char *restrict src);\n"
    parsed = parse(declaration, file_path="arrow/c/copy.h")
    assert parsed.language == "cpp"
    assert parsed.status is ParseStatus.SUCCESS
    assert [d.message for d in parsed.diagnostics if d.code == "header_language_retry"] == [
        "c grammar unavailable; cpp parsed cleanly"
    ]


def test_header_keeps_the_unavailable_result_when_the_other_grammar_errors(monkeypatch):
    _require_grammars()
    _without_grammar(monkeypatch, "cpp")
    parsed = parse(CLEAN_CPP_HEADER, file_path="arrow/array_base.h")  # the text says C++
    assert parsed.language == "cpp"
    assert parsed.status is ParseStatus.FALLBACK
    assert not [d for d in parsed.diagnostics if d.code == "header_language_retry"]


def test_genuinely_c_header_stays_c_when_only_the_cpp_grammar_is_installed(monkeypatch):
    """``new`` as an identifier is C; the C++ grammar cannot read it, so no retry."""
    _require_grammars()
    _without_grammar(monkeypatch, "c")
    parsed = parse(C_RESTRICT_HEADER, file_path="arrow/c/copy.h")
    assert parsed.language == "c"
    assert parsed.status is ParseStatus.FALLBACK
    assert not [d for d in parsed.diagnostics if d.code == "header_language_retry"]
