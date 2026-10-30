"""Header and Ruby interface aliases preserve automatic parsing."""

import pytest

from victor_codegraph import parse
from victor_codegraph.languages import detect_language
from victor_codegraph.treesitter_parser import GrammarUnavailable


@pytest.mark.parametrize(
    "extension,language,source",
    [
        (".hh", "cpp", "class Example { public: void run() {} };"),
        (".hxx", "cpp", "class Example { public: void run() {} };"),
        (".rbi", "ruby", "class Example\n  def run; end\nend\n"),
    ],
)
def test_header_and_ruby_interface_aliases_extract_symbols(extension, language, source):
    path = "example" + extension
    assert detect_language(path) == language
    try:
        parsed = parse(source, file_path=path)
    except GrammarUnavailable:
        pytest.skip(f"{language} grammar not installed")
    names = {symbol.simple_name for symbol in parsed.symbols}
    assert "run" in names
    if language == "cpp":
        assert "Example" in names
    # Aliases preserve existing extraction; they do not promise new Ruby class support.
    canonical = parse(source, file_path="example" + (".hpp" if language == "cpp" else ".rb"))
    assert {(s.simple_name, s.symbol_type) for s in parsed.symbols} == {
        (s.simple_name, s.symbol_type) for s in canonical.symbols
    }


def test_unknown_extension_remains_unknown():
    assert detect_language("example.unknown") is None
