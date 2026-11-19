# Copyright 2026 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""Text-pipeline differential parity: dedup, classifier, JSON repair (FOU-C3).

Every test here runs the same input through BOTH dispatch paths of the
``victor.processing.native`` wrappers — the native path (normal dispatch,
requires the Rust extension) and the forced pure-Python fallback (each
wrapper module binds ``_NATIVE_AVAILABLE`` by value at import, so forcing
means patching the flag ON THE CONSUMER MODULE — the same pattern
test_secrets_scanner.py uses for the scanner).

Documented divergences are pinned, not hidden:

- **Hash-valued dedup/signature ops** — Rust uses xxHash3, the Python
  fallback MD5. The digests can never agree; the parity contract is the
  STRUCTURE (identical normalization bytes, identical block splits, identical
  duplicate-flag sequences and duplicate indices, deterministic and
  key-order-independent signatures). Pinned in ``TestHashFunctionDivergence``.
- **``repair_json`` string-content** — formerly a pinned divergence (the
  Python fallback rewrote ``True``/``False``/``None`` inside string values);
  the fallback now ports the Rust state machine's quote/escape tracking, so
  both paths are byte-identical including string-content preservation.
  Covered by ``TestRepairJsonParity`` and
  ``TestRepairJsonStringContentParity``.
- **``coerce_string_type`` install-shape split** — RESOLVED: the sweep found
  the fallback mapping ``yes``/``on``/``1``/``0`` to bool while the native
  contract keeps them string/int. The fallback now runs the same shared
  coercion pipeline as the hot path (native semantics), asserted as strict
  output parity in ``TestCoerceStringTypeParity``.

Selected by the native-parity CI job (``pytest -m native_parity``).
"""

from __future__ import annotations

import pytest

from victor.processing.native import classifier as classifier_mod
from victor.processing.native import deduplication as dedup_mod
from victor.processing.native import json_repair as json_repair_mod
from victor.processing.native._base import _NATIVE_AVAILABLE, _native

pytestmark = pytest.mark.native_parity

_TEXT_OPS = (
    "normalize_block",
    "rolling_hash_blocks",
    "find_duplicate_blocks",
    "compute_signature",
    "signature_similarity",
)
_JSON_OPS = ("repair_json", "extract_json_objects", "coerce_string_type")
_CLASSIFIER_OPS = (
    "has_action_keywords",
    "has_analysis_keywords",
    "has_generation_keywords",
    "has_negation",
    "find_all_keywords",
)

_requires_native = pytest.mark.skipif(
    not (
        _NATIVE_AVAILABLE
        and all(hasattr(_native, op) for op in _TEXT_OPS + _JSON_OPS + _CLASSIFIER_OPS)
    ),
    reason="native wheel missing text-pipeline ops — rebuild with maturin",
)


def _forced_python(mod, fn_name, *args):
    """Call the wrapper's pure-Python fallback by disabling ITS flag."""
    saved = mod._NATIVE_AVAILABLE
    mod._NATIVE_AVAILABLE = False
    try:
        return getattr(mod, fn_name)(*args)
    finally:
        mod._NATIVE_AVAILABLE = saved


# =============================================================================
# Deduplication
# =============================================================================

_LONG_PARA = (
    "This is a sufficiently long paragraph block that exceeds the minimum "
    "block length threshold easily."
)
DEDUP_CONTENT = "\n\n".join([_LONG_PARA, "x" * 70, _LONG_PARA, "y" * 80, _LONG_PARA, "tiny"])

NORMALIZE_CORPUS = [
    "  Hello   World  ",
    "Test.",
    "Test,",
    "Test:",
    "Line1\n\nLine2\ttab  end.",
    "",
    "   ",
    "MixedCASE Words HERE",
]


class TestNormalizeBlockParity:
    @_requires_native
    @pytest.mark.parametrize("block", NORMALIZE_CORPUS)
    def test_normalize_block_byte_identical(self, block):
        assert dedup_mod.normalize_block(block) == _forced_python(
            dedup_mod, "normalize_block", block
        )


class TestBlockSplitParity:
    """The block segmentation and duplicate DETECTION must agree exactly;
    only the hash digests differ (see TestHashFunctionDivergence)."""

    @_requires_native
    def test_block_texts_identical(self):
        native = dedup_mod.rolling_hash_blocks(DEDUP_CONTENT)
        fallback = _forced_python(dedup_mod, "rolling_hash_blocks", DEDUP_CONTENT)
        assert [b for _, b, _ in native] == [b for _, b, _ in fallback]

    @_requires_native
    def test_duplicate_flags_identical(self):
        native = dedup_mod.rolling_hash_blocks(DEDUP_CONTENT)
        fallback = _forced_python(dedup_mod, "rolling_hash_blocks", DEDUP_CONTENT)
        assert [f for _, _, f in native] == [f for _, _, f in fallback]

    @_requires_native
    def test_below_min_length_blocks_unhashed_both_paths(self):
        native = dedup_mod.rolling_hash_blocks("tiny block", 50)
        fallback = _forced_python(dedup_mod, "rolling_hash_blocks", "tiny block")
        assert native == fallback == [("", "tiny block", False)]

    @_requires_native
    def test_duplicate_indices_identical(self):
        native = dedup_mod.find_duplicate_blocks(DEDUP_CONTENT)
        fallback = _forced_python(dedup_mod, "find_duplicate_blocks", DEDUP_CONTENT)
        assert [idx for idx, _ in native] == [idx for idx, _ in fallback]
        assert len(native) == 2  # the repeated paragraph, seen twice after first

    @_requires_native
    def test_empty_content_matches(self):
        assert dedup_mod.rolling_hash_blocks("") == _forced_python(
            dedup_mod, "rolling_hash_blocks", ""
        )


class TestSignatureParity:
    @_requires_native
    def test_signature_deterministic_and_length_bound(self):
        args = {"path": "/tmp/x", "line": 5, "flag": True, "none": None}
        for fn in (
            dedup_mod.compute_signature,
            lambda *a: _forced_python(dedup_mod, "compute_signature", *a),
        ):
            first = fn("read", args)
            second = fn("read", args)
            assert first == second
            assert len(first) == 16
            int(first, 16)  # hex

    @_requires_native
    def test_signature_key_order_independent_both_paths(self):
        args_a = {"path": "/tmp/x", "line": 5}
        args_b = {"line": 5, "path": "/tmp/x"}
        assert dedup_mod.compute_signature("read", args_a) == dedup_mod.compute_signature(
            "read", args_b
        )
        fallback = _forced_python(dedup_mod, "compute_signature", "read", args_a)
        assert fallback == _forced_python(dedup_mod, "compute_signature", "read", args_b)

    @_requires_native
    def test_batch_matches_single_both_paths(self):
        calls = [("read", {"path": "a"}), ("write", {"path": "b", "content": "x"})]
        native = dedup_mod.compute_batch_signatures(calls)
        fallback = _forced_python(dedup_mod, "compute_batch_signatures", calls)
        assert len(native) == len(fallback) == 2
        assert native[0] == dedup_mod.compute_signature(*calls[0])
        assert fallback[0] == _forced_python(dedup_mod, "compute_signature", *calls[0])

    @_requires_native
    def test_signature_similarity_extremes_both_paths(self):
        sig = "a" * 16
        for fn in (
            dedup_mod.signature_similarity,
            lambda *a: _forced_python(dedup_mod, "signature_similarity", *a),
        ):
            assert fn(sig, sig) == 1.0
            assert fn(sig, "b" * 8) == 0.0  # different length -> 0.0


class TestHashFunctionDivergence:
    """PINNED: the digest VALUES can never match — xxHash3 (Rust) vs MD5
    (Python fallback). The parity contract is the structure above; this pin
    documents the value divergence so it is visible, not discovered."""

    @_requires_native
    def test_block_hash_digests_differ_by_design(self):
        native = dedup_mod.rolling_hash_blocks(DEDUP_CONTENT)
        fallback = _forced_python(dedup_mod, "rolling_hash_blocks", DEDUP_CONTENT)
        native_hashes = [h for h, _, f in native if f or h]
        fallback_hashes = [h for h, _, f in fallback if f or h]
        assert len(native_hashes) == len(fallback_hashes)
        assert native_hashes != fallback_hashes

    @_requires_native
    def test_signature_digests_differ_by_design(self):
        args = {"path": "/tmp/x"}
        assert dedup_mod.compute_signature("read", args) != _forced_python(
            dedup_mod, "compute_signature", "read", args
        )


# =============================================================================
# Classifier keyword detection
# =============================================================================

CLASSIFIER_CORPUS = [
    "please run the tests",
    "analyze this code and explain",
    "create a new file",
    "do not deploy yet, avoid breaking prod",
    "don't touch the pipeline without review",
    "RESTART the service and COMMIT",
    "find where is the config located",
    "refactor and rename the module",
    "hello world",
    "",
]


class TestClassifierKeywordParity:
    @_requires_native
    @pytest.mark.parametrize("text", CLASSIFIER_CORPUS)
    @pytest.mark.parametrize(
        "fn_name",
        ["has_action_keywords", "has_analysis_keywords", "has_generation_keywords", "has_negation"],
    )
    def test_keyword_predicates_agree(self, text, fn_name):
        assert getattr(classifier_mod, fn_name)(text) == _forced_python(
            classifier_mod, fn_name, text
        )

    @_requires_native
    @pytest.mark.parametrize(
        "text",
        [
            "run and analyze then create, fix the search",
            "no keywords here at all",
            "UPDATE and modify, then look for where is",
            "implement write deploy",
        ],
    )
    def test_find_all_keywords_agree(self, text):
        native = sorted(classifier_mod.find_all_keywords(text))
        fallback = sorted(_forced_python(classifier_mod, "find_all_keywords", text))
        assert native == fallback


# =============================================================================
# JSON repair
# =============================================================================

REPAIR_CORPUS = [
    '{"key": "value"}',  # already valid
    "{'key': 'value'}",  # single quotes
    "{'a': True, 'b': False, 'c': None}",  # Python literals
    "{'nested': {'x': [1, 2, 3]}}",  # nesting
    '{"a": "it has an apostrophe: it\'s fine"}',  # apostrophe inside string
    '{"a": "quote\\" inside", \'b\': 2}',  # escaped quote + mixed quotes
    "[1, 2, 3]",
    "  {'spaced': 'key'}  ",
    # String-content preservation (FOU-C3 #1957 finding): literals inside
    # string VALUES must survive the repair on both paths.
    "{'msg': 'True facts about None'}",
    "{'msg': 'NoneTheLess TrueNorth', 'flag': False}",
    "{'a': 'TrueFalseNone'}",
    "{'s': 'braces {not json} here', 'n': None}",
    "{'a': 'escaped \\' single', 'b': True}",  # \' apostrophe escape
    "{'a': 'backslash \\\\ then None'}",  # backslash escape before a literal
    "{'k': \"double 'inner' single\", 'j': True}",  # quotes inside quotes
    "{'a': 'mixed \\' and \" both', 'b': None}",
    "{'apostrophe': \"it's True alright\"}",  # ' inside double-quoted string
    '{"key": "None value"}',  # literals in a valid-JSON string value
    "[True, False, None]",  # literals outside strings stay rewriteable
    "True",  # bare literal document
]

EXTRACT_CORPUS = [
    ('prefix {"a": 1} suffix', None),
    ("blah {'x': True} and [1,2] end", None),
    ('{"a": 1}{"b": 2}', None),
    ('text with {braces} not json {"ok": 1}', None),
    ("no json here", None),
]


class TestRepairJsonParity:
    @_requires_native
    @pytest.mark.parametrize("text", REPAIR_CORPUS)
    def test_repair_json_identical(self, text):
        assert json_repair_mod.repair_json(text) == _forced_python(
            json_repair_mod, "repair_json", text
        )


class TestExtractJsonObjectsParity:
    @_requires_native
    @pytest.mark.parametrize("text,_", EXTRACT_CORPUS, ids=lambda v: str(v)[:30])
    def test_extract_identical(self, text, _):
        native = json_repair_mod.extract_json_objects(text)
        fallback = _forced_python(json_repair_mod, "extract_json_objects", text)
        assert native == fallback

    @_requires_native
    def test_positions_are_exact_spans_both_paths(self):
        text = 'before {"a": 1} after'
        for result in (
            json_repair_mod.extract_json_objects(text),
            _forced_python(json_repair_mod, "extract_json_objects", text),
        ):
            start, end, extracted = result[0]
            assert text[start:end] == extracted


class TestRepairJsonStringContentParity:
    """The FOU-C3 #1957 finding, now closed: the Python fallback used to
    rewrite ``True``/``False``/``None`` byte-wise BEFORE quote tracking,
    corrupting string values (``{'msg': 'True facts about None'}`` became
    ``"true facts about null"``). The fallback now tracks quote/escape state
    exactly like the Rust machine, so both paths agree byte for byte AND
    parse back to the original string data."""

    @_requires_native
    def test_string_values_survive_repair_both_paths(self):
        text = "{'msg': 'True facts about None'}"
        native = json_repair_mod.repair_json(text)
        fallback = _forced_python(json_repair_mod, "repair_json", text)

        assert native == fallback
        expected = {"msg": "True facts about None"}
        assert json_repair_mod.json_loads(native) == expected
        assert json_repair_mod.json_loads(fallback) == expected

    @_requires_native
    @pytest.mark.parametrize(
        "text,expected",
        [
            # Mid-word occurrences inside string values are preserved.
            ("{'m': 'NoneTheLess', 'n': None}", {"m": "NoneTheLess", "n": None}),
            ("{'m': 'TrueNorth', 'f': False}", {"m": "TrueNorth", "f": False}),
            # Braces inside strings are not structural.
            ("{'s': 'braces {not json} here'}", {"s": "braces {not json} here"}),
            # Escaped quotes inside strings stay escaped and intact.
            ('{"a": "quote\\" inside"}', {"a": 'quote" inside'}),
            ("{'a': 'it\\'s True'}", {"a": "it's True"}),
            # A lone double quote inside a single-quoted string gets escaped.
            ('{\'a\': "say \\"hi\\""}', {"a": 'say "hi"'}),
        ],
    )
    def test_content_edge_cases_identical_and_parse(self, text, expected):
        native = json_repair_mod.repair_json(text)
        fallback = _forced_python(json_repair_mod, "repair_json", text)

        assert native == fallback
        assert json_repair_mod.json_loads(native) == expected


class TestCoerceStringTypeParity:
    """The C3 sweep pinned an install-shape split here: the pure-Python
    fallback mapped ``yes``/``on``/``1``/``0`` to bool and had no
    structured-type detection, while the native contract (the Rust hot path,
    mirrored by the scalar fast path in ``json_repair.py``) keeps only
    ``true``/``false`` as bool, coerces ``1``/``0`` to int, and validates
    JSON object/array candidates as dict/list. The fallback now runs the
    SAME shared coercion pipeline as the hot path, so the contract is strict
    parity: same input, same ``(type_name, coerced_str, error)`` tuple on
    either install shape."""

    # Values where the old fallback diverged from the native contract:
    DIVERGENT_FAMILY = (
        "yes",
        "on",
        "no",
        "off",
        "1",
        "0",
        '{"a": 1}',
        "[1, 2]",
    )

    # Broad corpus: truthiness words, case variants, null family, numeric
    # strings, whitespace, empty, plain strings, structured candidates.
    PARITY_CORPUS = DIVERGENT_FAMILY + (
        "YES",
        "On",
        "TRUE",
        "False",
        "true",
        "false",
        "null",
        "none",
        "nil",
        "NONE",
        "5",
        "-123",
        "007",
        "3.14",
        "-0.5",
        "1e3",
        "  true  ",
        "\t42\n",
        " hello ",
        "",
        "hello world",
        "True)",
        "/usr/bin",
        "0x10",
        '{"a"',
        "{invalid}",
        "  ",
    )

    @pytest.mark.parametrize("value", PARITY_CORPUS)
    @_requires_native
    def test_dispatch_equals_fallback(self, value):
        dispatch = json_repair_mod.coerce_string_type(value)
        fallback = _forced_python(json_repair_mod, "coerce_string_type", value)
        assert (
            dispatch == fallback
        ), f"install-shape fork for {value!r}: dispatch={dispatch!r} fallback={fallback!r}"

    @pytest.mark.parametrize(
        "value,expected",
        [
            # The old fallback's bool expansion must NOT come back:
            ("yes", ("string", "yes", None)),
            ("YES", ("string", "YES", None)),  # original case preserved
            ("on", ("string", "on", None)),
            ("no", ("string", "no", None)),
            ("off", ("string", "off", None)),
            ("1", ("int", "1", None)),
            ("0", ("int", "0", None)),
            # Native truthiness contract: only true/false are bool.
            ("true", ("bool", "true", None)),
            ("FALSE", ("bool", "false", None)),
            # Null family.
            ("None", ("null", "null", None)),
            # Output is trimmed like the Rust trim().
            ("  yes  ", ("string", "yes", None)),
            ("\ttrue\n", ("bool", "true", None)),
            # Empty / degenerate inputs stay strings.
            ("", ("string", "", None)),
            ("{invalid", ("string", "{invalid", None)),
            # Structured candidates are detected like serde_json does.
            ('{"a": 1}', ("dict", '{"a": 1}', None)),
            ("[1, 2]", ("list", "[1, 2]", None)),
        ],
    )
    def test_fallback_matches_native_contract_without_wheel(self, value, expected):
        """Run the pure-Python fallback directly (native dispatch forced off)
        and assert the NATIVE contract tuples — the exact expectations the
        Rust hot path produces. This fails against the pre-fix fallback."""
        assert _forced_python(json_repair_mod, "coerce_string_type", value) == expected
