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

"""Native-BPE/tiktoken parity for exact token counting (co-design review 17a).

``count_tokens`` was misrouted to the heuristic ``count_tokens_fast`` whenever
the native wheel was installed — the "exact" entry point returned approximate
counts. It now runs the Rust ``BpeTokenizer`` built from tiktoken's own
cl100k_base ranks, so exact and fast paths finally differ. These tests pin:

- native BPE counts equal tiktoken counts on ordinary text (the parity
  contract that makes the redirect safe), and
- the ONE documented divergence (the Rust pre-tokenizer approximates
  tiktoken's ``\\s+(?!\\S)`` lookahead as ``\\s+$`` because the Rust regex
  crate lacks lookahead, so mid-string runs of 2+ whitespace before a word
  split differently) stays bounded at ±1 token, and
- the fallback chain survives a wheel without ``BpeTokenizer`` and a
  tiktoken-less environment.
"""

from __future__ import annotations

import pytest

import victor.processing.native.tokenizer as tok_mod
from victor.processing.native._base import _NATIVE_AVAILABLE, _native
from victor.processing.native.tokenizer import (
    _get_bpe_tokenizer,
    _get_tiktoken_encoder,
    count_tokens,
    count_tokens_batch,
)

# Corpus agreed on by both engines (verified empirically): prose, code,
# unicode, contractions, numbers, newline runs, trailing single whitespace.
_PARITY_CORPUS = [
    "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "def tokenize(text: str) -> list[int]:\n    return enc.encode(text)\n",
    "don't can't won't it's",
    "values 3.14159 and 271828 and 1e9",
    "unicode: café naïve 日本語 emoji 🎉",
    "line\n\n\nbreak",
    "trailing space ",
    "",
    " ",
    "x",
]

# Inputs where the documented regex approximation actually diverges: a run of
# 2+ whitespace (tab or mixed) immediately before a word. tiktoken's
# ``\s+(?!\S)`` backtracks to leave the run attached to the following word;
# ``\s+$`` only fires at end-of-string, so the run here splits off alone.
_DIVERGENCE_CORPUS = [
    "tabs\t\there",
    "a\t\t\tb",
    "mixed \t\n x",
]


def _has_bpe_symbol() -> bool:
    return _NATIVE_AVAILABLE and hasattr(_native, "BpeTokenizer")


def _stale_wheel() -> bool:
    """True when the wheel has the symbol but can't build a working tokenizer.

    Behavioral probe (mirrors test_context_fitter_parity.py): construct with
    tiny ranks and count a known string — catches wheels where the class
    exists but its ABI has drifted, regardless of version strings.
    """
    if not _has_bpe_symbol():
        return False
    try:
        probe = _native.BpeTokenizer(
            "probe",
            [(b" a", 0), (b" b", 1), (b"ab", 2)],
            [("<|endoftext|>", 3)],
        )
        return probe.count_tokens("ab") != 1
    except Exception:
        return True


@pytest.fixture(autouse=True)
def _restore_tokenizer_globals():
    """Snapshot/restore the module's cached-singleton globals so tests that
    force rebuilds (or simulate unavailable tiktoken) don't leak state."""
    saved = (
        tok_mod._bpe_tokenizer,
        tok_mod._bpe_tokenizer_failed,
        tok_mod._tiktoken_encoder,
    )
    yield
    (
        tok_mod._bpe_tokenizer,
        tok_mod._bpe_tokenizer_failed,
        tok_mod._tiktoken_encoder,
    ) = saved


_requires_native_bpe = pytest.mark.skipif(
    not _has_bpe_symbol() or _stale_wheel(),
    reason="native wheel missing BpeTokenizer or stale — rebuild with maturin",
)


class TestNativeTiktokenParity:
    @_requires_native_bpe
    @pytest.mark.parametrize("text", _PARITY_CORPUS)
    def test_native_bpe_matches_tiktoken(self, text: str) -> None:
        encoder = _get_tiktoken_encoder()
        assert encoder is not None, "tiktoken required for parity comparison"
        tokenizer = _get_bpe_tokenizer()
        assert tokenizer is not None

        assert tokenizer.count_tokens(text) == len(encoder.encode(text))

    @_requires_native_bpe
    def test_batch_matches_single(self) -> None:
        texts = _PARITY_CORPUS + _DIVERGENCE_CORPUS
        assert count_tokens_batch(texts) == [count_tokens(t) for t in texts]


class TestExactCountRedirect:
    """The redirect itself: count_tokens must agree with tiktoken on ordinary
    text now that it runs real BPE instead of the heuristic."""

    @_requires_native_bpe
    @pytest.mark.parametrize("text", _PARITY_CORPUS)
    def test_count_tokens_matches_tiktoken(self, text: str) -> None:
        encoder = _get_tiktoken_encoder()
        assert encoder is not None
        assert count_tokens(text) == len(encoder.encode(text))

    @_requires_native_bpe
    def test_exact_and_fast_paths_actually_differ(self) -> None:
        """The whole point of item 17a: exact must no longer be the heuristic.
        On a whitespace-heavy input the two native functions disagree."""
        text = "word     gap\t\ttab\n\n\nruns"
        assert _get_bpe_tokenizer() is not None
        if hasattr(_native, "count_tokens_fast"):
            assert count_tokens(text) != _native.count_tokens_fast(text)


class TestKnownDivergence:
    """The documented ``\\s+(?!\\S)`` → ``\\s+$`` approximation diverges only
    on mid-string whitespace runs; when it does, the count gap must stay at 1."""

    @_requires_native_bpe
    @pytest.mark.parametrize("text", _DIVERGENCE_CORPUS)
    def test_divergence_is_bounded(self, text: str) -> None:
        encoder = _get_tiktoken_encoder()
        assert encoder is not None
        tokenizer = _get_bpe_tokenizer()
        assert tokenizer is not None
        diff = abs(tokenizer.count_tokens(text) - len(encoder.encode(text)))
        assert diff <= 1, (
            f"whitespace-run divergence for {text!r} exceeded the documented "
            f"±1 bound (diff={diff}); the pre-tokenizer approximation in "
            f"rust/crates/python-bindings/src/tokenizer.rs has changed"
        )

    @_requires_native_bpe
    def test_trailing_whitespace_does_not_diverge(self) -> None:
        """``\\s+$`` and ``\\s+(?!\\S)`` agree at end-of-string: trailing
        whitespace runs — the case the lookahead actually targets — must
        match tiktoken exactly."""
        encoder = _get_tiktoken_encoder()
        assert encoder is not None
        tokenizer = _get_bpe_tokenizer()
        assert tokenizer is not None
        for text in ["trailing   ", "tabs at end\t\t", "mixed  \t  "]:
            assert tokenizer.count_tokens(text) == len(encoder.encode(text))


class TestFallbackPreservation:
    """The redirect must not break the chain when native BPE can't engage."""

    def test_wheel_without_bpe_symbol_falls_through_to_tiktoken(self, monkeypatch):
        """A stale wheel that predates BpeTokenizer must leave count_tokens
        on the tiktoken path (still exact), not the heuristic."""
        import builtins

        encoder = _get_tiktoken_encoder()
        if encoder is None:
            pytest.skip("tiktoken unavailable")

        class _OldWheel:
            def __getattr__(self, name):
                if name == "BpeTokenizer":
                    raise AttributeError(name)
                return getattr(_native, name)

        monkeypatch.setattr(tok_mod, "_bpe_tokenizer", None)
        monkeypatch.setattr(tok_mod, "_bpe_tokenizer_failed", False)
        monkeypatch.setattr(tok_mod, "_native", _OldWheel())
        assert count_tokens("hello world") == len(encoder.encode("hello world"))

    def test_tiktoken_less_environment_falls_through_to_estimation(self, monkeypatch):
        """Without tiktoken OR native BPE, count_tokens must still return the
        word-based estimate rather than raising."""
        monkeypatch.setattr(tok_mod, "_TIKTOKEN_AVAILABLE", False)
        monkeypatch.setattr(tok_mod, "_tiktoken_encoder", None)
        monkeypatch.setattr(tok_mod, "_bpe_tokenizer", None)
        monkeypatch.setattr(tok_mod, "_bpe_tokenizer_failed", False)
        if _has_bpe_symbol():
            # Native present: construction needs tiktoken ranks, so it must
            # fail-and-remember rather than raise.
            assert count_tokens("hello world") == len("hello world".split()) * 13 // 10
            assert tok_mod._bpe_tokenizer_failed is True
        else:
            assert count_tokens("hello world") == len("hello world".split()) * 13 // 10

    def test_failed_construction_is_remembered(self, monkeypatch):
        """A construction failure must set the failed flag so we don't retry
        the (expensive) ranks handoff on every call."""
        if not _has_bpe_symbol():
            pytest.skip("exercises the native construction path")

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated construction failure")

        monkeypatch.setattr(tok_mod._native, "BpeTokenizer", _boom)
        monkeypatch.setattr(tok_mod, "_bpe_tokenizer", None)
        monkeypatch.setattr(tok_mod, "_bpe_tokenizer_failed", False)
        assert _get_bpe_tokenizer() is None
        assert tok_mod._bpe_tokenizer_failed is True
        # Second call short-circuits without re-raising/retrying.
        assert _get_bpe_tokenizer() is None
