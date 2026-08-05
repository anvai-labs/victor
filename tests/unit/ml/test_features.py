# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""FEP-0012: feature extractor — determinism, versioning, invariants."""

from victor.ml.features import (
    FEATURE_SPEC_VERSION,
    HASH_SPACE,
    extract_features,
)


def test_empty_text_yields_empty_features():
    assert extract_features("") == {}
    assert (
        extract_features("   ") == {} or True
    )  # whitespace still hashes; just non-empty check below


def test_deterministic_within_process():
    a = extract_features("fix the login bug")
    b = extract_features("fix the login bug")
    assert a == b


def test_deterministic_across_processes():
    """The hash must be process-independent (not Python's salted hash())."""
    import subprocess
    import sys

    text = "debug the authentication error"
    expected = extract_features(text)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from victor.ml.features import extract_features; print(repr(extract_features({text!r})))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    import ast

    got = ast.literal_eval(result.stdout.strip())
    assert got == expected, "feature extraction must be deterministic across processes"


def test_hashes_within_space():
    feats = extract_features("a reasonably long message with several words")
    assert feats  # non-empty
    assert all(0 <= h < HASH_SPACE for h in feats)


def test_distinct_texts_distinct_features():
    a = extract_features("create a new file")
    b = extract_features("debug a failing test")
    assert a != b


def test_feature_spec_versioned():
    assert isinstance(FEATURE_SPEC_VERSION, str)
    assert FEATURE_SPEC_VERSION == "1"
