# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Secret-scanner word-boundary tests (FEP-0048 FOU-A2b follow-up).

The generic keyword patterns (``generic_password``, ``generic_secret``,
``generic_api_key``) previously treated bare whitespace as a key-value
separator (Rust separator class ``['":\\s=]+``), so benign prose like
"the password policies are documented elsewhere" matched as a credential.
The tightened patterns require an explicit ``=`` or ``:`` delimiter —
a value-bearing shape — while staying fail-closed for real key-value
credentials, quoted or not.

Differential parity: every case here is asserted through BOTH dispatch
paths (the Rust-accelerated ``detect_secrets`` and the pure-Python
fallback), so CI without the native extension exercises the same
contract.
"""

import pytest

from victor.security.safety.secrets import (
    GENERIC_PASSWORD_PATTERN,
    GENERIC_SECRET_PATTERN,
    SecretScanner,
    detect_secrets,
)

_GITHUB_PAT = "ghp_" + "a" * 36

# Benign prose: the keyword appears as a WORD, never as key=value/# key: value.
BENIGN_PROSE = [
    "my secret project name",
    "the secret ingredient is patience",
    "password policies are documented elsewhere",
    "the password manager recommends vaults",
    "our secret sauce is the ranking strategy",
    "secret recipes live in the cookbook appendix",
    "the api key rotation ceremony happens monthly",
]

# Real credential shapes that must STILL be detected after the tightening.
# Expected values are SETS because the two dispatch paths name equivalent
# patterns differently (Rust folds secret= into generic_password and names
# GitHub PATs github_token; Python has a separate generic_secret and
# github_pat). Semantics must agree; labels may differ.
REAL_SHAPES = [
    ("password = hunter2hunter2", {"generic_password"}),
    ("PASSWORD:hunter2hunter2", {"generic_password"}),
    ('password = "correct horse battery staple"', {"generic_password"}),
    ("SECRET=abc12345def6", {"generic_secret", "generic_password"}),
    ("client secret: supersecretvalue1", {"generic_secret", "generic_password"}),
    (
        "api_key = ABCDEFGHIJKLMNOPQRSTUVWXYZ1234",
        {"generic_api_key", "generic_secret"},
    ),
    ("secret=ghp_" + "a" * 36, {"github_pat", "github_token"}),
    (_GITHUB_PAT, {"github_pat", "github_token"}),
]


def _matched_types(content: str, *, use_native: bool) -> set[str]:
    if use_native:
        return {m.secret_type for m in detect_secrets(content)}
    # Force the pure-Python scanner path.
    return {m.secret_type for m in SecretScanner().scan(content)}


def _native_has_generic_patterns() -> bool:
    """True when the INSTALLED native scanner knows the generic family.

    ``victor-native`` is a PyPI dependency: CI's compiled module can be
    older than this repo's Rust source (the generic password/secret
    family is not in every released wheel). Probe behaviorally — the
    Python fallback must always detect the probe shape, so a native
    miss means the installed build predates the generic patterns and
    dispatch-path assertions would test the wrong thing.
    """
    import victor.security.safety.secrets as sec

    if not sec._RUST_SECRETS_AVAILABLE:
        return False
    probe = {m.secret_type for m in sec.detect_secrets("password: hunter2hunter2")}
    return "generic_password" in probe


@pytest.fixture(params=[True, False], ids=["dispatch", "python-fallback"])
def scanner_path(request, monkeypatch):
    """Run assertions through both dispatch paths.

    Yields a callable that scans with either the normal dispatch (Rust
    when installed) or a forced pure-Python ``SecretScanner``. The
    dispatch path skips when the installed native build predates the
    generic patterns (PyPI wheel lag); the python-fallback assertions
    are unconditional — they are the correctness contract.
    """
    use_native = request.param
    if use_native and not _native_has_generic_patterns():
        pytest.skip(
            "installed victor-native wheel predates the generic "
            "secret patterns; python-fallback covers correctness"
        )
    if not use_native:
        monkeypatch.setattr("victor.security.safety.secrets._RUST_SECRETS_AVAILABLE", False)
    return use_native


class TestBenignProseNeverMatches:
    """The word-boundary regression: keyword prose is not a credential."""

    @pytest.mark.parametrize("content", BENIGN_PROSE)
    def test_benign_prose_is_not_flagged(self, scanner_path, content):
        assert _matched_types(content, use_native=scanner_path) == set()


class TestRealShapesStillDetected:
    """Fail-closed: tightening must not open a hole for real shapes."""

    @pytest.mark.parametrize(
        "content,expected_types",
        REAL_SHAPES,
        ids=[
            "unquoted-assign",
            "unquoted-colon",
            "quoted-assign",
            "unquoted-secret-env",
            "two-word-key-colon",
            "long-api-key",
            "secret-keyword-github-pat",
            "bare-github-pat",
        ],
    )
    def test_real_credential_shape_is_detected(self, scanner_path, content, expected_types):
        matched = _matched_types(content, use_native=scanner_path)
        assert expected_types & matched, f"none of {expected_types} present in {matched}"


class TestPatternShapeContract:
    """The pattern strings themselves encode the value-bearing shape."""

    def test_generic_password_requires_delimiter(self):
        import re

        compiled = re.compile(GENERIC_PASSWORD_PATTERN, re.IGNORECASE)
        assert compiled.search("password = hunter2hunter2")
        # No delimiter -> no match, regardless of word length.
        assert not compiled.search("password policies are documented elsewhere")

    def test_generic_secret_requires_delimiter(self):
        import re

        compiled = re.compile(GENERIC_SECRET_PATTERN, re.IGNORECASE)
        assert compiled.search("secret = abc12345def6")
        assert not compiled.search("my secret project name")
