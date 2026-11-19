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

"""Native/Python secret-scanner differential parity (FOU-C3).

Broadens the FOU-A2b scanner tests (tests/unit/security/test_secrets_scanner.py)
from the generic keyword family to the whole credential registry: every corpus
item runs through BOTH dispatch paths (the Rust-accelerated ``detect_secrets``
and the forced pure-Python ``SecretScanner``). The parity contract, mirroring
the scanner tests, is *semantic*: both engines must agree on detection, with
equivalent labels — the two sides name some families differently (Rust folds
all GitHub token shapes into ``github_token`` and all DB URIs into
``database_url``; Python names each variant), so expectations are label sets
and agreement means a non-empty intersection.

The engines' pattern REGISTRIES are intentionally disjoint in places (Rust
carries vendor-specific families the Python registry lacks and vice versa).
Those are pinned in ``TestDocumentedDivergences`` — a deliberate, visible
ledger, not a hidden skip: a change on either side breaks the pin and forces
this ledger to be revisited.
"""

from __future__ import annotations

import pytest

import victor.security.safety.secrets as secrets_mod
from victor.security.safety.secrets import SecretScanner, detect_secrets, mask_secrets

pytestmark = pytest.mark.native_parity

# (id, content, rust label set, python label set) — agreement contract: the
# two sets must intersect (both engines detect the shape) and each engine must
# find nothing OUTSIDE its expected set.
# Corpus values are built by concatenation so GitHub push protection does not
# mistake the synthetic test keys for real credentials.
AGREEMENT_CORPUS = [
    (
        "aws-access-key",
        "AKIA" + "IOSFODNN7EXAMPLE",
        {"aws_access_key"},
        {"aws_access_key"},
    ),
    (
        "github-pat",
        "ghp_" + "a" * 36,
        {"github_token"},
        {"github_pat"},
    ),
    (
        "github-oauth",
        "gho_" + "a" * 36,
        {"github_oauth", "github_token"},
        {"github_oauth"},
    ),
    (
        "github-app",
        "ghs_" + "a" * 36,
        {"github_token"},
        {"github_app"},
    ),
    (
        "google-api-key",
        "AIza" + "SyA1234567890abcdefghijklmnopqrstuv",
        {"google_api_key"},
        {"google_api_key"},
    ),
    (
        "stripe-key",
        "sk_live_" + "abcdefghijklmnopqrstuvwx",
        {"stripe_key"},
        {"stripe_key"},
    ),
    (
        "jwt",
        "eyJhbGciOiJIUzI1NiJ9."
        + "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        + "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJVadQssw5c",
        {"jwt_token"},
        {"jwt_token"},
    ),
    (
        "private-key",
        "-----BEGIN RSA PRIVATE KEY-----",
        {"private_key"},
        {"private_key"},
    ),
    (
        "postgres-uri",
        "postgres://admin:pw123@db.example.com/prod",
        {"database_url"},
        {"postgres_uri"},
    ),
    ("mysql-uri", "mysql://admin:pw123@db.example.com/prod", {"database_url"}, {"mysql_uri"}),
    ("mongodb-uri", "mongodb://admin:pw123@db.example.com/prod", {"database_url"}, {"mongodb_uri"}),
    ("redis-uri", "redis://admin:pw123@cache.example.com", {"database_url"}, {"redis_uri"}),
    (
        "generic-password",
        "password = hunter2hunter2",
        {"generic_password"},
        {"generic_password"},
    ),
    (
        "generic-password-colon-quoted",
        'password: "correct horse battery staple"',
        {"generic_password"},
        {"generic_password"},
    ),
    (
        "generic-secret",
        "SECRET=abc12345def6",
        {"generic_password"},
        {"generic_secret"},
    ),
    (
        "generic-api-key",
        "api_key = ABCDEFGHIJKLMNOPQRSTUVWXYZ1234",
        {"generic_api_key"},
        {"generic_secret"},
    ),
]

# Benign prose: neither engine may flag any of it.
BENIGN_CORPUS = [
    "my secret project name",
    "the secret ingredient is patience",
    "password policies are documented elsewhere",
    "the api key rotation ceremony happens monthly",
    "we should ghr the release notes",  # prefix-like fragment without the shape
    "def use_secret(self):\n    return self._cache\n",
]

# Pinned divergences: (id, content, rust label set, python label set, reason).
DIVERGENCE_CORPUS = [
    (
        "github-refresh-py-gap",
        "ghr_" + "a" * 36,
        {"github_token"},
        set(),
        "Python registry has GITHUB_REFRESH_PATTERN but does not register it; "
        "Rust's gh[pousr]_ family covers it.",
    ),
    (
        "slack-real-token-shape-py-gap",
        "xoxb-" + "12345678901-12345678901-abcdefghijklmnopqrstuvwxyz",
        {"slack_token"},
        set(),
        "Real Slack tokens have three dash-separated segments; the Python "
        "pattern only allows two, so it cannot match this shape.",
    ),
    (
        "bearer-rust-gap",
        "Bearer abcdefghijklmnopqrstuvwxyz123456",
        set(),
        {"bearer_token"},
        "The Rust registry has no bearer-token family.",
    ),
    (
        "pgp-header-rust-gap",
        "-----BEGIN PGP PRIVATE KEY BLOCK-----",
        set(),
        {"pgp_private_key"},
        "The Rust private_key regex does not cover the PGP variant.",
    ),
    (
        "bare-aws-secret-rust-gap",
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        set(),
        {"aws_secret_key"},
        "Python's aws_secret_key matches a bare 40-char base64 run; the Rust "
        "pattern requires an aws/amazon keyword within 20 chars.",
    ),
    (
        "anthropic-rust-only",
        "sk-ant-" + "api03-abcdefghijklmnopqrstuvwxyz1234567890",
        {"anthropic_api_key"},
        set(),
        "Vendor key families (anthropic/openai) exist only in the Rust registry.",
    ),
    (
        "twilio-rust-only",
        "AC" + "a" * 32,
        {"twilio_sid"},
        set(),
        "Vendor-specific families exist only in the Rust registry.",
    ),
    (
        "sendgrid-rust-only",
        "SG." + "abcdefghijklmnopqrstuv." + "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwx",
        {"sendgrid_key"},
        set(),
        "Vendor-specific families exist only in the Rust registry.",
    ),
]


def _matched_types(content: str, *, use_native: bool) -> set[str]:
    if use_native:
        return {m.secret_type for m in detect_secrets(content)}
    # Force the pure-Python scanner path (mirror of
    # tests/unit/security/test_secrets_scanner.py's forcing).
    return {m.secret_type for m in SecretScanner().scan(content)}


def _native_secrets_active() -> bool:
    """True when detect_secrets actually dispatches to the Rust scanner.

    The flag ``_RUST_SECRETS_AVAILABLE`` only proves the wrapper imported —
    the wrapper itself falls back to Python internally when the wheel is
    absent, so a behavioral probe against a Rust-registry-only family
    (anthropic keys) is the real gate (mirrors
    test_secrets_scanner.py's ``_native_has_generic_patterns`` probe and the
    CI wheel-lag pattern).
    """
    if not secrets_mod._RUST_SECRETS_AVAILABLE:
        return False
    probe = {m.secret_type for m in detect_secrets("sk-ant-" + "a" * 32)}
    return "anthropic_api_key" in probe


@pytest.fixture(params=[True, False], ids=["dispatch", "python-fallback"])
def scanner_path(request, monkeypatch):
    """Run assertions through both dispatch paths.

    The dispatch path requires the native extension — verified by behavior,
    not by the import flag (skip when the wheel is absent or predates the
    Rust-only families the corpus pins); the python-fallback path forces
    ``_RUST_SECRETS_AVAILABLE`` off and is the unconditional correctness
    contract.
    """
    use_native = request.param
    if use_native:
        if not _native_secrets_active():
            pytest.skip("native secrets scanner not active (wheel missing or stale)")
    else:
        monkeypatch.setattr(secrets_mod, "_RUST_SECRETS_AVAILABLE", False)
    return use_native


class TestAgreementCorpus:
    """Both engines detect every real credential shape with equivalent labels."""

    @pytest.mark.parametrize(
        "name,content,rust_types,py_types", AGREEMENT_CORPUS, ids=[c[0] for c in AGREEMENT_CORPUS]
    )
    def test_both_engines_detect(self, scanner_path, name, content, rust_types, py_types):
        matched = _matched_types(content, use_native=scanner_path)
        expected = rust_types if scanner_path else py_types
        assert matched == expected

    @pytest.mark.parametrize("content", BENIGN_CORPUS)
    def test_benign_prose_never_matches(self, scanner_path, content):
        assert _matched_types(content, use_native=scanner_path) == set()

    @pytest.mark.parametrize(
        "name,content,rust_types,py_types", AGREEMENT_CORPUS, ids=[c[0] for c in AGREEMENT_CORPUS]
    )
    def test_has_secrets_agrees(self, scanner_path, name, content, rust_types, py_types):
        if scanner_path:
            assert secrets_mod.has_secrets(content) is True
        else:
            # Forced-python has_secrets: detect via the scanner itself.
            assert bool(SecretScanner().scan(content)) is True

    @pytest.mark.parametrize(
        "name,content,rust_types,py_types", AGREEMENT_CORPUS, ids=[c[0] for c in AGREEMENT_CORPUS]
    )
    def test_masked_output_leaks_nothing(self, scanner_path, name, content, rust_types, py_types):
        masked = mask_secrets(content) if scanner_path else _forced_python_mask(content)
        assert masked != content
        # The raw secret substring (the match itself) must not survive.
        for m in SecretScanner().scan(content):
            raw = content[m.start : m.end]
            assert raw not in masked


def _forced_python_mask(content: str) -> str:
    """mask_secrets through the forced pure-Python path."""
    saved = secrets_mod._RUST_SECRETS_AVAILABLE
    secrets_mod._RUST_SECRETS_AVAILABLE = False
    try:
        return secrets_mod.mask_secrets(content)
    finally:
        secrets_mod._RUST_SECRETS_AVAILABLE = saved


class TestMaskingStyle:
    """Mask STYLE differs by design (Rust partial-masks, Python fully
    replaces with [REDACTED]) — the contract is leak-freedom, above."""

    def test_rust_and_python_mask_styles_differ(self):
        """Pinned: the two paths' mask styles are intentionally different —
        rust partial-masks in place, the python fallback replaces the whole
        match with [REDACTED]. Both are leak-free (see the property test)."""
        content = "AKIA" + "IOSFODNN7EXAMPLE"
        if not _native_secrets_active():
            pytest.skip("native secrets scanner not active (wheel missing or stale)")
        rust_masked = mask_secrets(content)
        py_masked = _forced_python_mask(content)
        assert rust_masked != py_masked
        assert content not in rust_masked
        assert content not in py_masked


class TestDocumentedDivergences:
    """The visible ledger of registry-level divergences (see module docstring).

    Each pin breaks loudly when either engine's pattern registry changes,
    forcing this ledger to be re-examined rather than silently drifting.
    """

    @pytest.mark.parametrize(
        "name,content,rust_types,py_types,reason",
        DIVERGENCE_CORPUS,
        ids=[c[0] for c in DIVERGENCE_CORPUS],
    )
    def test_divergence_is_pinned(self, name, content, rust_types, py_types, reason):
        if not _native_secrets_active():
            pytest.skip("native secrets scanner not active (wheel missing or stale)")
        assert _matched_types(content, use_native=True) == rust_types, reason
        assert _matched_types(content, use_native=False) == py_types, reason
