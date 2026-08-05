# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Guard: the generated feature-flag doc must match the ``FeatureFlag`` enum.

Facts restated in prose drift silently — ``USE_SMART_ROUTING`` was documented
"default OFF" in three places (TD-17, CLAUDE.md, flag-graduation-policy) while the
code defaulted it ON (F-016 class). This guard makes the flag inventory *generated,
not asserted*: ``docs/architecture/feature-flags.md`` is rendered from
``victor.core.feature_flags.get_flag_manifest()`` and this test fails if the committed
doc drifts from the enum (a flag added/removed, or an ``is_opt_in_by_default()``
change). Regenerate with ``python scripts/gen_feature_flag_doc.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from victor.core.feature_flags import FeatureFlag, get_flag_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC_PATH = REPO_ROOT / "docs" / "architecture" / "feature-flags.md"
GEN_SCRIPT = REPO_ROOT / "scripts" / "gen_feature_flag_doc.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_feature_flag_doc", GEN_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_flag_doc_matches_code() -> None:
    """The committed doc must be byte-identical to the code-derived render."""
    generator = _load_generator()
    expected = generator.render_feature_flag_markdown(get_flag_manifest())

    assert DOC_PATH.exists(), (
        f"{DOC_PATH.relative_to(REPO_ROOT)} is missing — run "
        "`python scripts/gen_feature_flag_doc.py`"
    )
    actual = DOC_PATH.read_text(encoding="utf-8")
    assert actual == expected, (
        f"{DOC_PATH.relative_to(REPO_ROOT)} is stale (flag added/removed or opt-in "
        "status changed). Regenerate with `python scripts/gen_feature_flag_doc.py`."
    )


def test_manifest_covers_every_flag_once() -> None:
    """Sanity: the manifest is a complete, 1:1 view of the enum."""
    manifest = get_flag_manifest()
    names = [entry["name"] for entry in manifest]
    assert sorted(names) == names, "manifest must be sorted by name"
    assert set(names) == {flag.value for flag in FeatureFlag}
    assert len(names) == len(set(names)) == len(list(FeatureFlag))


def test_default_is_off_iff_opt_in() -> None:
    """The invariant the doc encodes: default OFF ⇔ opt-in."""
    for entry in get_flag_manifest():
        assert entry["default"] == (not entry["opt_in"]), entry["name"]
