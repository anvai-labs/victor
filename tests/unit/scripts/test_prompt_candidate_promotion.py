# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Promoting a candidate must not quietly hardcode the completion markers.

``COMPLETION_GUIDANCE`` is the only f-string in the section module and the
section evolution targets most. A candidate's stored text is the *rendered*
output, so writing it back verbatim would bake ``VICTOR_FILE_DONE::`` into the
source and end ``completion_markers.py``'s role as the single definition of
those tokens — after which renaming a marker would change the detector and leave
the prompt still telling the model to emit the old one.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

from victor.core.completion_markers import (
    BLOCKED_MARKER,
    FILE_DONE_MARKER,
    SUMMARY_MARKER,
    TASK_DONE_MARKER,
)

_MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "prompt_candidates.py"
_spec = importlib.util.spec_from_file_location("prompt_candidates", _MODULE_PATH)
pc = importlib.util.module_from_spec(_spec)
# Register before executing: the module defines a dataclass, and dataclasses
# resolves annotations through sys.modules[cls.__module__].
sys.modules["prompt_candidates"] = pc
_spec.loader.exec_module(pc)


RENDERED = f"""TASK COMPLETION (MANDATORY):
1. {FILE_DONE_MARKER} Created/Modified <filename>
2. {TASK_DONE_MARKER} <what was fixed>
3. {SUMMARY_MARKER} <key findings>
4. {BLOCKED_MARKER} <reason>
- Signal completion ONCE."""


class TestRetemplatizing:
    def test_rendered_markers_become_interpolations_again(self):
        templatized, error = pc._retemplatize(RENDERED)

        assert error is None
        assert "{FILE_DONE_MARKER}" in templatized
        assert "{BLOCKED_MARKER}" in templatized
        assert FILE_DONE_MARKER not in templatized, "the literal value must not survive"

    def test_it_round_trips_exactly(self):
        """A lossy substitution would only surface at runtime, as a prompt the
        detector no longer matches."""
        templatized, error = pc._retemplatize(RENDERED)
        assert error is None

        rendered_again = templatized.format(
            FILE_DONE_MARKER=FILE_DONE_MARKER,
            TASK_DONE_MARKER=TASK_DONE_MARKER,
            SUMMARY_MARKER=SUMMARY_MARKER,
            BLOCKED_MARKER=BLOCKED_MARKER,
        )
        assert rendered_again == RENDERED

    def test_a_literal_brace_is_refused_rather_than_mangled(self):
        """An f-string would read it as an interpolation site."""
        _, error = pc._retemplatize(RENDERED + '\n- Emit {"status": "done"}.')
        assert error is not None
        assert "literal" in error

    def test_text_without_markers_is_refused(self):
        """If no marker survived the rewrite, the section lost its whole point."""
        _, error = pc._retemplatize("Some guidance with no markers at all.")
        assert error is not None
        assert "none of the completion markers" in error

    def test_longer_markers_are_substituted_first(self):
        """A marker that prefixes another must not corrupt it."""
        placeholders = list(pc._marker_placeholders())
        assert placeholders == sorted(placeholders, key=len, reverse=True)


class TestSectionRewriting:
    SOURCE = (
        'OTHER = """\nkeep me\n""".strip()\n\n'
        'COMPLETION_GUIDANCE = f"""\nold body\n""".strip()\n\n'
        "TRAILING = 1\n"
    )

    def test_it_replaces_only_the_named_section(self):
        out = pc._replace_section(self.SOURCE, "COMPLETION_GUIDANCE", "new body", True)

        assert "new body" in out
        assert "old body" not in out
        assert 'OTHER = """\nkeep me\n""".strip()' in out
        assert "TRAILING = 1" in out

    def test_the_f_prefix_is_preserved(self):
        out = pc._replace_section(self.SOURCE, "COMPLETION_GUIDANCE", "{FILE_DONE_MARKER}", True)
        assert 'COMPLETION_GUIDANCE = f"""' in out

    def test_a_plain_section_stays_plain(self):
        """Adding an f prefix to a section with braces would break the import."""
        out = pc._replace_section(self.SOURCE, "OTHER", "new", False)
        assert 'OTHER = """' in out
        assert 'OTHER = f"""' not in out

    def test_an_unknown_section_returns_none_rather_than_guessing(self):
        assert pc._replace_section(self.SOURCE, "NOT_A_SECTION", "x", False) is None

    def test_detects_which_sections_interpolate(self):
        assert pc._is_fstring_section(self.SOURCE, "COMPLETION_GUIDANCE") is True
        assert pc._is_fstring_section(self.SOURCE, "OTHER") is False


class TestTheShippedModuleSurvivesARoundTrip:
    """Rewriting the real section with its own text must be a no-op."""

    def test_promoting_the_shipped_text_reproduces_the_file(self):
        source = pc.SECTION_TEXTS_PATH.read_text()
        from victor.agent.prompt_section_texts import COMPLETION_GUIDANCE as shipped

        templatized, error = pc._retemplatize(shipped)
        assert error is None

        rewritten = pc._replace_section(source, "COMPLETION_GUIDANCE", templatized, True)
        assert rewritten is not None

        namespace: dict = {}
        exec(compile(rewritten, "rewritten", "exec"), namespace)
        assert namespace["COMPLETION_GUIDANCE"] == shipped

    def test_the_rewritten_module_still_imports(self):
        """A broken f-string here would take the whole agent down at import."""
        source = pc.SECTION_TEXTS_PATH.read_text()
        from victor.agent.prompt_section_texts import COMPLETION_GUIDANCE as shipped

        templatized, _ = pc._retemplatize(shipped)
        rewritten = pc._replace_section(source, "COMPLETION_GUIDANCE", templatized, True)

        compile(rewritten, "rewritten", "exec")  # raises SyntaxError if malformed


class TestPromotionRefusesUnprovenCandidates:
    def test_a_hold_candidate_is_refused_without_force(self, capsys, monkeypatch):
        cand = pc.Candidate(
            section_name="COMPLETION_GUIDANCE",
            provider="zai",
            text_hash="abc123def456",
            parent_hash="",
            text=RENDERED,
            generation=9,
            sample_count=51,
            is_active=0,
            requires_benchmark=1,
            benchmark_passed=0,
            benchmark_runs=0,
            benchmark_score=0.0,
            strategy_chain="gepa",
            created_at="2026-07-26",
        )
        monkeypatch.setattr(pc, "_load", lambda db: [cand])
        args = argparse.Namespace(db=Path("unused.db"), hash="abc123", apply=False, force=False)

        assert pc.cmd_promote(args) == 1
        assert "Refusing to promote a HOLD candidate" in capsys.readouterr().err
