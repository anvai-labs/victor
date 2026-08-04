# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the prompt-promotion codegen core.

Promotion writes a source-of-truth file (prompt_section_texts.py), so the
re-templatizing and section-replacement logic must be correct and must refuse
rather than silently corrupt.
"""

from victor.core.completion_markers import FILE_DONE_MARKER, TASK_DONE_MARKER
from victor.framework.rl.prompt_promotion import (
    build_promoted_source,
    is_fstring_section,
    marker_placeholders,
    replace_section,
    retemplatize,
)

# A minimal stand-in for prompt_section_texts.py: one f-string section (markers)
# and one plain section.
SOURCE = (
    'COMPLETION_GUIDANCE = f"""\n'
    "Signal {FILE_DONE_MARKER} when a file is done and {TASK_DONE_MARKER} when finished.\n"
    '""".strip()\n'
    "\n"
    'OTHER = """\n'
    "Plain guidance.\n"
    '""".strip()\n'
)


class TestRetemplatize:
    def test_round_trips_rendered_markers(self):
        rendered = f"Signal {FILE_DONE_MARKER} then {TASK_DONE_MARKER}."
        templatized, error = retemplatize(rendered)
        assert error is None
        assert "{FILE_DONE_MARKER}" in templatized and "{TASK_DONE_MARKER}" in templatized
        # And it renders back to exactly the input.
        assert (
            templatized.format(**{name: val for val, name in marker_placeholders().items()})
            == rendered
        )

    def test_refuses_when_no_markers(self):
        _, error = retemplatize("Guidance with no markers at all.")
        assert error and "none of the completion markers" in error

    def test_refuses_on_literal_brace(self):
        rendered = f'Signal {FILE_DONE_MARKER}, emit {{"status": "done"}}.'
        _, error = retemplatize(rendered)
        assert error and "literal" in error

    def test_marker_placeholders_longest_first(self):
        values = list(marker_placeholders())
        assert values == sorted(values, key=len, reverse=True)


class TestReplaceSection:
    def test_swaps_body_and_preserves_fstring_prefix(self):
        out = replace_section(SOURCE, "COMPLETION_GUIDANCE", "{FILE_DONE_MARKER} new", True)
        assert out is not None
        assert 'COMPLETION_GUIDANCE = f"""\n{FILE_DONE_MARKER} new\n""".strip()' in out
        # OTHER is untouched.
        assert 'OTHER = """\nPlain guidance.\n""".strip()' in out

    def test_plain_section_has_no_prefix(self):
        out = replace_section(SOURCE, "OTHER", "new body", False)
        assert out is not None
        assert 'OTHER = """\nnew body\n""".strip()' in out

    def test_missing_section_returns_none(self):
        assert replace_section(SOURCE, "NOPE", "x", False) is None

    def test_is_fstring_section(self):
        assert is_fstring_section(SOURCE, "COMPLETION_GUIDANCE") is True
        assert is_fstring_section(SOURCE, "OTHER") is False


class TestBuildPromotedSource:
    def test_promotes_fstring_section_with_provenance(self):
        candidate = f"Signal {FILE_DONE_MARKER} when done and {TASK_DONE_MARKER} when finished."
        updated, error = build_promoted_source(
            SOURCE, "COMPLETION_GUIDANCE", candidate, "# provenance line"
        )
        assert error is None and updated is not None
        # Markers are re-templatized, not hardcoded.
        assert "{FILE_DONE_MARKER}" in updated
        assert FILE_DONE_MARKER not in updated.split("COMPLETION_GUIDANCE")[1]
        # Provenance sits directly above the assignment.
        assert "# provenance line\nCOMPLETION_GUIDANCE = " in updated

    def test_promotes_plain_section_verbatim(self):
        updated, error = build_promoted_source(SOURCE, "OTHER", "fresh guidance", "# prov")
        assert error is None and updated is not None
        assert '# prov\nOTHER = """\nfresh guidance\n""".strip()' in updated

    def test_refuses_fstring_candidate_without_markers(self):
        # An f-string section whose candidate lost its markers would bake in a
        # marker-free prompt; retemplatize refuses.
        updated, error = build_promoted_source(
            SOURCE, "COMPLETION_GUIDANCE", "no markers here", "# prov"
        )
        assert updated is None and error and "none of the completion markers" in error

    def test_missing_section_is_an_error(self):
        updated, error = build_promoted_source(SOURCE, "NOPE", "x", "# prov")
        assert updated is None and error and "could not locate" in error
