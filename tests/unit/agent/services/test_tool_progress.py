"""LOOP-04: the shared "every tool call was rejected" rule."""

from __future__ import annotations

import pytest

from victor.agent.services.tool_progress import REJECTED_OUTCOME_KINDS, batch_all_rejected


def _skip(kind: str, *, success: bool = False) -> dict:
    return {"tool_name": "read", "skipped": True, "success": success, "outcome_kind": kind}


@pytest.mark.parametrize("kind", sorted(REJECTED_OUTCOME_KINDS))
def test_every_rejected_kind_counts(kind):
    assert batch_all_rejected([_skip(kind), _skip(kind)]) is True


@pytest.mark.parametrize(
    "results",
    [
        [],
        None,
        # An executed call (success or failure) is progress, so the batch is not rejected.
        [_skip("permission_denied"), {"tool_name": "read", "success": True}],
        [_skip("permission_denied"), {"tool_name": "read", "success": False, "error": "x"}],
        # Informative redirects hand back content.
        [_skip("duplicate_read")],
        [_skip("redundant_call", success=True)],
        [_skip("broad_code_read")],
        [_skip("result_withheld")],
        # A rejected kind that nevertheless reports success, or is not a skip.
        [_skip("permission_denied", success=True)],
        [{"tool_name": "read", "success": False, "outcome_kind": "permission_denied"}],
        # Unknown kinds and malformed entries never count.
        [_skip("something_new")],
        [{"tool_name": "read", "skipped": True, "success": False}],
        ["not a dict"],
    ],
)
def test_batches_with_progress_or_information_do_not_count(results):
    assert batch_all_rejected(results) is False


def test_accepts_a_generator():
    assert batch_all_rejected(_skip("tool_unavailable") for _ in range(2)) is True
