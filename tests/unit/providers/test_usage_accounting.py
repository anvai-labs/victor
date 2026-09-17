"""Direct contract coverage for the shared numeric usage accountant."""

import pytest

from victor.providers.usage_accounting import (
    accumulate_usage,
    billable_completion_tokens,
    usage_total_tokens,
)


@pytest.mark.parametrize(
    "usage,expected",
    [
        ({}, 0),
        ({"completion_tokens": 10}, 10),
        ({"completion_tokens": 10, "reasoning_tokens": 3, "reasoning_included": False}, 13),
        ({"completion_tokens": 10, "reasoning_tokens": 3, "reasoning_included": True}, 10),
        ({"completion_tokens": 2, "reasoning_tokens": 8, "reasoning_included": True}, 2),
        ({"completion_tokens": 10, "reasoning_tokens": 3}, 10),
        ({"completion_tokens": 2, "reasoning_tokens": 8}, 10),
        ({"completion_tokens": 10, "billable_completion_tokens": 0}, 0),
        ({"completion_tokens": 10, "reasoning_tokens": 50, "billable_completion_tokens": 12}, 12),
    ],
)
def test_billable_completion_preserves_per_call_convention(usage, expected):
    assert billable_completion_tokens(usage) == expected


@pytest.mark.parametrize("reported,expected", [(None, 18), (0, 18), (99, 99)])
def test_totals_preserve_reported_values_and_derive_missing(reported, expected):
    usage = {
        "prompt_tokens": 5,
        "completion_tokens": 10,
        "reasoning_tokens": 3,
        "reasoning_included": False,
    }
    if reported is not None:
        usage["total_tokens"] = reported
    assert usage_total_tokens(usage) == expected


def test_mixed_calls_accumulate_folded_output_without_reapplying_heuristic():
    total = {}
    accumulate_usage(
        total,
        {
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "reasoning_tokens": 3,
            "reasoning_included": False,
            "cache_read_input_tokens": 2,
        },
    )
    accumulate_usage(
        total,
        {
            "prompt_tokens": 7,
            "completion_tokens": 20,
            "reasoning_tokens": 4,
            "reasoning_included": True,
            "cache_creation_input_tokens": 6,
            "total_tokens": 30,
        },
    )
    assert total == {
        "prompt_tokens": 12,
        "completion_tokens": 30,
        "reasoning_tokens": 7,
        "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 6,
        "billable_completion_tokens": 33,
        "total_tokens": 48,
    }
    assert billable_completion_tokens(total) == 33
    assert usage_total_tokens(total) == 48

    combined = {}
    accumulate_usage(combined, total)
    assert combined == total
