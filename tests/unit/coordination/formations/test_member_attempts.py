"""Boundary guards for the shared reducer; formation suites own accounting behavior."""

import pytest

from victor.coordination.formations.member_attempts import aggregate_attempts
from victor.teams.types import MemberResult


@pytest.mark.parametrize("members", [[], ["first", "second"]])
def test_attempt_aggregation_rejects_empty_or_mixed_member_histories(members):
    with pytest.raises(ValueError, match="exactly one nonempty member history"):
        aggregate_attempts([MemberResult(name, True, "ready") for name in members], "consensus")
