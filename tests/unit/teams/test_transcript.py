from dataclasses import FrozenInstanceError

import pytest

from victor.teams.transcript import TeamTranscript


def test_roundtrip_and_immutable_snapshot():
    transcript = TeamTranscript(10)
    transcript.append("a", "one", "b")
    snapshot = transcript.snapshot()
    transcript.append("b", "two")
    assert len(snapshot) == 1
    with pytest.raises(FrozenInstanceError):
        snapshot[0].content = "mutate"
    assert TeamTranscript.from_list(transcript.to_list()).to_list() == transcript.to_list()
    with pytest.raises(ValueError, match="budget"):
        transcript.append("a", "too long")


@pytest.mark.parametrize(
    "records", [[{}], [{"sequence": 1, "member_id": "a", "content": "x", "handoff_to": None}]]
)
def test_invalid_serialized_transcript(records):
    with pytest.raises(ValueError):
        TeamTranscript.from_list(records)


@pytest.mark.parametrize(
    "member,content,target", [(None, "x", None), ("a", "", None), ("a", "x", False)]
)
def test_invalid_append(member, content, target):
    with pytest.raises(ValueError):
        TeamTranscript().append(member, content, target)
