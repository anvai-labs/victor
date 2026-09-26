"""Bounded, ordered team conversation records (FEP-0035)."""

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class PeerHandoff:
    source_member_id: str
    target_member_id: str
    sequence: int


@dataclass(frozen=True)
class TranscriptEntry:
    sequence: int
    member_id: str
    content: str
    handoff_to: Optional[str] = None


class TeamTranscript:
    """One append authority; selectors receive immutable snapshots."""

    def __init__(self, max_chars: int = 32768):
        if type(max_chars) is not int or max_chars < 1:
            raise ValueError("transcript_max_chars must be a positive integer")
        self.max_chars = max_chars
        self._entries: list[TranscriptEntry] = []
        self._chars = 0

    def append(
        self, member_id: str, content: str, handoff_to: Optional[str] = None
    ) -> TranscriptEntry:
        if not isinstance(member_id, str) or not member_id.strip():
            raise ValueError("Transcript member_id must be a nonempty string")
        if handoff_to is not None and (not isinstance(handoff_to, str) or not handoff_to.strip()):
            raise ValueError("Transcript handoff_to must be a nonempty string or null")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Transcript content must be a nonempty string")
        if self._chars + len(content) > self.max_chars:
            raise ValueError("Transcript character budget exceeded; use artifact references")
        entry = TranscriptEntry(len(self._entries), member_id, content, handoff_to)
        self._entries.append(entry)
        self._chars += len(content)
        return entry

    def snapshot(self) -> tuple[TranscriptEntry, ...]:
        return tuple(self._entries)

    def to_list(self) -> list[dict]:
        return [asdict(entry) for entry in self._entries]

    @classmethod
    def from_list(cls, records: list[dict], max_chars: int = 32768) -> "TeamTranscript":
        transcript = cls(max_chars)
        for record in records:
            if set(record) != {"sequence", "member_id", "content", "handoff_to"}:
                raise ValueError("Invalid transcript entry contract")
            if type(record["sequence"]) is not int or record["sequence"] != len(
                transcript._entries
            ):
                raise ValueError("Transcript sequence must be contiguous")
            transcript.append(record["member_id"], record["content"], record["handoff_to"])
        return transcript
