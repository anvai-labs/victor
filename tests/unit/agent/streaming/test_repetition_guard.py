"""Unit tests for the intra-turn streaming repetition detector (P5).

A single degenerate generation ("Let me check the state. " looped hundreds of
times) previously ran until max_tokens: victor only detected repetition BETWEEN
turns. The detector watches the running text of ONE generation.
"""

from __future__ import annotations

import random

from victor.agent.streaming.repetition_guard import IntraTurnRepetitionDetector


def make_detector(**overrides) -> IntraTurnRepetitionDetector:
    defaults = {
        "window_chars": 4000,
        "min_segment_chars": 24,
        "max_repeats": 6,
        "check_every_chars": 512,
    }
    defaults.update(overrides)
    return IntraTurnRepetitionDetector(**defaults)


def feed_all(detector: IntraTurnRepetitionDetector, text: str, chunk_size: int = 37):
    """Feed text in odd-sized chunks (mirrors arbitrary provider chunking)."""
    for i in range(0, len(text), chunk_size):
        repeated = detector.feed(text[i : i + chunk_size])
        if repeated is not None:
            return repeated
    return None


class TestTrigger:
    def test_looping_sentence_triggers(self):
        # The live failure shape: alternating short sentences repeated forever.
        loop = "Let me check the state. Let me check the remote tracking state. "
        repeated = feed_all(make_detector(), loop * 40)
        assert repeated is not None
        assert "let me check the" in repeated.lower()

    def test_single_repeated_line_triggers(self):
        line = "Checking the remote branch tracking status now...\n"
        assert feed_all(make_detector(), line * 30) is not None

    def test_varied_prose_does_not_trigger(self):
        text = " ".join(
            f"Sentence number {i} discusses a different topic entirely, item {i * 7}."
            for i in range(120)
        )
        assert feed_all(make_detector(), text) is None

    def test_repeated_short_code_lines_do_not_trigger(self):
        # Closing braces / short boilerplate lines repeat legitimately in code.
        code = ("    }\n" * 50) + ("    return x\n" * 10) + ("})\n" * 40)
        assert feed_all(make_detector(), code) is None

    def test_below_repeat_threshold_does_not_trigger(self):
        sentence = "This exact sentence appears a handful of times in the output. "
        varied = " ".join(f"Filler sentence {i} with distinct content here." for i in range(60))
        assert feed_all(make_detector(), sentence * 4 + varied) is None

    def test_long_period_block_loop_triggers(self):
        # A multi-sentence paragraph (~300 chars) looped — caught by the block rule
        # even though each inner sentence stays under max_repeats per window.
        para = (
            "First we examine the configuration files for the project setup. "
            "Then we validate every dependency against the recorded lockfile. "
            "Next we compile the sources and collect all emitted diagnostics. "
            "Finally we execute the verification suite and gather the results. "
        )
        assert feed_all(make_detector(), para * 12) is not None


class TestStaccatoLoop:
    """Rule 3: a loop built from phrases SHORTER than ``min_segment_chars``.

    The observed failure: the model narrated "Calling now. Executing. Stop.
    Call. Now. I'll make the call." for thousands of characters instead of
    emitting the tool call. Rule 1 cannot see it (every phrase is under the
    24-char floor) and rule 2 cannot either (the shuffled order means no
    256-char block ever recurs verbatim), so the stream ran to exhaustion.
    """

    PHRASES = [
        "Calling now.",
        "Executing.",
        "Stop.",
        "Call.",
        "Now.",
        "I'll make the call.",
        "Here is the call.",
        "Doing it.",
        "Proceeding.",
        "Done narrating.",
        "Tool call:",
        "I am calling.",
        "Here goes.",
        "Making it now.",
    ]

    def _shuffled_loop(self, count: int = 1200) -> str:
        # Seeded => deterministic, but NOT periodic: a modular index cycles, which
        # makes long blocks recur verbatim and hands the catch back to rule 2.
        rnd = random.Random(7)
        return " ".join(rnd.choice(self.PHRASES) for _ in range(count))

    def test_shuffled_short_phrase_loop_triggers(self):
        loop = self._shuffled_loop()
        assert feed_all(make_detector(), loop) is not None

    def test_shuffled_loop_is_invisible_to_the_other_rules(self):
        # Guards the premise: if this ever starts tripping rule 1 or 2, the
        # diversity rule is no longer the thing under test.
        loop = self._shuffled_loop()
        assert all(len(p) < 24 for p in self.PHRASES)  # rule 1 floor
        window = loop[-4000:]
        assert window.count(window[-256:]) == 1  # rule 2 block never recurs

    def test_stops_within_a_couple_of_kilobytes(self):
        # The whole point is bounding the waste; 200k chars used to stream.
        detector = make_detector()
        loop = self._shuffled_loop()
        consumed = 0
        for i in range(0, len(loop), 40):
            consumed += len(loop[i : i + 40])
            if detector.feed(loop[i : i + 40]) is not None:
                break
        assert consumed <= 2048, f"took {consumed} chars to notice the loop"

    def test_varied_short_lines_do_not_trigger(self):
        # Terse but legitimate output: many short lines, all distinct.
        text = "\n".join(f"- step {i}: {'ab' * (i % 9 + 3)} done" for i in range(120))
        assert feed_all(make_detector(), text) is None

    def test_short_output_below_sample_floor_does_not_trigger(self):
        # Too few segments to judge diversity — must not fire on a terse reply.
        assert feed_all(make_detector(), "Done. Ok. Done. Ok. Done. Ok. Done. Ok. ") is None


class TestWindowing:
    def test_repeats_far_apart_outside_window_do_not_trigger(self):
        sentence = "A rare marker sentence that shows up occasionally in output. "
        filler = " ".join(f"Unique padding sentence number {i} for spacing." for i in range(30))
        text = (sentence + filler + " ") * 8  # repeats spaced ~1.5k chars apart, window 4k
        detector = make_detector(window_chars=1000)
        assert feed_all(detector, text) is None


class TestTruncation:
    def test_truncation_point_keeps_first_occurrence(self):
        detector = make_detector()
        prefix = "Here is a legitimate answer about the topic at hand. "
        loop = "Let me check the remote tracking state of the branch. "
        text = prefix + loop * 20
        repeated = feed_all(detector, text)
        assert repeated is not None
        cut = detector.truncation_point(text)
        truncated = text[:cut]
        assert prefix.strip() in truncated
        # keeps at most a couple of instances of the loop, not twenty
        assert truncated.lower().count("remote tracking state") <= 2

    def test_truncation_point_defaults_to_full_length_when_not_found(self):
        detector = make_detector()
        assert detector.truncation_point("short unrelated text") == len("short unrelated text")
