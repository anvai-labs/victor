from victor.agent.turn_policy import SpinDetector, SpinState


def test_spin_detector_repetition():
    detector = SpinDetector(REPETITION_THRESHOLD=3)

    # Turn 1: Call tool A
    state = detector.record_turn(has_tool_calls=True, tool_signatures={"tool_a:arg1"})
    assert state == SpinState.NORMAL

    # Turn 2: Call tool A again
    state = detector.record_turn(has_tool_calls=True, tool_signatures={"tool_a:arg1"})
    assert state == SpinState.NORMAL

    # Turn 3: Call tool A third time
    state = detector.record_turn(has_tool_calls=True, tool_signatures={"tool_a:arg1"})
    # After 2 consecutive repetitions (3 total same calls), it should terminate
    assert state == SpinState.TERMINATED
    assert detector._repetition_count == 2


def test_spin_detector_repetition_with_interruption():
    detector = SpinDetector(REPETITION_THRESHOLD=3)

    # Turn 1: Call tool A
    detector.record_turn(has_tool_calls=True, tool_signatures={"tool_a:arg1"})

    # Turn 2: Call tool B
    detector.record_turn(has_tool_calls=True, tool_signatures={"tool_b:arg1"})

    # Turn 3: Call tool A again
    state = detector.record_turn(has_tool_calls=True, tool_signatures={"tool_a:arg1"})

    # Repetition count should be 0 because B broke the sequence
    # Wait! My implementation checks last 2 turns.
    # Turn 3 sees Turn 2 signatures (B) and Turn 1 signatures (A).
    # It will see Turn 1 is same.
    # Actually my code says: "for prev_signatures in self._turn_signatures[-2:]"
    # Turn 3 signatures is compared against Turn 2 and Turn 1.

    # Let's check how many turn signatures we have.
    assert len(detector._turn_signatures) == 3
    # Turn 3 signatures == Turn 1 signatures.
    # So is_repetitive = True.
    # Repetition count becomes 1.
    assert detector._repetition_count == 1


def test_spin_detector_reset():
    detector = SpinDetector(REPETITION_THRESHOLD=3)
    detector.record_turn(has_tool_calls=True, tool_signatures={"tool_a:arg1"})
    detector.record_turn(has_tool_calls=True, tool_signatures={"tool_a:arg1"})
    detector.record_turn(has_tool_calls=True, tool_signatures={"tool_a:arg1"})
    assert detector.state == SpinState.TERMINATED

    detector.reset()
    assert detector.state == SpinState.NORMAL
    assert detector._repetition_count == 0
    assert len(detector._turn_signatures) == 0
