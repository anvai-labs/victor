"""Tests for victor.agent.turn_policy module.

Shared turn policy used by both batch (AgenticLoop) and streaming
(StreamingChatExecutor) execution paths.
"""

from __future__ import annotations

import pytest

from victor.agent.action_authorizer import ActionIntent

from victor.agent.turn_policy import (
    FulfillmentCriteria,
    FulfillmentCriteriaBuilder,
    NudgeDecision,
    NudgePolicy,
    NudgeType,
    SpinDetector,
    SpinState,
    MAX_ALL_BLOCKED,
    MAX_NO_TOOL_TURNS,
    NUDGE_THRESHOLD,
    READ_ONLY_ESCALATION_THRESHOLD,
)

# ============================================================================
# SpinDetector tests
# ============================================================================


class TestSpinDetector:
    """Tests for SpinDetector."""

    def test_initial_state_is_normal(self):
        detector = SpinDetector()
        assert detector.state == SpinState.NORMAL

    def test_tool_calls_keep_normal(self):
        detector = SpinDetector()
        state = detector.record_turn(has_tool_calls=True, tool_count=2)
        assert state == SpinState.NORMAL
        assert detector.total_tool_calls == 2

    def test_no_tools_reaches_warning(self):
        detector = SpinDetector()
        detector.record_turn(has_tool_calls=False)
        state = detector.record_turn(has_tool_calls=False)
        assert state == SpinState.WARNING
        assert detector.consecutive_no_tool_turns == NUDGE_THRESHOLD

    def test_no_tools_reaches_terminated(self):
        detector = SpinDetector()
        for _ in range(MAX_NO_TOOL_TURNS):
            detector.record_turn(has_tool_calls=False)
        assert detector.state == SpinState.TERMINATED

    def test_all_blocked_reaches_blocked(self):
        detector = SpinDetector()
        detector.record_turn(has_tool_calls=True, all_blocked=True)
        state = detector.record_turn(has_tool_calls=True, all_blocked=True)
        assert state == SpinState.BLOCKED

    def test_all_blocked_three_turns_remains_blocked(self):
        detector = SpinDetector()
        for _ in range(3):
            detector.record_turn(has_tool_calls=True, all_blocked=True)
        assert detector.state == SpinState.BLOCKED

    def test_all_blocked_reaches_terminated(self):
        detector = SpinDetector()
        for _ in range(MAX_ALL_BLOCKED):
            detector.record_turn(has_tool_calls=True, all_blocked=True)
        assert detector.state == SpinState.TERMINATED

    def test_successful_tool_resets_no_tool_counter(self):
        detector = SpinDetector()
        detector.record_turn(has_tool_calls=False)
        detector.record_turn(has_tool_calls=False)
        detector.record_turn(has_tool_calls=True, tool_count=1)
        assert detector.consecutive_no_tool_turns == 0
        assert detector.state == SpinState.NORMAL

    def test_successful_tool_resets_blocked_counter(self):
        detector = SpinDetector()
        detector.record_turn(has_tool_calls=True, all_blocked=True)
        detector.record_turn(has_tool_calls=True, all_blocked=False, tool_count=1)
        assert detector.consecutive_all_blocked == 0

    def test_read_only_tracking(self):
        detector = SpinDetector()
        for _ in range(READ_ONLY_ESCALATION_THRESHOLD):
            detector.record_turn(
                has_tool_calls=True,
                tool_names={"read", "grep"},
                tool_count=2,
            )
        assert detector.consecutive_read_only_turns == READ_ONLY_ESCALATION_THRESHOLD
        assert detector.has_used_code_search is False

    def test_read_only_tracking_accepts_alias_names(self):
        detector = SpinDetector()
        detector.record_turn(
            has_tool_calls=True,
            tool_names={"read_file", "list_directory"},
            tool_count=2,
        )
        assert detector.consecutive_read_only_turns == 1

    def test_code_search_breaks_read_only(self):
        detector = SpinDetector()
        detector.record_turn(has_tool_calls=True, tool_names={"read"}, tool_count=1)
        detector.record_turn(has_tool_calls=True, tool_names={"code_search"}, tool_count=1)
        assert detector.has_used_code_search is True

    def test_non_read_only_resets_counter(self):
        detector = SpinDetector()
        detector.record_turn(has_tool_calls=True, tool_names={"read"}, tool_count=1)
        detector.record_turn(has_tool_calls=True, tool_names={"edit"}, tool_count=1)
        assert detector.consecutive_read_only_turns == 0

    def test_reset(self):
        detector = SpinDetector()
        detector.record_turn(has_tool_calls=False)
        detector.record_turn(has_tool_calls=True, all_blocked=True, tool_count=3)
        detector.reset()
        assert detector.state == SpinState.NORMAL
        assert detector.total_tool_calls == 0
        assert detector.consecutive_no_tool_turns == 0
        assert detector.consecutive_all_blocked == 0


# ============================================================================
# NudgePolicy tests
# ============================================================================


class TestNudgePolicy:
    """Tests for NudgePolicy."""

    def test_no_nudge_on_normal(self):
        policy = NudgePolicy()
        detector = SpinDetector()
        decision = policy.evaluate(detector)
        assert decision.nudge_type == NudgeType.NONE
        assert decision.should_inject is False

    def test_use_tools_nudge_on_warning(self):
        policy = NudgePolicy()
        detector = SpinDetector()
        # First no-tool turn (below threshold) - no nudge
        detector.record_turn(has_tool_calls=False)
        decision = policy.evaluate(detector)
        assert decision.nudge_type == NudgeType.NONE
        assert decision.should_inject is False

        # Second no-tool turn (at threshold) - gets REPETITION_BREAK nudge
        detector.record_turn(has_tool_calls=False)
        decision = policy.evaluate(detector)
        assert decision.nudge_type == NudgeType.REPETITION_BREAK
        assert decision.should_inject is True
        assert "MUST" in decision.message
        assert "repetition loop" in decision.message
        assert decision.role == "user"

    def test_different_tools_nudge_on_blocked(self):
        policy = NudgePolicy()
        detector = SpinDetector()
        detector.record_turn(has_tool_calls=True, all_blocked=True)
        detector.record_turn(has_tool_calls=True, all_blocked=True)
        decision = policy.evaluate(detector)
        assert decision.nudge_type == NudgeType.DIFFERENT_TOOLS
        assert decision.should_inject is True
        assert "blocked" in decision.message.lower()
        assert decision.role == "user"

    def test_code_search_nudge_on_read_only(self):
        policy = NudgePolicy()
        detector = SpinDetector()
        for _ in range(READ_ONLY_ESCALATION_THRESHOLD):
            detector.record_turn(has_tool_calls=True, tool_names={"read"}, tool_count=1)
        decision = policy.evaluate(detector)
        assert decision.nudge_type == NudgeType.CODE_SEARCH
        assert "code_search" in decision.message

    def test_no_code_search_nudge_after_use(self):
        policy = NudgePolicy()
        detector = SpinDetector()
        detector.record_turn(has_tool_calls=True, tool_names={"code_search"}, tool_count=1)
        for _ in range(READ_ONLY_ESCALATION_THRESHOLD):
            detector.record_turn(has_tool_calls=True, tool_names={"read"}, tool_count=1)
        decision = policy.evaluate(detector)
        # Should NOT nudge for code_search since it was already used
        assert decision.nudge_type != NudgeType.CODE_SEARCH

    def test_no_budget_warning_before_halfway(self):
        policy = NudgePolicy()
        decision = policy.budget_warning(iteration=3, max_iterations=10)
        assert decision.should_inject is False


# ============================================================================
# FulfillmentCriteriaBuilder tests
# ============================================================================


class TestFulfillmentCriteriaBuilder:
    """Tests for FulfillmentCriteriaBuilder."""

    def test_records_written_files(self):
        builder = FulfillmentCriteriaBuilder()
        builder.record_tool_result(
            {
                "tool_name": "write",
                "args": {"file_path": "/app/main.py"},
                "success": True,
            }
        )
        criteria = builder.build()
        assert "/app/main.py" in criteria.file_paths

    def test_records_edited_files(self):
        builder = FulfillmentCriteriaBuilder()
        builder.record_tool_result(
            {
                "tool_name": "edit",
                "args": {"file_path": "/app/utils.py"},
                "success": True,
            }
        )
        criteria = builder.build()
        assert "/app/utils.py" in criteria.file_paths

    def test_records_test_files_from_shell(self):
        builder = FulfillmentCriteriaBuilder()
        builder.record_tool_result(
            {
                "tool_name": "shell",
                "args": {"command": "pytest tests/test_main.py -v"},
                "success": True,
            }
        )
        criteria = builder.build()
        assert "tests/test_main.py" in criteria.test_files

    def test_records_test_files_from_shell_cmd(self):
        builder = FulfillmentCriteriaBuilder()
        builder.record_tool_result(
            {
                "tool_name": "shell",
                "args": {"cmd": "pytest tests/test_other.py -q"},
                "success": True,
            }
        )
        criteria = builder.build()
        assert "tests/test_other.py" in criteria.test_files

    def test_records_doc_files(self):
        builder = FulfillmentCriteriaBuilder()
        builder.record_tool_result(
            {
                "tool_name": "write",
                "args": {"file_path": "/app/README.md"},
                "success": True,
            }
        )
        criteria = builder.build()
        assert "/app/README.md" in criteria.doc_files

    def test_records_errors(self):
        builder = FulfillmentCriteriaBuilder()
        builder.record_tool_result(
            {
                "tool_name": "edit",
                "args": {"file_path": "/app/main.py"},
                "success": False,
                "error": "File not found",
            }
        )
        criteria = builder.build()
        assert criteria.original_error == "File not found"
        assert len(criteria.file_paths) == 0  # Failed edits not tracked

    def test_to_dict(self):
        criteria = FulfillmentCriteria(
            file_paths=["/app/main.py"],
            test_files=["tests/test_main.py"],
            original_error="KeyError",
        )
        d = criteria.to_dict()
        assert d["file_path"] == "/app/main.py"
        assert d["test_files"] == ["tests/test_main.py"]
        assert d["original_error"] == "KeyError"

    def test_reset(self):
        builder = FulfillmentCriteriaBuilder()
        builder.record_tool_result(
            {
                "tool_name": "write",
                "args": {"file_path": "/x.py"},
                "success": True,
            }
        )
        builder.reset()
        criteria = builder.build()
        assert len(criteria.file_paths) == 0

    def test_ignores_failed_results(self):
        builder = FulfillmentCriteriaBuilder()
        builder.record_tool_result(
            {
                "tool_name": "write",
                "args": {"file_path": "/fail.py"},
                "success": False,
                "error": "Permission denied",
            }
        )
        criteria = builder.build()
        assert len(criteria.file_paths) == 0


# ============================================================================
# Consistency tests (batch vs streaming should use same logic)
# ============================================================================


class TestBatchStreamingConsistency:
    """Verify batch and streaming paths produce consistent decisions."""

    def test_same_spin_detection_constants(self):
        """Both paths use same thresholds from turn_policy."""
        assert MAX_NO_TOOL_TURNS == 3
        assert MAX_ALL_BLOCKED == 4
        assert NUDGE_THRESHOLD == 2

    def test_spin_detector_deterministic(self):
        """Same input sequence produces same state."""
        d1 = SpinDetector()
        d2 = SpinDetector()

        # Simulate identical turn sequence
        sequence = [
            {"has_tool_calls": True, "all_blocked": False, "tool_count": 2},
            {"has_tool_calls": True, "all_blocked": True, "tool_count": 1},
            {"has_tool_calls": False},
        ]

        for turn in sequence:
            d1.record_turn(**turn)
            d2.record_turn(**turn)

        assert d1.state == d2.state
        assert d1.consecutive_no_tool_turns == d2.consecutive_no_tool_turns
        assert d1.consecutive_all_blocked == d2.consecutive_all_blocked
        assert d1.total_tool_calls == d2.total_tool_calls

    def test_nudge_policy_deterministic(self):
        """Same detector state produces same nudge decision."""
        policy = NudgePolicy()

        d1 = SpinDetector()
        d2 = SpinDetector()

        for _ in range(NUDGE_THRESHOLD):
            d1.record_turn(has_tool_calls=False)
            d2.record_turn(has_tool_calls=False)

        n1 = policy.evaluate(d1)
        n2 = policy.evaluate(d2)

        assert n1.nudge_type == n2.nudge_type
        assert n1.message == n2.message
        assert n1.role == n2.role


# ============================================================================
# Pagination Support tests
# ============================================================================


class TestPaginationSupport:
    """Verify pagination is allowed - same resource with different offsets is NOT a loop.

    Key principle: progress_params positive allowlist enables tools to explicitly
    declare which parameters indicate meaningful progress. Pagination parameters
    (offset, limit, k, etc.) are EXCLUDED from progress_params, allowing the LLM
    to read files in chunks without triggering false loop detection.
    """

    def test_read_pagination_same_signature(self, mocker):
        """Reading same file with different offsets generates SAME signature (not a loop)."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        tracker = UnifiedTaskTracker()

        # Mock get_signature_params to return ["path"] for read tool (offset/limit excluded)
        def mock_get_signature_params(tool_name):
            if tool_name == "read":
                return {"path"}
            return set()

        mocker.patch(
            "victor.agent.unified_task_tracker.get_signature_params",
            side_effect=mock_get_signature_params,
        )

        # Get signatures for pagination calls
        sig1 = tracker._get_signature("read", {"path": "src/main.py", "offset": 0, "limit": 500})
        sig2 = tracker._get_signature("read", {"path": "src/main.py", "offset": 500, "limit": 500})
        sig3 = tracker._get_signature("read", {"path": "src/main.py", "offset": 1000, "limit": 100})

        # All should have the SAME signature because offset/limit are NOT in signature_params
        assert (
            sig1 == sig2 == sig3
        ), f"Pagination should generate same signature: {sig1}, {sig2}, {sig3}"
        # Verify only path is in the signature
        assert "src/main.py" in sig1
        assert "offset" not in sig1.lower()
        assert "limit" not in sig1.lower()

    def test_read_different_path_different_signature(self, mocker):
        """Reading different files generates DIFFERENT signatures (actual loop detection)."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        tracker = UnifiedTaskTracker()

        # Mock get_signature_params to return ["path"] for read tool
        def mock_get_signature_params(tool_name):
            if tool_name == "read":
                return {"path"}
            return set()

        mocker.patch(
            "victor.agent.unified_task_tracker.get_signature_params",
            side_effect=mock_get_signature_params,
        )

        sig1 = tracker._get_signature("read", {"path": "src/main.py", "offset": 0})
        sig2 = tracker._get_signature("read", {"path": "src/utils.py", "offset": 0})
        sig3 = tracker._get_signature("read", {"path": "README.md", "offset": 0})

        # All should have DIFFERENT signatures because paths differ
        assert sig1 != sig2 != sig3, "Different paths should generate different signatures"
        assert "main.py" in sig1
        assert "utils.py" in sig2
        assert "README.md" in sig3

    def test_code_search_pagination_same_signature(self, mocker):
        """Code search with different k (result count) generates SAME signature."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        tracker = UnifiedTaskTracker()

        # Mock get_signature_params - note that "k" is NOT included (pagination param)
        def mock_get_signature_params(tool_name):
            if tool_name == "code_search":
                return {"query", "path", "mode"}
            return set()

        mocker.patch(
            "victor.agent.unified_task_tracker.get_signature_params",
            side_effect=mock_get_signature_params,
        )

        sig1 = tracker._get_signature("code_search", {"query": "dataclass", "path": ".", "k": 10})
        sig2 = tracker._get_signature("code_search", {"query": "dataclass", "path": ".", "k": 20})
        sig3 = tracker._get_signature("code_search", {"query": "dataclass", "path": ".", "k": 50})

        # All should have the SAME signature because k is NOT in signature_params
        assert (
            sig1 == sig2 == sig3
        ), f"Pagination with different k should generate same signature: {sig1}, {sig2}, {sig3}"

    def test_ls_depth_is_progress(self, mocker):
        """ls with different depth values generates DIFFERENT signatures (meaningful exploration)."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        tracker = UnifiedTaskTracker()

        # Mock get_signature_params - depth IS included (exploration param, not pagination)
        def mock_get_signature_params(tool_name):
            if tool_name == "ls":
                return {"path", "depth", "pattern"}
            return set()

        mocker.patch(
            "victor.agent.unified_task_tracker.get_signature_params",
            side_effect=mock_get_signature_params,
        )

        sig1 = tracker._get_signature("ls", {"path": "src", "depth": 1})
        sig2 = tracker._get_signature("ls", {"path": "src", "depth": 2})
        sig3 = tracker._get_signature("ls", {"path": "src", "depth": 3})

        # All should have DIFFERENT signatures because depth IS in signature_params
        assert sig1 != sig2 != sig3, "Different depth values should generate different signatures"


# ============================================================================
# Wave 2: SpinDetectorConfig configurability
# ============================================================================


class TestSpinDetectorConfig:
    """SpinDetectorConfig dataclass — Wave 2 addition."""

    def test_spin_detector_config_importable(self):
        from victor.agent.turn_policy import SpinDetectorConfig

        config = SpinDetectorConfig()
        assert config is not None

    def test_spin_detector_config_defaults_match_legacy_constants(self):
        from victor.agent.turn_policy import (
            MAX_NO_TOOL_TURNS,
            READ_ONLY_ESCALATION_THRESHOLD,
            SpinDetectorConfig,
        )

        config = SpinDetectorConfig()
        assert config.max_no_tool_turns == MAX_NO_TOOL_TURNS
        assert config.read_only_escalation_threshold == READ_ONLY_ESCALATION_THRESHOLD
        assert config.repetition_threshold == 3

    def test_spin_detector_default_config_preserves_legacy_behavior(self):
        """SpinDetector() with default config must behave identically to old code."""
        from victor.agent.turn_policy import SpinDetector, SpinState, MAX_NO_TOOL_TURNS

        detector = SpinDetector()
        for _ in range(MAX_NO_TOOL_TURNS):
            detector.record_turn(has_tool_calls=False)
        assert detector.state == SpinState.TERMINATED

    def test_spin_detector_respects_custom_max_no_tool_turns(self):
        from victor.agent.turn_policy import SpinDetector, SpinDetectorConfig, SpinState

        config = SpinDetectorConfig(max_no_tool_turns=2)
        detector = SpinDetector(config=config)
        detector.record_turn(has_tool_calls=False)
        assert detector.state != SpinState.TERMINATED
        detector.record_turn(has_tool_calls=False)
        assert detector.state == SpinState.TERMINATED

    def test_spin_detector_respects_custom_repetition_threshold(self):
        from victor.agent.turn_policy import SpinDetector, SpinDetectorConfig, SpinState

        config = SpinDetectorConfig(repetition_threshold=2)
        detector = SpinDetector(config=config)
        sig = frozenset({"read:test.py"})
        detector.record_turn(has_tool_calls=True, tool_signatures=sig)
        detector.record_turn(has_tool_calls=True, tool_signatures=sig)
        # With threshold=2, two identical turns should terminate
        assert detector.state == SpinState.TERMINATED

    def test_nudge_policy_uses_config_read_only_escalation_threshold(self):
        from victor.agent.turn_policy import (
            NudgePolicy,
            NudgeType,
            SpinDetector,
            SpinDetectorConfig,
        )

        # Use a lower threshold so we can trigger it easily
        config = SpinDetectorConfig(read_only_escalation_threshold=2)
        detector = SpinDetector(config=config)
        policy = NudgePolicy(config=config)

        read_only_sig = frozenset({"read:a.py"})
        detector.record_turn(
            has_tool_calls=True,
            tool_names={"read"},
            tool_signatures=read_only_sig,
        )
        detector.consecutive_read_only_turns = 2  # force to threshold

        decision = policy.evaluate(detector, iteration=1, max_iterations=10)
        assert decision.nudge_type == NudgeType.CODE_SEARCH

    def test_module_level_constants_still_importable(self):
        """Legacy module-level constants must remain importable for third-party code."""
        from victor.agent.turn_policy import (
            MAX_NO_TOOL_TURNS,
            READ_ONLY_ESCALATION_THRESHOLD,
        )

        assert MAX_NO_TOOL_TURNS == 3
        assert READ_ONLY_ESCALATION_THRESHOLD == 5


@pytest.mark.parametrize("intent", [None, *ActionIntent])
@pytest.mark.parametrize("trigger", ["budget", "repetition", "blocked", "reading"])
def test_nudges_preserve_task_intent(intent, trigger):
    policy = NudgePolicy()
    detector = SpinDetector()
    if trigger == "budget":
        decision = policy.budget_warning(7, 10, intent=intent)
    else:
        if trigger == "repetition":
            for _ in range(2):
                detector.record_turn(has_tool_calls=False)
        elif trigger == "blocked":
            for _ in range(2):
                detector.record_turn(has_tool_calls=True, all_blocked=True)
        else:
            for _ in range(READ_ONLY_ESCALATION_THRESHOLD):
                detector.record_turn(has_tool_calls=True, tool_names={"read"}, tool_count=1)
        decision = policy.evaluate(detector, intent=intent)
    assert decision.should_inject
    assert decision.role == "user"
    if trigger == "budget":
        assert decision.nudge_type is NudgeType.BUDGET_WARNING
        assert "3 turns remaining out of 10" in decision.message
    if intent is ActionIntent.WRITE_ALLOWED:
        if trigger != "blocked":
            assert "edit" in decision.message.lower()
    else:
        assert "edit" not in decision.message.lower()
        assert "write" not in decision.message.lower()
        assert "shell" not in decision.message.lower()
        assert "apply the fix" not in decision.message.lower()
