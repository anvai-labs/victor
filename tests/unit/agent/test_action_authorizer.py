# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Tests for the action_authorizer module.

Tests intent detection for action authorization:
- Display-only intent detection
- Write-allowed intent detection
- Read-only intent detection
- Ambiguous intent handling
- Custom signals
- Safe actions filtering
"""

from victor.agent.action_authorizer import (
    ActionIntent,
    ActionAuthorizer,
    IntentClassification,
    IntentDetector,
    detect_intent,
    has_explicit_readonly_shell_request,
    is_tool_blocked_for_intent,
    is_write_authorized,
    get_prompt_guard,
    get_safe_tools,
    SAFE_ACTIONS,
    PROMPT_GUARDS,
)


class TestActionIntent:
    """Tests for ActionIntent enum."""

    def test_intent_values(self):
        """Test all intent enum values exist."""
        assert ActionIntent.DISPLAY_ONLY.value == "display_only"
        assert ActionIntent.WRITE_ALLOWED.value == "write_allowed"
        assert ActionIntent.AMBIGUOUS.value == "ambiguous"
        assert ActionIntent.READ_ONLY.value == "read_only"


class TestIntentClassification:
    """Tests for IntentClassification dataclass."""

    def test_classification_creation(self):
        """Test creating an IntentClassification."""
        classification = IntentClassification(
            intent=ActionIntent.WRITE_ALLOWED,
            confidence=0.9,
            matched_signals=["save_to_file", "create_file"],
            safe_actions={"write", "read"},
            prompt_guard="",
        )
        assert classification.intent == ActionIntent.WRITE_ALLOWED
        assert classification.confidence == 0.9
        assert len(classification.matched_signals) == 2
        assert "write" in classification.safe_actions


class TestIntentDetector:
    """Tests for the IntentDetector class."""

    def test_display_only_detection(self):
        """Test detection of display-only signals."""
        detector = IntentDetector()

        # "show me" patterns
        result = detector.detect("Show me a function that calculates factorial")
        assert result.intent == ActionIntent.DISPLAY_ONLY
        assert "show_me" in result.matched_signals

        # "just show" patterns
        result = detector.detect("just show me the code")
        assert result.intent == ActionIntent.DISPLAY_ONLY

        # "display only" patterns
        result = detector.detect("display only, don't save")
        assert result.intent == ActionIntent.DISPLAY_ONLY

        # "without saving" patterns
        result = detector.detect("show me code without saving to disk")
        assert result.intent == ActionIntent.DISPLAY_ONLY

    def test_write_allowed_detection(self):
        """Test detection of write-allowed signals."""
        detector = IntentDetector()

        # "save to file" patterns
        result = detector.detect("save this to main.py")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "save_to_file" in result.matched_signals

        # "create file" patterns
        result = detector.detect("create a new file called utils.py")
        assert result.intent == ActionIntent.WRITE_ALLOWED

        # "update file" patterns
        result = detector.detect("update the file with this code")
        assert result.intent == ActionIntent.WRITE_ALLOWED

        # "add to file" patterns
        result = detector.detect("add this function into helpers.py")
        assert result.intent == ActionIntent.WRITE_ALLOWED

        result = detector.detect(
            "review and find using tools all information needed to apply fixes"
        )
        assert result.intent == ActionIntent.WRITE_ALLOWED

        result = detector.detect("continue to fix remaining 3 files.")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "continuation_with_payload" in result.matched_signals

        result = detector.detect("continue implementing the remaining files")
        assert result.intent == ActionIntent.WRITE_ALLOWED

        result = detector.detect("adress idetified gaps compehensively")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "address_generic" in result.matched_signals

    def test_read_only_detection(self):
        """Test detection of read-only signals."""
        detector = IntentDetector()

        # "list files" patterns
        result = detector.detect("list all files in the directory")
        assert result.intent == ActionIntent.READ_ONLY
        assert "list_files" in result.matched_signals

        # "explain" patterns
        result = detector.detect("explain the file contents")
        assert result.intent == ActionIntent.READ_ONLY

        # "git status" patterns
        result = detector.detect("git status")
        assert result.intent == ActionIntent.READ_ONLY
        assert "git_read" in result.matched_signals

        # "summarize" patterns
        result = detector.detect("summarize this module")
        assert result.intent == ActionIntent.READ_ONLY

    def test_ambiguous_detection(self):
        """Test ambiguous intent detection."""
        detector = IntentDetector()

        # No clear signals - should fall back to default (safe)
        result = detector.detect("do something with the code")
        # Should be ambiguous, write, or display (safe default)
        assert result.intent in (
            ActionIntent.AMBIGUOUS,
            ActionIntent.WRITE_ALLOWED,
            ActionIntent.DISPLAY_ONLY,
        )

    def test_default_intent_ambiguous(self):
        """Test that default intent is ambiguous (to error on side of caution)."""
        detector = IntentDetector()

        # No signals match
        result = detector.detect("hello world")
        assert result.intent == ActionIntent.AMBIGUOUS
        assert result.confidence < 0.5

    def test_custom_default_intent(self):
        """Test setting custom default intent."""
        detector = IntentDetector(default_intent=ActionIntent.AMBIGUOUS)

        result = detector.detect("something random")
        assert result.intent == ActionIntent.AMBIGUOUS

    def test_write_overrides_display_when_strong(self):
        """Test that strong write signal results in write-allowed."""
        detector = IntentDetector()

        # Clear write signal - should be write allowed
        result = detector.detect("create a new file called config.py")
        assert result.intent == ActionIntent.WRITE_ALLOWED

    def test_confidence_scoring(self):
        """Test confidence scores are reasonable."""
        detector = IntentDetector()

        # Clear signal should have high confidence
        result = detector.detect("show me a function")
        assert result.confidence > 0.5

        # No signal should have low confidence
        result = detector.detect("hello")
        assert result.confidence < 0.5

    def test_is_write_authorized(self):
        """Test the is_write_authorized convenience method."""
        detector = IntentDetector()

        assert detector.is_write_authorized("save this to file.py") is True
        assert detector.is_write_authorized("show me a function") is False
        assert detector.is_write_authorized("list files") is False

    def test_custom_display_signals(self):
        """Test adding custom display signals."""
        custom_signals = [
            (r"\bpreview\b", 1.0, "preview"),
            (r"\bdemonstrate\b", 0.9, "demonstrate"),
        ]
        detector = IntentDetector(custom_display_signals=custom_signals)

        result = detector.detect("preview the code")
        assert result.intent == ActionIntent.DISPLAY_ONLY
        assert "preview" in result.matched_signals

    def test_custom_write_signals(self):
        """Test adding custom write signals."""
        custom_signals = [
            (r"\bpersist\b", 1.0, "persist"),
            (r"\bcommit\s+to\s+disk\b", 0.9, "commit_to_disk"),
        ]
        detector = IntentDetector(custom_write_signals=custom_signals)

        result = detector.detect("persist this configuration")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "persist" in result.matched_signals

    def test_custom_detector_function(self):
        """Test custom detector functions take precedence."""

        def custom_detector(message: str):
            if "magic" in message.lower():
                return IntentClassification(
                    intent=ActionIntent.WRITE_ALLOWED,
                    confidence=1.0,
                    matched_signals=["magic_word"],
                    safe_actions=set(),
                    prompt_guard="",
                )
            return None

        detector = IntentDetector(custom_detectors=[custom_detector])

        result = detector.detect("use the magic word")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "magic_word" in result.matched_signals

    def test_safe_actions_for_each_intent(self):
        """Test that safe actions are correctly assigned for each intent."""
        detector = IntentDetector()

        # Display only should have read + generate but not write
        result = detector.detect("show me a function")
        assert "read" in result.safe_actions
        assert "generate_response" in result.safe_actions
        assert "write" not in result.safe_actions

        # Write allowed should have all actions
        result = detector.detect("save this to file.py")
        assert "write" in result.safe_actions
        assert "edit" in result.safe_actions

    def test_prompt_guard_assignment(self):
        """Test that prompt guards are correctly assigned."""
        detector = IntentDetector()

        result = detector.detect("show me a function")
        assert "IMPORTANT" in result.prompt_guard
        assert "write" in result.prompt_guard or "NOT" in result.prompt_guard

        result = detector.detect("save this to file.py")
        # WRITE_ALLOWED now has positive guidance
        assert len(result.prompt_guard) > 0
        assert (
            "file modifications" in result.prompt_guard.lower()
            or "readonly=False" in result.prompt_guard
        )

    def test_write_allowed_has_positive_guidance(self):
        """WRITE_ALLOWED intent should have constructive guidance, not empty."""
        detector = IntentDetector()

        # Test consolidation prompt (the original bug trigger)
        result = detector.detect("consolidate documentation and replace adoc files with md")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert len(result.prompt_guard) > 0  # Not empty
        assert (
            "file modifications" in result.prompt_guard.lower()
            or "readonly=False" in result.prompt_guard
        )

        # Test other write patterns
        result = detector.detect("reduce proliferation and consolidate files")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert len(result.prompt_guard) > 0

    def test_case_insensitive_matching(self):
        """Test that pattern matching is case insensitive."""
        detector = IntentDetector()

        result1 = detector.detect("SHOW ME a function")
        result2 = detector.detect("show me a function")

        assert result1.intent == result2.intent

    def test_invalid_regex_pattern_handling(self):
        """Test that invalid regex patterns are handled gracefully."""
        # Invalid regex should not crash, just log warning
        custom_signals = [
            (r"[invalid", 1.0, "invalid_pattern"),  # Invalid regex
        ]
        detector = IntentDetector(custom_display_signals=custom_signals)

        # Should still work with valid patterns
        result = detector.detect("show me a function")
        assert result.intent == ActionIntent.DISPLAY_ONLY


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_detect_intent(self):
        """Test the detect_intent convenience function."""
        result = detect_intent("show me a function")
        assert isinstance(result, IntentClassification)
        assert result.intent == ActionIntent.DISPLAY_ONLY

    def test_is_write_authorized_function(self):
        """Test the is_write_authorized convenience function."""
        assert is_write_authorized("save this to file.py") is True
        assert is_write_authorized("show me a function") is False

    def test_get_prompt_guard(self):
        """Test the get_prompt_guard convenience function."""
        guard = get_prompt_guard("show me a function")
        assert "IMPORTANT" in guard

        guard = get_prompt_guard("save this to file.py")
        # WRITE_ALLOWED now has positive guidance
        assert len(guard) > 0
        assert "file modifications" in guard.lower() or "readonly=False" in guard

    def test_get_safe_tools(self):
        """Test the get_safe_tools convenience function."""
        all_tools = {
            "read",
            "write",
            "ls",
            "shell",
            "code_search",
        }

        # Display only should filter out write
        safe = get_safe_tools("show me a function", all_tools)
        assert "read" in safe
        assert "write" not in safe

        # Write allowed should include write
        safe = get_safe_tools("save this to file.py", all_tools)
        assert "write" in safe


class TestSafeActionsMapping:
    """Tests for the SAFE_ACTIONS constant."""

    def test_all_intents_have_safe_actions(self):
        """Test that all intents have defined safe actions."""
        for intent in ActionIntent:
            assert intent in SAFE_ACTIONS
            assert isinstance(SAFE_ACTIONS[intent], set)

    def test_read_only_actions(self):
        """Test read-only has minimal actions."""
        actions = SAFE_ACTIONS[ActionIntent.READ_ONLY]
        assert "read" in actions
        assert "write" not in actions
        assert "shell" not in actions

    def test_write_allowed_actions(self):
        """Test write-allowed has all actions."""
        actions = SAFE_ACTIONS[ActionIntent.WRITE_ALLOWED]
        assert "read" in actions
        assert "write" in actions
        assert "edit" in actions
        assert "shell" in actions


class TestPromptGuards:
    """Tests for the PROMPT_GUARDS constant."""

    def test_all_intents_have_prompt_guards(self):
        """Test that all intents have defined prompt guards."""
        for intent in ActionIntent:
            assert intent in PROMPT_GUARDS

    def test_write_allowed_has_positive_guard(self):
        """Test that write-allowed has positive guidance for file operations."""
        guard = PROMPT_GUARDS[ActionIntent.WRITE_ALLOWED]
        assert len(guard) > 0  # Not empty
        assert "file modifications" in guard.lower() or "readonly=False" in guard

    def test_display_only_has_guard(self):
        """Test that display-only has protective guard."""
        guard = PROMPT_GUARDS[ActionIntent.DISPLAY_ONLY]
        assert "NOT" in guard or "Do not" in guard.lower()


class TestAlias:
    """Tests for the ActionAuthorizer alias."""

    def test_alias_works(self):
        """Test that ActionAuthorizer is an alias for IntentDetector."""
        assert ActionAuthorizer is IntentDetector

        authorizer = ActionAuthorizer()
        assert isinstance(authorizer, IntentDetector)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_message(self):
        """Test handling of empty message."""
        detector = IntentDetector()
        result = detector.detect("")
        assert result.intent == ActionIntent.AMBIGUOUS
        assert result.confidence < 0.5

    def test_very_long_message(self):
        """Test handling of very long message."""
        detector = IntentDetector()
        long_message = "show me " * 1000
        result = detector.detect(long_message)
        assert result.intent == ActionIntent.DISPLAY_ONLY

    def test_special_characters(self):
        """Test handling of special characters."""
        detector = IntentDetector()
        result = detector.detect("show me a function! @#$%^&*()")
        assert result.intent == ActionIntent.DISPLAY_ONLY

    def test_unicode_message(self):
        """Test handling of unicode characters."""
        detector = IntentDetector()
        result = detector.detect("show me a function with unicode")
        assert result.intent == ActionIntent.DISPLAY_ONLY

    def test_multiple_signals_accumulate(self):
        """Test that multiple matching signals accumulate score."""
        detector = IntentDetector()

        # Multiple display signals
        result = detector.detect("just show me an example, display only please")
        assert result.confidence >= 1.0  # High confidence from multiple matches


class TestCompoundWriteSignals:
    """Tests for compound write signal detection (analyze and fix patterns)."""

    def test_analyze_then_fix(self):
        """Test 'analyze and then fix' is detected as WRITE_ALLOWED."""
        detector = IntentDetector()
        result = detector.detect("Analyze the codebase and then fix any bugs you find")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "analyze_then_fix" in result.matched_signals

    def test_find_and_fix(self):
        """Test 'find and fix' is detected as WRITE_ALLOWED."""
        detector = IntentDetector()
        result = detector.detect("Find and fix the memory leak")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "find_and_fix" in result.matched_signals

    def test_fix_bugs(self):
        """Test 'fix the bugs' is detected as WRITE_ALLOWED."""
        detector = IntentDetector()
        # "Fix all bugs" pattern requires specific format
        result = detector.detect("Fix any bugs in the codebase")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "fix_bugs" in result.matched_signals

    def test_fix_issue_in(self):
        """Test 'fix the issue in' is detected as WRITE_ALLOWED."""
        detector = IntentDetector()
        result = detector.detect("Fix the bug in the auth module")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "fix_in" in result.matched_signals

    def test_apply_fix(self):
        """Test 'apply the fix' is detected as WRITE_ALLOWED."""
        detector = IntentDetector()
        result = detector.detect("Apply the fix to the codebase")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "apply_fix" in result.matched_signals

    def test_make_changes(self):
        """Test 'make the changes' is detected as WRITE_ALLOWED."""
        detector = IntentDetector()
        result = detector.detect("Make the changes to improve performance")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "make_changes" in result.matched_signals

    def test_refactor_improve(self):
        """Test 'refactor and improve' is detected as WRITE_ALLOWED."""
        detector = IntentDetector()
        result = detector.detect("Refactor and improve the code structure")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "refactor_improve" in result.matched_signals

    def test_create_dockerfile(self):
        """Test 'create a Dockerfile' is detected as WRITE_ALLOWED."""
        detector = IntentDetector()
        result = detector.detect("Create a Dockerfile for this project")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "create_infra_file" in result.matched_signals

    def test_create_pipeline(self):
        """Test 'create a CI/CD pipeline' is detected as WRITE_ALLOWED."""
        detector = IntentDetector()
        result = detector.detect("Create a CI/CD pipeline for this repo")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "create_pipeline" in result.matched_signals

    def test_implement_feature(self):
        """Test 'implement a feature' is detected as WRITE_ALLOWED."""
        detector = IntentDetector()
        # Pattern requires: implement [optional_single_word] feature/functionality/module/etc
        result = detector.detect("Implement the feature")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "implement_feature" in result.matched_signals

    def test_add_feature_to(self):
        """Test 'add feature to' is detected as WRITE_ALLOWED."""
        detector = IntentDetector()
        result = detector.detect("Add caching support to the API")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "add_feature" in result.matched_signals

    def test_setup_configure(self):
        """Test 'set up' is detected as WRITE_ALLOWED."""
        detector = IntentDetector()
        result = detector.detect("Set up the testing framework")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "setup_configure" in result.matched_signals

    def test_write_tests(self):
        """Test 'write tests for' is detected as WRITE_ALLOWED."""
        detector = IntentDetector()
        result = detector.detect("Write tests for the user service")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "write_tests" in result.matched_signals

    def test_compound_overrides_read_only(self):
        """Test that compound write signals override read-only signals."""
        detector = IntentDetector()
        # "Review" would normally be read-only, but "and fix" makes it write
        result = detector.detect("Review the code and then fix any issues")
        assert result.intent == ActionIntent.WRITE_ALLOWED


class TestWeakWriteSignals:
    """Tests for weak write signal detection (ambiguous intent)."""

    def test_weak_write_becomes_write_allowed(self):
        """Test that weak write signals with low score (but still signals) become authorized."""
        detector = IntentDetector()
        # "modify" is now a generic write intent signal (0.7)
        result = detector.detect("modify something somehow")
        assert result.intent == ActionIntent.WRITE_ALLOWED

    def test_edit_without_file_extension(self):
        """Test edit without file extension has lower weight."""
        detector = IntentDetector()
        result = detector.detect("edit the file")
        # "edit the file" matches with 0.8 weight
        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "edit_file" in result.matched_signals

    def test_weak_custom_write_signal_becomes_ambiguous(self):
        """Test that a weak custom write signal (score <= 0.5) becomes AMBIGUOUS."""
        # Add a custom write signal with very low weight
        custom_signals = [(r"\bmaybe\s+store\b", 0.3, "maybe_store")]
        detector = IntentDetector(custom_write_signals=custom_signals)
        # This matches only the custom low-weight signal
        result = detector.detect("maybe store something")
        # Score is 0.3 which is <= 0.5, so should be AMBIGUOUS
        assert result.intent == ActionIntent.AMBIGUOUS
        assert result.confidence == 0.3
        assert "maybe_store" in result.matched_signals


class TestScorePatterns:
    """Tests for the _score_patterns method."""

    def test_score_patterns_empty_patterns(self):
        """Test scoring with empty pattern list."""
        detector = IntentDetector()
        score, matched = detector._score_patterns("any message", [])
        assert score == 0.0
        assert matched == []

    def test_score_patterns_no_match(self):
        """Test scoring when no patterns match."""
        detector = IntentDetector()
        score, matched = detector._score_patterns("hello world", detector._write_patterns)
        assert score == 0.0
        assert matched == []

    def test_score_patterns_single_match(self):
        """Test scoring with a single matching pattern."""
        detector = IntentDetector()
        score, matched = detector._score_patterns("show me the code", detector._display_patterns)
        assert score > 0
        assert "show_me" in matched

    def test_score_patterns_multiple_matches(self):
        """Test scoring with multiple matching patterns."""
        detector = IntentDetector()
        # This should match multiple display signals
        message = "just show me, display only please"
        score, matched = detector._score_patterns(message, detector._display_patterns)
        assert score > 1.0  # Multiple matches accumulate
        assert len(matched) >= 2

    def test_score_patterns_accumulates_weights(self):
        """Test that weights are accumulated correctly."""
        detector = IntentDetector()
        message = "show me this, just display it"
        score, matched = detector._score_patterns(message, detector._display_patterns)
        # Each match adds its weight
        assert score >= 2.0  # At least two patterns with weight 1.0


class TestDetectMethodBranches:
    """Tests for all branches in the detect method."""

    def test_custom_detector_returns_result(self):
        """Test that custom detector result is used when returned."""
        custom_result = IntentClassification(
            intent=ActionIntent.WRITE_ALLOWED,
            confidence=1.0,
            matched_signals=["custom"],
            safe_actions={"write_file"},
            prompt_guard="",
        )

        def custom_detector(msg):
            if "magic" in msg.lower():
                return custom_result
            return None

        detector = IntentDetector(custom_detectors=[custom_detector])
        result = detector.detect("do the magic thing")
        assert result == custom_result

    def test_custom_detector_returns_none(self):
        """Test that normal detection is used when custom detector returns None."""

        def custom_detector(msg):
            return None  # Always returns None

        detector = IntentDetector(custom_detectors=[custom_detector])
        result = detector.detect("show me a function")
        assert result.intent == ActionIntent.DISPLAY_ONLY

    def test_multiple_custom_detectors(self):
        """Test multiple custom detectors are tried in order."""
        first_called = []
        second_called = []

        def first_detector(msg):
            first_called.append(True)
            return None

        def second_detector(msg):
            second_called.append(True)
            if "special" in msg:
                return IntentClassification(
                    intent=ActionIntent.WRITE_ALLOWED,
                    confidence=1.0,
                    matched_signals=["second"],
                    safe_actions=set(),
                    prompt_guard="",
                )
            return None

        detector = IntentDetector(custom_detectors=[first_detector, second_detector])
        result = detector.detect("special case")
        assert first_called == [True]
        assert second_called == [True]
        assert "second" in result.matched_signals

    def test_compound_write_highest_precedence(self):
        """Test compound write signals take highest precedence."""
        detector = IntentDetector()
        # Message with both display and compound write signals
        result = detector.detect("review the code and then fix any bugs")
        assert result.intent == ActionIntent.WRITE_ALLOWED
        # Compound pattern should match
        assert any("fix" in signal for signal in result.matched_signals)

    def test_strong_write_over_display(self):
        """Test strong write signal wins over display signal."""
        detector = IntentDetector()
        result = detector.detect("create a new file and save it to config.py")
        assert result.intent == ActionIntent.WRITE_ALLOWED

    def test_read_only_over_display(self):
        """Test read-only signals take precedence when equal or higher."""
        detector = IntentDetector()
        result = detector.detect("list all files in the directory")
        assert result.intent == ActionIntent.READ_ONLY

    def test_display_only_when_only_display_signals(self):
        """Test display-only when only display signals match."""
        detector = IntentDetector()
        result = detector.detect("show me a Python function")
        assert result.intent == ActionIntent.DISPLAY_ONLY

    def test_default_intent_when_no_signals(self):
        """Test default intent is used when no signals match."""
        detector = IntentDetector(default_intent=ActionIntent.AMBIGUOUS)
        result = detector.detect("xyzzy foobar")
        assert result.intent == ActionIntent.AMBIGUOUS
        assert result.confidence == 0.2
        assert result.matched_signals == []

    def test_confidence_capped_at_1(self):
        """Test confidence is capped at 1.0."""
        detector = IntentDetector()
        # Multiple signals that would sum to > 1.0
        result = detector.detect("just show me, display only, without saving")
        assert result.confidence <= 1.0


class TestGetSafeToolsFunction:
    """Tests for the get_safe_tools convenience function."""

    def test_get_safe_tools_display_only(self):
        """Test get_safe_tools filters for display-only intent."""
        all_tools = {
            "read_file",
            "write_file",
            "list_directory",
            "code_search",
            "semantic_code_search",
            "execute_bash",
        }
        safe = get_safe_tools("show me a function", all_tools)
        assert "read_file" in safe
        assert "list_directory" in safe
        assert "write_file" not in safe

    def test_get_safe_tools_write_allowed(self):
        """Test get_safe_tools allows write tools for write-allowed intent."""
        all_tools = {
            "read_file",
            "write_file",
            "create_file",
            "list_directory",
            "execute_bash",
        }
        safe = get_safe_tools("save this to config.py", all_tools)
        assert "read_file" in safe
        assert "write_file" in safe
        assert "create_file" in safe
        assert "execute_bash" in safe

    def test_get_safe_tools_read_only(self):
        """Test get_safe_tools for read-only intent."""
        all_tools = {
            "read_file",
            "write_file",
            "list_directory",
            "code_search",
            "semantic_code_search",
        }
        safe = get_safe_tools("list all files", all_tools)
        assert "read_file" in safe
        assert "list_directory" in safe
        assert "write_file" not in safe

    def test_get_safe_tools_intersection(self):
        """Test get_safe_tools only returns tools from input set."""
        all_tools = {"read_file", "custom_tool"}
        safe = get_safe_tools("show me code", all_tools)
        # Only read_file should be in result since custom_tool is not in mapping
        assert "read_file" in safe
        assert "custom_tool" not in safe

    def test_get_safe_tools_empty_input(self):
        """Test get_safe_tools with empty input set."""
        safe = get_safe_tools("show me code", set())
        assert safe == set()

    def test_get_safe_tools_code_search_mapping(self):
        """Test get_safe_tools maps code_search action to tools."""
        all_tools = {"code_search", "semantic_code_search", "read_file"}
        safe = get_safe_tools("show me the implementation", all_tools)
        assert "code_search" in safe
        assert "semantic_code_search" in safe

    def test_get_safe_tools_accepts_canonical_tool_sets(self):
        """Canonical safe actions should map cleanly to canonical tool inventories."""
        all_tools = {"read", "write", "edit", "ls", "shell", "code_search"}
        safe = get_safe_tools("save this to config.py", all_tools)
        assert "read" in safe
        assert "write" in safe
        assert "edit" in safe
        assert "ls" in safe
        assert "shell" in safe


class TestInitializationCoverage:
    """Tests for IntentDetector initialization coverage."""

    def test_init_with_all_custom_signals(self):
        """Test initialization with both custom display and write signals."""
        custom_display = [(r"\bpreview\b", 1.0, "preview")]
        custom_write = [(r"\bpersist\b", 1.0, "persist")]
        detector = IntentDetector(
            custom_display_signals=custom_display,
            custom_write_signals=custom_write,
        )
        # Verify custom patterns were added
        result = detector.detect("preview the code")
        assert "preview" in result.matched_signals

        result = detector.detect("persist the changes")
        assert "persist" in result.matched_signals

    def test_init_custom_display_signals_only(self):
        """Test initialization with only custom display signals."""
        custom_display = [(r"\bdemonstrate\b", 0.9, "demonstrate")]
        detector = IntentDetector(custom_display_signals=custom_display)
        result = detector.detect("demonstrate the algorithm")
        assert "demonstrate" in result.matched_signals
        assert result.intent == ActionIntent.DISPLAY_ONLY

    def test_init_custom_write_signals_only(self):
        """Test initialization with only custom write signals."""
        custom_write = [(r"\bcommit\s+changes\b", 1.0, "commit_changes")]
        detector = IntentDetector(custom_write_signals=custom_write)
        result = detector.detect("commit changes to the repo")
        assert "commit_changes" in result.matched_signals
        assert result.intent == ActionIntent.WRITE_ALLOWED

    def test_init_empty_custom_detectors(self):
        """Test initialization with empty custom detectors list."""
        detector = IntentDetector(custom_detectors=[])
        result = detector.detect("show me a function")
        assert result.intent == ActionIntent.DISPLAY_ONLY


class TestPatternCompilationErrors:
    """Tests for pattern compilation error handling."""

    def test_multiple_invalid_patterns(self):
        """Test handling multiple invalid patterns."""
        custom_signals = [
            (r"[invalid", 1.0, "invalid1"),
            (r"(unclosed", 1.0, "invalid2"),
            (r"\bvalid\b", 1.0, "valid"),
        ]
        detector = IntentDetector(custom_display_signals=custom_signals)
        # Valid pattern should still work
        result = detector.detect("this is valid text")
        assert "valid" in result.matched_signals

    def test_all_patterns_invalid(self):
        """Test when all custom patterns are invalid."""
        custom_signals = [
            (r"[invalid", 1.0, "invalid1"),
            (r"(unclosed", 1.0, "invalid2"),
        ]
        detector = IntentDetector(custom_display_signals=custom_signals)
        # Should still work with built-in patterns
        result = detector.detect("show me a function")
        assert result.intent == ActionIntent.DISPLAY_ONLY


class TestToolCategories:
    """Tests for tool category constants."""

    def test_write_tools_constant(self):
        """Test WRITE_TOOLS constant contains expected tools."""
        from victor.agent.action_authorizer import WRITE_TOOLS

        assert "write" in WRITE_TOOLS
        assert "edit" in WRITE_TOOLS
        assert "shell" in WRITE_TOOLS
        assert "git" in WRITE_TOOLS
        assert isinstance(WRITE_TOOLS, frozenset)

    def test_read_only_tools_constant(self):
        """Test READ_ONLY_TOOLS constant contains expected tools."""
        from victor.agent.action_authorizer import READ_ONLY_TOOLS

        assert "read" in READ_ONLY_TOOLS
        assert "ls" in READ_ONLY_TOOLS
        assert "code_search" in READ_ONLY_TOOLS
        assert isinstance(READ_ONLY_TOOLS, frozenset)

    def test_generation_tools_constant(self):
        """Test GENERATION_TOOLS constant contains expected tools."""
        from victor.agent.action_authorizer import GENERATION_TOOLS

        assert "generate_code" in GENERATION_TOOLS
        assert "generate_docs" in GENERATION_TOOLS
        assert isinstance(GENERATION_TOOLS, frozenset)

    def test_intent_blocked_tools_mapping(self):
        """Test INTENT_BLOCKED_TOOLS mapping is correct (now permissive)."""
        from victor.agent.action_authorizer import (
            INTENT_BLOCKED_TOOLS,
        )

        # Permissive policy allows tools, relies on prompt guards
        assert INTENT_BLOCKED_TOOLS[ActionIntent.DISPLAY_ONLY] == frozenset()
        assert INTENT_BLOCKED_TOOLS[ActionIntent.READ_ONLY] == frozenset()
        assert INTENT_BLOCKED_TOOLS[ActionIntent.WRITE_ALLOWED] == frozenset()
        assert INTENT_BLOCKED_TOOLS[ActionIntent.AMBIGUOUS] == frozenset()


class TestIsWriteAuthorizedMethod:
    """Tests specifically for is_write_authorized method coverage."""

    def test_is_write_authorized_method_true(self):
        """Test is_write_authorized method returns True for write intent."""
        detector = IntentDetector()
        assert detector.is_write_authorized("save this to file.py") is True
        assert detector.is_write_authorized("create a new file") is True
        assert detector.is_write_authorized("update the file") is True

    def test_is_write_authorized_method_false(self):
        """Test is_write_authorized method returns False for non-write intent."""
        detector = IntentDetector()
        assert detector.is_write_authorized("show me a function") is False
        assert detector.is_write_authorized("list files") is False
        assert detector.is_write_authorized("explain the code") is False

    def test_is_write_authorized_method_ambiguous(self):
        """Test is_write_authorized method with ambiguous message."""
        detector = IntentDetector()
        # Messages that don't clearly indicate write
        result = detector.is_write_authorized("do something")
        assert result is False  # Default is safe (not authorized)


class TestConvenienceFunctionsCoverage:
    """Additional tests for convenience functions to ensure full coverage."""

    def test_detect_intent_returns_classification(self):
        """Test detect_intent returns proper IntentClassification."""
        result = detect_intent("explain the file contents")
        assert isinstance(result, IntentClassification)
        assert result.intent == ActionIntent.READ_ONLY
        assert isinstance(result.safe_actions, set)

    def test_is_write_authorized_function_compound_signals(self):
        """Test is_write_authorized with compound signals."""
        assert is_write_authorized("review and fix the bugs") is True
        assert is_write_authorized("implement the feature") is True
        assert is_write_authorized("create a Dockerfile") is True

    def test_get_prompt_guard_for_all_intents(self):
        """Test get_prompt_guard for different intents."""
        # Display only
        guard = get_prompt_guard("show me the code")
        assert "IMPORTANT" in guard
        assert "SEE code" in guard or "NOT" in guard

        # Read only
        guard = get_prompt_guard("list files in directory")
        assert "read-only" in guard

        # Write allowed
        guard = get_prompt_guard("save this to test.py")
        # WRITE_ALLOWED now has positive guidance
        assert len(guard) > 0
        assert "file modifications" in guard.lower() or "readonly=False" in guard

    def test_get_safe_tools_with_git_tools(self):
        """Test get_safe_tools with git-related tools."""
        all_tools = {"execute_bash", "read_file", "write_file", "git_status"}
        safe = get_safe_tools("show me git status", all_tools)
        # git_status action maps to execute_bash
        assert "execute_bash" in safe or "git_status" not in safe


class TestIntentClassificationDataclass:
    """Additional tests for IntentClassification dataclass."""

    def test_classification_with_empty_matched_signals(self):
        """Test classification with empty matched signals."""
        classification = IntentClassification(
            intent=ActionIntent.AMBIGUOUS,
            confidence=0.2,
            matched_signals=[],
            safe_actions=set(),
            prompt_guard="",
        )
        assert classification.matched_signals == []
        assert classification.safe_actions == set()

    def test_classification_safe_actions_copy(self):
        """Test that safe_actions is returned as a copy in detect."""
        detector = IntentDetector()
        result = detector.detect("show me code")
        original = result.safe_actions.copy()
        result.safe_actions.add("new_action")
        # Modifying result should not affect SAFE_ACTIONS constant
        assert "new_action" not in SAFE_ACTIONS[ActionIntent.DISPLAY_ONLY]


class TestReadonlyShellSignals:
    """Tests for readonly shell/SQLite authorization signals."""

    def test_misspelled_sqllite_request_is_recognized(self):
        assert has_explicit_readonly_shell_request(
            "also review the sqllite db for evolved prompts and explain significance"
        )

    def test_review_db_request_keeps_shell_available(self):
        blocked = is_tool_blocked_for_intent(
            "shell",
            ActionIntent.DISPLAY_ONLY,
            user_message="review the sqllite db directly",
        )

        assert blocked is False


class TestGitReadonlySignalSourcedFromBash:
    """The git read-only intent signal is derived from bash.py's authoritative
    GIT_READONLY_SUBCOMMANDS set, so it cannot drift from actual command validation."""

    def test_git_signal_covers_subcommands_beyond_the_old_hardcoded_six(self):
        # blame / rev-parse are read-only in bash.py but were missing from the
        # previous hardcoded (status|log|show|branch|tag|diff) regex.
        assert has_explicit_readonly_shell_request("can you git blame this file")
        assert has_explicit_readonly_shell_request("please git rev-parse HEAD")

    def test_git_signal_still_matches_core_subcommands(self):
        assert has_explicit_readonly_shell_request("run git status")
        assert has_explicit_readonly_shell_request("show me the git log")

    def test_git_write_subcommands_do_not_signal_readonly(self):
        assert not has_explicit_readonly_shell_request("git push the branch")
        assert not has_explicit_readonly_shell_request("git commit the changes")

    def test_signal_set_matches_bash_authority(self):
        from victor.tools.bash import GIT_READONLY_SUBCOMMANDS

        # Every bash read-only git subcommand should be recognized as a readonly signal.
        for sub in sorted(GIT_READONLY_SUBCOMMANDS):
            assert has_explicit_readonly_shell_request(f"run git {sub} now"), sub
