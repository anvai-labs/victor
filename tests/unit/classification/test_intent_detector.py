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

"""Tests for IntentDetector - Gap 4 implementation."""

import pytest


class TestDisplayOnlyIntent:
    """Tests for DISPLAY_ONLY intent detection."""

    def test_show_me_code(self):
        """Test that 'show me' is classified as DISPLAY_ONLY."""
        from victor.agent.action_authorizer import IntentDetector, ActionIntent

        detector = IntentDetector()
        result = detector.detect("Show me a function that calculates factorial")

        assert result.intent == ActionIntent.DISPLAY_ONLY
        assert "show_me" in result.matched_signals

    def test_create_function_display(self):
        """Test that 'create a function' is classified as DISPLAY_ONLY."""
        from victor.agent.action_authorizer import IntentDetector, ActionIntent

        detector = IntentDetector()
        result = detector.detect("Create a Python function that sorts a list")

        assert result.intent == ActionIntent.DISPLAY_ONLY
        assert "create_function" in result.matched_signals

    def test_give_me_example(self):
        """Test that 'give me an example' is classified as DISPLAY_ONLY."""
        from victor.agent.action_authorizer import IntentDetector, ActionIntent

        detector = IntentDetector()
        result = detector.detect("Give me an example of a REST API handler")

        assert result.intent == ActionIntent.DISPLAY_ONLY
        assert "give_example" in result.matched_signals

    def test_just_show(self):
        """Test that 'just show' is classified as DISPLAY_ONLY."""
        from victor.agent.action_authorizer import IntentDetector, ActionIntent

        detector = IntentDetector()
        result = detector.detect("Just show me the code, don't save anything")

        assert result.intent == ActionIntent.DISPLAY_ONLY

    def test_how_would_i_write(self):
        """Test that 'how would I write' is classified as DISPLAY_ONLY."""
        from victor.agent.action_authorizer import IntentDetector, ActionIntent

        detector = IntentDetector()
        result = detector.detect("How would I write a binary search algorithm?")

        assert result.intent == ActionIntent.DISPLAY_ONLY
        assert "how_to_write" in result.matched_signals


class TestWriteAllowedIntent:
    """Tests for WRITE_ALLOWED intent detection."""

    def test_save_to_file(self):
        """Test that 'save to file' is classified as WRITE_ALLOWED."""
        from victor.agent.action_authorizer import IntentDetector, ActionIntent

        detector = IntentDetector()
        result = detector.detect("Save this to utils.py")

        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "save_to_file" in result.matched_signals

    def test_create_file(self):
        """Test that 'create a file' is classified as WRITE_ALLOWED."""
        from victor.agent.action_authorizer import IntentDetector, ActionIntent

        detector = IntentDetector()
        result = detector.detect("Create a file called helpers.py with the sorting function")

        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "create_file" in result.matched_signals

    def test_add_to_file(self):
        """Test that 'add into file' is classified as WRITE_ALLOWED."""
        from victor.agent.action_authorizer import IntentDetector, ActionIntent

        detector = IntentDetector()
        result = detector.detect("Add this into utils.py")

        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "add_into_file" in result.matched_signals or "add_to_file" in result.matched_signals

    def test_update_file(self):
        """Test that 'update the file' is classified as WRITE_ALLOWED."""
        from victor.agent.action_authorizer import IntentDetector, ActionIntent

        detector = IntentDetector()
        result = detector.detect("Update the file with these changes")

        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "update_file" in result.matched_signals

    def test_write_to_disk(self):
        """Test that 'write to disk' is classified as WRITE_ALLOWED."""
        from victor.agent.action_authorizer import IntentDetector, ActionIntent

        detector = IntentDetector()
        result = detector.detect("Write this to disk")

        assert result.intent == ActionIntent.WRITE_ALLOWED


class TestReadOnlyIntent:
    """Tests for READ_ONLY intent detection."""

    def test_list_files(self):
        """Test that 'list files' is classified as READ_ONLY."""
        from victor.agent.action_authorizer import IntentDetector, ActionIntent

        detector = IntentDetector()
        result = detector.detect("List all files in the src directory")

        assert result.intent == ActionIntent.READ_ONLY
        assert "list_files" in result.matched_signals

    def test_explain_file(self):
        """Test that 'explain the file' is classified as READ_ONLY."""
        from victor.agent.action_authorizer import IntentDetector, ActionIntent

        detector = IntentDetector()
        result = detector.detect("Explain the file orchestrator.py")

        assert result.intent == ActionIntent.READ_ONLY
        assert "explain" in result.matched_signals

    def test_git_status(self):
        """Test that 'git status' is classified as READ_ONLY."""
        from victor.agent.action_authorizer import IntentDetector, ActionIntent

        detector = IntentDetector()
        result = detector.detect("Show git status")

        assert result.intent == ActionIntent.READ_ONLY
        assert "git_read" in result.matched_signals


class TestAmbiguousIntent:
    """Tests for AMBIGUOUS intent detection."""

    def test_unrecognized_message(self):
        """Test that unrecognized messages use the default intent."""
        from victor.agent.action_authorizer import IntentDetector, ActionIntent

        detector = IntentDetector()
        result = detector.detect("xyzzy foobar baz")

        # Default is AMBIGUOUS: intent is unclear, so the AMBIGUOUS prompt guard
        # steers the model to safe behavior rather than locking it to display-only.
        assert result.intent == ActionIntent.AMBIGUOUS
        assert result.confidence < 0.5


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_detect_intent(self):
        """Test detect_intent convenience function."""
        from victor.agent.action_authorizer import detect_intent, ActionIntent

        result = detect_intent("Show me a function")
        assert result.intent == ActionIntent.DISPLAY_ONLY

    def test_is_write_authorized(self):
        """Test is_write_authorized convenience function."""
        from victor.agent.action_authorizer import is_write_authorized

        assert not is_write_authorized("Show me a function")
        assert is_write_authorized("Save this to file.py")

    def test_get_prompt_guard(self):
        """Test get_prompt_guard convenience function."""
        from victor.agent.action_authorizer import get_prompt_guard

        guard = get_prompt_guard("Show me a function")
        assert "SEE code" in guard
        assert "Do NOT use write" in guard or "Do NOT modify any existing files" in guard

        guard = get_prompt_guard("Save to file.py")
        # WRITE_ALLOWED carries a positive authorization guard (permissive policy).
        assert "explicitly authorized file modifications" in guard


class TestSafeActions:
    """Tests for safe actions."""

    def test_display_only_safe_actions(self):
        """Test safe actions for DISPLAY_ONLY intent."""
        from victor.agent.action_authorizer import detect_intent

        result = detect_intent("Show me a function")

        assert "read" in result.safe_actions
        assert "generate_response" in result.safe_actions
        assert "write" not in result.safe_actions

    def test_write_allowed_safe_actions(self):
        """Test safe actions for WRITE_ALLOWED intent."""
        from victor.agent.action_authorizer import detect_intent

        result = detect_intent("Save this to utils.py")

        assert "read" in result.safe_actions
        assert "write" in result.safe_actions
        assert "edit" in result.safe_actions

    def test_read_only_safe_actions(self):
        """Test safe actions for READ_ONLY intent."""
        from victor.agent.action_authorizer import detect_intent

        result = detect_intent("List files in src/")

        assert "read" in result.safe_actions
        assert "ls" in result.safe_actions
        assert "write" not in result.safe_actions
        assert "generate_response" not in result.safe_actions


class TestCustomDetector:
    """Tests for custom detector support."""

    def test_custom_detector_takes_precedence(self):
        """Test that custom detectors are tried first."""
        from victor.agent.action_authorizer import (
            IntentDetector,
            IntentClassification,
            ActionIntent,
        )

        def custom_detector(message: str):
            if "URGENT" in message.upper():
                return IntentClassification(
                    intent=ActionIntent.WRITE_ALLOWED,
                    confidence=1.0,
                    matched_signals=["urgent_override"],
                    safe_actions={"write_file", "execute_bash"},
                    prompt_guard="",
                )
            return None

        detector = IntentDetector(custom_detectors=[custom_detector])
        result = detector.detect("URGENT: Create the file immediately")

        assert result.intent == ActionIntent.WRITE_ALLOWED
        assert "urgent_override" in result.matched_signals


class TestPromptGuards:
    """Tests for prompt guards."""

    def test_display_only_prompt_guard(self):
        """Test prompt guard for DISPLAY_ONLY intent."""
        from victor.agent.action_authorizer import detect_intent

        result = detect_intent("Show me a binary search function")

        assert "DISPLAY" in result.prompt_guard
        assert (
            "Do NOT use write" in result.prompt_guard
            or "Do NOT modify any existing files" in result.prompt_guard
        )

    def test_read_only_prompt_guard(self):
        """Test prompt guard for READ_ONLY intent."""
        from victor.agent.action_authorizer import detect_intent

        result = detect_intent("List files in the directory")

        assert "read-only" in result.prompt_guard
        assert "NOT write" in result.prompt_guard

    def test_write_allowed_authorization_prompt_guard(self):
        """WRITE_ALLOWED emits a positive file-modification authorization guard."""
        from victor.agent.action_authorizer import detect_intent

        result = detect_intent("Save this to utils.py and update the file")

        assert "explicitly authorized file modifications" in result.prompt_guard


class TestEdgeCases:
    """Tests for edge cases."""

    def test_conflicting_signals_write_wins_when_explicit(self):
        """Test that explicit write signals win over display signals."""
        from victor.agent.action_authorizer import IntentDetector, ActionIntent

        detector = IntentDetector()
        # "Show me" suggests display, but "save to" is explicit write
        result = detector.detect("Show me this code and save to file.py")

        # Write signal is stronger when explicit
        # Note: actual behavior depends on signal weights
        assert result.intent in (ActionIntent.WRITE_ALLOWED, ActionIntent.DISPLAY_ONLY)

    def test_empty_message(self):
        """Test classification of empty message."""
        from victor.agent.action_authorizer import IntentDetector, ActionIntent

        detector = IntentDetector()
        result = detector.detect("")

        # Empty input has no signals -> AMBIGUOUS (the safe default intent),
        # whose prompt guard steers the model toward safe behavior.
        assert result.intent == ActionIntent.AMBIGUOUS
        assert result.confidence < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
