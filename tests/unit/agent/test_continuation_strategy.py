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

"""Tests for continuation strategy."""

import pytest
from unittest.mock import MagicMock, patch

from victor.agent.continuation_strategy import (
    ContinuationActionType,
    ContinuationStrategy,
)
from victor.agent.decisions.schemas import (
    ContinuationAction,
    ContinuationDecision,
    DecisionType,
)
from victor.agent.services.protocols.decision_service import DecisionResult
from victor.storage.embeddings.intent_classifier import IntentType
from victor.storage.embeddings.question_classifier import (
    QuestionType,
    QuestionClassificationResult,
)

# =============================================================================
# DETECT MENTIONED TOOLS TESTS
# =============================================================================


class TestDetectMentionedTools:
    """Tests for detect_mentioned_tools static method."""

    def test_detects_call_pattern(self):
        """Test detects 'call tool' pattern for non-ambiguous tool names."""
        text = "Let me call code_search to see the file"
        tools = ["code_search", "read_file", "write_file"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "code_search" in result

    def test_detects_use_pattern(self):
        """Test detects 'use tool' pattern for non-ambiguous tool names."""
        text = "I'll use web_search to find it"
        tools = ["read_file", "web_search", "write_file"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "web_search" in result

    def test_detects_execute_pattern(self):
        """Test detects 'execute tool' pattern for non-ambiguous tool names."""
        text = "Let me execute grep"
        tools = ["grep", "read_file", "write_file"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "grep" in result

    def test_ambiguous_tool_requires_parenthesis(self):
        """Test ambiguous tool names (common English words) need parenthesis or 'tool' suffix."""
        # "call read" should NOT trigger for ambiguous names
        text = "Let me call read to see the file"
        tools = ["read", "search", "write"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)
        assert "read" not in result, "Ambiguous 'read' should not match 'call read'"

        # "implementation plan" should NOT trigger plan tool
        text2 = "I reviewed the implementation plan and the project roadmap."
        tools2 = ["plan", "read", "search"]
        result2 = ContinuationStrategy.detect_mentioned_tools(text2, tools2, {})
        assert "plan" not in result2, "Natural language 'plan' should not trigger tool detection"

        # But "read(" SHOULD still trigger
        text3 = "I'll call read() to get the content"
        result3 = ContinuationStrategy.detect_mentioned_tools(text3, tools, aliases)
        assert "read" in result3, "read() with parenthesis should still trigger"

    def test_detects_parenthesis_pattern(self):
        """Test detects 'tool()' pattern."""
        text = "I'll call read() to get the content"
        tools = ["read", "search"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "read" in result

    def test_detects_the_tool_pattern(self):
        """Test detects 'the X tool' pattern."""
        text = "I'll use the read tool to check"
        tools = ["read", "search"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "read" in result

    def test_resolves_aliases(self):
        """Test resolves tool aliases to canonical names."""
        text = "Let me call read_file"
        tools = ["read_file"]
        aliases = {"read_file": "read"}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "read" in result

    def test_no_duplicates(self):
        """Test no duplicate tool names returned."""
        text = "I'll call code_search and use code_search to check"
        tools = ["code_search"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert len(result) == 1
        assert "code_search" in result

    def test_returns_empty_for_no_matches(self):
        """Test returns empty list when no tools mentioned."""
        text = "Just a simple response without any tools"
        tools = ["read", "search", "write"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert result == []

    def test_case_insensitive(self):
        """Test detection is case insensitive for non-ambiguous tools."""
        text = "Let me CALL CODE_SEARCH to check"
        tools = ["code_search"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "code_search" in result

    def test_case_insensitive_ambiguous_parenthesis(self):
        """Test ambiguous tool detection is case insensitive with parenthesis."""
        text = "Let me call READ() to check"
        tools = ["read"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "read" in result

    def test_run_pattern(self):
        """Test detects 'run tool' pattern for non-ambiguous tools."""
        text = "Let me run execute_bash to run the command"
        tools = ["execute_bash", "read_file"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "execute_bash" in result

    def test_invoke_pattern(self):
        """Test detects 'invoke tool' pattern for non-ambiguous tools."""
        text = "I need to invoke web_search"
        tools = ["web_search", "read_file"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "web_search" in result

    def test_perform_pattern(self):
        """Test detects 'perform tool' pattern for non-ambiguous tools."""
        text = "Let me perform write_file to save"
        tools = ["write_file", "read_file"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "write_file" in result

    def test_multiple_tools(self):
        """Test detects multiple non-ambiguous tools mentioned."""
        text = "I'll call code_search and then use file_edit"
        tools = ["code_search", "file_edit", "web_search"]
        aliases = {}

        result = ContinuationStrategy.detect_mentioned_tools(text, tools, aliases)

        assert "code_search" in result
        assert "file_edit" in result


# =============================================================================
# CONTINUATION STRATEGY INITIALIZATION
# =============================================================================


class TestContinuationStrategyInit:
    """Tests for ContinuationStrategy initialization."""

    def test_init_with_event_bus(self):
        """Test initialization with custom event bus."""
        mock_bus = MagicMock()
        strategy = ContinuationStrategy(event_bus=mock_bus)

        assert strategy._event_bus is mock_bus

    def test_init_without_event_bus(self):
        """Test initialization without event bus uses singleton."""
        strategy = ContinuationStrategy()

        assert strategy._event_bus is not None


# =============================================================================
# DETERMINE CONTINUATION ACTION TESTS
# =============================================================================


class TestDetermineContinuationAction:
    """Tests for determine_continuation_action method."""

    @pytest.fixture
    def strategy(self):
        """Create a strategy with mocked event bus."""
        mock_bus = MagicMock()
        return ContinuationStrategy(event_bus=mock_bus)

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.max_continuation_prompts_analysis = 6
        settings.max_continuation_prompts_action = 5
        settings.max_continuation_prompts_default = 3
        settings.continuation_prompt_overrides = {}
        return settings

    @pytest.fixture
    def base_kwargs(self, mock_settings):
        """Create base kwargs for determine_continuation_action."""
        from victor.agent.task_completion import TaskCompletionDetector

        detector = TaskCompletionDetector()

        return {
            "is_analysis_task": False,
            "is_action_task": False,
            "content_length": 100,
            "full_content": "Some content",
            "continuation_prompts": 0,
            "asking_input_prompts": 0,
            "one_shot_mode": False,
            "mentioned_tools": None,
            "max_prompts_summary_requested": False,
            "settings": mock_settings,
            "rl_coordinator": None,
            "provider_name": "openai",
            "model": "gpt-4",
            "tool_budget": 50,
            "unified_tracker_config": {"max_total_iterations": 50},
        }

    def test_finish_when_summary_already_requested(self, strategy, base_kwargs):
        """Test finishes when summary was already requested."""
        base_kwargs["max_prompts_summary_requested"] = True

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.CONTINUATION

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.FINISH
        assert "already requested" in result["reason"]

    def test_request_summary_on_stuck_loop(self, strategy, base_kwargs):
        """Test requests summary on stuck loop intent."""
        mock_intent = MagicMock()
        mock_intent.intent = IntentType.STUCK_LOOP

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.REQUEST_SUMMARY
        assert "STUCK_LOOP" in result["reason"]

    def test_runtime_intelligence_guides_low_confidence_continuation(
        self, mock_settings, base_kwargs
    ):
        """Low-confidence continuation decisions should use runtime intelligence."""
        runtime_intelligence = MagicMock()
        runtime_intelligence.decide_sync.return_value = DecisionResult(
            decision_type=DecisionType.CONTINUATION_ACTION,
            result=ContinuationDecision(
                action=ContinuationAction.REQUEST_SUMMARY,
                reason="Need a concise handoff",
            ),
            source="llm",
            confidence=0.8,
        )
        strategy = ContinuationStrategy(
            event_bus=MagicMock(),
            runtime_intelligence=runtime_intelligence,
        )
        mock_intent = MagicMock()
        mock_intent.intent = "other"
        base_kwargs["continuation_prompts"] = 3
        mock_settings.max_continuation_prompts_default = 6

        with patch("victor.agent.decisions.chain.should_use_llm", return_value=True):
            result = strategy.determine_continuation_action(
                intent_result=mock_intent,
                **base_kwargs,
            )

        assert result["action"] is ContinuationActionType.REQUEST_SUMMARY
        assert result["reason"] == "LLM: Need a concise handoff"
        runtime_intelligence.decide_sync.assert_called_once()

    def test_force_tool_execution_on_hallucinated_calls(self, strategy, base_kwargs):
        """Test forces tool execution when tools mentioned but not called."""
        base_kwargs["mentioned_tools"] = ["read", "search"]

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.CONTINUATION

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.FORCE_TOOL_EXECUTION
        assert "Hallucinated" in result["reason"]

    def test_force_tool_execution_message_avoids_pseudo_call_examples(self, strategy, base_kwargs):
        """Recovery guidance should ask for real tool calls, not text pseudo-calls."""
        base_kwargs["mentioned_tools"] = ["read", "search"]

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.CONTINUATION

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.FORCE_TOOL_EXECUTION
        assert "plain text" in result["message"].lower()
        assert "write(path=" not in result["message"]
        assert "edit(path=" not in result["message"]
        assert "shell(command=" not in result["message"]

    def test_short_preamble_on_action_task_requests_one_real_tool_call(self, strategy, base_kwargs):
        """Interrupted action turns should get a concrete single-tool recovery prompt."""
        mock_intent = MagicMock()
        mock_intent.intent = IntentType.CONTINUATION

        base_kwargs.update(
            {
                "is_action_task": True,
                "content_length": 87,
                "full_content": "I'll continue implementing the remaining fixes. Let me read the files I need to modify:",
                "continuation_prompts": 1,
            }
        )

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.PROMPT_TOOL_CALL
        assert "one actual tool call" in result["message"].lower()
        assert "read" in result["message"].lower()
        assert result["updates"]["continuation_prompts"] == 2

    def test_short_preamble_includes_degraded_resume_summary(self, strategy, base_kwargs):
        """Degraded resume state should be surfaced in the recovery prompt."""
        mock_intent = MagicMock()
        mock_intent.intent = IntentType.CONTINUATION

        base_kwargs.update(
            {
                "is_action_task": True,
                "content_length": 96,
                "full_content": "I'll continue implementing this. Let me check the files I still need to modify:",
                "continuation_prompts": 0,
                "degraded_resume_state": True,
                "resume_summary": "2 tool call(s) used; files examined: victor/framework/graph.py; previous provider request required tool-history repair",
            }
        )

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.PROMPT_TOOL_CALL
        assert "resume state:" in result["message"].lower()
        assert "tool-history repair" in result["message"].lower()

    def test_return_to_user_for_asking_input_one_shot(self, strategy, base_kwargs):
        """Test returns to user when asking input in one-shot mode."""
        base_kwargs["one_shot_mode"] = True

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.ASKING_INPUT

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.RETURN_TO_USER

    def test_finishes_direct_qa_response_in_one_shot_mode(self, strategy, base_kwargs):
        """One-shot direct-answer turns should not be pushed into continuation loops."""
        base_kwargs["one_shot_mode"] = True
        base_kwargs["content_length"] = 5
        base_kwargs["full_content"] = "READY"
        base_kwargs["task_completion_signals"] = {
            "direct_response_requested": True,
            "required_files": set(),
            "explicit_database_query_requested": False,
        }

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.NEUTRAL

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.FINISH
        assert "direct-response" in result["reason"].lower()

    def test_finishes_direct_response_in_interactive_mode(self, strategy, base_kwargs):
        """Interactive direct-response turns should also terminate immediately."""
        base_kwargs["one_shot_mode"] = False
        base_kwargs["content_length"] = 5
        base_kwargs["full_content"] = "READY"
        base_kwargs["task_completion_signals"] = {
            "direct_response_requested": True,
            "required_files": set(),
            "explicit_database_query_requested": False,
        }

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.NEUTRAL

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.FINISH
        assert "direct-response" in result["reason"].lower()

    def test_direct_response_finish_preempts_stuck_loop_recovery(self, strategy, base_kwargs):
        """Direct-response turns should finish before stuck-loop recovery fires."""
        base_kwargs["content_length"] = 120
        base_kwargs["full_content"] = (
            "The user is asking me to reply with exactly READY, so the answer is READY"
        )
        base_kwargs["task_completion_signals"] = {
            "direct_response_requested": True,
            "required_files": set(),
            "explicit_database_query_requested": False,
        }

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.STUCK_LOOP

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.FINISH
        assert "direct-response" in result["reason"].lower()

    def test_direct_response_finish_preempts_hallucinated_tool_recovery(
        self, strategy, base_kwargs
    ):
        """Direct-response turns should finish before generic tool-recovery prompts."""
        base_kwargs["content_length"] = 120
        base_kwargs["full_content"] = (
            "This is a simple acknowledgement request that does not require tool calls. READY"
        )
        base_kwargs["mentioned_tools"] = ["search"]
        base_kwargs["task_completion_signals"] = {
            "direct_response_requested": True,
            "required_files": set(),
            "explicit_database_query_requested": False,
        }

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.NEUTRAL

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.FINISH
        assert "direct-response" in result["reason"].lower()

    def test_continue_asking_input_auto_respond(self, strategy, base_kwargs):
        """Test auto-responds to asking input intent when question is rhetorical."""
        mock_intent = MagicMock()
        mock_intent.intent = IntentType.ASKING_INPUT

        # Mock classify_question to return a rhetorical question that should auto-continue
        # Note: should_auto_continue is a property computed from question_type and confidence
        mock_result = QuestionClassificationResult(
            question_type=QuestionType.RHETORICAL,
            confidence=0.9,  # >= 0.6 so should_auto_continue will be True
            matched_pattern="Should I proceed?",
        )
        with patch(
            "victor.agent.continuation_strategy.classify_question",
            return_value=mock_result,
        ):
            result = strategy.determine_continuation_action(
                intent_result=mock_intent, **base_kwargs
            )

        assert result["action"] is ContinuationActionType.CONTINUE_ASKING_INPUT
        assert "asking_input_prompts" in result["updates"]

    def test_return_to_user_max_asking_input(self, strategy, base_kwargs):
        """Test returns to user when max asking input reached."""
        base_kwargs["asking_input_prompts"] = 3

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.ASKING_INPUT

        # Mock classify_question to return a rhetorical question (which would normally auto-continue)
        # But since we hit max prompts, it should return to user instead
        # Note: should_auto_continue is a property computed from question_type and confidence
        mock_result = QuestionClassificationResult(
            question_type=QuestionType.RHETORICAL,
            confidence=0.9,  # >= 0.6 so should_auto_continue will be True
            matched_pattern="Should I proceed?",
        )
        with patch(
            "victor.agent.continuation_strategy.classify_question",
            return_value=mock_result,
        ):
            result = strategy.determine_continuation_action(
                intent_result=mock_intent, **base_kwargs
            )

        assert result["action"] is ContinuationActionType.RETURN_TO_USER

    def test_finish_on_completion_intent(self, strategy, base_kwargs):
        """Test finishes on completion intent."""
        mock_intent = MagicMock()
        mock_intent.intent = IntentType.COMPLETION

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.FINISH
        assert "completion" in result["reason"].lower()

    def test_prompt_tool_call_for_analysis_task(self, strategy, base_kwargs):
        """Test prompts for tool calls on analysis task."""
        base_kwargs["is_analysis_task"] = True

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.CONTINUATION

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.PROMPT_TOOL_CALL
        assert "analysis" in result["message"].lower()

    def test_prompt_tool_call_for_action_task(self, strategy, base_kwargs):
        """Test prompts for tool calls on action task."""
        base_kwargs["is_action_task"] = True

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.CONTINUATION

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.PROMPT_TOOL_CALL
        assert "implementation" in result["message"].lower()

    def test_request_summary_max_continuation_prompts(self, strategy, base_kwargs):
        """Test requests summary when max continuation prompts reached."""
        base_kwargs["is_analysis_task"] = True
        base_kwargs["continuation_prompts"] = 6

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.CONTINUATION

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.REQUEST_SUMMARY
        assert result["updates"].get("max_prompts_summary_requested") is True

    def test_prompt_tool_call_for_yes_let_me_short_preamble(self, strategy, base_kwargs):
        """Interjections before 'let me' should still trigger continuation."""
        base_kwargs["content_length"] = 83
        base_kwargs["full_content"] = (
            "Yes! Let me find and query the SQLite databases directly. Let me locate them first:"
        )

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.NEUTRAL

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.PROMPT_TOOL_CALL
        assert "didn't complete" in result["message"]

    def test_prompt_tool_call_when_explicit_database_request_unmet(self, strategy, base_kwargs):
        """Explicit direct DB inspection requests should block completion until satisfied."""
        base_kwargs["task_completion_signals"] = {
            "explicit_database_query_requested": True,
            "database_query_satisfied": False,
            "executed_tool_names": set(),
            "original_user_message": "review the sqllite db directly",
        }

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.NEUTRAL

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.PROMPT_TOOL_CALL
        assert "database" in result["message"].lower()
        assert result["updates"]["continuation_prompts"] == 1

    def test_finish_when_explicit_database_request_is_satisfied(self, strategy, base_kwargs):
        """A real db/shell call should satisfy the direct inspection requirement."""
        base_kwargs["task_completion_signals"] = {
            "explicit_database_query_requested": True,
            "database_query_satisfied": True,
            "executed_tool_names": {"shell"},
            "original_user_message": "review the sqllite db directly",
        }

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.NEUTRAL

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.FINISH

    def test_finish_default(self, strategy, base_kwargs):
        """Test finishes by default when no conditions met."""
        mock_intent = MagicMock()
        # Use a neutral intent that doesn't match specific handlers
        mock_intent.intent = IntentType.NEUTRAL

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        assert result["action"] is ContinuationActionType.FINISH


class TestRLCoordinator:
    """Tests for RL coordinator integration."""

    @pytest.fixture
    def strategy(self):
        mock_bus = MagicMock()
        return ContinuationStrategy(event_bus=mock_bus)

    def test_uses_rl_recommendations(self, strategy):
        """Test uses RL coordinator recommendations."""
        from victor.agent.task_completion import TaskCompletionDetector

        mock_settings = MagicMock()
        mock_settings.max_continuation_prompts_analysis = 6
        mock_settings.max_continuation_prompts_action = 5
        mock_settings.max_continuation_prompts_default = 3
        mock_settings.continuation_prompt_overrides = {}

        mock_rl = MagicMock()
        mock_recommendation = MagicMock()
        mock_recommendation.value = 10
        mock_recommendation.confidence = 0.9
        mock_rl.get_recommendation.return_value = mock_recommendation

        detector = TaskCompletionDetector()

        base_kwargs = {
            "is_analysis_task": True,
            "is_action_task": False,
            "content_length": 100,
            "full_content": "Content",
            "continuation_prompts": 0,
            "asking_input_prompts": 0,
            "one_shot_mode": False,
            "mentioned_tools": None,
            "max_prompts_summary_requested": False,
            "settings": mock_settings,
            "rl_coordinator": mock_rl,
            "provider_name": "openai",
            "model": "gpt-4",
            "tool_budget": 50,
            "unified_tracker_config": {"max_total_iterations": 50},
        }

        mock_intent = MagicMock()
        mock_intent.intent = IntentType.CONTINUATION

        result = strategy.determine_continuation_action(intent_result=mock_intent, **base_kwargs)

        # RL coordinator should have been called
        mock_rl.get_recommendation.assert_called()
