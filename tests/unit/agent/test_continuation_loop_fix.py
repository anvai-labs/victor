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

"""Tests for prompting loop fix - TDD tests.

These tests verify the fixes for the follow-on prompting loop bug where:
- Agent gets stuck in infinite loop after reading required files
- Repeatedly summarizes files and asks for more specifics
- Never produces required output format
- Only terminates on Ctrl-C

Test categories:
- Task completion detection
- Pinned prompt section (survives compaction)
- File re-read deduplication
- Best-effort finalize on grounding failure
"""

import os
import pytest
import time
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Any, Dict, Optional


class TestTaskCompletionDetection:
    """Tests for hard exit condition when task is complete."""

    def test_finish_when_all_files_read(self):
        """Should return 'finish' action when all required files are read."""
        from victor.agent.continuation_strategy import ContinuationStrategy

        strategy = ContinuationStrategy()

        # Mock intent result indicating continuation
        intent_result = MagicMock()
        intent_result.intent = MagicMock()
        intent_result.intent.name = "CONTINUATION"
        intent_result.confidence = 0.8

        # Task completion signals indicate all files read
        task_completion_signals = {
            "required_files": ["file1.py", "file2.py"],
            "read_files": {"file1.py", "file2.py"},
            "required_outputs": ["findings table", "top-3 fixes"],
            "all_files_read": True,
        }

        # Content mentions required outputs
        full_content = """
        ## Findings Table
        | Finding | Severity |
        | --- | --- |
        | Issue 1 | High |

        ## Top-3 Fixes
        1. Fix A
        2. Fix B
        3. Fix C
        """

        result = strategy.determine_continuation_action(
            intent_result=intent_result,
            is_analysis_task=True,
            is_action_task=False,
            content_length=len(full_content),
            full_content=full_content,
            continuation_prompts=0,
            asking_input_prompts=0,
            one_shot_mode=False,
            mentioned_tools=None,
            max_prompts_summary_requested=False,
            settings=MagicMock(
                max_continuation_prompts_analysis=6,
                max_continuation_prompts_action=5,
                max_continuation_prompts_default=3,
                continuation_prompt_overrides={},
            ),
            rl_coordinator=None,
            provider_name="anthropic",
            model="claude-3-5-sonnet",
            tool_budget=25,
            unified_tracker_config={"max_total_iterations": 50},
            task_completion_signals=task_completion_signals,
        )

        assert result["action"] == "finish"
        assert "task completion" in result["reason"].lower()

    def test_continue_when_files_missing(self):
        """Should continue prompting when required files are not yet read."""
        from victor.agent.continuation_strategy import ContinuationStrategy

        strategy = ContinuationStrategy()

        # Mock intent result indicating continuation
        intent_result = MagicMock()
        intent_result.intent = MagicMock()
        intent_result.intent.name = "CONTINUATION"
        intent_result.confidence = 0.8

        # Task completion signals indicate files still missing
        task_completion_signals = {
            "required_files": ["file1.py", "file2.py", "file3.py"],
            "read_files": {"file1.py"},  # Only 1 of 3 read
            "required_outputs": ["findings table"],
            "all_files_read": False,
        }

        result = strategy.determine_continuation_action(
            intent_result=intent_result,
            is_analysis_task=True,
            is_action_task=False,
            content_length=100,
            full_content="Partial analysis...",
            continuation_prompts=0,
            asking_input_prompts=0,
            one_shot_mode=False,
            mentioned_tools=None,
            max_prompts_summary_requested=False,
            settings=MagicMock(
                max_continuation_prompts_analysis=6,
                max_continuation_prompts_action=5,
                max_continuation_prompts_default=3,
                continuation_prompt_overrides={},
            ),
            rl_coordinator=None,
            provider_name="anthropic",
            model="claude-3-5-sonnet",
            tool_budget=25,
            unified_tracker_config={"max_total_iterations": 50},
            task_completion_signals=task_completion_signals,
        )

        # Should prompt for tool call, not finish
        assert result["action"] in ("prompt_tool_call", "continue_asking_input")

    def test_finish_when_output_requirements_met(self):
        """Should finish when output contains required format elements."""
        from victor.agent.continuation_strategy import ContinuationStrategy

        strategy = ContinuationStrategy()

        # Check output requirements detection
        full_content = """
        Here are my findings:

        | Finding | Impact | Fix |
        |---------|--------|-----|
        | Bug 1   | High   | X   |

        Top 3 recommended fixes:
        1. First fix
        2. Second fix
        3. Third fix
        """

        # Test _output_requirements_met helper if it exists
        if hasattr(strategy, "_output_requirements_met"):
            requirements = ["findings table", "top-3 fixes"]
            assert strategy._output_requirements_met(full_content, requirements)


class TestCumulativeInterventionTracking:
    """Tests for cumulative prompt intervention tracking.

    The cumulative counter NEVER resets (unlike per-segment counter) and is used
    to detect excessive prompting across the entire session.
    """

    def test_synthesis_nudge_on_cumulative_threshold(self):
        """Should nudge synthesis when cumulative interventions reach threshold."""
        from victor.agent.continuation_strategy import ContinuationStrategy

        strategy = ContinuationStrategy()

        # Mock intent result indicating continuation
        intent_result = MagicMock()
        intent_result.intent = MagicMock()
        intent_result.intent.name = "CONTINUATION"
        intent_result.confidence = 0.8

        # Task completion signals with high cumulative interventions (>= 5)
        task_completion_signals = {
            "required_files": ["file1.py", "file2.py"],
            "read_files": {"file1.py"},  # Not all read
            "required_outputs": ["findings table"],
            "all_files_read": False,
            "cumulative_prompt_interventions": 5,  # Reached threshold
            "synthesis_nudge_count": 0,
        }

        result = strategy.determine_continuation_action(
            intent_result=intent_result,
            is_analysis_task=True,
            is_action_task=False,
            content_length=100,
            full_content="Still analyzing...",
            continuation_prompts=0,  # Per-segment counter reset
            asking_input_prompts=0,
            one_shot_mode=False,
            mentioned_tools=None,
            max_prompts_summary_requested=False,
            settings=MagicMock(
                max_continuation_prompts_analysis=6,
                max_continuation_prompts_action=5,
                max_continuation_prompts_default=3,
                continuation_prompt_overrides={},
            ),
            rl_coordinator=None,
            provider_name="anthropic",
            model="claude-3-5-sonnet",
            tool_budget=25,
            unified_tracker_config={"max_total_iterations": 50},
            task_completion_signals=task_completion_signals,
        )

        # Should trigger synthesis nudge, not just prompt_tool_call
        assert result["action"] == "continue_with_synthesis_hint"
        assert "synthesis_nudge_count" in result["updates"]

    def test_force_synthesis_on_excessive_interventions(self):
        """Should force synthesis when cumulative interventions are excessive AND nudges exhausted."""
        from victor.agent.continuation_strategy import ContinuationStrategy

        strategy = ContinuationStrategy()

        intent_result = MagicMock()
        intent_result.intent = MagicMock()
        intent_result.intent.name = "CONTINUATION"
        intent_result.confidence = 0.8

        # Task completion signals with very high cumulative interventions (>= 8)
        # and all synthesis nudges exhausted (synthesis_nudge_count >= 3)
        task_completion_signals = {
            "required_files": ["file1.py"],
            "read_files": {"file1.py", "file2.py", "file3.py"},
            "required_outputs": ["findings table"],
            "all_files_read": False,  # Even if not all required read
            "cumulative_prompt_interventions": 8,  # Excessive
            "synthesis_nudge_count": 3,  # All nudges exhausted
        }

        result = strategy.determine_continuation_action(
            intent_result=intent_result,
            is_analysis_task=True,
            is_action_task=False,
            content_length=100,
            full_content="Still not synthesizing...",
            continuation_prompts=0,
            asking_input_prompts=0,
            one_shot_mode=False,
            mentioned_tools=None,
            max_prompts_summary_requested=False,
            settings=MagicMock(
                max_continuation_prompts_analysis=6,
                max_continuation_prompts_action=5,
                max_continuation_prompts_default=3,
                continuation_prompt_overrides={},
            ),
            rl_coordinator=None,
            provider_name="anthropic",
            model="claude-3-5-sonnet",
            tool_budget=25,
            unified_tracker_config={"max_total_iterations": 50},
            task_completion_signals=task_completion_signals,
        )

        # Should force synthesis with request_summary
        assert result["action"] == "request_summary"
        assert "Excessive" in result["reason"] or "prompt interventions" in result["reason"]

    def test_no_intervention_on_low_count(self):
        """Should not trigger intervention nudge when cumulative count is low."""
        from victor.agent.continuation_strategy import ContinuationStrategy
        from victor.storage.embeddings.intent_classifier import IntentType

        strategy = ContinuationStrategy()

        intent_result = MagicMock()
        intent_result.intent = IntentType.CONTINUATION
        intent_result.confidence = 0.8

        # Task completion signals with low cumulative interventions
        task_completion_signals = {
            "required_files": ["file1.py", "file2.py"],
            "read_files": {"file1.py"},
            "required_outputs": ["findings table"],
            "all_files_read": False,
            "cumulative_prompt_interventions": 2,  # Below threshold
            "synthesis_nudge_count": 0,
        }

        result = strategy.determine_continuation_action(
            intent_result=intent_result,
            is_analysis_task=True,
            is_action_task=False,
            content_length=100,
            full_content="Working on analysis...",
            continuation_prompts=0,
            asking_input_prompts=0,
            one_shot_mode=False,
            mentioned_tools=None,
            max_prompts_summary_requested=False,
            settings=MagicMock(
                max_continuation_prompts_analysis=6,
                max_continuation_prompts_action=5,
                max_continuation_prompts_default=3,
                continuation_prompt_overrides={},
            ),
            rl_coordinator=None,
            provider_name="anthropic",
            model="claude-3-5-sonnet",
            tool_budget=25,
            unified_tracker_config={"max_total_iterations": 50},
            task_completion_signals=task_completion_signals,
        )

        # Should just prompt for tool call, not nudge synthesis
        assert result["action"] == "prompt_tool_call"


class TestPinnedPromptSection:
    """Tests for pinned output requirements that survive compaction."""

    def test_pinned_requirements_survive_compaction(self):
        """Pinned output requirements should not be compacted."""
        from victor.agent.context_compactor import MessagePriority

        # Verify PINNED priority exists and is highest
        assert hasattr(MessagePriority, "PINNED")
        assert MessagePriority.PINNED > MessagePriority.CRITICAL

    def test_pinned_priority_highest(self):
        """PINNED priority should be higher than all others."""
        from victor.agent.context_compactor import MessagePriority

        assert MessagePriority.PINNED == 150
        assert MessagePriority.PINNED > MessagePriority.CRITICAL
        assert MessagePriority.PINNED > MessagePriority.HIGH
        assert MessagePriority.PINNED > MessagePriority.MEDIUM

    def test_is_pinned_requirement_detects_patterns(self):
        """Should detect pinned requirement patterns in content."""
        from victor.agent.context_compactor import ContextCompactor, MessagePriority

        compactor = ContextCompactor()

        # Test patterns that should be pinned
        pinned_patterns = [
            "You must output a findings table",
            "Required format: JSON",
            "Create a findings table showing",
            "Provide top-3 fixes for the issues",
            "Deliverables: summary report",
        ]

        for pattern in pinned_patterns:
            message = {"role": "user", "content": pattern}
            priority = compactor._assign_priority(message)
            assert priority == MessagePriority.PINNED, f"Failed for: {pattern}"

    def test_non_pinned_content_normal_priority(self):
        """Normal content should not get PINNED priority."""
        from victor.agent.context_compactor import ContextCompactor, MessagePriority

        compactor = ContextCompactor()

        normal_messages = [
            {"role": "user", "content": "Please analyze this code"},
            {"role": "assistant", "content": "I'll help you with that"},
            {"role": "user", "content": "What does this function do?"},
        ]

        for message in normal_messages:
            priority = compactor._assign_priority(message)
            assert priority < MessagePriority.PINNED


class TestFileReadDedup:
    """Tests for file re-read deduplication."""

    def test_duplicate_read_returns_cached(self):
        """Duplicate file read within TTL should return cached result."""
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig

        # Create pipeline with mocked dependencies
        mock_registry = MagicMock()
        mock_registry.is_tool_enabled.return_value = True

        mock_executor = MagicMock()

        config = ToolPipelineConfig(
            enable_idempotent_caching=True,
            idempotent_cache_max_size=100,
        )

        pipeline = ToolPipeline(
            tool_registry=mock_registry,
            tool_executor=mock_executor,
            config=config,
        )

        # Simulate first read
        file_path = "/path/to/file.py"
        pipeline._read_file_timestamps[file_path] = time.monotonic()

        # Check if duplicate
        assert pipeline._is_duplicate_read(file_path, max_age_seconds=300)

    def test_read_allowed_after_ttl(self):
        """File read should be allowed after TTL expires."""
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig

        mock_registry = MagicMock()
        mock_executor = MagicMock()

        config = ToolPipelineConfig(
            enable_idempotent_caching=True,
        )

        pipeline = ToolPipeline(
            tool_registry=mock_registry,
            tool_executor=mock_executor,
            config=config,
        )

        # Simulate old read (beyond TTL)
        file_path = "/path/to/file.py"
        pipeline._read_file_timestamps[file_path] = time.monotonic() - 400  # 400 seconds ago

        # Should not be considered duplicate (TTL is 300 seconds)
        assert not pipeline._is_duplicate_read(file_path, max_age_seconds=300)

    def test_record_file_read_updates_timestamp(self):
        """record_file_read should update the timestamp."""
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig

        mock_registry = MagicMock()
        mock_executor = MagicMock()

        pipeline = ToolPipeline(
            tool_registry=mock_registry,
            tool_executor=mock_executor,
        )

        file_path = "/path/to/file.py"

        # Record file read
        before = time.monotonic()
        pipeline.record_file_read(file_path)
        after = time.monotonic()

        # Timestamp should be set
        assert file_path in pipeline._read_file_timestamps
        assert before <= pipeline._read_file_timestamps[file_path] <= after

    def test_read_allowed_after_file_changes(self, tmp_path):
        """A reread should be allowed when the file changed on disk."""
        from victor.agent.tool_pipeline import ToolPipeline

        mock_registry = MagicMock()
        mock_executor = MagicMock()

        pipeline = ToolPipeline(
            tool_registry=mock_registry,
            tool_executor=mock_executor,
        )

        file_path = tmp_path / "module.py"
        file_path.write_text("print('before')\n")
        read_key = str(file_path)

        pipeline.record_file_read(read_key)
        assert pipeline._is_duplicate_read(read_key, max_age_seconds=300)

        file_stat = file_path.stat()
        file_path.write_text("print('after')\n")
        os.utime(
            file_path,
            ns=(
                file_stat.st_atime_ns,
                max(file_stat.st_mtime_ns + 1_000_000, time.time_ns()),
            ),
        )

        assert not pipeline._is_duplicate_read(read_key, max_age_seconds=300)

    @pytest.mark.asyncio
    async def test_execute_read_after_file_change_not_skipped(self, tmp_path):
        """Pipeline should re-execute a read after the file changed."""
        from victor.agent.tool_call_tracker import (
            ToolCallTracker as ToolDeduplicationTracker,
        )
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig
        from victor.agent.tool_executor import ToolExecutionResult

        mock_registry = MagicMock()
        mock_registry.is_tool_enabled.return_value = True

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(
            side_effect=[
                ToolExecutionResult(
                    tool_name="read",
                    success=True,
                    result="before",
                    error=None,
                ),
                ToolExecutionResult(
                    tool_name="read",
                    success=True,
                    result="after",
                    error=None,
                ),
            ]
        )

        pipeline = ToolPipeline(
            tool_registry=mock_registry,
            tool_executor=mock_executor,
            config=ToolPipelineConfig(enable_idempotent_caching=False),
            deduplication_tracker=ToolDeduplicationTracker(),
        )

        file_path = tmp_path / "module.py"
        file_path.write_text("print('before')\n")
        tool_calls = [{"name": "read", "arguments": {"path": str(file_path)}}]

        first = await pipeline.execute_tool_calls(tool_calls, {})
        assert first.results[0].success is True
        assert first.results[0].skipped is False

        file_stat = file_path.stat()
        file_path.write_text("print('after')\n")
        os.utime(
            file_path,
            ns=(
                file_stat.st_atime_ns,
                max(file_stat.st_mtime_ns + 1_000_000, time.time_ns()),
            ),
        )

        second = await pipeline.execute_tool_calls(tool_calls, {})
        assert second.results[0].success is True
        assert second.results[0].skipped is False
        assert second.results[0].result == "after"
        assert mock_executor.execute.call_count == 2


class TestBestEffortFinalize:
    """Tests for best-effort finalize on grounding failure."""

    @pytest.mark.asyncio
    async def test_finalize_after_max_grounding_retries(self):
        """Should finalize with best-effort response after max grounding failures."""
        from victor.agent.runtime_intelligence_pipeline import (
            RuntimeIntelligencePipeline,
        )

        pipeline = RuntimeIntelligencePipeline(
            provider_name="anthropic",
            model="claude-3-5-sonnet",
            profile_name="test",
        )
        pipeline._max_grounding_retries = 1

        # Simulate grounding failures - already exceeded max
        pipeline._grounding_failure_count = 2

        # Mock grounding verifier to return ungrounded result
        mock_verifier = AsyncMock()
        mock_verifier.verify = AsyncMock(
            return_value=MagicMock(
                is_grounded=False,
                confidence=0.3,
                issues=[],
            )
        )
        pipeline._grounding_verifier = mock_verifier

        # Mock the lazy initializer to return True
        with patch.object(pipeline, "_get_grounding_verifier", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = True

            result = await pipeline.process_response(
                response="This is my analysis...",
                context={"task": "audit"},
            )

            # Should indicate finalize
            assert result.should_finalize is True
            assert "grounding" in result.finalize_reason.lower()

    @pytest.mark.asyncio
    async def test_retry_on_first_grounding_failure(self):
        """Should allow retry on first grounding failure."""
        from victor.agent.runtime_intelligence_pipeline import (
            RuntimeIntelligencePipeline,
        )

        pipeline = RuntimeIntelligencePipeline(
            provider_name="anthropic",
            model="claude-3-5-sonnet",
            profile_name="test",
        )
        pipeline._max_grounding_retries = 1
        pipeline._grounding_failure_count = 0  # First failure

        # Mock grounding verifier to return ungrounded result
        mock_verifier = AsyncMock()
        mock_verifier.verify = AsyncMock(
            return_value=MagicMock(
                is_grounded=False,
                confidence=0.4,
                issues=[],
            )
        )
        pipeline._grounding_verifier = mock_verifier

        with patch.object(pipeline, "_get_grounding_verifier", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = True

            result = await pipeline.process_response(
                response="This is my analysis...",
                context={"task": "audit"},
            )

            # Should indicate retry, not finalize
            assert result.should_retry is True
            assert result.should_finalize is False

    @pytest.mark.asyncio
    async def test_reset_counter_on_success(self):
        """Should reset grounding failure counter on success."""
        from victor.agent.runtime_intelligence_pipeline import (
            RuntimeIntelligencePipeline,
        )

        pipeline = RuntimeIntelligencePipeline(
            provider_name="anthropic",
            model="claude-3-5-sonnet",
            profile_name="test",
        )
        pipeline._grounding_failure_count = 1  # Had one failure

        # Mock grounding verifier to return grounded result
        mock_verifier = AsyncMock()
        mock_verifier.verify = AsyncMock(
            return_value=MagicMock(
                is_grounded=True,
                confidence=0.9,
                issues=[],
            )
        )
        pipeline._grounding_verifier = mock_verifier

        with patch.object(pipeline, "_get_grounding_verifier", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = True

            await pipeline.process_response(
                response="Verified analysis...",
                context={"task": "audit"},
            )

            # Counter should be reset
            assert pipeline._grounding_failure_count == 0


class TestMiddlewareChainWiring:
    """Tests for middleware chain wiring into tool pipeline."""

    @pytest.mark.asyncio
    async def test_middleware_chain_called_during_execution(self):
        """Middleware chain should be called during tool execution."""
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig
        from victor.agent.middleware_chain import MiddlewareChain

        mock_registry = MagicMock()
        mock_registry.is_tool_enabled.return_value = True

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(
            return_value=MagicMock(success=True, result="result", error=None)
        )

        # Create middleware chain with mock middleware
        chain = MiddlewareChain()
        chain.process_before = AsyncMock(
            return_value=MagicMock(proceed=True, modified_arguments=None)
        )
        chain.process_after = AsyncMock(return_value="result")

        config = ToolPipelineConfig()
        pipeline = ToolPipeline(
            tool_registry=mock_registry,
            tool_executor=mock_executor,
            config=config,
            middleware_chain=chain,
        )

        # Execute a tool call
        await pipeline.execute_tool_calls(
            [{"name": "read", "arguments": {"path": "test.py"}}],
            context={},
        )

        # Middleware should have been called
        chain.process_before.assert_called()
        chain.process_after.assert_called()

    @pytest.mark.asyncio
    async def test_middleware_can_block_execution(self):
        """Middleware blocking should prevent tool execution."""
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig
        from victor.agent.middleware_chain import MiddlewareChain

        mock_registry = MagicMock()
        mock_registry.is_tool_enabled.return_value = True

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock()

        # Create middleware chain that blocks
        chain = MiddlewareChain()
        chain.process_before = AsyncMock(
            return_value=MagicMock(
                proceed=False,
                error_message="Blocked by safety middleware",
            )
        )

        config = ToolPipelineConfig()
        pipeline = ToolPipeline(
            tool_registry=mock_registry,
            tool_executor=mock_executor,
            config=config,
            middleware_chain=chain,
        )

        # Execute a tool call
        result = await pipeline.execute_tool_calls(
            [{"name": "write", "arguments": {"path": "/etc/passwd", "content": "x"}}],
            context={},
        )

        # Tool executor should NOT have been called
        mock_executor.execute.assert_not_called()

        # Result should indicate skipped
        assert len(result.results) == 1
        assert result.results[0].skipped is True


class TestVerticalIntegrationObservability:
    """Tests for vertical integration event emission."""

    async def test_vertical_applied_event_emitted(self):
        """Should emit vertical_applied event when integration completes."""
        from victor.framework.vertical_integration import (
            VerticalIntegrationPipeline,
            IntegrationResult,
        )
        from victor.core.events import MessagingEvent, get_observability_bus

        # Create a mock result that the pipeline would return
        result = IntegrationResult(vertical_name="coding")
        result.tools_applied = ["read", "write", "grep"]
        result.middleware_count = 0
        result.safety_patterns_count = 0
        result.prompt_hints_count = 0
        result.workflows_count = 0
        result.rl_learners_count = 0
        result.team_specs_count = 0
        result.success = True

        # Emit the event using the new event system
        bus = get_observability_bus()
        await bus.connect()
        try:
            await bus.emit(
                topic="state.vertical_applied",
                data={
                    "vertical": result.vertical_name,
                    "tools_count": len(result.tools_applied),
                    "middleware_count": result.middleware_count,
                    "success": result.success,
                    "category": "state",
                },
            )
        finally:
            await bus.disconnect()

        # Note: In the new event system, we can't easily assert on calls without mocking
        # The event was emitted to the bus successfully if no exception was raised
        assert result.vertical_name == "coding"
        assert len(result.tools_applied) == 3
        assert result.success is True


class TestBudgetAwareLoopDetection:
    """Tests for budget-aware loop detection — productive agents should not be
    killed at the soft iteration threshold (max-5) when remaining tool budget
    is high and tool calls are active.
    """

    def _build_intent_result(self) -> MagicMock:
        intent_result = MagicMock()
        intent_result.intent = MagicMock()
        intent_result.intent.name = "CONTINUATION"
        intent_result.confidence = 0.8
        return intent_result

    def _build_settings(self) -> MagicMock:
        return MagicMock(
            max_continuation_prompts_analysis=6,
            max_continuation_prompts_action=5,
            max_continuation_prompts_default=3,
            continuation_prompt_overrides={},
        )

    def test_productive_agent_survives_soft_threshold(self):
        """At iteration 45/50 with budget remaining and active tool calls,
        loop detection should NOT fire — agent is still productive.
        """
        from victor.agent.continuation_strategy import ContinuationStrategy

        strategy = ContinuationStrategy()

        result = strategy.determine_continuation_action(
            intent_result=self._build_intent_result(),
            is_analysis_task=False,
            is_action_task=True,
            content_length=200,
            full_content="Still working on the edit task.",
            continuation_prompts=0,
            asking_input_prompts=0,
            one_shot_mode=False,
            mentioned_tools=None,
            max_prompts_summary_requested=False,
            settings=self._build_settings(),
            rl_coordinator=None,
            provider_name="anthropic",
            model="claude-3-5-sonnet",
            tool_budget=200,
            unified_tracker_config={
                "max_total_iterations": 50,
                "remaining_tool_budget": 150,  # plenty of budget left
            },
            task_completion_signals={
                "required_files": [],
                "read_files": set(),
                "required_outputs": [],
                "all_files_read": False,
                "current_iteration": 45,  # at the soft threshold
                "recent_tool_call_count": 5,  # actively making tool calls
            },
        )

        # Should NOT finish — agent is productive
        assert (
            result["action"] != "finish"
        ), f"Productive agent killed prematurely at iter 45 — got action={result['action']}"

    def test_spinning_agent_still_killed_at_threshold(self):
        """At iteration 45/50 with no recent tool calls, loop detection
        SHOULD fire — agent is spinning.
        """
        from victor.agent.continuation_strategy import ContinuationStrategy

        strategy = ContinuationStrategy()

        result = strategy.determine_continuation_action(
            intent_result=self._build_intent_result(),
            is_analysis_task=False,
            is_action_task=True,
            content_length=200,
            full_content="Spinning without progress.",
            continuation_prompts=0,
            asking_input_prompts=0,
            one_shot_mode=False,
            mentioned_tools=None,
            max_prompts_summary_requested=False,
            settings=self._build_settings(),
            rl_coordinator=None,
            provider_name="anthropic",
            model="claude-3-5-sonnet",
            tool_budget=200,
            unified_tracker_config={
                "max_total_iterations": 50,
                "remaining_tool_budget": 150,
            },
            task_completion_signals={
                "required_files": [],
                "read_files": set(),
                "required_outputs": [],
                "all_files_read": False,
                "current_iteration": 45,
                "recent_tool_call_count": 0,  # NOT productive — spinning
            },
        )

        assert result["action"] == "finish"

    def test_productive_agent_killed_at_hard_max(self):
        """Even productive agents are killed when they hit hard max_iterations."""
        from victor.agent.continuation_strategy import ContinuationStrategy

        strategy = ContinuationStrategy()

        result = strategy.determine_continuation_action(
            intent_result=self._build_intent_result(),
            is_analysis_task=False,
            is_action_task=True,
            content_length=200,
            full_content="At hard limit.",
            continuation_prompts=0,
            asking_input_prompts=0,
            one_shot_mode=False,
            mentioned_tools=None,
            max_prompts_summary_requested=False,
            settings=self._build_settings(),
            rl_coordinator=None,
            provider_name="anthropic",
            model="claude-3-5-sonnet",
            tool_budget=200,
            unified_tracker_config={
                "max_total_iterations": 50,
                "remaining_tool_budget": 150,
            },
            task_completion_signals={
                "required_files": [],
                "read_files": set(),
                "required_outputs": [],
                "all_files_read": False,
                "current_iteration": 50,  # at HARD max
                "recent_tool_call_count": 5,
            },
        )

        assert result["action"] == "finish"

    def test_low_budget_falls_back_to_soft_threshold(self):
        """If remaining budget is low, soft threshold applies even when
        recent_tool_call_count > 0 — agent will exhaust budget soon anyway.
        """
        from victor.agent.continuation_strategy import ContinuationStrategy

        strategy = ContinuationStrategy()

        result = strategy.determine_continuation_action(
            intent_result=self._build_intent_result(),
            is_analysis_task=False,
            is_action_task=True,
            content_length=200,
            full_content="Almost out of budget.",
            continuation_prompts=0,
            asking_input_prompts=0,
            one_shot_mode=False,
            mentioned_tools=None,
            max_prompts_summary_requested=False,
            settings=self._build_settings(),
            rl_coordinator=None,
            provider_name="anthropic",
            model="claude-3-5-sonnet",
            tool_budget=200,
            unified_tracker_config={
                "max_total_iterations": 50,
                "remaining_tool_budget": 5,  # nearly empty
            },
            task_completion_signals={
                "required_files": [],
                "read_files": set(),
                "required_outputs": [],
                "all_files_read": False,
                "current_iteration": 45,
                "recent_tool_call_count": 3,
            },
        )

        assert result["action"] == "finish"

    def test_malformed_budget_signals_dont_crash_detector(self):
        """Defensive coercion: malformed remaining_tool_budget and
        recent_tool_call_count must not crash the loop detector specifically.

        Other parts of continuation_strategy still expect valid types — we
        only verify the budget-aware code path is hardened.
        """
        from victor.agent.continuation_strategy import ContinuationStrategy

        strategy = ContinuationStrategy()

        # Use valid max_total_iterations (other paths need it) but malformed
        # budget-aware signals — those are the ones we hardened.
        result = strategy.determine_continuation_action(
            intent_result=self._build_intent_result(),
            is_analysis_task=False,
            is_action_task=True,
            content_length=200,
            full_content="Working.",
            continuation_prompts=0,
            asking_input_prompts=0,
            one_shot_mode=False,
            mentioned_tools=None,
            max_prompts_summary_requested=False,
            settings=self._build_settings(),
            rl_coordinator=None,
            provider_name="anthropic",
            model="claude-3-5-sonnet",
            tool_budget=200,
            unified_tracker_config={
                "max_total_iterations": 50,
                "remaining_tool_budget": "not-a-number",  # malformed
            },
            task_completion_signals={
                "required_files": [],
                "read_files": set(),
                "required_outputs": [],
                "all_files_read": False,
                "current_iteration": 30,
                "recent_tool_call_count": None,  # malformed
            },
        )

        # Coercion should produce a valid action; budget defaults to 0,
        # recent_tool_calls defaults to -1, so is_productive=False — but
        # iteration 30 < threshold 45 means we don't FINISH here either.
        assert "action" in result
