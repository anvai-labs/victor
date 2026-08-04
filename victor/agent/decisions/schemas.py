"""Pydantic models for structured LLM decision output.

Each schema is designed for minimal token usage (5-20 tokens) to keep
LLM decision calls fast and cheap. Fields use constrained types to
enable strict validation of LLM responses.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# =============================================================================
# Enums for LLM Decisions
# =============================================================================


class DecisionType(str, Enum):
    """Types of decisions the LLM decision service can make."""

    TASK_COMPLETION = "task_completion"
    INTENT_CLASSIFICATION = "intent_classification"
    TASK_TYPE_CLASSIFICATION = "task_type_classification"
    QUESTION_CLASSIFICATION = "question_classification"
    LOOP_DETECTION = "loop_detection"
    ERROR_CLASSIFICATION = "error_classification"
    CONTINUATION_ACTION = "continuation_action"
    TOOL_SELECTION = "tool_selection"
    PROMPT_FOCUS = "prompt_focus"
    STAGE_DETECTION = "stage_detection"
    SKILL_SELECTION = "skill_selection"
    MULTI_SKILL_DECOMPOSITION = "multi_skill_decomposition"
    TOOL_NECESSITY = "tool_necessity"
    COMPACTION = "compaction"
    TURN_AUDIT = "turn_audit"  # EVR-6: per-turn continue/alarm judgment


class ErrorType(str, Enum):
    """Classification of errors for retry decisions.

    Determines whether an error is retryable:
    - PERMANENT: Never retry (auth failures, invalid input)
    - TRANSIENT: Retry after delay (rate limits, timeouts)
    - RETRYABLE: Retry immediately (network blips)
    """

    PERMANENT = "permanent"  # Never retry (auth, invalid input)
    TRANSIENT = "transient"  # Retry after delay (rate limits, timeouts)
    RETRYABLE = "retryable"  # Retry immediately (network blips)


class TaskPhase(str, Enum):
    """Current phase of task execution.

    Progress tracking for task completion:
    - WORKING: Actively working on the task
    - FINALIZING: Wrapping up, last steps
    - DONE: Task complete
    - STUCK: Blocked or needs input
    """

    WORKING = "working"  # Actively working
    FINALIZING = "finalizing"  # Wrapping up
    DONE = "done"  # Task complete
    STUCK = "stuck"  # Blocked or needs input


class IntentType(str, Enum):
    """Intent classification for model responses.

    Determines what the model is trying to do:
    - CONTINUATION: Continue working on current task
    - COMPLETION: Task is complete, wrapping up
    - ASKING_INPUT: Requesting input from user
    - STUCK_LOOP: Stuck in repetitive loop
    """

    CONTINUATION = "continuation"  # Continue working
    COMPLETION = "completion"  # Task complete
    ASKING_INPUT = "asking_input"  # Requesting input
    STUCK_LOOP = "stuck_loop"  # Stuck in loop


class TaskCategoryType(str, Enum):
    """Category of task being performed.

    High-level task classification:
    - ANALYSIS: Analyzing code or data
    - ACTION: Executing commands or actions
    - GENERATION: Creating new content
    - SEARCH: Finding information
    - EDIT: Modifying existing content
    """

    ANALYSIS = "analysis"  # Analyzing code/data
    ACTION = "action"  # Executing commands
    GENERATION = "generation"  # Creating content
    SEARCH = "search"  # Finding information
    EDIT = "edit"  # Modifying content


class QuestionType(str, Enum):
    """Type of question being asked.

    Determines how to handle questions:
    - RHETORICAL: Not a real question, continue
    - CONTINUATION: Question about continuing work
    - CLARIFICATION: Needs clarification
    - INFO: Informational question
    """

    RHETORICAL = "rhetorical"  # Not a real question
    CONTINUATION = "continuation"  # About continuing work
    CLARIFICATION = "clarification"  # Needs clarification
    INFO = "info"  # Informational


class LoopType(str, Enum):
    """Type of loop detected.

    Classification of repetitive patterns:
    - STALLING: Intent without action
    - CIRCULAR: Circular reasoning
    - REPETITION: Exact repetition
    - NONE: No loop detected
    """

    STALLING = "stalling"  # Intent without action
    CIRCULAR = "circular"  # Circular reasoning
    REPETITION = "repetition"  # Exact repetition
    NONE = "none"  # No loop


class ContinuationAction(str, Enum):
    """Action to take for continuation decisions.

    Determines how to proceed:
    - CONTINUE_ASKING_INPUT: Ask a short follow-up and continue
    - RETURN_TO_USER: Return control to user
    - PROMPT_TOOL_CALL: Prompt to use tools
    - CONTINUE_WITH_SYNTHESIS_HINT: Continue with a synthesis nudge
    - REQUEST_SUMMARY: Request summary
    - REQUEST_COMPLETION: Request explicit completion
    - EXECUTE_EXTRACTED_TOOL: Execute a recovered tool call
    - FORCE_TOOL_EXECUTION: Force the model to emit a real tool call
    - FINISH: Task is complete
    """

    CONTINUE_ASKING_INPUT = "continue_asking_input"
    RETURN_TO_USER = "return_to_user"
    PROMPT_TOOL_CALL = "prompt_tool_call"
    CONTINUE_WITH_SYNTHESIS_HINT = "continue_with_synthesis_hint"
    REQUEST_SUMMARY = "request_summary"
    REQUEST_COMPLETION = "request_completion"
    EXECUTE_EXTRACTED_TOOL = "execute_extracted_tool"
    FORCE_TOOL_EXECUTION = "force_tool_execution"
    FINISH = "finish"  # Task complete


class DeliverableType(str, Enum):
    """Types of deliverables a task can produce.

    Expected outputs from task execution:
    - FILE_CREATED: New file created
    - FILE_MODIFIED: Existing file modified
    - ANALYSIS_PROVIDED: Analysis output provided
    - ANSWER_PROVIDED: Answer to question
    - PLAN_PROVIDED: Plan created
    - CODE_EXECUTED: Code was executed
    """

    FILE_CREATED = "file_created"  # New file created
    FILE_MODIFIED = "file_modified"  # File modified
    ANALYSIS_PROVIDED = "analysis_provided"  # Analysis output
    ANSWER_PROVIDED = "answer_provided"  # Answer provided
    PLAN_PROVIDED = "plan_provided"  # Plan created
    CODE_EXECUTED = "code_executed"  # Code executed


class ConversationStage(str, Enum):
    """Stage in conversation execution.

    Progress tracking through conversation lifecycle:
    - INITIAL: Starting point
    - PLANNING: Creating plan
    - READING: Reading files
    - ANALYSIS: Analyzing information
    - EXECUTION: Executing actions
    - VERIFICATION: Verifying results
    """

    INITIAL = "initial"  # Starting point
    PLANNING = "planning"  # Creating plan
    READING = "reading"  # Reading files
    ANALYSIS = "analysis"  # Analyzing information
    EXECUTION = "execution"  # Executing actions
    VERIFICATION = "verification"  # Verifying results


class ComplexityLevel(str, Enum):
    """Task complexity level.

    Based on conversation length:
    - SIMPLE: ≤8 messages
    - COMPLEX: >8 messages
    """

    SIMPLE = "simple"  # ≤8 messages
    COMPLEX = "complex"  # >8 messages


class TierType(str, Enum):
    """Recommended tier for handling.

    Which tier should handle the request:
    - EDGE: Fast tier for simple requests
    - BALANCED: Balanced tier for normal requests
    - PERFORMANCE: Powerful tier for complex requests
    """

    EDGE = "edge"  # Fast tier
    BALANCED = "balanced"  # Balanced tier
    PERFORMANCE = "performance"  # Powerful tier


class TaskCompletionDecision(BaseModel):
    """Is the task done?"""

    is_complete: bool = Field(description="Whether the task appears complete")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the assessment")
    phase: TaskPhase = Field(
        default=TaskPhase.WORKING, description="Current phase of task execution"
    )


class TurnAuditDecision(BaseModel):
    """EVR-6: is this agent turn healthy, or should the loop be alarmed?"""

    verdict: Literal["continue", "alarm"] = Field(
        description="continue = healthy turn; alarm = degenerate/stuck/off-track turn"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the verdict")
    reason: str = Field(default="", description="Brief reason when the verdict is alarm")


class IntentDecision(BaseModel):
    """What is the model doing?"""

    intent: IntentType = Field(description="Classified intent of the model's response")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the classification")


class TaskTypeDecision(BaseModel):
    """What kind of task is this, and what deliverables are expected?"""

    task_type: TaskCategoryType = Field(description="Classified type of the task")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the classification")
    deliverables: list[DeliverableType] = Field(
        default_factory=list,
        description="Expected deliverable types (empty = infer from task_type)",
    )


class QuestionTypeDecision(BaseModel):
    """Should we auto-continue past this question?"""

    question_type: QuestionType = Field(description="Type of question being asked")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the classification")


class LoopDetection(BaseModel):
    """Is the model stuck in a loop?"""

    is_loop: bool = Field(description="Whether a loop is detected")
    loop_type: LoopType = Field(default=LoopType.NONE, description="Type of loop detected")


class ErrorClassDecision(BaseModel):
    """Can we retry this error?"""

    error_type: ErrorType = Field(
        default=ErrorType.TRANSIENT, description="Classification of the error"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the classification")


class ToolSelectionDecision(BaseModel):
    """Which tools are most relevant for this task?"""

    tools: list[str] = Field(description="Selected tool names")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in selection")


class PromptFocusDecision(BaseModel):
    """Which system prompt sections to include?"""

    sections: list[str] = Field(description="Selected section names")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in selection")


class StageDetectionDecision(BaseModel):
    """What conversation stage is this?"""

    stage: ConversationStage = Field(
        default=ConversationStage.INITIAL, description="Detected conversation stage"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in detection")


class ToolNecessityDecision(BaseModel):
    """Does this request require tools?"""

    requires_tools: bool = Field(description="Whether tools/file operations are needed")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in assessment")


class ContinuationDecision(BaseModel):
    """What action should we take next?"""

    action: ContinuationAction = Field(description="Recommended next action")
    reason: str = Field(max_length=100, description="Brief reason for the recommendation")


class SkillSelectionDecision(BaseModel):
    """Which skill best matches this user request?"""

    skill: str = Field(description="Selected skill name, or empty string if none match")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in selection")


class MultiSkillDecision(BaseModel):
    """How to decompose a complex request into multiple skills?"""

    skills: list[str] = Field(description="Ordered list of skill names to execute")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in decomposition")


class CompactionDecision(BaseModel):
    """Decision for context compaction routing.

    Determines which tier should handle compaction based on complexity
    and estimated token count. Simple compaction (≤8 messages) can use
    edge tier for speed, while complex compaction (>8 messages) should
    use performance tier for quality.

    Attributes:
        complexity: simple (≤8 messages) or complex (>8 messages)
        recommended_tier: Which tier should handle compaction (edge, balanced, performance)
        estimated_tokens: Estimated size of content to compact
        confidence: Decision confidence (0.0-1.0)
        reason: Brief explanation of the decision
    """

    complexity: ComplexityLevel = Field(
        default=ComplexityLevel.SIMPLE,
        description="simple (≤8 messages) or complex (>8 messages)",
    )
    recommended_tier: TierType = Field(
        default=TierType.BALANCED, description="Recommended tier for compaction"
    )
    estimated_tokens: int = Field(description="Estimated token count of content to compact")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the decision")
    reason: str = Field(max_length=200, description="Brief explanation of the decision")


class SystemPromptOptimizationDecision(BaseModel):
    """Decision for system prompt optimization during compaction.

    When system_prompt_strategy is 'dynamic', this decision determines
    which prompt sections to include based on current context and task state.

    Attributes:
        include_sections: Which prompt sections to include (subset of all sections)
        add_context_reminder: Whether to add compaction summary reminder
        add_failure_hints: Whether to include recent failure patterns
        adjust_for_complexity: Tailor prompt for task complexity
        confidence: Decision confidence (0.0-1.0)
        reason: Brief explanation of the decision
    """

    include_sections: list[str] = Field(
        description="List of section names to include in the prompt"
    )
    add_context_reminder: bool = Field(
        description="Whether to add a reminder about recent compaction"
    )
    add_failure_hints: bool = Field(description="Whether to include hints based on recent failures")
    adjust_for_complexity: bool = Field(
        description="Whether to tailor prompt based on task complexity"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the decision")
    reason: str = Field(max_length=200, description="Brief explanation of the decision")
