"""Prompt templates for LLM decision calls.

Each decision type maps to a focused system prompt and user template
designed to elicit a minimal JSON response (5-20 tokens).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

from pydantic import BaseModel

from victor.agent.decisions.schemas import (
    CompactionDecision,
    ContinuationDecision,
    DecisionType,
    ErrorClassDecision,
    IntentDecision,
    LoopDetection,
    MultiSkillDecision,
    PromptFocusDecision,
    QuestionTypeDecision,
    SkillSelectionDecision,
    StageDetectionDecision,
    TaskCompletionDecision,
    TaskTypeDecision,
    ToolNecessityDecision,
    ToolSelectionDecision,
    TurnAuditDecision,
)


@dataclass(frozen=True)
class DecisionPrompt:
    """Configuration for a single decision type's LLM call."""

    system: str
    user_template: str
    schema: Type[BaseModel]
    max_tokens: int


DECISION_PROMPTS: Dict[DecisionType, DecisionPrompt] = {
    DecisionType.TASK_COMPLETION: DecisionPrompt(
        system=(
            "You are a task completion classifier. "
            "Respond ONLY with a JSON object, no other text."
        ),
        user_template=(
            "Given this AI assistant response tail and context, is the task complete?\n\n"
            "Response tail:\n{response_tail}\n\n"
            "Deliverables found: {deliverable_count}\n"
            "Completion signals: {signal_count}\n\n"
            'Respond with JSON: {{"is_complete": bool, "confidence": 0.0-1.0, '
            '"phase": "working"|"finalizing"|"done"|"stuck"}}'
        ),
        schema=TaskCompletionDecision,
        max_tokens=30,
    ),
    DecisionType.INTENT_CLASSIFICATION: DecisionPrompt(
        system=(
            "You are an intent classifier for AI assistant responses. "
            "Respond ONLY with a JSON object, no other text."
        ),
        user_template=(
            "Classify the intent of this AI response:\n\n"
            "Response tail:\n{text_tail}\n\n"
            "Has tool calls: {has_tool_calls}\n\n"
            'Respond with JSON: {{"intent": '
            '"continuation"|"completion"|"asking_input"|"stuck_loop", '
            '"confidence": 0.0-1.0}}'
        ),
        schema=IntentDecision,
        max_tokens=20,
    ),
    DecisionType.TASK_TYPE_CLASSIFICATION: DecisionPrompt(
        system=(
            "You classify user requests for an AI coding assistant. "
            "Respond ONLY with a JSON object, no other text."
        ),
        user_template=(
            "Classify this user request:\n\n"
            "{message_excerpt}\n\n"
            "Task types: analysis (review/investigate), action (fix/modify), "
            "generation (create new), search (find/locate), edit (refactor/update)\n\n"
            "Deliverables (pick all that apply): "
            "file_modified, file_created, analysis_provided, "
            "answer_provided, plan_provided, code_executed\n\n"
            'Respond with JSON: {{"task_type": "...", "confidence": 0.0-1.0, '
            '"deliverables": ["..."]}}'
        ),
        schema=TaskTypeDecision,
        max_tokens=50,
    ),
    DecisionType.QUESTION_CLASSIFICATION: DecisionPrompt(
        system=(
            "You are a question type classifier. " "Respond ONLY with a JSON object, no other text."
        ),
        user_template=(
            "Classify this question from an AI assistant:\n\n"
            "{question_text}\n\n"
            'Respond with JSON: {{"question_type": '
            '"rhetorical"|"continuation"|"clarification"|"info", '
            '"confidence": 0.0-1.0}}'
        ),
        schema=QuestionTypeDecision,
        max_tokens=20,
    ),
    DecisionType.LOOP_DETECTION: DecisionPrompt(
        system=(
            "You are a loop detector for AI assistant behavior. "
            "Respond ONLY with a JSON object, no other text."
        ),
        user_template=(
            "Is this AI assistant stuck in a loop?\n\n"
            "Recent content:\n{content_excerpt}\n\n"
            "Recent thinking blocks: {recent_blocks}\n\n"
            'Respond with JSON: {{"is_loop": bool, '
            '"loop_type": "stalling"|"circular"|"repetition"|"none"}}'
        ),
        schema=LoopDetection,
        max_tokens=20,
    ),
    DecisionType.ERROR_CLASSIFICATION: DecisionPrompt(
        system=("You are an error classifier. " "Respond ONLY with a JSON object, no other text."),
        user_template=(
            "Classify this error:\n\n"
            "{error_message}\n\n"
            'Respond with JSON: {{"error_type": '
            '"permanent"|"transient"|"retryable", '
            '"confidence": 0.0-1.0}}'
        ),
        schema=ErrorClassDecision,
        max_tokens=20,
    ),
    DecisionType.CONTINUATION_ACTION: DecisionPrompt(
        system=(
            "You are a continuation action selector for an AI assistant. "
            "Respond ONLY with a JSON object, no other text."
        ),
        user_template=(
            "What should the AI assistant do next?\n\n"
            "Response excerpt:\n{response_excerpt}\n\n"
            "Continuation prompts sent: {continuation_prompts}\n"
            "Task type: {task_type}\n\n"
            'Respond with JSON: {{"action": '
            '"finish"|"prompt_tool_call"|"request_summary"|"return_to_user", '
            '"reason": "brief reason"}}'
        ),
        schema=ContinuationDecision,
        max_tokens=40,
    ),
    DecisionType.TOOL_SELECTION: DecisionPrompt(
        system="You select tools for a coding assistant. Respond ONLY with JSON.",
        user_template=(
            "Pick the most relevant tools for this request.\n\n"
            "Request: {message_excerpt}\n"
            "Stage: {stage}\n"
            "Available: {available_tools}\n\n"
            'JSON: {{"tools": ["tool1", "tool2", ...], "confidence": 0.0-1.0}}'
        ),
        schema=ToolSelectionDecision,
        max_tokens=60,
    ),
    DecisionType.PROMPT_FOCUS: DecisionPrompt(
        system="You optimize system prompts. Respond ONLY with JSON.",
        user_template=(
            "Select prompt sections needed for this task.\n\n"
            "Task type: {task_type}\n"
            "Request: {message_excerpt}\n"
            "Available: {available_sections}\n\n"
            'JSON: {{"sections": ["section1", ...], "confidence": 0.0-1.0}}'
        ),
        schema=PromptFocusDecision,
        max_tokens=40,
    ),
    DecisionType.STAGE_DETECTION: DecisionPrompt(
        system="You classify conversation stages. Respond ONLY with JSON.",
        user_template=(
            "What stage is this conversation in?\n\n"
            "Current stage: {current_stage}\n"
            "Recent tools: {last_tools}\n"
            "Files read: {files_observed}, Files modified: {files_modified}\n"
            "Tool calls so far: {tool_call_count}\n"
            "Heuristic guess: {detected_stage_heuristic}\n"
            "Request: {message_excerpt}\n\n"
            "Stages: initial, planning, reading, analysis, execution, verification\n"
            "Rule: Stay in execution if files are being read/edited actively.\n\n"
            'JSON: {{"stage": "...", "confidence": 0.0-1.0}}'
        ),
        schema=StageDetectionDecision,
        max_tokens=25,
    ),
    DecisionType.SKILL_SELECTION: DecisionPrompt(
        system=(
            "You match user requests to available skills. "
            "Respond ONLY with a JSON object, no other text."
        ),
        user_template=(
            "Which skill best matches this request?\n\n"
            "Request: {message_excerpt}\n\n"
            "Available skills: {available_skills}\n\n"
            'Respond: {{"skill": "name_or_empty", "confidence": 0.0-1.0}}'
        ),
        schema=SkillSelectionDecision,
        max_tokens=30,
    ),
    DecisionType.MULTI_SKILL_DECOMPOSITION: DecisionPrompt(
        system=(
            "You decompose complex requests into ordered skill sequences. "
            "Respond ONLY with a JSON object, no other text."
        ),
        user_template=(
            "Decompose this request into skills to execute in order:\n\n"
            "Request: {message_excerpt}\n\n"
            "Available skills: {available_skills}\n\n"
            'Respond: {{"skills": ["skill1", "skill2"], "confidence": 0.0-1.0}}'
        ),
        schema=MultiSkillDecision,
        max_tokens=50,
    ),
    DecisionType.TOOL_NECESSITY: DecisionPrompt(
        system="You assess if a coding request needs file/tool operations. Respond JSON only.",
        user_template=(
            "Does this need tools (file read/write/search/execute)?\n\n"
            "Request: {message_excerpt}\n\n"
            'JSON: {{"requires_tools": true/false, "confidence": 0.0-1.0}}'
        ),
        schema=ToolNecessityDecision,
        max_tokens=30,
    ),
    DecisionType.COMPACTION: DecisionPrompt(
        system="You classify conversation complexity for context compaction. Respond JSON only.",
        user_template=(
            "Classify this conversation for compaction routing.\n\n"
            "Message count: {message_count}\n"
            "Estimated tokens: {estimated_tokens}\n"
            "Task complexity: {task_complexity}\n\n"
            "Complexity: simple (≤8 messages) or complex (>8 messages)\n"
            "Tiers: edge (fast), balanced (normal), performance (quality)\n\n"
            'JSON: {{"complexity": "simple"|"complex", '
            '"recommended_tier": "edge"|"balanced"|"performance", '
            '"estimated_tokens": int, "confidence": 0.0-1.0, "reason": "brief explanation"}}'
        ),
        schema=CompactionDecision,
        max_tokens=50,
    ),
    DecisionType.TURN_AUDIT: DecisionPrompt(
        system=(
            "You are a per-turn auditor for an AI coding agent. Judge whether the agent's latest "
            "turn is healthy progress or is degenerate/stuck/off-track. "
            "Respond ONLY with a JSON object, no other text."
        ),
        user_template=(
            "Judge the agent's latest turn.\n\n"
            "Task context: {task_context}\n\n"
            "Recent trace prefix (oldest to newest):\n{recent_prefix}\n\n"
            "Turn output (tail):\n{turn_tail}\n\n"
            "Made tool calls: {has_tool_calls}\n"
            "All tool calls failed: {all_tools_failed}\n\n"
            "Alarm if the turn is degenerate (claims progress but did nothing), stuck in a loop, "
            "or clearly off-track; otherwise continue.\n"
            'Respond with JSON: {{"verdict": "continue"|"alarm", "confidence": 0.0-1.0, '
            '"reason": "brief reason if alarm"}}'
        ),
        schema=TurnAuditDecision,
        max_tokens=128,
    ),
}
