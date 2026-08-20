from __future__ import annotations

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

"""Action Authorization via Intent Detection.

This module determines whether user requests authorize file modifications:
- File writes when user only asked to "show" or "create" code
- Destructive operations without explicit consent

Purpose: Action Authorization (can the agent write files?)
Note: For model continuation detection (is the model done?),
      see victor.embeddings.intent_classifier (IntentClassifier).

Design Principles:
- Default to safe (display-only) when intent is ambiguous
- Explicit write signals required for file modifications
- Extensible pattern system for custom intents
- Clear audit trail for authorization decisions

Aliases: ActionAuthorizer = IntentDetector (for semantic clarity)
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, List, Optional, Set, Tuple

from victor.agent.safety import get_write_tool_names
from victor.tools.core_tool_aliases import canonicalize_core_tool_name
from victor.tools.tool_names import ToolNames, get_canonical_name

logger = logging.getLogger(__name__)


READ_TOOL_ALIASES = frozenset({ToolNames.READ, "read_file"})
LIST_TOOL_ALIASES = frozenset({ToolNames.LS, "list_directory"})
WRITE_TOOL_ALIASES = frozenset({ToolNames.WRITE, "write_file", "create_file"})
EDIT_TOOL_ALIASES = frozenset({ToolNames.EDIT, "edit_files", "edit_file", "patch_file"})
SHELL_TOOL_ALIASES = frozenset({ToolNames.SHELL, "execute_bash", "bash"})


def _canonicalize_tool_set(tools: Set[str]) -> Set[str]:
    """Canonicalize only the compact file/shell tool aliases."""
    return {canonicalize_core_tool_name(tool) for tool in tools}


def _canonicalize_tool_name(tool_name: str) -> str:
    """Canonicalize compact file/shell aliases for shared policy lookups."""
    return canonicalize_core_tool_name(tool_name)


def normalize_tool_name_for_policy(tool_name: str) -> str:
    """Normalize a tool name for shared policy/classification lookups."""
    return get_canonical_name(_canonicalize_tool_name(tool_name))


class ActionIntent(Enum):
    """User intent regarding actions.

    Values:
        DISPLAY_ONLY: User wants to see output but not modify files
        WRITE_ALLOWED: User explicitly requested file modifications
        AMBIGUOUS: Intent unclear, requires clarification or safe default
        READ_ONLY: User is only asking for information, no generation
    """

    DISPLAY_ONLY = "display_only"
    WRITE_ALLOWED = "write_allowed"
    AMBIGUOUS = "ambiguous"
    READ_ONLY = "read_only"


# =============================================================================
# Tool Categories for Intent-Based Filtering
# =============================================================================
# These constants define which tools are allowed/blocked for each intent.
# Single source of truth for tool filtering - update here when adding new tools.

# Tools that modify files - blocked for DISPLAY_ONLY and READ_ONLY intents
# IMPORTANT: Keep in sync with victor.agent.safety.WRITE_TOOL_NAMES
WRITE_TOOLS: frozenset[str] = frozenset(
    {
        # Direct file modifications
        ToolNames.WRITE,
        ToolNames.EDIT,
        "notebook_edit",
        # Patch/diff application
        "apply_patch",  # patch_tool.py
        # Bash execution (can modify files)
        ToolNames.SHELL,
        # Git write operations
        "git",  # git_tool.py (commit, push, etc.)
        # Refactoring (modifies files)
        "refactor_rename_symbol",
        "refactor_extract_function",
        "refactor_inline_variable",
        "refactor_organize_imports",
        "rename_symbol",
        # Scaffolding (creates files)
        "scaffold",
        # Batch operations
        "batch",
    }
)

# Tools that are safe for all intents (read-only operations)
READ_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        ToolNames.READ,
        ToolNames.LS,
        "code_search",
        "semantic_code_search",
        "grep_search",
        "find_files",
        "git_status",
        "git_log",
        "git_diff",
        "analyze_code",
        "analyze_docs",
        "web_search",
        "web_fetch",
    }
)

# Tools blocked for READ_ONLY intent (no code generation at all)
GENERATION_TOOLS: frozenset[str] = frozenset(
    {
        "generate_code",
        "generate_docs",
        "refactor_code",
    }
)

# Git read-only subcommands are sourced from bash.py — the authority for actual
# command-level read-only validation — so this intent signal cannot drift from what the
# shell tool will actually permit. (The binary/test signals below are intentionally a
# narrow NL-intent subset, NOT bash's full read-only set, whose common-word entries like
# "file"/"find"/"test" would false-positive when scanning natural-language messages.)
try:
    from victor.tools.bash import GIT_READONLY_SUBCOMMANDS as _GIT_READONLY_SUBCOMMANDS
except Exception:  # pragma: no cover - defensive fallback if bash is unavailable at import
    _GIT_READONLY_SUBCOMMANDS = {"status", "log", "show", "branch", "tag", "diff"}

_GIT_READONLY_SIGNAL = r"\bgit\s+(" + "|".join(sorted(_GIT_READONLY_SUBCOMMANDS)) + r")\b"

# Signals that indicate the user explicitly wants readonly shell/SQLite inspection.
READONLY_SHELL_SIGNALS: List[Tuple[str, str]] = [
    (r"\b(sqlite|sqlite3|sqllite|sqllite3)\b", "sqlite"),
    (r"\b(ls|list|dir|find|grep|cat|head|tail|wc|stat|file)\b", "read_only_binary"),
    (_GIT_READONLY_SIGNAL, "git_read_only"),
    (r"\b(pytest|npm\s+test|cargo\s+test|go\s+test|tox)\b", "test_execution"),
    (
        r"\b(use|run|query|inspect|review|check|look\s+at)\b.*\b(shell|bash|terminal|sqlite3?)\b",
        "explicit_shell_request",
    ),
    (
        r"\b(shell|bash|terminal)\b.*\b(read|review|inspect|query|check|look\s+at)\b",
        "explicit_shell_request_reverse",
    ),
    # Database-related signals - allow shell for database operations
    (
        r"\b(read|query|inspect|review|check|view|explore)\b.*\b(database|db|table|schema)\b",
        "database_read",
    ),
    (
        r"\b(database|db|table|schema)\b.*\b(read|query|inspect|review|check|view|list)\b",
        "database_read_reverse",
    ),
    (r"\b(list|show)\s+(all\s+)?(tables|rows|data)\b", "database_list"),
    (r"\b\.tables\b", "sqlite_dot_tables"),
    (r"\bSELECT\b.*\bFROM\b", "sql_query"),
    # File extensions that indicate databases
    (r"\.(db|sqlite|sqlite3)\b", "database_file"),
]

CONTINUATION_KEYWORDS: tuple[str, ...] = (
    "continue please",
    "keep it up",
    "keep going",
    "go ahead",
    "carry on",
    "yes please",
    "resume",
    "continue",
    "proceed",
    "go on",
    "okay",
    "next",
    "do it",
    "yes",
    "ok",
    "go",
)

_CONTINUATION_CONNECTORS: tuple[str, ...] = (
    "to ",
    "and ",
    "with ",
    "by ",
    "on ",
    "please ",
)

# Mapping of intent to blocked tool sets
INTENT_BLOCKED_TOOLS: dict[ActionIntent, frozenset[str]] = {
    ActionIntent.DISPLAY_ONLY: frozenset(),  # [PERMISSIVE] Allow tools, rely on prompt guard
    ActionIntent.READ_ONLY: frozenset(),  # [PERMISSIVE] Allow tools, rely on prompt guard
    ActionIntent.WRITE_ALLOWED: frozenset(),  # No restrictions
    ActionIntent.AMBIGUOUS: frozenset(),  # Rely on prompt guard
}


@dataclass
class IntentClassification:
    """Result of intent detection.

    Attributes:
        intent: The detected user intent
        confidence: Confidence score (0.0-1.0)
        matched_signals: List of signal names that matched
        safe_actions: Set of action categories that are safe to perform
        prompt_guard: Text to inject into system prompt if needed
    """

    intent: ActionIntent
    confidence: float
    matched_signals: List[str]
    safe_actions: Set[str]
    prompt_guard: str


READ_ONLY_INTENTS: frozenset[ActionIntent] = frozenset(
    {ActionIntent.DISPLAY_ONLY, ActionIntent.READ_ONLY}
)


def _coerce_action_intent(intent: Optional[object]) -> Optional[ActionIntent]:
    """Normalize enums, strings, or mock-like objects into ActionIntent."""
    if isinstance(intent, ActionIntent):
        return intent
    if isinstance(intent, str):
        try:
            return ActionIntent(intent.lower())
        except ValueError:
            try:
                return ActionIntent[intent.upper()]
            except KeyError:
                return None
    intent_name = getattr(intent, "name", None)
    if isinstance(intent_name, str):
        try:
            return ActionIntent[intent_name.upper()]
        except KeyError:
            return None
    return None


def get_write_tools_for_policy() -> frozenset[str]:
    """Return the canonical write/execute tool set used by intent policy."""
    return build_write_tool_set()


def build_write_tool_set(*extra_tool_names: str) -> frozenset[str]:
    """Build a shared write-tool set with canonical and compatibility spellings."""
    tool_names: Set[str] = set(get_write_tool_names()) | set(WRITE_TOOLS)
    tool_names.update(extra_tool_names)
    normalized = {normalize_tool_name_for_policy(name) for name in tool_names if name}
    raw = {name for name in tool_names if name}
    canonicalized = {_canonicalize_tool_name(name) for name in tool_names if name}
    return frozenset(normalized | raw | canonicalized)


def get_intent_blocked_tools(intent: ActionIntent) -> frozenset[str]:
    """Return blocked tools for an intent using shared, registry-aware policy."""
    intent = _coerce_action_intent(intent)
    if intent is None:
        return frozenset()
    if intent == ActionIntent.DISPLAY_ONLY:
        return get_write_tools_for_policy()
    if intent == ActionIntent.READ_ONLY:
        return frozenset(set(get_write_tools_for_policy()) | set(GENERATION_TOOLS))
    return frozenset()


def has_explicit_readonly_shell_request(message: Optional[str]) -> bool:
    """Return True when the user explicitly requests readonly shell/SQLite access."""
    if not message:
        return False
    lowered = message.lower()
    return any(re.search(pattern, lowered, re.IGNORECASE) for pattern, _ in READONLY_SHELL_SIGNALS)


def split_continuation_request(message: Optional[str]) -> tuple[bool, str]:
    """Return whether the message is a continuation request and its trailing payload."""
    if not message:
        return False, ""

    stripped = message.strip()
    lowered = stripped.lower()

    for keyword in CONTINUATION_KEYWORDS:
        if lowered == keyword:
            return True, ""

        if not lowered.startswith(keyword):
            continue

        next_char_index = len(keyword)
        if next_char_index >= len(lowered):
            return True, ""

        next_char = lowered[next_char_index]
        if next_char.isalnum():
            continue

        remainder = stripped[next_char_index:].lstrip(" \t,.:;-")
        lowered_remainder = remainder.lower()
        for connector in _CONTINUATION_CONNECTORS:
            if lowered_remainder.startswith(connector):
                remainder = remainder[len(connector) :].lstrip()
                break
        return True, remainder

    return False, ""


def is_continuation_request(message: Optional[str]) -> bool:
    """Return True when the message resumes or carries forward the prior task."""
    is_continuation, _ = split_continuation_request(message)
    return is_continuation


def should_allow_shell_for_read_only_intent(
    intent: Optional[ActionIntent],
    message: Optional[str],
) -> bool:
    """Allow the single shell tool on read-only turns when the user explicitly asks for it."""
    intent = _coerce_action_intent(intent)
    if intent not in READ_ONLY_INTENTS:
        return False
    return has_explicit_readonly_shell_request(message)


def is_tool_blocked_for_intent(
    tool_name: str,
    intent: Optional[ActionIntent],
    user_message: Optional[str] = None,
) -> bool:
    """Check whether a tool is blocked for a given intent under shared policy."""
    intent = _coerce_action_intent(intent)
    if intent is None:
        return False
    canonical_tool_name = normalize_tool_name_for_policy(tool_name)
    if canonical_tool_name == ToolNames.SHELL and should_allow_shell_for_read_only_intent(
        intent, user_message
    ):
        return False
    return canonical_tool_name in get_intent_blocked_tools(intent)


# Signals that indicate "display only" intent
DISPLAY_SIGNALS: List[Tuple[str, float, str]] = [
    (r"\bshow\s+me\b", 1.0, "show_me"),
    (r"\bgive\s+me\s+(an?\s+)?(example|code)\b", 0.9, "give_example"),
    (
        r"\bcreate\s+(a|an)\s+\w*\s*(function|class|method|script)\b",
        0.8,
        "create_function",
    ),
    (r"\bwrite\s+(a|an)\s+\w*\s*(function|class|method)\b", 0.7, "write_function"),
    (r"\bhow\s+(do|would)\s+(i|you)\s+(write|create|implement)\b", 0.9, "how_to_write"),
    (r"\bwhat\s+(does|would)\s+\w+\s+look\s+like\b", 0.8, "what_looks_like"),
    (r"\bjust\s+(show|display|print)\b", 1.0, "just_show"),
    (
        r"\bdon'?t\s+(save|write|create)\s+(any\s+)?(files?|to\s+disk)\b",
        1.0,
        "dont_write",
    ),
    (r"\bdisplay\s+(only|it)\b", 1.0, "display_only"),
    (r"\bwithout\s+(saving|writing)\b", 0.9, "without_saving"),
]

# Signals that explicitly authorize file writing
WRITE_SIGNALS: List[Tuple[str, float, str]] = [
    (r"\b(save|write)\s+(\w+\s+)?(to|as)\s+\w+", 1.0, "save_to_file"),
    (r"\bsave\s+(this|it)\s+(to|as)\b", 1.0, "save_this_to"),
    (r"\bcreate\s+(a\s+)?(new\s+)?file\b", 1.0, "create_file"),
    (
        r"\b(add|put|place)\s+(\w+\s+)?(in|into)\s+\w+\.(py|js|ts|java|cpp|go|rs)",
        0.9,
        "add_to_file",
    ),
    (
        r"\badd\s+(this|it)\s+(function\s+)?(into|to)\s+\w+\.(py|js|ts)",
        0.9,
        "add_into_file",
    ),
    (r"\bupdate\s+(the\s+)?file\b", 0.9, "update_file"),
    (r"\bmodify\s+(the\s+)?\w+\.(py|js|ts|java|cpp)", 0.9, "modify_file"),
    (r"\b(edit|change)\s+(the\s+)?file\b", 0.8, "edit_file"),
    (r"\bsave\s+changes\b", 0.9, "save_changes"),
    (r"\bwrite\s+(this|it)\s+to\s+disk\b", 1.0, "write_to_disk"),
    (r"\badd\s+(this\s+)?to\s+\w+\.(py|js|ts)", 0.8, "add_to_ext"),
    (
        r"\bimplement\s+(this\s+)?(in|into)\s+\w+\.(py|js|ts)",
        0.8,
        "implement_into_file",
    ),
    # "Address/handle/tackle the rest/findings/issues" - action-oriented continuation
    (
        r"\b(a(?:d+ress|dress)|handle|tackle)\s+(the\s+)?(rest\s+of\s+)?"
        r"(findings|issues|concerns|problems|tasks|gaps|bugs)\b",
        0.95,
        "address_findings",
    ),
    # Generic "address/handle/fix X" patterns
    (
        r"\b(a(?:d+ress|dress)|handle|tackle|resolve)\s+\w+",
        0.85,
        "address_generic",
    ),
    # Generic write-action keywords as a broad signal
    (
        r"\b(fix|update|modify|edit|change|patch|apply|implement|add)\b",
        0.7,
        "generic_write_intent",
    ),
    # "gitignore X" as a verb (add to .gitignore)
    (
        r"\bgitignore\s+[\w\./]+",
        0.9,
        "gitignore_files",
    ),
    # "consolidate/merge/eliminate/deduplicate" - refactoring actions
    (
        r"\b(consolidate|merge|eliminate|deduplicate|remove|reduce)\s+\w+",
        0.85,
        "consolidate_action",
    ),
    (
        r"\b(fix(?:ing)?|update(?:ing)?|modify(?:ing)?|edit(?:ing)?|implement(?:ing)?|"
        r"apply(?:ing)?|finish(?:ing)?|complete(?:ing)?)\b.*\b(remaining|rest|last|final|"
        r"these|those|the)\b.*\b(file|files|module|modules|artifact|artifacts|task|tasks)\b",
        0.9,
        "complete_remaining_artifacts",
    ),
]

# Signals that indicate read-only intent (no generation)
READ_ONLY_SIGNALS: List[Tuple[str, float, str]] = [
    (r"\b(list|show)\s+(all\s+)?(files?|directories?)\b", 0.9, "list_files"),
    (r"\bwhat\s+(are|is)\s+(in|inside)\b", 0.8, "what_is_in"),
    (r"\bexplain\s+(the\s+)?(file|code|function)\b", 0.9, "explain"),
    (r"\bsummarize\b", 0.9, "summarize"),
    (r"\bdescribe\b", 0.8, "describe"),
    (r"\bread\s+(and\s+)?\w+\b", 0.7, "read"),
    (r"\bgit\s+(status|log|branch|diff)\b", 1.0, "git_read"),
]

# Compound signals: "analyze AND fix" patterns that authorize writes after analysis
# These take precedence over READ_ONLY signals when both analysis and action words present
COMPOUND_WRITE_SIGNALS: List[Tuple[str, float, str]] = [
    # Analyze/review + fix/update/modify
    (
        r"\b(analyze|review|check|find)\b.*\b(and|then)\s+(fix|update|modify|correct|improve)\b",
        1.0,
        "analyze_then_fix",
    ),
    # "Identify/find areas for consolidation/improvement" - refactoring context
    (
        r"\b(identify|find|locate|detect)\b.*\b(for)\s+(consolidation|refactoring|improvement|optimization|reduction)\b",
        0.95,
        "identify_for_improvement",
    ),
    # "Identify/find areas that need/require/should (be) X" - implies action should be taken
    # Matches verb forms: consolidate, consolidated, consolidating; remove, removed, removal; etc.
    (
        r"\b(identify|find|locate|detect)\b.*?\s+(that\s+)?(needs?|requires?|should(\s+be)?)\s+(?:consolidat|refactor|improv|fix|remov|eliminat|reduc)(?:e|ed|ing|ement|ement|al|ation|ion|tion)?\b",
        0.9,
        "identify_needing_action",
    ),
    # "Review... and consolidate/eliminate/remove/reduce"
    (
        r"\b(review|analyze|examine)\b.*\b(and\s+)?(consolidate|eliminate|remove|reduce|deduplicate|merge)\b",
        0.9,
        "review_and_consolidate",
    ),
    # =============================================================================
    # Review + output/documentation patterns - when review results need to be written
    # =============================================================================
    # "Review... and document/save/report/output the findings"
    (
        r"\b(review|analyze|examine|audit|inspect)\b.*\b(and|then)\s+(document|save|report|output|write)\b",
        0.95,
        "review_and_document",
    ),
    # "Generate/create a review/analysis report/document"
    (
        r"\b(generate|create|write|produce)\s+(a\s+)?(review|analysis|audit|summary)\s+(report|document|documentation)\b",
        0.95,
        "generate_review_report",
    ),
    # "Document the findings/issues/results"
    (
        r"\b(document|write|record|log)\s+(the\s+)?(findings|issues|results|analysis|review)\b",
        0.9,
        "document_findings",
    ),
    # "Save/output findings to file"
    (
        r"\b(save|output|export)\s+(the\s+)?(findings|results|analysis|review)\s+(to|as)\b",
        0.9,
        "save_findings_to_file",
    ),
    # "Write up the review/analysis/findings"
    (
        r"\bwrite\s+(up\s+)?(the\s+)?(review|analysis|findings|results)\b",
        0.85,
        "write_up_findings",
    ),
    # "Create summary/document of review"
    (
        r"\b(create|make)\s+(a\s+)?(summary|document|report)\s+(of|for|from)\s+(the\s+)?(review|analysis|audit)\b",
        0.85,
        "create_summary_document",
    ),
    (
        r"\b(review|analyze|find|gather)\b.*\b(apply|implement)\s+(the\s+)?fix(?:es)?\b",
        1.0,
        "analyze_then_apply_fixes",
    ),
    (
        r"\b(identify|detect|find)\b.*\b(and\s+)?(fix|correct|resolve)\b",
        1.0,
        "find_and_fix",
    ),
    (
        r"\b(fix|correct|resolve)\s+(any|all|the)?\s*(bugs?|issues?|errors?|problems?)\b",
        0.9,
        "fix_bugs",
    ),
    # "Fix the bug in X" patterns
    (r"\bfix\s+(the\s+)?(bug|issue|error|problem)\s+(in|with)\b", 0.9, "fix_in"),
    # "Apply the fix/changes" patterns
    (
        r"\b(apply|implement)\s+(the\s+)?(fix(?:es)?|changes?|improvements?)\b",
        0.9,
        "apply_fix",
    ),
    # "Make the changes" / "make it work"
    (r"\bmake\s+(the\s+)?(changes?|it\s+work|corrections?)\b", 0.8, "make_changes"),
    # "Refactor and improve"
    (
        r"\b(refactor|clean\s*up)\s+(and\s+)?(improve|optimize)?\b",
        0.8,
        "refactor_improve",
    ),
    # =============================================================================
    # Implementation task patterns - tasks that require creating/modifying artifacts
    # =============================================================================
    # "Create/Write a Dockerfile/docker-compose/manifest/config" - infrastructure files
    (
        r"\b(create|write|generate|build)\s+(a\s+)?(dockerfile|docker-compose|makefile|"
        r"jenkinsfile|gitlab-ci|github\s*actions?|kubernetes|k8s|helm|terraform|ansible)",
        0.9,
        "create_infra_file",
    ),
    # "Generate/Create a CI/CD/pipeline/workflow" - pipeline configuration
    (
        r"\b(create|write|generate|set\s*up)\s+(a\s+)?(ci/?cd|pipeline|workflow|"
        r"deployment|build)\s*(configuration|config|file|yaml|yml)?",
        0.85,
        "create_pipeline",
    ),
    # "Create a {report/analysis/document}" - document creation tasks
    (
        r"\b(create|write|generate)\s+(a\s+)?([\w\s]+)?(report|analysis|document|documentation)",
        0.8,
        "create_document",
    ),
    # "Create/Write/Generate a {artifact} for {project/this}" - project-specific creation
    (
        r"\b(create|write|generate)\s+(a\s+)?[\w\s]+\s+(for|in)\s+(this|the|my)\s+(project|repo|codebase)",
        0.85,
        "create_for_project",
    ),
    # "Implement {feature}" - explicit implementation request
    (
        r"\bimplement\s+(\w+\s+)?(feature|functionality|module|component|system|service)",
        0.85,
        "implement_feature",
    ),
    # "Implement {something}" - any implementation task (user authentication, login, etc.)
    (
        r"\bimplement\s+(the\s+)?[\w\s]+(authentication|login|logout|auth|api|endpoint|handler|service)",
        0.85,
        "implement_auth",
    ),
    # Generic "implement" followed by a noun phrase
    (r"\bimplement\s+[\w\s]{3,30}$", 0.75, "implement_generic"),
    # "Add {feature/functionality} to" - adding to existing codebase
    (
        r"\badd\s+(\w+\s+)?(feature|functionality|support|capability|module)\s+(to|for)\b",
        0.85,
        "add_feature",
    ),
    # "Set up/Configure {tool/system}" - setup tasks
    (
        r"\b(set\s*up|configure|initialize|bootstrap)\s+(a\s+)?\w+",
        0.8,
        "setup_configure",
    ),
    # "Build/Deploy {service/app}" - deployment tasks
    (
        r"\b(build|deploy|release)\s+(the\s+)?\w+\s*(service|app|application|container)?",
        0.8,
        "build_deploy",
    ),
    # "Create a {type} that {does something}" - functional creation for project artifacts
    # EXCLUDES function/class/method/script (those are DISPLAY_ONLY patterns)
    (
        r"\bcreate\s+(a\s+)?(?!.*\b(function|class|method|script)\b)[\w\s]+\s+that\s+",
        0.75,
        "create_functional",
    ),
    # "Write/Create tests for" - test creation
    (
        r"\b(write|create|add|implement)\s+(unit\s+|integration\s+)?tests?\s+(for|to)\b",
        0.85,
        "write_tests",
    ),
]

# Safe actions by intent type
SAFE_ACTIONS: dict[ActionIntent, Set[str]] = {
    ActionIntent.READ_ONLY: {
        ToolNames.READ,
        ToolNames.LS,
        "code_search",
        "git_status",
    },
    ActionIntent.DISPLAY_ONLY: {
        ToolNames.READ,
        ToolNames.LS,
        "code_search",
        "git_status",
        "generate_response",  # Can generate code in response, just not write
    },
    ActionIntent.WRITE_ALLOWED: {
        ToolNames.READ,
        ToolNames.LS,
        "code_search",
        "git_status",
        "generate_response",
        ToolNames.WRITE,
        ToolNames.EDIT,
        ToolNames.SHELL,
    },
    ActionIntent.AMBIGUOUS: {
        ToolNames.READ,
        ToolNames.LS,
        "code_search",
        "git_status",
        "generate_response",
    },
}

# Prompt guards for different intents
PROMPT_GUARDS: dict[ActionIntent, str] = {
    ActionIntent.DISPLAY_ONLY: """
IMPORTANT: The user wants to SEE code, not save it to a file.
- Generate and DISPLAY the requested code directly in your response
- Do NOT use write or edit tools
- Do NOT modify any existing files
- Present code in markdown code blocks
- Be concise and complete the task in one response
- Do NOT ask follow-up questions like "Would you like me to elaborate?" or "Should I explain more?"
- Simply provide the answer and stop
""",
    ActionIntent.READ_ONLY: """
IMPORTANT: This is a read-only query.
- Only use read operations (read, ls, grep)
- Be concise and answer directly
- Do NOT ask follow-up questions - just provide the information requested
- Do NOT write, create, or modify any files
- Do NOT generate new code unless specifically asked
""",
    ActionIntent.WRITE_ALLOWED: """
IMPORTANT: The user has explicitly authorized file modifications.
- You MAY write, edit, create, and modify files to accomplish this task
- For shell commands that write files (using redirection >, >>, or file creation), use shell(cmd=..., readonly=False)
- Use write() and edit() tools freely for file modifications
- The task requires creating/updating content - proceed with file operations
""",
    ActionIntent.AMBIGUOUS: """
NOTE: User intent is unclear. Default to safe behavior:
- Present code in markdown blocks rather than writing to files
- Ask for confirmation before creating or modifying files
""",
}


class IntentDetector:
    """Detects user intent for action authorization.

    This detector analyzes user messages to determine whether file
    modifications are authorized or if output should be display-only.

    Example:
        detector = IntentDetector()
        result = detector.detect("Show me a function that calculates factorial")
        print(result.intent)  # ActionIntent.DISPLAY_ONLY
        print(result.safe_actions)  # {"read", "generate_response", ...}
    """

    def __init__(
        self,
        custom_display_signals: Optional[List[Tuple[str, float, str]]] = None,
        custom_write_signals: Optional[List[Tuple[str, float, str]]] = None,
        custom_detectors: Optional[List[Callable[[str], Optional[IntentClassification]]]] = None,
        default_intent: ActionIntent = ActionIntent.AMBIGUOUS,
    ):
        """Initialize the intent detector.

        Args:
            custom_display_signals: Additional patterns for display-only intent
            custom_write_signals: Additional patterns for write-allowed intent
            custom_detectors: Custom detector functions to try first
            default_intent: Default when no patterns match (safe by default)
        """
        self.default_intent = default_intent
        self.custom_detectors = custom_detectors or []

        # Compile patterns
        self._display_patterns: List[Tuple[re.Pattern, float, str]] = []
        self._write_patterns: List[Tuple[re.Pattern, float, str]] = []
        self._read_only_patterns: List[Tuple[re.Pattern, float, str]] = []
        self._compound_write_patterns: List[Tuple[re.Pattern, float, str]] = []

        self._compile_patterns(DISPLAY_SIGNALS, self._display_patterns)
        self._compile_patterns(WRITE_SIGNALS, self._write_patterns)
        self._compile_patterns(READ_ONLY_SIGNALS, self._read_only_patterns)
        self._compile_patterns(COMPOUND_WRITE_SIGNALS, self._compound_write_patterns)

        if custom_display_signals:
            self._compile_patterns(custom_display_signals, self._display_patterns)
        if custom_write_signals:
            self._compile_patterns(custom_write_signals, self._write_patterns)

    def _compile_patterns(
        self,
        signals: List[Tuple[str, float, str]],
        target: List[Tuple[re.Pattern, float, str]],
    ) -> None:
        """Compile regex patterns."""
        for pattern_str, weight, name in signals:
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE)
                target.append((compiled, weight, name))
            except re.error as e:
                logger.warning(f"Invalid pattern '{pattern_str}': {e}")

    def detect(self, message: str) -> IntentClassification:
        """Detect user intent from message.

        Args:
            message: User's input message

        Returns:
            IntentClassification with intent, confidence, and safe actions
        """
        # Try custom detectors first
        for detector in self.custom_detectors:
            result = detector(message)
            if result is not None:
                return result

        # Short-circuit: bare continuation commands resume an in-progress task and
        # must not receive a DISPLAY_ONLY guard that blocks tool use and file writes.
        is_continuation, continuation_payload = split_continuation_request(message)
        if is_continuation and not continuation_payload:
            return IntentClassification(
                intent=ActionIntent.WRITE_ALLOWED,
                confidence=0.9,
                matched_signals=["continuation_keyword"],
                safe_actions=SAFE_ACTIONS[ActionIntent.WRITE_ALLOWED].copy(),
                prompt_guard=PROMPT_GUARDS[ActionIntent.WRITE_ALLOWED],
            )

        scoring_message = continuation_payload or message

        # Score each intent type
        display_score, display_matched = self._score_patterns(
            scoring_message, self._display_patterns
        )
        write_score, write_matched = self._score_patterns(scoring_message, self._write_patterns)
        read_only_score, read_only_matched = self._score_patterns(
            scoring_message, self._read_only_patterns
        )
        compound_score, compound_matched = self._score_patterns(
            scoring_message, self._compound_write_patterns
        )

        # Determine intent based on scores
        # Compound write signals (e.g., "analyze and fix") take highest precedence
        # They indicate user wants both analysis AND file modifications
        if compound_score > 0:
            intent = ActionIntent.WRITE_ALLOWED
            confidence = min(1.0, compound_score)
            matched = compound_matched
            logger.debug(f"Compound write intent detected: {compound_matched}")
        # Write signals are explicit and take precedence when strong
        elif write_score > 0.5 and write_score > display_score:
            intent = ActionIntent.WRITE_ALLOWED
            confidence = min(1.0, write_score)
            matched = write_matched
        elif read_only_score > 0.5 and read_only_score >= display_score:
            intent = ActionIntent.READ_ONLY
            confidence = min(1.0, read_only_score)
            matched = read_only_matched
        elif display_score > 0.5:
            intent = ActionIntent.DISPLAY_ONLY
            confidence = min(1.0, display_score)
            matched = display_matched
        elif write_score > 0:
            # Weak write signal, but no display signal
            intent = ActionIntent.AMBIGUOUS
            confidence = 0.3
            matched = write_matched
        else:
            # No signals - use safe default
            intent = self.default_intent
            confidence = 0.2
            matched = []

        if is_continuation and continuation_payload:
            matched = ["continuation_with_payload", *matched]

        return IntentClassification(
            intent=intent,
            confidence=confidence,
            matched_signals=matched,
            safe_actions=SAFE_ACTIONS[intent].copy(),
            prompt_guard=PROMPT_GUARDS[intent],
        )

    def _score_patterns(
        self,
        message: str,
        patterns: List[Tuple[re.Pattern, float, str]],
    ) -> Tuple[float, List[str]]:
        """Score patterns against message.

        Args:
            message: User message
            patterns: List of (pattern, weight, name) tuples

        Returns:
            Tuple of (total_score, matched_names)
        """
        total_score = 0.0
        matched = []
        for pattern, weight, name in patterns:
            if pattern.search(message):
                total_score += weight
                matched.append(name)
        return total_score, matched

    def is_write_authorized(self, message: str) -> bool:
        """Quick check if file writing is authorized.

        Args:
            message: User message

        Returns:
            True if writing is explicitly authorized
        """
        result = self.detect(message)
        return result.intent == ActionIntent.WRITE_ALLOWED


def detect_intent(message: str) -> IntentClassification:
    """Convenience function to detect intent.

    Args:
        message: User message

    Returns:
        IntentClassification
    """
    detector = IntentDetector()
    return detector.detect(message)


def is_write_authorized(message: str) -> bool:
    """Quick check if file writing is authorized.

    Args:
        message: User message

    Returns:
        True if writing is explicitly authorized
    """
    detector = IntentDetector()
    return detector.is_write_authorized(message)


def get_prompt_guard(message: str) -> str:
    """Get prompt guard text for a message.

    Args:
        message: User message

    Returns:
        Prompt guard text to inject into system prompt
    """
    result = detect_intent(message)
    return result.prompt_guard


def get_safe_tools(message: str, all_tools: Set[str]) -> Set[str]:
    """Get tools that are safe to use based on intent.

    Args:
        message: User message
        all_tools: Set of all available tool names

    Returns:
        Set of tool names that are safe to use
    """
    result = detect_intent(message)

    # Map safe action categories to actual tool names
    # This is a simple mapping - can be extended
    action_to_tools = {
        ToolNames.READ: READ_TOOL_ALIASES,
        ToolNames.LS: LIST_TOOL_ALIASES,
        "code_search": {"code_search", "semantic_code_search"},
        "git_status": SHELL_TOOL_ALIASES | {"git_status"},
        "generate_response": set(),  # Not a tool, just allow text response
        ToolNames.WRITE: WRITE_TOOL_ALIASES,
        ToolNames.EDIT: EDIT_TOOL_ALIASES,
        ToolNames.SHELL: SHELL_TOOL_ALIASES,
    }

    safe = set()
    for action in result.safe_actions:
        tools = action_to_tools.get(action, set())
        safe.update(tools)

    safe_canonical = _canonicalize_tool_set(safe)
    return {
        tool
        for tool in all_tools
        if tool in safe or canonicalize_core_tool_name(tool) in safe_canonical
    }


# Semantic alias for clarity
ActionAuthorizer = IntentDetector
