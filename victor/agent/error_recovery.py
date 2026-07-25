"""
Error Recovery Chain of Responsibility for Tool Execution Errors.

This module handles tool-level execution errors such as:
- Missing required parameters
- Type conversion errors
- File not found errors
- Permission errors
- Network/timeout errors

NOTE: This module is COMPLEMENTARY to victor/agent/recovery/, not a duplicate:
- THIS module: Handles TOOL execution errors (parameter issues, file errors)
- recovery/ module: Handles LLM response failures (empty responses, stuck loops)

Both modules implement Chain of Responsibility but for different error domains.

SOLID Principles Applied:
- Single Responsibility: Each handler handles one type of error
- Open/Closed: New handlers can be added without modifying existing ones
- Liskov Substitution: All handlers are interchangeable
- Interface Segregation: ErrorRecoveryHandler defines minimal interface
- Dependency Inversion: Handlers depend on abstractions, not concrete classes

Implements GAP-10 from Grok/DeepSeek provider testing.
"""

import ast
from abc import ABC, abstractmethod
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
import asyncio
import logging
import os
import re

logger = logging.getLogger(__name__)

# Minimum basename similarity for a "Did you mean" suggestion to be trusted as a
# silent substitute for the requested path. Typo-class matches score >= 0.94;
# different-document matches score <= 0.7. Below the threshold the suggestion is
# still surfaced to the model in the error text, but never auto-retried: a weak
# match silently swaps in the wrong file and poisons the session path-redirect
# cache, which the model cannot observe or escape.
PATH_SUGGESTION_CONFIDENCE_THRESHOLD = 0.85


def path_suggestion_confidence(original_path: str, suggested_path: str) -> float:
    """Similarity ratio (0.0-1.0) between requested and suggested path basenames."""
    original_name = os.path.basename(original_path.rstrip("/").replace("\\", "/"))
    suggested_name = os.path.basename(suggested_path.rstrip("/").replace("\\", "/"))
    if not original_name or not suggested_name:
        return 0.0
    return SequenceMatcher(None, original_name.lower(), suggested_name.lower()).ratio()


def is_confident_path_suggestion(original_path: str, suggested_path: str) -> bool:
    """True when a suggested path is close enough to auto-substitute for the original."""
    return (
        path_suggestion_confidence(original_path, suggested_path)
        >= PATH_SUGGESTION_CONFIDENCE_THRESHOLD
    )


class ErrorRecoveryAction(Enum):
    """Actions that can be taken to recover from a tool error.

    Renamed from RecoveryAction to be semantically distinct:
    - ErrorRecoveryAction (here): Tool error recovery actions (string values)
    - StrategyRecoveryAction (victor.agent.recovery.protocols): Recovery strategy actions (auto)
    - OrchestratorRecoveryAction (victor.agent.orchestrator_recovery): Orchestrator recovery dataclass
    """

    RETRY = "retry"
    RETRY_WITH_DEFAULTS = "retry_with_defaults"
    RETRY_WITH_INFERRED = "retry_with_inferred"
    SKIP = "skip"
    FALLBACK_TOOL = "fallback_tool"
    ASK_USER = "ask_user"
    ABORT = "abort"


# Backward compatibility alias
RecoveryAction = ErrorRecoveryAction


@dataclass
class RecoveryResult:
    """Result of attempting to recover from an error."""

    action: RecoveryAction
    modified_args: Optional[Dict[str, Any]] = None
    fallback_tool: Optional[str] = None
    user_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def should_retry(self) -> bool:
        """Check if action is a retry variant."""
        return self.action in (
            RecoveryAction.RETRY,
            RecoveryAction.RETRY_WITH_DEFAULTS,
            RecoveryAction.RETRY_WITH_INFERRED,
        )

    @property
    def can_retry(self) -> bool:
        """Check if more retries are allowed."""
        return self.retry_count < self.max_retries


class ErrorRecoveryHandler(ABC):
    """Abstract handler in the chain of responsibility."""

    def __init__(self) -> None:
        self._next_handler: Optional["ErrorRecoveryHandler"] = None
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def set_next(self, handler: "ErrorRecoveryHandler") -> "ErrorRecoveryHandler":
        """Set the next handler in the chain."""
        self._next_handler = handler
        return handler

    @abstractmethod
    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        """Check if this handler can handle the error."""
        pass

    @abstractmethod
    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        """Handle the error and return recovery result."""
        pass

    def process(
        self,
        error: Exception,
        tool_name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> RecoveryResult:
        """Process the error through the chain."""
        if self.can_handle(error, tool_name, args):
            self._logger.info(
                f"Handler {self.__class__.__name__} handling error for {tool_name}: {error}"
            )
            result = self.handle(error, tool_name, args)
            result.metadata["handler"] = self.__class__.__name__
            return result
        elif self._next_handler:
            return self._next_handler.process(error, tool_name, args, context)
        else:
            self._logger.warning(f"No handler could process error for {tool_name}: {error}")
            return RecoveryResult(
                action=RecoveryAction.ABORT,
                user_message=f"Unrecoverable error in {tool_name}: {error}",
            )


class MissingParameterHandler(ErrorRecoveryHandler):
    """Handle missing required parameter errors."""

    # Default values for common parameters
    DEFAULTS: Dict[str, Any] = {
        "file_path": ".",
        "path": ".",
        "directory": ".",
        "limit": 100,
        "offset": 0,
        "start_line": 1,
        "end_line": -1,  # -1 means end of file
        "recursive": False,
        "max_depth": 3,
    }

    # Patterns to detect missing parameter errors
    PATTERNS = [
        r"missing \d+ required positional argument[s]?: '(\w+)'",
        r"required parameter '(\w+)' was not provided",
        r"'(\w+)' is a required property",
        r"missing required argument[s]?: (\w+)",
    ]

    def _extract_missing_params(self, error_str: str) -> List[str]:
        """Extract all missing parameter names from an error message."""
        list_patterns = [
            r"missing \d+ required positional argument[s]?:\s*(.+)$",
            r"required parameter[s]?\s+(.+?)\s+was not provided",
            r"required parameter[s]?\s+(.+?)\s+were not provided",
            r"missing required argument[s]?:\s*(.+)$",
            r"(.+?)\s+is a required property",
            r"(.+?)\s+are required properties",
        ]

        for pattern in list_patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if not match:
                continue

            raw_params = match.group(1).strip().rstrip(".")
            params = [
                token
                for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", raw_params)
                if token.lower()
                not in {
                    "and",
                    "or",
                    "required",
                    "require",
                    "parameter",
                    "parameters",
                    "argument",
                    "arguments",
                    "property",
                    "properties",
                    "positional",
                    "provided",
                    "was",
                    "were",
                    "missing",
                }
            ]
            if params:
                return list(dict.fromkeys(params))

        quoted_params = re.findall(r"'(\w+)'", error_str)
        if quoted_params:
            return list(dict.fromkeys(quoted_params))

        return []

    def _recover_wrapped_value_arguments(
        self, args: Dict[str, Any], missing_params: List[str], tool_name: str
    ) -> Optional[Dict[str, Any]]:
        """Recover structured tool args wrapped inside a generic value envelope.

        Enhanced with better diagnostics for large payloads and partial recovery.
        """
        if set(args.keys()) != {"value"}:
            return None

        # Single authority: coercion + schema-aware value-envelope recovery +
        # alias normalization all happen in ArgumentNormalizer.parse_tool_arguments
        # (no duplicate json/ast/greedy ladder here).
        from victor.agent.argument_normalizer import ArgumentNormalizer

        try:
            recovered, _ = ArgumentNormalizer(provider_name="recovery").parse_tool_arguments(
                args.get("value"), tool_name
            )
        except Exception as exc:
            self._logger.debug("Value envelope recovery failed: %s", str(exc)[:100])
            return None

        if isinstance(recovered, dict) and all(param in recovered for param in missing_params):
            self._logger.info(
                "Recovered wrapped value envelope for %s with params: %s",
                tool_name,
                ", ".join(sorted(recovered.keys())),
            )
            return recovered

        self._logger.debug(
            "Recovered payload missing required params: have=%s, need=%s",
            (list(recovered.keys()) if isinstance(recovered, dict) else type(recovered).__name__),
            missing_params,
        )
        return None

    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        error_str = str(error).lower()
        # Check for various missing parameter/argument patterns
        has_missing = "missing" in error_str
        has_required = "required" in error_str
        has_argument = "argument" in error_str
        has_parameter = "parameter" in error_str
        has_property = "property" in error_str

        # Missing argument pattern: "missing 1 required positional argument"
        if has_missing and has_argument:
            return True
        # Required parameter pattern: "required parameter 'x' was not provided"
        if has_required and has_parameter:
            return True
        # Required property pattern: "'x' is a required property"
        if has_required and has_property:
            return True
        # Missing parameter pattern: "missing required argument"
        if has_missing and has_required:
            return True

        return False

    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        error_str = str(error)
        missing_params = self._extract_missing_params(error_str)

        if missing_params:
            recovered_args = self._recover_wrapped_value_arguments(args, missing_params, tool_name)
            if recovered_args is not None:
                return RecoveryResult(
                    action=RecoveryAction.RETRY_WITH_INFERRED,
                    modified_args=recovered_args,
                    user_message=("Recovered structured arguments from wrapped value payload"),
                )

        if missing_params and all(param in self.DEFAULTS for param in missing_params):
            defaults = {param: self.DEFAULTS[param] for param in missing_params}
            self._logger.info(
                "Providing default values for missing params %s",
                defaults,
            )
            return RecoveryResult(
                action=RecoveryAction.RETRY_WITH_DEFAULTS,
                modified_args={**args, **defaults},
                user_message=(
                    "Using default values for "
                    + ", ".join(f"'{param}'" for param in missing_params)
                ),
            )

        param_name = missing_params[0] if missing_params else None

        # Build a helpful error message with context
        if "value" in args and len(missing_params) == 1:
            # Special case: value envelope unwrapping failed
            error_detail = (
                f"Parameter '{param_name}' was wrapped in a value envelope but couldn't be extracted. "
                f"This can happen with large payloads containing special characters. "
                f"Try breaking the command into smaller parts or using different quoting."
            )
        else:
            error_detail = (
                "Cannot infer value for required parameter(s): "
                + ", ".join(missing_params or ([param_name] if param_name else ["unknown"]))
                + ". Provide explicit values for these parameters."
            )

        return RecoveryResult(
            action=RecoveryAction.SKIP,
            user_message=error_detail,
        )


class ToolNotFoundHandler(ErrorRecoveryHandler):
    """Handle tool not found errors with fallback alternatives."""

    # Mapping from tools to their fallbacks
    FALLBACKS: Dict[str, str] = {
        "symbol": "grep",  # Fallback from symbol lookup to grep
        "get_symbol": "grep",
        "semantic_search": "grep",
        "semantic_code_search": "code_search",
        "code_search": "grep",  # Second-level fallback
        "tree": "ls",
        "find_files": "glob",
        "analyze_dependencies": "grep",
    }

    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        error_str = str(error).lower()
        return (
            "not found" in error_str
            or "unknown tool" in error_str
            or "unregistered" in error_str
            or "dependencies missing" in error_str
            or "no module named" in error_str
            or "import error" in error_str
            or "not installed" in error_str
            or isinstance(error, (ImportError, ModuleNotFoundError))
        )

    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        if tool_name in self.FALLBACKS:
            fallback = self.FALLBACKS[tool_name]
            self._logger.info(f"Falling back from {tool_name} to {fallback}")
            return RecoveryResult(
                action=RecoveryAction.FALLBACK_TOOL,
                fallback_tool=fallback,
                user_message=f"Tool '{tool_name}' not available, using '{fallback}' instead",
            )

        return RecoveryResult(
            action=RecoveryAction.SKIP,
            user_message=f"Tool '{tool_name}' not found and no fallback available",
        )


class NetworkErrorHandler(ErrorRecoveryHandler):
    """Handle network-related errors with retry logic."""

    NETWORK_ERROR_PATTERNS = [
        "timeout",
        "connection",
        "network",
        "refused",
        "unreachable",
        "dns",
        "socket",
        "ssl",
        "certificate",
    ]

    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in self.NETWORK_ERROR_PATTERNS)

    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        return RecoveryResult(
            action=RecoveryAction.RETRY,
            max_retries=3,
            user_message="Network error, retrying...",
            metadata={"retry_delay_seconds": 1.0},
        )


class FileNotFoundHandler(ErrorRecoveryHandler):
    """Handle file not found errors by attempting path variations."""

    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        # Check error class first
        if isinstance(error, FileNotFoundError):
            return True
        error_str = str(error).lower()
        return (
            "file not found" in error_str
            or "no such file" in error_str
            or "does not exist" in error_str
        )

    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        path = args.get("path") or args.get("file_path") or args.get("file")

        if path:
            suggested_paths = self._extract_suggested_paths(str(error))
            suggested_retry = self._choose_best_suggested_path(path, suggested_paths, tool_name)
            if suggested_retry:
                self._logger.info(f"Trying suggested path: {suggested_retry}")
                path_key = self._get_path_arg_key(args)
                return RecoveryResult(
                    action=RecoveryAction.RETRY_WITH_INFERRED,
                    modified_args={**args, path_key: suggested_retry},
                    user_message=f"Trying suggested path: {suggested_retry}",
                    metadata={"suggested_paths": suggested_paths},
                )
            if suggested_paths:
                return RecoveryResult(
                    action=RecoveryAction.SKIP,
                    user_message="File not found and suggestions did not provide a new path",
                    metadata={"suggested_paths": suggested_paths},
                )

            variations = self._get_path_variations(path)
            if variations:
                self._logger.info(f"Trying path variation: {variations[0]}")
                path_key = self._get_path_arg_key(args)
                return RecoveryResult(
                    action=RecoveryAction.RETRY_WITH_INFERRED,
                    modified_args={**args, path_key: variations[0]},
                    user_message=f"Trying alternate path: {variations[0]}",
                    metadata={"tried_variations": variations},
                )

        return RecoveryResult(
            action=RecoveryAction.SKIP,
            user_message="File not found and no alternatives discovered",
        )

    def _get_path_variations(self, path: str) -> List[str]:
        """Generate path variations to try."""
        import os

        variations = []

        # Remove leading ./ or ./
        if path.startswith("./"):
            variations.append(path[2:])
        elif not path.startswith("/"):
            variations.append(f"./{path}")

        # Try with/without extension
        base, ext = os.path.splitext(path)
        if ext == ".py":
            variations.append(f"{base}/__init__.py")
        elif not ext:
            variations.append(f"{path}.py")
            variations.append(f"{path}/index.py")
            variations.append(f"{path}/__init__.py")

        return variations

    def _get_path_arg_key(self, args: Dict[str, Any]) -> str:
        """Select the canonical path-like argument key for retries."""
        if "path" in args:
            return "path"
        if "file_path" in args:
            return "file_path"
        if "file" in args:
            return "file"
        return "path"

    def _extract_suggested_paths(self, error_str: str) -> List[str]:
        """Parse candidate paths from "Did you mean" style error text."""
        suggested_paths = re.findall(r"^\s*-\s+(.+?)\s*$", error_str, re.MULTILINE)
        if suggested_paths:
            return suggested_paths

        inline_match = re.search(r"Did you mean:\s*(.+)", error_str)
        if not inline_match:
            return []

        return [
            candidate.strip() for candidate in inline_match.group(1).split(",") if candidate.strip()
        ]

    def _choose_best_suggested_path(
        self, original_path: str, suggestions: List[str], tool_name: str
    ) -> Optional[str]:
        """Prefer the most relevant suggestion for the tool being retried."""
        if not suggestions:
            return None

        normalized_original = original_path.rstrip("/").replace("\\", "/")
        base_path, _ = os.path.splitext(normalized_original)
        filtered_suggestions = [
            candidate
            for candidate in suggestions
            if candidate.rstrip("/").replace("\\", "/") != normalized_original
        ]
        if not filtered_suggestions:
            return None

        file_suggestions = [
            candidate for candidate in filtered_suggestions if not candidate.endswith("/")
        ]
        directory_suggestions = [
            candidate for candidate in filtered_suggestions if candidate.endswith("/")
        ]

        if tool_name in {"read", "open", "cat"}:
            package_file_matches = [
                candidate
                for candidate in file_suggestions
                if candidate.replace("\\", "/").startswith(f"{base_path}/")
            ]
            if package_file_matches:
                # Module-entry recovery (foo.py -> foo/__init__.py): the path-prefix
                # containment is its own confidence signal; basename gate not applied.
                return package_file_matches[0]
            return self._first_confident(
                normalized_original, file_suggestions or filtered_suggestions
            )

        if tool_name in {"ls", "find"} and directory_suggestions:
            return self._first_confident(normalized_original, directory_suggestions)

        return self._first_confident(normalized_original, filtered_suggestions)

    @staticmethod
    def _first_confident(original_path: str, candidates: List[str]) -> Optional[str]:
        """First candidate similar enough to auto-substitute; None keeps the honest error."""
        for candidate in candidates:
            if is_confident_path_suggestion(original_path, candidate):
                return candidate
        return None


class RateLimitHandler(ErrorRecoveryHandler):
    """Handle rate limit errors with exponential backoff."""

    RATE_LIMIT_PATTERNS = [
        "rate limit",
        "too many requests",
        "429",
        "throttle",
        "quota exceeded",
    ]

    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in self.RATE_LIMIT_PATTERNS)

    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        # Extract retry-after if present
        retry_after = 5.0  # default
        match = re.search(r"retry.after[:\s]*(\d+)", str(error), re.IGNORECASE)
        if match:
            retry_after = float(match.group(1))

        return RecoveryResult(
            action=RecoveryAction.RETRY,
            max_retries=3,
            user_message=f"Rate limited, waiting {retry_after}s...",
            metadata={"retry_delay_seconds": retry_after, "exponential_backoff": True},
        )


class ResourceBudgetTimeoutHandler(ErrorRecoveryHandler):
    """Handle resource-budget timeouts from bounded local tools."""

    BUDGET_TIMEOUT_PATTERNS = [
        r"exceeded\s+\d+(?:\.\d+)?s?\s+budget",
        r"exceeded\s+.*\btime\s+budget\b",
        r"\bresource[-_\s]?budget\b",
        r"\bdeadline exceeded\b",
        r"\btime limit exceeded\b",
    ]

    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        error_str = str(error).lower()
        return any(
            re.search(pattern, error_str, re.IGNORECASE) for pattern in self.BUDGET_TIMEOUT_PATTERNS
        )

    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        return RecoveryResult(
            action=RecoveryAction.SKIP,
            user_message=(
                f"Tool '{tool_name}' exceeded its resource budget. "
                "Retry with a narrower path, smaller top_k, or a more specific mode."
            ),
            metadata={"error_kind": "resource_budget_timeout"},
        )


class TimeoutErrorHandler(ErrorRecoveryHandler):
    """Handle tool timeout errors.

    Timeouts are expected for slow tools (code_search, web_search, etc.)
    and do not require handler intervention beyond clear messaging.
    """

    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        if isinstance(error, asyncio.TimeoutError):
            return True
        error_str = str(error).lower()
        return "timed out" in error_str or "timeout" in error_str

    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        return RecoveryResult(
            action=RecoveryAction.SKIP,
            user_message=(
                f"Tool '{tool_name}' timed out. "
                "Consider increasing the per-tool timeout setting or simplifying the operation."
            ),
        )


class PermissionErrorHandler(ErrorRecoveryHandler):
    """Handle permission errors."""

    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        # Check error class first
        if isinstance(error, PermissionError):
            return True
        error_str = str(error).lower()
        return (
            "permission denied" in error_str
            or "access denied" in error_str
            or "forbidden" in error_str
        )

    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        # For permission errors, we typically can't recover automatically
        return RecoveryResult(
            action=RecoveryAction.ASK_USER,
            user_message="Permission denied. Please check file permissions or run with elevated privileges.",
        )


class GraphDatabaseErrorHandler(ErrorRecoveryHandler):
    """Handle graph database errors with intelligent fallbacks."""

    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        """Check if this is a graph database error we can handle."""
        if tool_name != "graph":
            return False
        error_str = str(error).lower()
        return (
            "project graph database is empty" in error_str
            or "project graph database is unavailable" in error_str
            or "graph data unavailable" in error_str
            or "graph database is empty" in error_str
        )

    def handle(
        self, error: Exception, tool_name: str, args: Dict[str, Any]
    ) -> Optional[RecoveryResult]:
        """Handle graph database errors with helpful suggestions."""
        error_str = str(error).lower()
        path = args.get("path", ".")

        if "empty" in error_str:
            return RecoveryResult(
                action=RecoveryAction.ASK_USER,
                user_message=(
                    f"Graph database is empty for path '{path}'. "
                    f"Build the index with: graph(mode='stats', path='{path}', reindex=True). "
                    f"Or use ls(path='{path}', depth=2) for file operations."
                ),
            )

        if "unavailable" in error_str:
            return RecoveryResult(
                action=RecoveryAction.ASK_USER,
                user_message=(
                    f"Graph index unavailable for path '{path}'. "
                    f"Try: graph(mode='stats', path='{path}', reindex=True) "
                    "or use code_search(mode='literal', ...) for text search."
                ),
            )

        return RecoveryResult(
            action=RecoveryAction.ASK_USER,
            user_message=f"Graph database error: {error}",
        )


class TypeErrorHandler(ErrorRecoveryHandler):
    """Handle type errors in tool arguments."""

    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        return isinstance(error, TypeError) or "type" in str(error).lower()

    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        # Try to fix common type issues
        modified_args = dict(args)
        fixed = False

        for key, value in args.items():
            # Convert string booleans
            if isinstance(value, str) and value.lower() in ("true", "false"):
                modified_args[key] = value.lower() == "true"
                fixed = True
            # Convert string numbers
            elif isinstance(value, str):
                try:
                    if "." in value:
                        modified_args[key] = float(value)
                    else:
                        modified_args[key] = int(value)
                    fixed = True
                except ValueError:
                    pass

        if fixed:
            return RecoveryResult(
                action=RecoveryAction.RETRY_WITH_INFERRED,
                modified_args=modified_args,
                user_message="Converted argument types",
            )

        return RecoveryResult(
            action=RecoveryAction.SKIP,
            user_message=f"Type error in arguments: {error}",
        )


class ShellGrepRedirectHandler(ErrorRecoveryHandler):
    """Handle shell tool errors that suggest using code_search instead.

    When the shell tool rejects grep/rg commands for project code and suggests
    using code_search, this handler extracts the search query and suggests
    using code_search with the appropriate parameters.
    """

    # Pattern to detect the shell tool's code_search suggestion
    CODE_SEARCH_SUGGESTION = "use code_search(query='..."

    # Pattern to extract grep pattern from shell commands
    GREP_PATTERN_PATTERNS = [
        # grep/rg -r "pattern" [path]
        r'(?:grep|rg|ag|ack)\s+(?:-[a-zA-Z]*\s+)*[\'"]([^\']+)[\'"]',
        # grep/rg "pattern" file
        r'(?:grep|rg|ag|ack)\s+[\'"]([^\']+)[\'"]',
        # grep -e pattern
        r'-e\s+[\'"]?([^\'"\s]+)[\'"]?',
    ]

    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        """Check if this is a shell tool error suggesting code_search."""
        if tool_name != "shell":
            return False

        error_str = str(error).lower()
        return "code_search" in error_str and "instead of shell" in error_str

    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        """Extract the search query and suggest code_search."""
        cmd = args.get("cmd", "")

        # Try to extract the search pattern from the command
        query = self._extract_search_query(cmd)

        if query:
            self._logger.info(f"Redirecting shell grep to code_search with query: {query}")
            return RecoveryResult(
                action=RecoveryAction.FALLBACK_TOOL,
                fallback_tool="code_search",
                modified_args={"query": query, "mode": "semantic"},
                user_message=f"Using code_search instead of shell grep for query: {query}",
            )

        # If we couldn't extract a query, still suggest code_search
        return RecoveryResult(
            action=RecoveryAction.FALLBACK_TOOL,
            fallback_tool="code_search",
            user_message="Use code_search for project code instead of shell grep",
        )

    def _extract_search_query(self, cmd: str) -> Optional[str]:
        """Extract the search pattern from a grep/rg command."""
        for pattern in self.GREP_PATTERN_PATTERNS:
            match = re.search(pattern, cmd, re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                # Clean up the query - remove regex special chars if it looks like a literal search
                if query and len(query) < 200:  # Reasonable query length
                    return query
        return None


def build_recovery_chain() -> ErrorRecoveryHandler:
    """Build the default error recovery chain.

    Chain order matters - more specific handlers should come first.
    """
    chain = MissingParameterHandler()
    (
        chain.set_next(ShellGrepRedirectHandler())
        .set_next(TypeErrorHandler())
        .set_next(FileNotFoundHandler())
        .set_next(ToolNotFoundHandler())
        .set_next(RateLimitHandler())
        .set_next(ResourceBudgetTimeoutHandler())
        .set_next(NetworkErrorHandler())
        .set_next(TimeoutErrorHandler())
        .set_next(PermissionErrorHandler())
        .set_next(GraphDatabaseErrorHandler())
    )
    return chain


# Singleton instance for convenience
_default_chain: Optional[ErrorRecoveryHandler] = None


def get_recovery_chain() -> ErrorRecoveryHandler:
    """Get the default recovery chain (singleton)."""
    global _default_chain
    if _default_chain is None:
        _default_chain = build_recovery_chain()
    return _default_chain


def reset_recovery_chain() -> None:
    """Reset the recovery chain singleton.

    Forces the chain to be rebuilt on the next call to get_recovery_chain().
    Useful when new handlers have been added dynamically.
    """
    global _default_chain
    _default_chain = None


def recover_from_error(
    error: Exception,
    tool_name: str,
    args: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> RecoveryResult:
    """Convenience function to recover from an error using the default chain."""
    chain = get_recovery_chain()
    return chain.process(error, tool_name, args, context)
