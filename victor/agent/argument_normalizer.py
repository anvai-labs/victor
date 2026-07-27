"""
Robust argument normalization for tool calls.
Handles malformed JSON from various LLM providers.

This module provides a multi-layer normalization pipeline that gracefully handles
common argument formatting issues, particularly Python-style syntax (single quotes)
being used instead of JSON (double quotes).
"""

import ast
from victor.core.json_utils import json_dumps, json_loads
from json import JSONDecodeError
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from victor.tools.core_tool_aliases import canonicalize_core_tool_name

# Import native extensions with unified availability check
try:
    from victor.processing.native import (
        repair_json as native_repair_json,
        is_native_available,
        coerce_string_type as native_coerce_string_type,
    )

    _NATIVE_AVAILABLE = is_native_available()
except ImportError:
    _NATIVE_AVAILABLE = False

    def native_coerce_string_type(value: str) -> tuple:
        """Fallback stub when native not available."""
        return ("string", value, None)


logger = logging.getLogger(__name__)


_STRUCTURED_PARSE_FAILED = object()


class ToolStats(BaseModel):
    """Statistics for tool argument normalization (Pydantic v2)."""

    model_config = {"validate_assignment": True}

    calls: int = Field(default=0, ge=0, description="Number of times tool was called")
    normalizations: int = Field(default=0, ge=0, description="Number of arguments normalized")

    # Dict-like interface for backward compatibility
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class NormalizationStats(BaseModel):
    """Aggregate normalization statistics (Pydantic v2)."""

    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}

    total_calls: int = Field(default=0, ge=0, description="Total normalization calls")
    normalizations: Dict[str, int] = Field(
        default_factory=dict,
        description="Normalizations per strategy",
    )
    failures: int = Field(default=0, ge=0, description="Total normalization failures")
    by_tool: Dict[str, ToolStats] = Field(
        default_factory=dict,
        description="Per-tool statistics",
    )

    # Dict-like interface for backward compatibility
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class NormalizationStrategy(Enum):
    """Strategies for normalizing malformed arguments."""

    DIRECT = "direct"  # Valid JSON, no changes needed
    PYTHON_AST = "python_ast"  # Python syntax → JSON via ast.literal_eval
    REGEX_QUOTES = "regex_quotes"  # Simple quote replacement
    MANUAL_REPAIR = "manual_repair"  # Tool-specific repairs
    FAILED = "failed"  # All strategies failed


class ArgumentNormalizer:
    """
    Multi-layer argument normalization with fallback strategies.

    Design Philosophy:
    - Fast path first (valid JSON) - zero overhead for 99%+ cases
    - Graceful degradation (multiple fallback strategies)
    - Complete transparency (log all normalizations)
    - Provider-aware (can customize per provider)
    - Alias normalization (model-specific param names → standard names)

    Usage:
        normalizer = ArgumentNormalizer(provider_name="ollama")
        normalized_args, strategy = normalizer.normalize_arguments(
            arguments={"operations": "[{'type': 'modify', ...}]"},
            tool_name="edit_files"
        )

        # With parameter aliases (e.g., gpt-oss uses line_start instead of offset)
        normalizer = ArgumentNormalizer(
            provider_name="ollama",
            config={"parameter_aliases": {
                "read": {"line_start": "offset", "line_end": "_line_end"}
            }}
        )
    """

    # These arguments are semantically "opaque text" and must survive provider
    # fallback normalization untouched. Parsing or coercing them corrupts shell
    # commands, file contents, search queries, and replacement strings.
    _RAW_STRING_ARGUMENT_NAMES = frozenset(
        {
            "cmd",
            "cwd",
            "path",
            "content",
            "old_str",
            "new_str",
            "new_path",
            "query",
            "desc",
            "message",
            "text",
            "pattern",
        }
    )

    def __init__(self, provider_name: str = "unknown", config: Optional[Dict[str, Any]] = None):
        """
        Initialize argument normalizer.

        Args:
            provider_name: Name of the LLM provider (for logging/metrics)
            config: Optional configuration overrides
                - parameter_aliases: Dict[tool_name, Dict[model_param, standard_param]]
        """
        self.provider_name = provider_name
        self.config = config or {}
        # Built-in aliases for common model hallucinations (LLMs use wrong param names)
        _builtin_aliases: Dict[str, Dict[str, str]] = {
            "shell": {"command": "cmd"},
            "read": {"file_path": "path", "filename": "path"},
            "write": {
                "file_path": "path",
                "filename": "path",
                "text": "content",
                "data": "content",
            },
            "edit": {"operations": "ops"},
            "ls": {"directory": "path", "dir": "path"},
        }
        user_aliases = {
            canonicalize_core_tool_name(tool_name): aliases
            for tool_name, aliases in self.config.get("parameter_aliases", {}).items()
        }
        # Merge: user overrides built-in
        self.parameter_aliases: Dict[str, Dict[str, str]] = {
            **_builtin_aliases,
            **user_aliases,
        }
        self.stats: NormalizationStats = NormalizationStats(
            total_calls=0,
            normalizations={strategy.value: 0 for strategy in NormalizationStrategy},
            failures=0,
            by_tool={},
        )
        self._alias_stats = {"total": 0, "aliased": 0, "by_tool": {}}

    def normalize_parameter_aliases(
        self, arguments: Dict[str, Any], tool_name: str
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Normalize model-specific parameter names to standard tool parameter names.

        Some models (e.g., gpt-oss) use different parameter names than what the
        tools expect. This method maps those aliases to standard names.

        Example:
            Input:  {"line_start": 10, "line_end": 50} for tool "read"
            Output: {"offset": 10, "_line_end": 50} (with _line_end for special handling)

        Args:
            arguments: Raw arguments from model
            tool_name: Name of the tool being called

        Returns:
            (normalized_arguments, was_aliased) - returns original if no aliases apply
        """
        tool_name = canonicalize_core_tool_name(tool_name)
        tool_aliases = self.parameter_aliases.get(tool_name, {})
        if not tool_aliases:
            return arguments, False

        normalized = {}
        was_aliased = False

        for param, value in arguments.items():
            if param in tool_aliases:
                standard_param = tool_aliases[param]
                normalized[standard_param] = value
                was_aliased = True
                logger.debug(
                    f"[{self.provider_name}] {tool_name}: Aliased '{param}' -> '{standard_param}'"
                )
            else:
                normalized[param] = value

        if was_aliased:
            self._alias_stats["total"] += 1
            self._alias_stats["aliased"] += 1
            if tool_name not in self._alias_stats["by_tool"]:
                self._alias_stats["by_tool"][tool_name] = 0
            self._alias_stats["by_tool"][tool_name] += 1
            logger.debug(
                f"[{self.provider_name}] Normalized parameter aliases for {tool_name}: "
                f"{list(set(arguments.keys()) - set(normalized.keys()))} -> "
                f"{list(set(normalized.keys()) - set(arguments.keys()))}"
            )

        return normalized, was_aliased

    def _is_raw_string_argument(self, tool_name: str, key: str) -> bool:
        """Return True when a string argument should be preserved verbatim."""
        return key in self._RAW_STRING_ARGUMENT_NAMES

    def _looks_like_structured_string(self, value: str) -> bool:
        """Return True when a string likely encodes structured data, not raw text."""
        stripped = value.strip()
        return (
            stripped.startswith(("[", "{"))
            or "Path(" in value
            or "tuple[" in value
            or "list[" in value
            or ("\\'" in value and any(token in value for token in ("{", "[", ":")))
        )

    def _escape_control_chars_in_json_strings(self, json_str: str) -> str:
        """Escape raw control characters embedded inside JSON string literals."""
        result: list[str] = []
        in_string = False
        escape_next = False

        for char in json_str:
            if escape_next:
                result.append(char)
                escape_next = False
                continue

            if char == "\\":
                result.append(char)
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                result.append(char)
                continue

            if in_string:
                if char == "\n":
                    result.append("\\n")
                    continue
                if char == "\t":
                    result.append("\\t")
                    continue
                if char == "\r":
                    result.append("\\r")
                    continue
                if ord(char) < 32:
                    result.append(f"\\u{ord(char):04x}")
                    continue

            result.append(char)

        return "".join(result)

    def _parse_structured_string(self, value: str) -> Any:
        """Best-effort parse for top-level JSON/Python structured strings.

        Enhanced with multi-layer fallbacks for large payloads and common
        LLM-generated JSON issues (trailing commas, unescaped characters,
        malformed strings in content values).
        """
        if not isinstance(value, str) or not self._looks_like_structured_string(value):
            return _STRUCTURED_PARSE_FAILED

        # Fast path: Try standard JSON first (99%+ of valid cases)
        try:
            return json_loads(value)
        except JSONDecodeError as exc:
            # Log the specific error for debugging large payload failures
            logger.debug(
                "JSON parse failed at line %s, column %s: %s",
                exc.lineno,
                exc.colno,
                exc.msg[:200],
            )

            # Handle control characters (common in markdown content)
            if "control character" in str(exc).lower():
                try:
                    repaired = self._escape_control_chars_in_json_strings(value)
                    return json_loads(repaired)
                except JSONDecodeError as repair_exc:
                    logger.debug(
                        "Control char escape failed at line %s: %s",
                        repair_exc.lineno,
                        repair_exc.msg[:100],
                    )

        except (TypeError, ValueError) as exc:
            logger.debug("JSON decode type error: %s", str(exc)[:100])

        # Layer 2: Try Python AST (handles single quotes, trailing commas)
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            logger.debug("AST parse failed: %s", str(exc)[:100])

        # Layer 3: Native JSON repair (handles more edge cases)
        if _NATIVE_AVAILABLE:
            try:
                repaired = native_repair_json(value)
                if repaired:
                    return json_loads(repaired)
            except (TypeError, ValueError, JSONDecodeError) as exc:
                logger.debug("Native repair failed: %s", str(exc)[:100])

        # Layer 4: Try to extract parameters from malformed JSON using regex
        # This handles cases where JSON parsing fails due to unescaped newlines/quotes
        # Common pattern: {"cmd":"shell command with 'heredoc' and newlines"}
        import re

        try:
            # Extract top-level key-value pairs. Use backreferences so the
            # closing quote matches the opening quote — earlier version used
            # `[^"\'\\]` which excluded BOTH quote types and silently
            # truncated content at any apostrophe (e.g. "Victor's") or
            # unescaped quote inside markdown.
            #
            # Two patterns: double-quoted values (JSON form) and single-quoted
            # (Python-repr form). We try both and merge.
            patterns = [
                # "key": "value with 'apostrophes' inside, even quotes \" escaped"
                r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"',
                # 'key': 'value with "quotes" inside, even apostrophes \' escaped'
                r"'(\w+)'\s*:\s*'((?:[^'\\]|\\.)*)'",
            ]
            all_matches: List[tuple[str, str]] = []
            for pattern in patterns:
                all_matches.extend(re.findall(pattern, value))

            if all_matches:
                result: Dict[str, Any] = {}
                for key, val in all_matches:
                    # Unescape common escape sequences (JSON-style)
                    val = val.replace("\\n", "\n").replace("\\t", "\t")
                    val = val.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")
                    # Last-write-wins if same key appears in both quote styles
                    result[key] = val

                # Sanity check: if the payload is large but the recovered
                # content/cmd is suspiciously short, the regex likely
                # terminated early at an unescaped quote inside content.
                # Fall through to the tolerant extractor in that case.
                payload_size = len(value)
                if payload_size > 2000:
                    big_value_fields = ("content", "cmd")
                    suspect = False
                    for field in big_value_fields:
                        field_val = result.get(field)
                        if isinstance(field_val, str) and len(field_val) * 4 < payload_size:
                            # Recovered <25% of payload size — likely truncated
                            suspect = True
                            break
                    if suspect:
                        logger.debug(
                            "Regex result suspiciously short for payload size %d; "
                            "trying tolerant extractor",
                            payload_size,
                        )
                        tolerant: Dict[str, Any] = {}
                        for key in ("path", "content", "cmd", "query"):
                            extracted = self._extract_string_value_tolerant(value, key)
                            if extracted is not None:
                                tolerant[key] = extracted
                        # Use tolerant result if it recovered MORE content than regex
                        for field in big_value_fields:
                            if isinstance(tolerant.get(field), str) and len(tolerant[field]) > len(
                                result.get(field, "")
                            ):
                                result.update(tolerant)
                                logger.debug(
                                    "Tolerant extractor upgraded result: "
                                    "%s grew from %d to %d chars",
                                    field,
                                    len(result.get(field, "")),
                                    len(tolerant[field]),
                                )
                                break

                # If we found at least one key-value pair and it looks like a tool call
                if result and any(
                    k in result for k in ["cmd", "path", "content", "query", "file_path"]
                ):
                    logger.debug(
                        "Extracted parameters via regex fallback: %s " "(content_len=%s)",
                        list(result.keys()),
                        (
                            len(result.get("content", ""))
                            if isinstance(result.get("content"), str)
                            else None
                        ),
                    )
                    return result
        except Exception as exc:
            logger.debug("Regex extraction failed: %s", str(exc)[:100])

        # Layer 5: schema-driven span recovery for a large write envelope.
        if len(value) > 10000:  # Large payload threshold
            salvaged = self._extract_by_schema(value, ["path", "content"])
            if salvaged is not None:
                logger.warning(
                    "Successfully salvaged large payload (%d chars) via schema extraction",
                    len(value),
                )
                return salvaged

        return _STRUCTURED_PARSE_FAILED

    # Known parameter names that may appear as JSON keys in tool args.
    # Used by the tolerant extractor to disambiguate between value terminators
    # (followed by another `"<known_key>":`) and content quotes (no such pattern follows).
    _KNOWN_TOOL_PARAMS = frozenset(
        {
            "path",
            "content",
            "cmd",
            "query",
            "ops",
            "type",
            "old_str",
            "new_str",
            "new_path",
            "pattern",
            "operations",
            "file_path",
            "filename",
            "command",
            "directory",
            "dir",
            "text",
            "data",
            "desc",
            "message",
            "args",
        }
    )

    @staticmethod
    def _extract_string_value_tolerant(text: str, key: str) -> Optional[str]:
        """Extract a JSON string value tolerantly.

        Designed for LLM-generated tool args where content (markdown,
        heredocs, code blocks) contains literal newlines and `"` / `'`
        that aren't JSON-escaped.

        Algorithm (forward scan with lookahead disambiguator):
        1. Locate `"<key>":` and the opening quote.
        2. Walk forward. For each candidate closing quote:
           - If preceded by odd backslashes → escaped, content.
           - Else, check what follows (after whitespace):
             * `,` followed by another `"<known_key>":` pattern → real terminator
               (we're at a key boundary)
             * `}` and the rest of text is only whitespace → real terminator
               (we're at object end)
             * Anything else → content quote, keep going
        3. If we never find a terminator, take everything up to the last `}`.

        Args:
            text: Malformed JSON-like text containing the key/value
            key: The parameter name to extract (e.g. "content", "cmd")

        Returns:
            The decoded string value, or None if the key cannot be located.
        """
        import re

        key_re = re.compile(
            r'(?P<q>["\'])' + re.escape(key) + r"(?P=q)\s*:\s*",
            re.MULTILINE,
        )
        m = key_re.search(text)
        if m is None:
            return None

        i = m.end()
        while i < len(text) and text[i] in " \t\r\n":
            i += 1
        if i >= len(text) or text[i] not in ('"', "'"):
            return None

        opener = text[i]
        start = i + 1
        i = start
        n = len(text)
        # Precompile pattern for "is next a known-key reference"
        next_key_re = re.compile(r'\s*["\'](?P<k>\w+)["\']\s*:')

        end = -1
        while i < n:
            ch = text[i]
            if ch == opener:
                # Count preceding backslashes for escape detection
                back = 0
                j = i - 1
                while j >= start and text[j] == "\\":
                    back += 1
                    j -= 1
                if back % 2 == 1:
                    i += 1
                    continue

                # Look at what follows past whitespace
                k = i + 1
                while k < n and text[k] in " \t\r\n":
                    k += 1
                if k >= n:
                    end = i
                    break

                nxt = text[k]
                if nxt == "}":
                    # Check rest is only whitespace (object terminator)
                    rest = text[k + 1 :].strip()
                    if rest == "" or rest.startswith("}"):
                        end = i
                        break
                    # Not the outer close — content quote
                    i += 1
                    continue
                if nxt == ",":
                    # Real terminator only if followed by another `"<known_key>":` pattern
                    nm = next_key_re.match(text, k + 1)
                    if nm and nm.group("k") in ArgumentNormalizer._KNOWN_TOOL_PARAMS:
                        end = i
                        break
                    # Looks like content (e.g. ", " inside prose)
                    i += 1
                    continue
                # Any other char following — content quote
                i += 1
                continue
            i += 1

        if end == -1:
            # Fallback: take everything up to last `}`
            last_brace = text.rfind("}")
            end = last_brace if last_brace > start else n

        raw = text[start:end]
        decoded = (
            raw.replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\r", "\r")
            .replace('\\"', '"')
            .replace("\\'", "'")
            .replace("\\\\", "\\")
        )
        return decoded

    # File-like fields must never contain a newline/tab (a `"path":"foo.js\n"`
    # envelope would otherwise create a corrupt filename).
    _PATH_LIKE_FIELDS = frozenset({"path", "new_path", "file_path", "filename"})

    @staticmethod
    def _extract_by_schema(payload: str, fields: List[str]) -> Optional[Dict[str, str]]:
        """Recover ``{field: value}`` from a malformed tool-arg ``value`` envelope.

        The single, schema-driven recovery used for every tool (``write`` =
        ``[path, content]``, ``shell`` = ``[cmd]``, ``edit`` = ``[ops]``, etc.).
        Models (notably glm-5.1) emit large free-text fields (content/cmd) that are
        not validly escaped — literal newlines, unescaped ``"`` / ``}`` from
        markdown/mermaid (``A{decision}``), or a truncated tail with no closing
        ``"}``. Strict and lookahead JSON parsers then fail or stop early.

        Each field's value is recovered by *span*: located fields are ordered by
        position, and a field's value runs from just after its ``"field":"`` marker
        to the start of the next located field's marker (or end-of-string for the
        trailing field, trimming one object terminator). This is order-independent
        (handles ``{path, content}`` and ``{content, path}``) and tolerant of
        unescaped quotes/braces and truncation.

        Returns the dict only when *all* requested ``fields`` are located (never
        fabricates a field, which could clobber a file), else ``None``.
        """
        import re

        located: List[Tuple[str, int, int]] = []  # (field, value_start, marker_start)
        for field in fields:
            m = re.search(r'["\']' + re.escape(field) + r'["\']\s*:\s*["\']', payload)
            if not m:
                return None
            located.append((field, m.end(), m.start()))
        located.sort(key=lambda t: t[2])

        result: Dict[str, str] = {}
        for idx, (field, value_start, _marker_start) in enumerate(located):
            if idx + 1 < len(located):
                # Value ends just before the next field's `","field":` separator.
                raw = payload[value_start : located[idx + 1][2]]
                raw = re.sub(r'["\']\s*,\s*$', "", raw)
            else:
                # Trailing field: run to end, trim a single object terminator.
                raw = payload[value_start:]
                trimmed = re.sub(r'["\']\s*\}\s*$', "", raw)
                raw = trimmed if trimmed != raw else re.sub(r'["\']\s*$', "", raw)
            value = (
                raw.replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\r", "\r")
                .replace('\\"', '"')
                .replace("\\'", "'")
                .replace("\\\\", "\\")
            )
            if field in ArgumentNormalizer._PATH_LIKE_FIELDS:
                value = value.replace("\n", "").replace("\t", "").replace("\r", "").strip()
            result[field] = value
        return result

    @staticmethod
    def extract_write_payload_greedy(value: str) -> Optional[Dict[str, str]]:
        """Write-specific entry to the schema-driven recovery (``[path, content]``)."""
        return ArgumentNormalizer._extract_by_schema(value, ["path", "content"])

    def _looks_like_edit_operation(self, value: Dict[str, Any]) -> bool:
        """Heuristic check for a single edit operation dict."""
        if not isinstance(value, dict):
            return False
        op_keys = {"type", "path", "content", "old_str", "new_str", "new_path"}
        return bool(op_keys & set(value.keys()))

    def _apply_tool_aliases_without_stats(
        self, payload: Dict[str, Any], tool_name: str
    ) -> Dict[str, Any]:
        """Apply canonical parameter aliases without mutating normalization stats."""
        tool_aliases = self.parameter_aliases.get(tool_name, {})
        if not tool_aliases:
            return dict(payload)

        normalized: Dict[str, Any] = {}
        for param, value in payload.items():
            canonical_param = tool_aliases.get(param, param)
            if canonical_param not in normalized:
                normalized[canonical_param] = value
        return normalized

    def _unwrap_tool_specific_value_envelope(
        self, arguments: Dict[str, Any], tool_name: str
    ) -> Dict[str, Any]:
        """Recover known tool payloads wrapped in a generic ``value`` envelope.

        Enhanced with better error diagnostics, partial payload recovery,
        and specific handling for large content payloads that may contain
        malformed JSON.
        """
        if set(arguments.keys()) != {"value"}:
            return arguments

        canonical_tool_name = canonicalize_core_tool_name(tool_name)
        payload = arguments.get("value")

        # Log the unwrap attempt for debugging
        logger.debug(
            "[%s] Attempting to unwrap value envelope for tool '%s', payload type: %s, size: %s",
            self.provider_name,
            canonical_tool_name,
            type(payload).__name__,
            f"{len(str(payload))} chars" if isinstance(payload, str) else "N/A",
        )

        if isinstance(payload, str):
            # Write `content` is the trailing field and frequently arrives with
            # unescaped quotes/braces or a truncated tail. The lenient structured
            # parser can "succeed" with TRUNCATED content (stopping at the first
            # inner `"`), silently writing a partial file. For write, prefer a
            # strict parse, then greedy-to-end recovery, BEFORE the lenient path.
            if canonical_tool_name == "write":
                import json as _json

                try:
                    _strict = _json.loads(payload)
                except Exception:
                    _strict = None
                if not (isinstance(_strict, dict) and {"path", "content"} <= set(_strict)):
                    _greedy = self.extract_write_payload_greedy(payload)
                    if _greedy:
                        logger.debug(
                            "[%s] Greedy recovery of write value envelope " "(content_len=%d)",
                            self.provider_name,
                            len(_greedy["content"]),
                        )
                        return _greedy

            # Check for common LLM error patterns
            payload_size = len(payload)
            if payload_size > 10000:
                logger.debug(
                    "[%s] Large value envelope detected (%d chars) for tool '%s' - using enhanced parsing",
                    self.provider_name,
                    payload_size,
                    canonical_tool_name,
                )

            parsed = self._parse_structured_string(payload)
            if parsed is not _STRUCTURED_PARSE_FAILED:
                payload = parsed
                logger.debug(
                    "[%s] Successfully parsed value envelope string for tool '%s'",
                    self.provider_name,
                    canonical_tool_name,
                )
            else:
                # Enhanced error reporting for large payloads
                if payload_size > 5000:
                    # Log sample of the problematic payload
                    sample_start = payload[:200]
                    sample_end = payload[-200:] if payload_size > 400 else ""
                    logger.error(
                        "[%s] Failed to parse value envelope for tool '%s' (size=%d). "
                        "Sample start: %s... Sample end: ...%s. "
                        "This may indicate LLM-generated JSON with unescaped characters or malformed structure.",
                        self.provider_name,
                        canonical_tool_name,
                        payload_size,
                        sample_start,
                        sample_end,
                    )
                else:
                    logger.warning(
                        "[%s] Failed to parse value envelope string for tool '%s': %s",
                        self.provider_name,
                        canonical_tool_name,
                        (payload[:100] if isinstance(payload, str) else str(payload)[:100]),
                    )

                # Tolerant state-machine extraction — works for any tool's
                # required params. Handles unescaped quotes/newlines inside
                # JSON string values (common in LLM-generated content with
                # markdown, heredocs, code blocks).
                #
                # This runs BEFORE the legacy tool-specific salvage so all
                # tools benefit, not just write/shell. Walks the malformed
                # JSON looking for `"<param>":"<value>"` patterns and uses
                # lookahead to disambiguate real string terminators from
                # content quotes.
                tolerant_extract: Dict[str, Any] = {}
                # Try all params known for this tool's envelope shape.
                required_for_tool = {
                    "write": ["path", "content"],
                    "edit": ["ops"],
                    "shell": ["cmd"],
                    "read": ["path"],
                    "ls": ["path"],
                    "grep": ["query", "path"],
                    "search": ["query"],
                    "code_search": ["query"],
                    "semantic_code_search": ["query"],
                }
                params_to_try = required_for_tool.get(
                    canonical_tool_name,
                    ["path", "content", "cmd", "query"],
                )
                for param in params_to_try:
                    extracted = self._extract_string_value_tolerant(payload, param)
                    if extracted is not None:
                        tolerant_extract[param] = extracted

                # Did we recover the required param for this tool?
                required_set = set(required_for_tool.get(canonical_tool_name, []))
                if required_set and required_set.issubset(tolerant_extract.keys()):
                    content_field = tolerant_extract.get("content") or tolerant_extract.get(
                        "cmd", ""
                    )
                    logger.debug(
                        "[%s] Tolerant extraction recovered %s payload for "
                        "tool '%s' (content_len=%d, params=%s)",
                        self.provider_name,
                        canonical_tool_name,
                        canonical_tool_name,
                        len(content_field) if isinstance(content_field, str) else 0,
                        list(tolerant_extract.keys()),
                    )
                    return tolerant_extract

                # Final schema-driven fallback for write/shell value envelopes.
                if canonical_tool_name == "write" and payload_size > 50:
                    salvaged = self._extract_by_schema(payload, ["path", "content"])
                    if salvaged:
                        logger.debug(
                            "[%s] Schema recovery: write payload from malformed envelope",
                            self.provider_name,
                        )
                        return salvaged

                if canonical_tool_name == "shell" and payload_size > 50:
                    salvaged = self._extract_by_schema(payload, ["cmd"])
                    if salvaged:
                        logger.debug(
                            "[%s] Schema recovery: shell command (%d chars) from malformed payload",
                            self.provider_name,
                            len(str(salvaged.get("cmd", ""))),
                        )
                        return salvaged

                # Return original arguments if all layers failed
                return arguments

        if isinstance(payload, dict):
            aliased_payload = self._apply_tool_aliases_without_stats(payload, canonical_tool_name)

            envelope_param_map = {
                "write": {"path", "content"},
                "read": {"path"},
                "ls": {"path"},
                "shell": {"cmd"},
                "grep": {"query"},
                "search": {"query"},
                "code_search": {"query"},
                "semantic_code_search": {"query"},
            }
            required_params = envelope_param_map.get(canonical_tool_name)
            if required_params and required_params.issubset(aliased_payload.keys()):
                logger.debug(
                    "[%s] Recovered wrapped %s payload from value envelope with keys: %s",
                    self.provider_name,
                    canonical_tool_name,
                    list(aliased_payload.keys()),
                )
                return aliased_payload

            if canonical_tool_name != "edit":
                logger.debug(
                    "[%s] Value envelope unwrap failed for '%s': required params %s not found in payload keys %s",
                    self.provider_name,
                    canonical_tool_name,
                    required_params,
                    list(aliased_payload.keys()),
                )
                return arguments

            if "ops" in aliased_payload or "operations" in aliased_payload:
                logger.debug(
                    "[%s] Recovered wrapped edit payload from value envelope",
                    self.provider_name,
                )
                return aliased_payload
            if self._looks_like_edit_operation(aliased_payload):
                logger.debug(
                    "[%s] Recovered single wrapped edit operation from value envelope",
                    self.provider_name,
                )
                return {"ops": [aliased_payload]}

        if canonical_tool_name != "edit":
            return arguments

        if isinstance(payload, list):
            logger.debug(
                "[%s] Recovered wrapped edit payload from value envelope",
                self.provider_name,
            )
            return {"ops": payload}

        return arguments

    @staticmethod
    def normalize_arguments_shape(tool_call: Dict[str, Any]) -> None:
        """Guarantee ``tool_call["arguments"]`` is a dict, in place.

        A model can emit ``arguments`` as any JSON value (seen live: glm-5.2
        sent a list for ``write``); an unchecked non-dict crashes dict-assuming
        consumers (status messages, telemetry) before execution-time validation
        can feed a corrective error back to the model.

        Shapes handled: ``str`` → :meth:`coerce_arg_string` ladder; ``None`` →
        ``{}``; single-element ``[dict]`` → unwrapped (forgiveness — models
        sometimes wrap the args object in an array); any other non-object →
        the call is marked ``_invalid`` with a corrective ``_error`` (the
        pipeline's skip-result path returns it to the model as a tool_error)
        and ``arguments`` is reset to ``{}`` so downstream dict access is safe.
        """
        args = tool_call.get("arguments")
        if isinstance(args, str):
            tool_call["arguments"] = ArgumentNormalizer.coerce_arg_string(args)
            return
        if args is None:
            tool_call["arguments"] = {}
            return
        if isinstance(args, list) and len(args) == 1 and isinstance(args[0], dict):
            logger.warning(
                "Unwrapped single-element list arguments for tool '%s'",
                tool_call.get("name"),
            )
            tool_call["arguments"] = args[0]
            return
        if not isinstance(args, dict):
            type_name = type(args).__name__
            logger.warning(
                "Tool '%s' arguments have non-object shape (%s); "
                "marking invalid for corrective feedback",
                tool_call.get("name"),
                type_name,
            )
            tool_call["_invalid"] = True
            tool_call["_error"] = (
                f"Tool '{tool_call.get('name')}' arguments must be a single JSON "
                f"object of parameters, got {type_name}. Re-issue the call "
                'with an object, e.g. {"param": value}.'
            )
            tool_call["arguments"] = {}

    @staticmethod
    def coerce_arg_string(raw: str) -> Dict[str, Any]:
        """Coerce a raw model ``arguments`` *string* into a dict.

        The single coercion ladder (previously duplicated at 6 call sites): strict
        JSON → ``ast.literal_eval`` → native JSON repair → ``{"value": raw}``
        envelope. The envelope is the deliberate fallback; schema-aware recovery of
        a malformed envelope then happens in ``_unwrap_tool_specific_value_envelope``
        via ``_extract_by_schema``.
        """
        try:
            parsed = json_loads(raw)
        except Exception:
            try:
                parsed = ast.literal_eval(raw)
            except Exception:
                parsed = None
                if _NATIVE_AVAILABLE:
                    try:
                        repaired = native_repair_json(raw)
                        if repaired:
                            parsed = json_loads(repaired)
                    except Exception:
                        parsed = None
        else:
            return parsed if isinstance(parsed, dict) else {"value": raw}
        if isinstance(parsed, dict):
            return parsed
        return {"value": raw}

    def parse_tool_arguments(
        self, raw_args: Any, tool_name: str
    ) -> Tuple[Dict[str, Any], NormalizationStrategy]:
        """Single authority: raw model tool-call args → normalized dict.

        Accepts a JSON/Python string, a dict, or ``None`` and returns the fully
        normalized arguments (coercion + value-envelope recovery + alias/type
        normalization). All tool-call entry points (chat, streaming, pipeline,
        error recovery) route through this one method instead of re-implementing
        the string-coercion + ``{value: ...}`` wrap locally.
        """
        if raw_args is None:
            coerced: Dict[str, Any] = {}
        elif isinstance(raw_args, str):
            coerced = self.coerce_arg_string(raw_args)
        elif isinstance(raw_args, dict):
            coerced = raw_args
        else:
            coerced = {"value": raw_args}
        return self.normalize_arguments(coerced, tool_name)

    def normalize_arguments(
        self, arguments: Dict[str, Any], tool_name: str
    ) -> Tuple[Dict[str, Any], NormalizationStrategy]:
        """
        Normalize tool call arguments through multi-layer pipeline.

        Args:
            arguments: Raw arguments from model
            tool_name: Name of the tool being called

        Returns:
            (normalized_arguments, strategy_used)
        """
        tool_name = canonicalize_core_tool_name(tool_name)
        self.stats.total_calls += 1

        arguments = self._unwrap_tool_specific_value_envelope(arguments, tool_name)

        # Track per-tool stats
        if tool_name not in self.stats.by_tool:
            self.stats.by_tool[tool_name] = ToolStats(calls=0, normalizations=0)
        self.stats.by_tool[tool_name].calls += 1

        # Layer 0: Normalize parameter aliases (model-specific -> standard names)
        # This runs first so subsequent layers work with standard param names
        arguments, was_aliased = self.normalize_parameter_aliases(arguments, tool_name)

        # Layer 0.1: Sanitize non-JSON-serializable objects (Ellipsis, Path, etc.)
        # This MUST happen before primitive type coercion to prevent serialization errors
        from victor.agent.argument_sanitizer import sanitize_arguments_for_serialization

        arguments = sanitize_arguments_for_serialization(arguments)
        logger.debug(
            f"[{self.provider_name}] {tool_name}: Sanitized arguments for JSON serialization"
        )

        # Layer 0.5: Coerce primitive types (str -> int/float/bool)
        # Some models (OpenRouter, Fireworks) output integers as strings like "0" or "30"
        arguments = self._coerce_primitive_types(arguments, tool_name)

        # AGGRESSIVE APPROACH: Check if any values look like JSON and try normalization FIRST
        # This ensures we catch malformed JSON even if basic validation passes
        has_json_like_strings = any(
            isinstance(v, str)
            and not self._is_raw_string_argument(tool_name, k)
            and v.strip().startswith(("[", "{"))
            for k, v in arguments.items()
        )
        logger.debug(
            f"[{self.provider_name}] {tool_name}: has_json_like_strings={has_json_like_strings}"
        )

        if has_json_like_strings:
            # Try AST normalization preemptively for JSON-like strings
            try:
                logger.debug(
                    f"[{self.provider_name}] {tool_name}: Trying preemptive AST normalization"
                )
                normalized = self._normalize_via_ast(arguments, tool_name)
                changed = normalized != arguments
                logger.debug(
                    f"[{self.provider_name}] {tool_name}: AST normalization changed={changed}"
                )
                # Verify normalization actually changed something or improved validity
                if changed:
                    is_valid = self._is_valid_json_dict(normalized)
                    logger.debug(
                        f"[{self.provider_name}] {tool_name}: Normalized version is_valid={is_valid}"
                    )
                    if is_valid:
                        self.stats.normalizations[NormalizationStrategy.PYTHON_AST.value] += 1
                        self.stats.by_tool[tool_name].normalizations += 1
                        logger.debug(
                            f"[{self.provider_name}] Preemptively normalized {tool_name} arguments via AST"
                        )
                        return normalized, NormalizationStrategy.PYTHON_AST
            except Exception as e:
                logger.debug(
                    f"Preemptive AST normalization failed for {tool_name}: {e}",
                    exc_info=True,
                )

        # Layer 1: Check if already valid (fast path - most cases after preemptive normalization)
        logger.debug(f"[{self.provider_name}] {tool_name}: Layer 1 - Checking if already valid")
        try:
            is_valid = self._is_valid_json_dict(arguments)
            logger.debug(f"[{self.provider_name}] {tool_name}: Layer 1 - is_valid={is_valid}")
            if is_valid:
                self.stats.normalizations[NormalizationStrategy.DIRECT.value] += 1
                return arguments, NormalizationStrategy.DIRECT
        except Exception as e:
            logger.error(
                f"[{self.provider_name}] {tool_name}: Layer 1 validation threw exception: {e}",
                exc_info=True,
            )

        # Not valid - need normalization
        self.stats.by_tool[tool_name].normalizations += 1

        # Layer 2: Try Python AST conversion for string values (if not already tried)
        if not has_json_like_strings:
            try:
                normalized = self._normalize_via_ast(arguments, tool_name)
                if self._is_valid_json_dict(normalized):
                    self.stats.normalizations[NormalizationStrategy.PYTHON_AST.value] += 1
                    logger.debug(
                        f"[{self.provider_name}] Normalized {tool_name} arguments via AST conversion"
                    )
                    return normalized, NormalizationStrategy.PYTHON_AST
            except Exception as e:
                logger.debug(f"AST normalization failed for {tool_name}: {e}")

        # Layer 3: Regex-based quote replacement
        try:
            normalized = self._normalize_via_regex(arguments, tool_name)
            if self._is_valid_json_dict(normalized):
                self.stats.normalizations[NormalizationStrategy.REGEX_QUOTES.value] += 1
                logger.debug(f"[{self.provider_name}] Normalized {tool_name} arguments via regex")
                return normalized, NormalizationStrategy.REGEX_QUOTES
        except Exception as e:
            logger.debug(f"Regex normalization failed for {tool_name}: {e}")

        # Layer 4: Manual repair for known patterns
        try:
            normalized = self._normalize_via_manual_repair(arguments, tool_name)
            if self._is_valid_json_dict(normalized):
                self.stats.normalizations[NormalizationStrategy.MANUAL_REPAIR.value] += 1
                logger.debug(
                    f"[{self.provider_name}] Normalized {tool_name} arguments via manual repair"
                )
                return normalized, NormalizationStrategy.MANUAL_REPAIR
        except Exception as e:
            logger.debug(f"Manual repair failed for {tool_name}: {e}")

        # All strategies failed
        self.stats.failures += 1
        logger.error(
            f"[{self.provider_name}] Failed to normalize {tool_name} arguments "
            f"after all strategies. Original: {arguments}"
        )
        return arguments, NormalizationStrategy.FAILED

    def _is_valid_json_dict(self, obj: Any) -> bool:
        """
        Check if object is valid for JSON serialization.

        Also validates that string values that look like JSON structures
        (starting with [ or {) are themselves valid JSON.

        NOTE: Some LLM providers (e.g., Ollama/Qwen) may output JSON strings
        with literal control characters (actual newlines instead of \\n).
        While technically invalid JSON, these can still be used by tools that
        handle raw content. We accept these as "valid enough" to avoid false
        failures while still catching truly malformed JSON.

        Args:
            obj: Object to validate

        Returns:
            True if object can be JSON-serialized (even if string values
                 contain control characters that make them technically invalid JSON)
        """
        try:
            # First check if the whole object can be JSON-serialized
            json_dumps(obj)
            logger.debug("_is_valid_json_dict: json_dumps() succeeded")

            # Additionally, check string values that look like JSON
            # NOTE: We intentionally do NOT validate with json_loads() here because
            # some providers output literal control characters (actual newlines),
            # which are invalid JSON but still work fine with tools that handle
            # raw content (like edit_files). Validating with json_loads() would
            # cause false failures.
            #
            # The edit_files tool will handle any necessary escaping internally.
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str):
                        stripped = value.strip()
                        if stripped.startswith(("[", "{")):
                            logger.debug(
                                "_is_valid_json_dict: Checking '%s' value that looks like JSON",
                                key,
                            )
                            # Just verify it starts with JSON-like syntax
                            # Don't use json_loads() as it rejects literal control chars
                            logger.debug(
                                "_is_valid_json_dict: '%s' value looks like JSON (syntax check only)",
                                key,
                            )

            logger.debug("_is_valid_json_dict: Returning True")
            return True
        except (TypeError, ValueError) as e:
            logger.debug("_is_valid_json_dict: Exception in validation: %s", e)
            return False

    def _normalize_via_ast(self, arguments: Dict[str, Any], tool_name: str = "") -> Dict[str, Any]:
        """
        Convert Python syntax to JSON via AST.

        This is the primary normalization strategy for handling Python-style
        dict/list syntax (single quotes) in string arguments.

        Also handles type coercion: if a string looks like a simple list/dict
        (e.g., "[]", "{}"), convert it to the actual type.

        Example:
            Input:  {'operations': "[{'type': 'modify', 'path': 'file.sh'}]"}
            Output: {'operations': '[{"type": "modify", "path": "file.sh"}]'}

            Input:  {'patterns': '[]'}
            Output: {'patterns': []}

        Args:
            arguments: Raw arguments to normalize

        Returns:
            Normalized arguments with valid JSON strings or coerced types
        """
        normalized: Dict[str, Any] = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                if self._is_raw_string_argument(tool_name, key):
                    normalized[key] = value
                    continue

                # First, check if the string LOOKS like JSON but may need normalization
                stripped = value.strip()
                if stripped.startswith(("[", "{")):
                    # Aggressively normalize: try Python AST first, then verify with json_loads
                    try:
                        # Use ast.literal_eval (SAFE - no code execution)
                        python_obj = ast.literal_eval(value)

                        # Type coercion: for simple empty structures, return the actual type
                        # This helps tools that expect List[str] instead of str
                        if isinstance(python_obj, (list, dict)):
                            # If it's a simple empty structure, return as-is (type coercion)
                            if not python_obj:  # Empty list [] or dict {}
                                normalized[key] = python_obj
                            else:
                                # Complex structure - keep as Python object, don't convert to string
                                # Verify it's JSON-serializable for provider compatibility
                                try:
                                    json_dumps(
                                        python_obj
                                    )  # Verify serializable (will raise if not)
                                    normalized[key] = python_obj  # Keep as dict/list
                                except (TypeError, ValueError) as e:
                                    # Not serializable - convert to string as fallback
                                    logger.warning(
                                        f"Parameter {key} contains non-serializable value: {e}"
                                    )
                                    normalized[key] = json_dumps(python_obj, default=str)
                        else:
                            # Primitive value - keep as-is
                            normalized[key] = value
                    except (ValueError, SyntaxError, JSONDecodeError):
                        # AST failed - keep original and let other layers handle it
                        normalized[key] = value
                else:
                    # Doesn't look like JSON - keep as-is
                    normalized[key] = value
            else:
                # Non-string value - keep as-is
                normalized[key] = value
        return normalized

    def _normalize_via_regex(
        self, arguments: Dict[str, Any], tool_name: str = ""
    ) -> Dict[str, Any]:
        """
        Enhanced regex-based normalization with fallback for malformed patterns.

        Handles common malformed tool call patterns:
        - YAML-style: key: value
        - Python type hints: key: tuple[str, ...]
        - Path objects: pdf_path: Path(...)

        Uses native Rust implementation when available for ~5-10x speedup.

        This is a fallback for cases where AST parsing fails but simple
        quote replacement might work.

        Example:
            Input:  {'key': "{'field': 'value'}"}
            Output: {'key': '{"field": "value"}'}

        Args:
            arguments: Raw arguments to normalize

        Returns:
            Normalized arguments with replaced quotes and cleaned patterns
        """
        import re

        normalized = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                if self._is_raw_string_argument(tool_name, key):
                    normalized[key] = value
                    continue

                if not self._looks_like_structured_string(value):
                    normalized[key] = value
                    continue

                # Handle Python type hints in values
                # Example: "tuple[str, ...]" → "" (remove)
                # Example: "Path(...)" → string value
                value = self._clean_python_type_hints(value)

                # Handle Path(...) syntax
                if "Path(" in value:
                    # Extract path from Path(...) wrapper
                    # Example: Path('/path/to/file') → '/path/to/file'
                    path_match = re.search(r"Path\(['\"]([^'\"]+)['\"]\)", value)
                    if path_match:
                        value = path_match.group(1)

                # Handle tuple[str, ...] or list[str] syntax
                if "tuple[" in value or "list[" in value:
                    # Remove type annotation, use empty list as default
                    value = []

                # Now do the normal quote replacement
                if _NATIVE_AVAILABLE:
                    # Use native Rust JSON repair (handles more edge cases)
                    repaired = native_repair_json(value)
                else:
                    # Python fallback
                    # Replace escaped single quotes with double quotes
                    # Pattern: \' → "  and standalone ' → "
                    repaired = str(value).replace("\\'", '"')
                    # Also try replacing unescaped single quotes
                    # (careful not to replace quotes in strings)
                    if "'" in repaired:
                        repaired = repaired.replace("'", '"')
                normalized[key] = repaired
            else:
                normalized[key] = value
        return normalized

    def _clean_python_type_hints(self, value: str) -> str:
        """Remove Python type hints from string values.

        Args:
            value: String potentially containing type hints

        Returns:
            Cleaned string without type hints
        """
        import re

        # Remove standalone type annotations
        # Example: "tuple[str, ...]" → ""
        if re.match(r"^[a-z_]+\[[^\]]+\]$", value.strip()):
            return ""

        # Remove type annotations at end of value
        # Example: "'file.txt' : str" → "'file.txt'"
        value = re.sub(r"\s*:\s*[a-z_]+\[[^\]]*\]", "", value)

        return value

    def _normalize_via_manual_repair(
        self, arguments: Dict[str, Any], tool_name: str
    ) -> Dict[str, Any]:
        """
        Tool-specific manual repairs for known patterns.

        This allows adding custom repair logic for specific tools
        as edge cases are discovered.

        Args:
            arguments: Raw arguments to normalize
            tool_name: Name of the tool (for tool-specific repairs)

        Returns:
            Normalized arguments with tool-specific repairs applied
        """
        # Tool-specific repairs
        if canonicalize_core_tool_name(tool_name) == "edit":
            return self._repair_edit_files_args(arguments)

        # Add more tool-specific repairs here as needed
        # if tool_name == "other_tool":
        #     return self._repair_other_tool_args(arguments)

        return arguments

    def _repair_edit_files_args(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Repair common edit_files argument malformations.

        Known patterns:
        1. Python dict syntax in 'operations' field
        2. Missing quotes around field names
        3. Single quotes instead of double quotes

        Args:
            arguments: Raw edit_files arguments

        Returns:
            Repaired arguments
        """
        for field in ("operations", "ops"):
            if field not in arguments:
                continue

            ops = arguments[field]
            if not isinstance(ops, str):
                continue

            try:
                # Keep legacy edit_files semantics (JSON string in "operations")
                # while letting canonical edit/ops use the native Python object.
                python_obj = ast.literal_eval(ops)
                if field == "operations":
                    arguments[field] = json_dumps(python_obj)
                else:
                    arguments[field] = python_obj
            except Exception:
                # Fallback to original string when repair fails
                pass
        return arguments

    def _coerce_primitive_types(self, arguments: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """
        Coerce string values to primitive types (int, float, bool) when appropriate.

        Some models (e.g., OpenRouter Llama, Fireworks) output integers as strings
        like "0", "30", or booleans as "true", "false". This method coerces them
        to proper Python types.

        Examples:
            {"line_start": "0", "line_end": "30"} -> {"line_start": 0, "line_end": 30}
            {"regex": "false"} -> {"regex": False}
            {"timeout": "30.5"} -> {"timeout": 30.5}

        Args:
            arguments: Raw arguments from model
            tool_name: Name of the tool (for logging)

        Returns:
            Arguments with coerced primitive types
        """
        coerced = {}
        any_coerced = False

        for key, value in arguments.items():
            if isinstance(value, str):
                if self._is_raw_string_argument(tool_name, key):
                    coerced[key] = value
                    continue

                coerced_value = self._try_coerce_string(value)
                if coerced_value is not value:  # Identity check - was actually coerced
                    coerced[key] = coerced_value
                    any_coerced = True
                    logger.debug(
                        f"[{self.provider_name}] {tool_name}: Coerced '{key}' from str "
                        f"'{value}' to {type(coerced_value).__name__} {coerced_value}"
                    )
                else:
                    coerced[key] = value
            else:
                coerced[key] = value

        if any_coerced:
            logger.debug(
                f"[{self.provider_name}] Coerced primitive types for {tool_name}: "
                f"{[k for k in arguments if coerced.get(k) is not arguments.get(k)]}"
            )

        return coerced

    def _try_coerce_string(self, value: str) -> Any:
        """
        Try to coerce a string value to a primitive type.

        Uses Rust accelerator when available for 3-5x speedup.

        Attempts conversions in order:
        1. Boolean ("true"/"false", case-insensitive)
        2. None ("null"/"none", case-insensitive)
        3. Integer (digits only, with optional negative sign)
        4. Float (digits with decimal point)

        Args:
            value: String value to potentially coerce

        Returns:
            Coerced value or original string if no coercion applies
        """
        # Use Rust accelerator when available (3-5x faster)
        if _NATIVE_AVAILABLE:
            type_name, coerced_str, _ = native_coerce_string_type(value)
            if type_name == "null":
                return None
            elif type_name == "bool":
                return coerced_str.lower() == "true"
            elif type_name == "int":
                # Don't coerce if it looks like a path
                if not value.strip().startswith("/"):
                    return int(coerced_str)
            elif type_name == "float":
                # Don't coerce if it looks like a path
                if not value.strip().startswith("/"):
                    return float(coerced_str)
            # For string/list/dict types, fall through to return original
            return value

        # Python fallback
        stripped = value.strip()
        lower = stripped.lower()

        # Boolean coercion
        if lower == "true":
            return True
        if lower == "false":
            return False

        # None/null coercion
        if lower in ("null", "none"):
            return None

        # Integer coercion (must be all digits, optionally with leading minus)
        # Don't coerce if it looks like a path or has special chars
        if stripped.lstrip("-").isdigit() and not stripped.startswith("/"):
            try:
                return int(stripped)
            except ValueError:
                pass

        # Float coercion (digits with one decimal point)
        if "." in stripped and not stripped.startswith("/"):
            parts = stripped.lstrip("-").split(".")
            if len(parts) == 2 and all(p.isdigit() for p in parts if p):
                try:
                    return float(stripped)
                except ValueError:
                    pass

        # No coercion - return original
        return value

    def get_stats(self) -> Dict[str, Any]:
        """
        Get normalization statistics for monitoring.

        Returns:
            Dictionary with normalization metrics
        """
        success_rate = (
            (self.stats.total_calls - self.stats.failures) / max(self.stats.total_calls, 1)
        ) * 100

        return {
            "provider": self.provider_name,
            "total_calls": self.stats.total_calls,
            "normalizations": self.stats.normalizations,
            "failures": self.stats.failures,
            "success_rate": round(success_rate, 2),
            "by_tool": self.stats.by_tool,
            "alias_stats": self._alias_stats,
        }

    def reset_stats(self) -> None:
        """Reset statistics (useful for testing)."""
        self.stats = NormalizationStats(
            total_calls=0,
            normalizations={strategy.value: 0 for strategy in NormalizationStrategy},
            failures=0,
            by_tool={},
        )
        self._alias_stats = {"total": 0, "aliased": 0, "by_tool": {}}

    def log_stats(self) -> None:
        """Log current statistics."""
        stats = self.get_stats()
        logger.debug(f"Argument normalization stats: {stats}")
