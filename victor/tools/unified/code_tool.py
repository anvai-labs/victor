import argparse
import asyncio
import os
from pathlib import Path
import shlex
import sys
from io import StringIO
from typing import Optional

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool
from victor.tools.unified.parser import (
    detect_shell_operators,
    shell_operator_rejection,
    split_command,
)


async def run_tests(runner: str, path: str):
    """Run tests through the production shell surface."""
    from victor.tools.bash import shell

    cmd = f"{shlex.quote(runner)} {shlex.quote(path)}"
    return await shell(cmd=cmd, readonly=True, action="read")


async def execute_python(code: str):
    """Execute a short Python snippet through the production shell surface."""
    from victor.tools.bash import shell

    result = await shell(
        cmd=f"python -c {shlex.quote(code)}",
        readonly=False,
        action="execute",
    )
    if isinstance(result, dict):
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        if result.get("success") is False:
            return stderr or result.get("error", "")
        return stdout or stderr
    return str(result)


async def analyze_metrics(path: str):
    """Return lightweight code metrics for Python files under a path."""
    root = Path(path).expanduser()
    files = [root] if root.is_file() else list(root.rglob("*.py"))
    loc = 0
    functions = 0
    classes = 0
    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = text.splitlines()
        loc += sum(1 for line in lines if line.strip())
        functions += sum(1 for line in lines if line.lstrip().startswith("def "))
        classes += sum(1 for line in lines if line.lstrip().startswith("class "))
    return {"files": len(files), "loc": loc, "functions": functions, "classes": classes}


# Wall-clock budget for literal scans so a pathological tree can never stall a
# turn for minutes (env-overridable, mirroring the semantic path's
# VICTOR_TIMEOUT_* convention).
_LITERAL_SEARCH_TIMEOUT_ENV = "VICTOR_TIMEOUT_LITERAL_SEARCH"
_LITERAL_SEARCH_TIMEOUT_DEFAULT = 30.0

# Wall-clock budget for semantic/graph delegation so a hung vertical backend
# (measured live: victor-coding code_search stalls past the 60s pipeline cap
# and burns the turn) degrades to literal grep instead of timing the whole
# tool out. Same env-overridable convention as the literal budget.
_SEMANTIC_SEARCH_TIMEOUT_ENV = "VICTOR_TIMEOUT_SEMANTIC_SEARCH"
_SEMANTIC_SEARCH_TIMEOUT_DEFAULT = 45.0


def _literal_search_timeout() -> float:
    raw = os.environ.get(_LITERAL_SEARCH_TIMEOUT_ENV, "")
    try:
        return float(raw) if raw else _LITERAL_SEARCH_TIMEOUT_DEFAULT
    except ValueError:
        return _LITERAL_SEARCH_TIMEOUT_DEFAULT


async def _timed_grep_search(
    query: str,
    path: str,
    regex: bool = False,
    case_sensitive: bool = False,
    include_glob: Optional[str] = None,
):
    """Run :func:`grep_search` under the literal-search wall-clock budget."""
    from victor.tools.unified._search_helpers import grep_search

    kwargs = {"query": query, "path": path, "regex": regex, "case_sensitive": case_sensitive}
    if include_glob is not None:
        # Forwarded only when set so existing grep_search doubles/monkeypatches
        # without the parameter keep working unchanged.
        kwargs["include_glob"] = include_glob
    return await asyncio.wait_for(
        grep_search(**kwargs),
        timeout=_literal_search_timeout(),
    )


# --- grep-idiom forgiveness -------------------------------------------------
#
# `code grep` is not GNU grep, but models call it as if it were (measured: 96%
# of `code` errors were grep-style flags). Pre-parse normalization ABSORBS
# flags whose behavior is always-on, MAPS flags with a supported equivalent,
# and REJECTS everything else with a corrective hint.

_CODE_SUBCOMMANDS = ("search", "grep", "test", "python", "execute", "metrics")

# Always-on behaviors: line numbers, recursion, case-insensitive matching.
_GREP_ABSORB_SHORT = frozenset("nrRi")
_GREP_MAP_SHORT = {"l": "--files-only", "E": "--regex", "P": "--regex"}
_GREP_ABSORB_LONG = frozenset({"--line-number", "--recursive", "--ignore-case"})
_GREP_MAP_LONG = {
    "--files-with-matches": "--files-only",
    "--extended-regexp": "--regex",
    "--perl-regexp": "--regex",
}
_GREP_PASSTHROUGH_LONG = frozenset({"--regex", "--case-sensitive", "--files-only", "--include"})


def _unsupported_grep_flag(flag: str) -> str:
    return (
        "### ⚠️ UNSUPPORTED GREP FLAG\n"
        f"`code grep` is not GNU grep — `{flag}` is not supported here.\n"
        'Supported: code grep "PATTERN" [PATH] [--regex] [-C|--case-sensitive] '
        "[-l|--files-only] [--include GLOB]\n"
        "Line numbers and recursive search are always on; matching is "
        "case-insensitive by default.\n"
        f"  • For full grep syntax use the shell tool:  shell(cmd='grep {flag} ...')\n"
        "  • Or drop the flag and rerun this `code grep` call."
    )


def _unknown_subcommand_redirect(name: str) -> str:
    return (
        "### ⚠️ UNKNOWN SUBCOMMAND\n"
        f"`code` has no `{name}` subcommand. Supported: "
        f"{', '.join(_CODE_SUBCOMMANDS)}.\n"
        "  • To list files, use the `ls` tool or shell(cmd='ls ...').\n"
        "  • To find files by name, use shell(cmd='find ...').\n"
        '  • To search file contents: code grep "PATTERN" [PATH] [--include GLOB].'
    )


def _normalize_grep_flags(tokens: list) -> tuple[list, Optional[str]]:
    """Normalize grep-style flags in ``tokens`` (everything after ``grep``).

    Returns ``(normalized_tokens, rejection)``: on success ``rejection`` is
    ``None``; on an unsupported flag it is the full corrective message and the
    call must not proceed. Tokens at and after a literal ``--`` are untouched.
    """
    out: list = []
    for i, token in enumerate(tokens):
        if token == "--":
            out.extend(tokens[i:])
            return out, None
        if token.startswith("--"):
            name, _, _ = token.partition("=")
            if name in _GREP_ABSORB_LONG:
                continue
            if name in _GREP_MAP_LONG:
                mapped = _GREP_MAP_LONG[name]
                if mapped not in out:
                    out.append(mapped)
                continue
            if name in _GREP_PASSTHROUGH_LONG:
                out.append(token)  # keeps the `--include=X` form intact
                continue
            return [], _unsupported_grep_flag(token)
        if token.startswith("-") and len(token) > 1:
            # Short-flag run like `-rn`: absorb/map letter by letter.
            mapped_flags: list = []
            keep_case_sensitive = False
            for letter in token[1:]:
                if letter in _GREP_ABSORB_SHORT:
                    continue
                if letter in _GREP_MAP_SHORT:
                    mapped_flags.append(_GREP_MAP_SHORT[letter])
                    continue
                if letter == "C":
                    # Note: `-C` collides with GNU grep's context-lines flag;
                    # here it is `code grep`'s own --case-sensitive alias and
                    # passes through unchanged.
                    keep_case_sensitive = True
                    continue
                return [], _unsupported_grep_flag(token)
            for mapped in mapped_flags:
                if mapped not in out:
                    out.append(mapped)
            if keep_case_sensitive:
                out.append("-C")
            continue
        out.append(token)
    return out, None


def _semantic_search_timeout() -> float:
    raw = os.environ.get(_SEMANTIC_SEARCH_TIMEOUT_ENV, "")
    try:
        return float(raw) if raw else _SEMANTIC_SEARCH_TIMEOUT_DEFAULT
    except ValueError:
        return _SEMANTIC_SEARCH_TIMEOUT_DEFAULT


def _semantic_timeout_message(query: str) -> str:
    return (
        f"### ❌ ERROR\nSemantic search for {query!r} exceeded "
        f"{_semantic_search_timeout():.0f}s and was cancelled. Narrow the path "
        f"argument, use --mode literal, or raise {_SEMANTIC_SEARCH_TIMEOUT_ENV}."
    )


def _literal_timeout_message(query: str) -> str:
    return (
        f"### ❌ ERROR\nLiteral search for {query!r} exceeded "
        f"{_literal_search_timeout():.0f}s and was cancelled. Narrow the path "
        f"argument, or raise {_LITERAL_SEARCH_TIMEOUT_ENV}."
    )


class UnifiedCodeParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        raise ValueError(f"Argument parsing error: {message}")


def create_code_parser() -> UnifiedCodeParser:
    parser = UnifiedCodeParser(
        prog="code", description="Unified code operations.", exit_on_error=False
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="The operation to perform")

    # `test` subcommand
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("runner", help="Test runner (e.g. pytest)")
    test_parser.add_argument("path", nargs="?", default=".", help="Path to tests")

    # `execute` / `python` subcommands
    exec_parser = subparsers.add_parser("execute", help="Execute python code")
    exec_parser.add_argument("code", help="Python code to execute")

    python_parser = subparsers.add_parser("python", help="Execute python code")
    python_parser.add_argument("code", help="Python code to execute")

    # `metrics` subcommand
    metrics_parser = subparsers.add_parser("metrics", help="Analyze code metrics")
    metrics_parser.add_argument("path", nargs="?", default=".", help="Directory to analyze")

    # `grep` subcommand — literal content search (the canonical grep surface)
    grep_parser = subparsers.add_parser("grep", help="Search file contents (literal/regex)")
    grep_parser.add_argument("query", help="Text or regex to search for")
    grep_parser.add_argument("path", nargs="?", default=".", help="File or directory to search")
    grep_parser.add_argument("--regex", action="store_true", help="Treat query as a regex")
    grep_parser.add_argument(
        "--case-sensitive",
        "-C",
        action="store_true",
        help="Case-sensitive match (default is case-insensitive)",
    )
    grep_parser.add_argument(
        "-l",
        "--files-only",
        action="store_true",
        dest="files_only",
        help="Print only file names with matches",
    )
    grep_parser.add_argument(
        "--include",
        default=None,
        metavar="GLOB",
        help="Only search files whose name matches GLOB (e.g. *.py)",
    )

    # `search` subcommand — semantic/graph code search (delegates to victor-coding)
    search_parser = subparsers.add_parser("search", help="Semantic code search")
    search_parser.add_argument("query", help="Natural-language search query")
    search_parser.add_argument("path", nargs="?", default=".", help="Scope of the search")
    search_parser.add_argument(
        "--mode",
        choices=["semantic", "literal", "hybrid", "graph"],
        default="semantic",
        help="Search mode (default: semantic)",
    )
    search_parser.add_argument("--k", type=int, default=10, help="Max results")

    return parser


@tool(
    name="code",
    access_mode=AccessMode.MIXED,
    danger_level=DangerLevel.HIGH,
    execution_category=ExecutionCategory.MIXED,
    priority=Priority.HIGH,
)
async def code_tool(cmd: str) -> str:
    """Standalone code-intelligence tool. Call it directly as code(cmd='...').
    Do NOT pass code commands to the shell tool.

    Example commands:
      code search "authentication logic"       # semantic code search
      code grep "class RST"                    # literal pattern search
      code grep "def run" src --include *.py   # only files matching a glob
      code test pytest tests/
      code python "print('hello')"
      code metrics src/

    grep flags you do NOT need: -n, -r, -R, -i are always on (absorbed if
    passed). Use --regex for regex, -C for case-sensitive, -l for file names
    only.

    Args:
        cmd: A code-intelligence subcommand string (e.g. 'grep "pattern"').
            This is NOT a shell command — call the code tool directly,
            do NOT pass it to shell(). Subcommands: search, grep, test,
            python, execute, metrics.
    """
    parser = create_code_parser()

    try:
        args_list = split_command(cmd)
        if args_list and args_list[0] == "code":
            args_list = args_list[1:]
        operator = detect_shell_operators(args_list)
        if operator is not None:
            return shell_operator_rejection("code", operator)
        if args_list and not args_list[0].startswith("-") and args_list[0] not in _CODE_SUBCOMMANDS:
            # Hallucinated subcommands (`ls`, `find`, ...) get a redirect
            # instead of an argparse "invalid choice" dump; `help` gets usage.
            if args_list[0] == "help":
                return parser.format_help()
            return _unknown_subcommand_redirect(args_list[0])
        if args_list and args_list[0] == "grep":
            normalized, rejection = _normalize_grep_flags(args_list[1:])
            if rejection is not None:
                return rejection
            args_list = ["grep"] + normalized
        parsed_args = parser.parse_args(args_list)
    except ValueError as e:
        return f"### ❌ ERROR\n{e}"
    except Exception as e:
        return f"### ❌ ERROR\nUnexpected error: {e}"

    if parsed_args.subcommand == "test":
        try:
            results = await run_tests(runner=parsed_args.runner, path=parsed_args.path)
            if isinstance(results, dict) and "output" in results:
                return str(results["output"])
            if isinstance(results, dict):
                stdout = results.get("stdout", "").strip()
                stderr = results.get("stderr", "").strip()
                if stdout or stderr:
                    return "\n\n".join(part for part in (stdout, stderr) if part)
                if results.get("success") is False:
                    return f"### ❌ ERROR\n{results.get('error', 'Test execution failed')}"
            return str(results)
        except Exception as e:
            return f"### ❌ ERROR\nTest execution failed: {e}"

    elif parsed_args.subcommand in {"execute", "python"}:
        try:
            return str(await execute_python(parsed_args.code))
        except Exception as e:
            return f"### ❌ ERROR\nCode execution failed: {e}"

    elif parsed_args.subcommand == "metrics":
        try:
            results = await analyze_metrics(parsed_args.path)
            if not isinstance(results, dict):
                return str(results)

            # Format dict as yaml-like markdown
            out = ["### Code Metrics"]
            for k, v in results.items():
                out.append(f"- **{k}**: {v}")
            return "\n".join(out)
        except Exception as e:
            return f"### ❌ ERROR\nMetrics analysis failed: {e}"

    elif parsed_args.subcommand == "grep":
        from victor.tools.unified._search_helpers import format_grep_results

        try:
            results = await _timed_grep_search(
                query=parsed_args.query,
                path=parsed_args.path,
                regex=parsed_args.regex,
                case_sensitive=parsed_args.case_sensitive,
                include_glob=parsed_args.include,
            )
            return format_grep_results(results, files_only=parsed_args.files_only)
        except asyncio.TimeoutError:
            return _literal_timeout_message(parsed_args.query)
        except Exception as e:
            return f"### ❌ ERROR\ngrep failed: {e}"

    elif parsed_args.subcommand == "search":
        return await _code_search(parsed_args)

    else:
        old_stdout = sys.stdout
        sys.stdout = capture = StringIO()
        parser.print_help()
        sys.stdout = old_stdout
        return f"### ❌ ERROR\nInvalid subcommand '{parsed_args.subcommand}'.\n\n```text\n{capture.getvalue()}```"


async def _code_search(parsed_args) -> str:
    """``code search`` — semantic/graph code search with a literal fallback.

    Delegates to ``victor_coding.code_search`` when the package is available;
    otherwise (and for ``--mode literal``) falls back to a literal grep so the
    surface always works, per the graceful-degradation principle.
    """
    from victor.tools.unified._search_helpers import format_grep_results

    if parsed_args.mode == "literal":
        try:
            results = await _timed_grep_search(query=parsed_args.query, path=parsed_args.path)
        except asyncio.TimeoutError:
            return _literal_timeout_message(parsed_args.query)
        return format_grep_results(results)

    from victor.tools.unified._vertical_resolver import resolve_vertical_callable

    search_fn, _src = resolve_vertical_callable(
        "code_search",
        fallback_module="victor_coding.tools.code_search_tool",
        fallback_attr="code_search",
    )
    if search_fn is None:
        # No semantic backend available — degrade to literal with a hint.
        try:
            results = await _timed_grep_search(query=parsed_args.query, path=parsed_args.path)
        except asyncio.TimeoutError:
            return _literal_timeout_message(parsed_args.query)
        return (
            "### 💡 SYSTEM HINT\nSemantic code search requires the victor-coding "
            "package, which is not installed. Showing literal matches instead.\n\n"
            + format_grep_results(results)
        )

    try:
        result = await asyncio.wait_for(
            search_fn(
                query=parsed_args.query,
                path=parsed_args.path,
                mode=parsed_args.mode,
                k=parsed_args.k,
            ),
            timeout=_semantic_search_timeout(),
        )
    except asyncio.TimeoutError:
        # Hung backend — degrade to literal grep rather than surfacing a bare
        # timeout the caller cannot act on (the pipeline would otherwise kill
        # the whole tool at its own cap after a longer stall).
        try:
            results = await _timed_grep_search(query=parsed_args.query, path=parsed_args.path)
        except asyncio.TimeoutError:
            return _literal_timeout_message(parsed_args.query)
        return (
            f"### 💡 SYSTEM HINT\nSemantic search timed out after "
            f"{_semantic_search_timeout():.0f}s (raise {_SEMANTIC_SEARCH_TIMEOUT_ENV} "
            "if the index is worth waiting for); showing literal grep results.\n\n"
            + format_grep_results(results)
        )
    except TypeError:
        # Signature drift between versions — retry without the ``k`` kwarg.
        try:
            result = await asyncio.wait_for(
                search_fn(query=parsed_args.query, path=parsed_args.path, mode=parsed_args.mode),
                timeout=_semantic_search_timeout(),
            )
        except asyncio.TimeoutError:
            try:
                results = await _timed_grep_search(query=parsed_args.query, path=parsed_args.path)
            except asyncio.TimeoutError:
                return _literal_timeout_message(parsed_args.query)
            return (
                f"### 💡 SYSTEM HINT\nSemantic search timed out after "
                f"{_semantic_search_timeout():.0f}s (raise {_SEMANTIC_SEARCH_TIMEOUT_ENV} "
                "if the index is worth waiting for); showing literal grep results.\n\n"
                + format_grep_results(results)
            )
        except Exception as e:
            return f"### ❌ ERROR\nSemantic search failed: {e}"
    except Exception as e:
        return f"### ❌ ERROR\nSemantic search failed: {e}"

    if isinstance(result, dict) and result.get("success") is False:
        # Backend resolved but errored (e.g. "Settings not available in tool
        # context") — degrade to literal grep, mirroring the absent-backend path.
        reason = result.get("error", "unknown error")
        try:
            results = await _timed_grep_search(query=parsed_args.query, path=parsed_args.path)
        except asyncio.TimeoutError:
            return _literal_timeout_message(parsed_args.query)
        return (
            f"### 💡 SYSTEM HINT\nSemantic backend unavailable ({reason}); "
            "showing literal grep results.\n\n" + format_grep_results(results)
        )

    return _format_code_search_result(result)


def _format_code_search_result(result: object) -> str:
    """Normalize a ``code_search`` result (dict/list/str) into a display string."""
    if isinstance(result, dict):
        if result.get("success") is False:
            return f"### ❌ ERROR\n{result.get('error', 'code search failed')}"
        results = result.get("results") or result.get("matches") or result.get("output")
        if isinstance(results, list):
            out = []
            for item in results:
                if isinstance(item, dict):
                    loc = item.get("file") or item.get("path") or item.get("location", "?")
                    line = item.get("line", "")
                    score = item.get("score")
                    snippet = (item.get("content") or item.get("snippet") or "").strip()
                    head = f"{loc}:{line}" if line else str(loc)
                    if score is not None:
                        head += f" (score: {score})"
                    out.append(f"- {head}\n  {snippet}" if snippet else f"- {head}")
                else:
                    out.append(f"- {item}")
            return "\n".join(out) if out else "No matches found."
        return str(results) if results is not None else "No matches found."
    if isinstance(result, list):
        return "\n".join(f"- {item}" for item in result) or "No matches found."
    return str(result)
