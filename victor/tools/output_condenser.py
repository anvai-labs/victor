"""Command-aware condensation of shell output before LLM injection.

Inspired by rtk (github.com/rtk-ai/rtk), which compresses dev-command output
60-90% by parsing it structurally instead of truncating it generically. rtk
sits outside the agent as a CLI proxy and needs hook-based command rewriting;
Victor owns the tool-execution pipeline, so the same idea is applied natively
at shell-tool emission time.

Design contract (accuracy-first, matching the LLM-full-output decision in
``tool_service.py``):

- **Diagnostic content is never dropped.** Condensers keep failures, errors,
  tracebacks, and summary lines verbatim; only pass/progress/noise lines are
  collapsed to counts.
- **Fail-open.** A condenser that cannot confidently parse the output returns
  ``None`` and the raw output passes through unchanged.
- **Lossless escape hatch.** Whenever output is condensed, the raw output is
  teed to ``.victor/tool_output/`` and the condensed text ends with a pointer
  so the model can read the full log on demand.
- **Output still looks like real command output** so the model is not
  confused by a novel format.

Two condenser styles mirror rtk's two tiers:

1. Structural parsers (pytest, git status) that reformat output.
2. Declarative line filters (git network commands, pip/npm/cargo) that strip
   known noise lines and keep everything else — see ``LineFilterSpec``.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Pattern, Sequence, Tuple

logger = logging.getLogger(__name__)

# Outputs smaller than this are not worth condensing or teeing.
MIN_CONDENSE_LINES = 30
MIN_TEE_CHARS = 2_000
MAX_TEE_FILES = 20
MAX_TEE_BYTES = 1_048_576  # 1 MB per raw log

# Caps applied inside structural condensers (diagnostic sections are kept
# verbatim up to these bounds, then head+tail trimmed).
MAX_FAILURE_SECTION_LINES = 400
MAX_GROUP_ENTRIES = 20


@dataclass
class CondensationResult:
    """Result of condensing one shell invocation's output."""

    stdout: str
    stderr: str
    condenser: str
    original_chars: int
    condensed_chars: int
    raw_log_path: Optional[str] = None

    @property
    def savings_pct(self) -> float:
        if self.original_chars <= 0:
            return 0.0
        return 100.0 * (1 - self.condensed_chars / self.original_chars)


# ---------------------------------------------------------------------------
# Command matching
# ---------------------------------------------------------------------------

_ENV_PREFIX_RE = re.compile(r"^(?:\w+=(?:'[^']*'|\"[^\"]*\"|\S*)\s+)+")
_WRAPPER_TOKENS = ("uv run", "poetry run", "pipenv run", "hatch run")


def _effective_command(command: str) -> Optional[str]:
    """Return the segment of *command* whose output the shell will surface.

    Piped commands are excluded entirely: the pipe already transformed the
    output, and condensing it again risks double-mangling. For ``&&``/``;``
    chains the last segment wins (its output dominates and earlier segments
    are usually ``cd``/setup).
    """
    if "|" in command and "||" not in command:
        return None
    # Drop `||` alternates conservatively: match on the first alternative.
    command = command.split("||")[0]
    segments = re.split(r"&&|;", command)
    segment = segments[-1].strip() if segments else ""
    if not segment:
        return None
    segment = _ENV_PREFIX_RE.sub("", segment)
    for wrapper in _WRAPPER_TOKENS:
        if segment.startswith(wrapper + " "):
            segment = segment[len(wrapper) + 1 :]
    return segment.strip() or None


# ---------------------------------------------------------------------------
# Raw-output tee (lossless escape hatch)
# ---------------------------------------------------------------------------


def _tee_dir() -> Optional[Path]:
    try:
        from victor.config.settings import load_settings

        base = load_settings().project_victor_dir
    except Exception:
        base = Path.cwd() / ".victor"
    try:
        tee = base / "tool_output"
        tee.mkdir(parents=True, exist_ok=True)
        return tee
    except OSError:
        return None


def _rotate_tee_files(tee_dir: Path) -> None:
    try:
        logs = sorted(tee_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
        for old in logs[: max(0, len(logs) - (MAX_TEE_FILES - 1))]:
            old.unlink(missing_ok=True)
    except OSError:
        pass


def tee_raw_output(command: str, stdout: str, stderr: str) -> Optional[str]:
    """Persist raw output for later retrieval; return the file path or None."""
    combined_len = len(stdout) + len(stderr)
    if combined_len < MIN_TEE_CHARS:
        return None
    tee_dir = _tee_dir()
    if tee_dir is None:
        return None
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", command)[:40].strip("_") or "cmd"
    path = tee_dir / f"{slug}_{int(time.time() * 1000)}.log"
    body = stdout if not stderr else f"{stdout}\n--- stderr ---\n{stderr}"
    try:
        _rotate_tee_files(tee_dir)
        path.write_text(body[:MAX_TEE_BYTES], encoding="utf-8", errors="replace")
        return str(path)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Structural condenser: pytest
# ---------------------------------------------------------------------------

_PYTEST_RE = re.compile(r"^(?:python3?\s+-m\s+)?pytest\b")
_PYTEST_SUMMARY_BARE_RE = re.compile(
    r"^\d+ (?:passed|failed|skipped|error|errors|xfailed|xpassed|deselected|warning)"
)


def _cap_section(lines: List[str], cap: int) -> List[str]:
    """Head+tail cap that keeps both ends of an over-long section."""
    if len(lines) <= cap:
        return lines
    head = cap * 6 // 10
    tail = cap - head
    omitted = len(lines) - head - tail
    return lines[:head] + [f"... [{omitted} lines omitted] ..."] + lines[-tail:]


def _condense_pytest(
    command: str, stdout: str, stderr: str, return_code: int
) -> Optional[Tuple[str, str]]:
    """Keep FAILURES + short summary + final summary; collapse pass noise."""
    lines = stdout.splitlines()
    failures: List[str] = []
    short_summary: List[str] = []
    summary_line = ""
    collected_line = ""
    section = "header"

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("===") and "FAILURES" in stripped:
            section = "failures"
            continue
        if stripped.startswith("===") and "ERRORS" in stripped:
            section = "failures"
            failures.append(stripped)
            continue
        if stripped.startswith("===") and "short test summary" in stripped:
            section = "short_summary"
            continue
        if stripped.startswith("===") and "warnings summary" in stripped:
            section = "warnings"
            continue
        if stripped.startswith("===") and re.search(
            r"\d+ (?:passed|failed|error|skipped|no tests ran|xfailed|xpassed)", stripped
        ):
            summary_line = stripped
            section = "done"
            continue
        if (
            not summary_line
            and section != "failures"
            and _PYTEST_SUMMARY_BARE_RE.match(stripped)
            and " in " in stripped
        ):
            summary_line = stripped
            continue
        if "collected" in stripped and stripped.endswith(("item", "items")):
            collected_line = stripped
            continue
        if section == "failures":
            failures.append(line)
        elif section == "short_summary":
            short_summary.append(line)

    if not summary_line:
        return None  # Fail open: run broke before reporting, or unparseable.

    parts: List[str] = []
    if collected_line:
        parts.append(collected_line)
    if failures:
        parts.append("=== FAILURES (full) ===")
        parts.extend(_cap_section([ln for ln in failures if ln.strip()], MAX_FAILURE_SECTION_LINES))
    if short_summary:
        parts.append("=== short test summary ===")
        parts.extend(ln for ln in short_summary if ln.strip())
    parts.append(summary_line)
    if not failures and return_code == 0:
        parts.append("[per-test progress omitted: all tests passed]")
    condensed = "\n".join(parts)
    if len(condensed) >= len(stdout):
        return None
    return condensed, stderr


# ---------------------------------------------------------------------------
# Structural condenser: git status (long format)
# ---------------------------------------------------------------------------

_GIT_STATUS_RE = re.compile(r"^git\s+(?:-[^\s]+\s+)*status\b")
_GIT_STATUS_SECTIONS = {
    "Changes to be committed": "staged",
    "Changes not staged for commit": "modified",
    "Untracked files": "untracked",
    "Unmerged paths": "unmerged",
}


def _condense_git_status(
    command: str, stdout: str, stderr: str, return_code: int
) -> Optional[Tuple[str, str]]:
    if "--porcelain" in command or "-s" in command.split():
        return None  # Already compact.
    lines = stdout.splitlines()
    branch_lines = [ln for ln in lines[:3] if ln.startswith(("On branch", "HEAD detached"))]
    ahead_behind = [ln.strip() for ln in lines[:5] if "Your branch" in ln]
    groups: dict[str, List[str]] = {}
    current: Optional[str] = None
    recognized = False
    for line in lines:
        header = next((k for k in _GIT_STATUS_SECTIONS if line.startswith(k)), None)
        if header:
            current = _GIT_STATUS_SECTIONS[header]
            groups[current] = []
            recognized = True
            continue
        if current and line.startswith(("\t", "        ")):
            entry = line.strip()
            if entry and not entry.startswith("("):
                groups[current].append(entry)
        elif current and not line.strip():
            current = None
    if not recognized:
        if "nothing to commit" in stdout:
            summary = "\n".join(
                branch_lines + ahead_behind + ["nothing to commit, working tree clean"]
            )
            return (summary, stderr) if len(summary) < len(stdout) else None
        return None

    parts = branch_lines + ahead_behind
    for name, entries in groups.items():
        if not entries:
            continue
        shown = entries[:MAX_GROUP_ENTRIES]
        suffix = f", +{len(entries) - len(shown)} more" if len(entries) > len(shown) else ""
        parts.append(f"{name} ({len(entries)}):")
        parts.extend(f"\t{e}" for e in shown)
        if suffix:
            parts.append(f"\t...{suffix}")
    condensed = "\n".join(parts)
    if len(condensed) >= len(stdout):
        return None
    return condensed, stderr


# ---------------------------------------------------------------------------
# Declarative line filters (rtk TOML-filter analogue)
# ---------------------------------------------------------------------------


@dataclass
class LineFilterSpec:
    """Strip known noise lines; keep everything else (safe on any exit code)."""

    name: str
    match_command: Pattern[str]
    strip_lines: Sequence[Pattern[str]] = field(default_factory=list)
    max_lines: Optional[int] = None
    on_empty: str = ""

    def apply(self, text: str) -> str:
        kept = [
            line for line in text.splitlines() if not any(p.search(line) for p in self.strip_lines)
        ]
        if self.max_lines is not None:
            kept = _cap_section(kept, self.max_lines)
        result = "\n".join(kept).strip("\n")
        if not result and self.on_empty:
            return self.on_empty
        return result


def _p(*patterns: str) -> List[Pattern[str]]:
    return [re.compile(p) for p in patterns]


LINE_FILTERS: List[LineFilterSpec] = [
    LineFilterSpec(
        name="git-network",
        match_command=re.compile(r"^git\s+(?:-[^\s]+\s+)*(?:push|pull|fetch|clone)\b"),
        strip_lines=_p(
            r"^remote:\s+(?:Enumerating|Counting|Compressing|Resolving|Total)\b",
            r"^(?:Enumerating|Counting|Compressing|Resolving|Receiving|Unpacking|Writing)\s+(?:objects|deltas)",
            r"\d{1,3}%\s+\(\d+/\d+\)",
        ),
    ),
    LineFilterSpec(
        name="pip-install",
        match_command=re.compile(r"^(?:python3?\s+-m\s+)?(?:pip3?|uv\s+pip)\s+install\b"),
        strip_lines=_p(
            r"^\s*(?:Collecting|Downloading|Using cached|Requirement already satisfied)\b",
            r"^\s*(?:Preparing metadata|Getting requirements|Installing build dependencies)\b",
            r"^\s*[-\\|/]\s*$",
            r"^\s*[━╸\s]+[\d.]+/[\d.]+\s+[kMG]?B",
        ),
        on_empty="pip install: ok",
    ),
    LineFilterSpec(
        name="npm-install",
        match_command=re.compile(r"^npm\s+(?:install|ci|i)\b"),
        strip_lines=_p(
            r"^npm\s+(?:WARN\s+deprecated|notice)\b",
            r"^\s*(?:⠋|⠙|⠹|⠸|⠼|⠴|⠦|⠧|⠇|⠏)",
        ),
    ),
    LineFilterSpec(
        name="cargo-build",
        match_command=re.compile(r"^cargo\s+(?:build|check|test|clippy)\b"),
        strip_lines=_p(
            r"^\s*(?:Compiling|Downloaded|Downloading|Fresh|Checking)\s+\S+",
            r"^\s*Updating\s+crates\.io index",
        ),
    ),
]


def _apply_line_filter(
    spec: LineFilterSpec, command: str, stdout: str, stderr: str, return_code: int
) -> Optional[Tuple[str, str]]:
    new_stdout = spec.apply(stdout) if stdout else stdout
    new_stderr = spec.apply(stderr) if stderr else stderr
    if len(new_stdout) + len(new_stderr) >= len(stdout) + len(stderr):
        return None
    return new_stdout, new_stderr


# ---------------------------------------------------------------------------
# Registry + entry point
# ---------------------------------------------------------------------------

Condenser = Callable[[str, str, str, int], Optional[Tuple[str, str]]]

_STRUCTURAL: List[Tuple[Pattern[str], str, Condenser]] = [
    (_PYTEST_RE, "pytest", _condense_pytest),
    (_GIT_STATUS_RE, "git-status", _condense_git_status),
]


def condense_shell_output(
    command: str,
    stdout: str,
    stderr: str,
    return_code: int,
    *,
    tee_enabled: bool = True,
) -> Optional[CondensationResult]:
    """Condense *command*'s output if a condenser matches and helps.

    Returns None (passthrough) when: no condenser matches, output is small,
    the output is piped, parsing fails, or condensation would not shrink
    the output.
    """
    original_chars = len(stdout) + len(stderr)
    total_lines = stdout.count("\n") + stderr.count("\n") + 2
    if total_lines < MIN_CONDENSE_LINES:
        return None
    effective = _effective_command(command)
    if not effective:
        return None

    name: Optional[str] = None
    outcome: Optional[Tuple[str, str]] = None
    try:
        for pattern, cname, fn in _STRUCTURAL:
            if pattern.match(effective):
                outcome = fn(effective, stdout, stderr, return_code)
                name = cname
                break
        if outcome is None:
            for spec in LINE_FILTERS:
                if spec.match_command.match(effective):
                    outcome = _apply_line_filter(spec, effective, stdout, stderr, return_code)
                    name = spec.name
                    break
    except Exception:  # Fail open: condensation must never break tool output.
        logger.debug("Output condensation failed for %r", command, exc_info=True)
        return None

    if outcome is None or name is None:
        return None

    new_stdout, new_stderr = outcome
    raw_path = tee_raw_output(command, stdout, stderr) if tee_enabled else None
    if raw_path:
        new_stdout = f"{new_stdout}\n[condensed by victor; full output: {raw_path}]"
    condensed_chars = len(new_stdout) + len(new_stderr)
    result = CondensationResult(
        stdout=new_stdout,
        stderr=new_stderr,
        condenser=name,
        original_chars=original_chars,
        condensed_chars=condensed_chars,
        raw_log_path=raw_path,
    )
    logger.info(
        "Condensed %s output: %d→%d chars (%.0f%% saved)",
        name,
        original_chars,
        condensed_chars,
        result.savings_pct,
    )
    return result
