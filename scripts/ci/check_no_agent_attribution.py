#!/usr/bin/env python3
"""Block third-party AI-agent authorship attribution in commit messages and PR text.

Policy: commit titles/bodies and PR title/body must not carry *third-party* agent
authorship attribution — co-author trailers naming another vendor's agent or a
bot account, `Generated-by:`/`Assisted-by:` trailers, "Generated with ..."
taglines, the robot emoji, agent-email trailers, or agent signature strings (e.g.
"Claude Opus 4.8", "Claude Code", "Gemini Pro").

**Our own agent is the exception.** Victor is anvai-labs' first-party agent and we
credit it deliberately, so `Generated-by: victor-code-ai` and a `victor-code-ai`
co-author trailer pass (see ALLOWED_PATTERNS). This mirrors the Sandhi policy
(sandhi/scripts/check_no_agent_attribution.py) and the policy documented in
Victor's CLAUDE.md, which a pre-commit hook alone never enforced — this script
makes it real locally (``--message-file``) and in CI (``--range``).

This targets *attribution*, not mere mention: legitimate references such as the
`CLAUDE.md`/`GEMINI.md`/`AGENTS.md` rule files or integrating the Anthropic/OpenAI
APIs are allowed. Adjust FORBIDDEN_PATTERNS if you want a stricter bare-word rule.

Modes:
  --message-file <path>   check a single commit message file (commit-msg hook)
  --range <base>..<head>  check every commit message in the range (CI)
  --stdin                 check text read from stdin (e.g. PR title+body)

Exit 0 = clean, 1 = violation(s) found, 2 = usage error.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

# Case-insensitive attribution patterns. Each entry: (compiled regex, human label).
#
# A brand list alone is a losing game — a new agent ships under a new name and
# walks straight past it (a `Generated-by: victor-code-ai` trailer did exactly
# that). So the rules are layered: the AGENT-AGNOSTIC ones below catch the
# *shape* of machine attribution (a `Generated-by:` trailer, a bot/AI-suffixed
# co-author) regardless of which tool wrote it; the named list stays for prose
# signatures that carry no structural tell.
# Word-anchored: an unanchored alternation matches inside ordinary words (a bare
# `amp` hits "ex-amp-le", `aider` hits "raider") and would fail a human's commit.
_AGENTS = (
    r"\b(?:claude|codex|gemini|copilot|chatgpt|gpt-\d|anthropic|openai|bard|llm"
    r"|cursor|devin|aider|cline|windsurf|junie|jules|sourcegraph)\b"
)

# Our own agent is the exception. Victor (anvai-labs' agent) is first-party
# tooling we credit deliberately, so `Generated-by: victor-code-ai` and a
# `victor-code-ai` co-author trailer are ALLOWED; the rules below still block
# third-party agent attribution. Any violation whose matched text names the
# first-party identity is dropped in `scan`.
ALLOWED_PATTERNS = [
    (re.compile(r"\bvictor(?:[-_.]code)?(?:[-_.]ai)?\b", re.I), "first-party agent (victor)"),
]
# Names that mark a machine author regardless of vendor: an `-ai`/`ai-` segment,
# or a `-bot`/`[bot]` marker. Deliberately NOT keyed on `users.noreply.github.com`
# — that is the normal privacy address for human co-authors.
_BOT_NAME = r"(?:[-_.]ai\b|\bai[-_.]|\[bot\]|[-_.]bot\b|\bbot\b)"
FORBIDDEN_PATTERNS = [
    (
        re.compile(rf"^\s*co-authored-by:.*(?:{_AGENTS})", re.I | re.M),
        "co-author trailer naming an AI agent",
    ),
    (
        re.compile(rf"^\s*co-authored-by:.*{_BOT_NAME}", re.I | re.M),
        "co-author trailer naming a bot/AI account",
    ),
    # These consume the REST OF THE LINE on purpose: the reported match is what
    # ALLOWED_PATTERNS is tested against, so a one-char `\S` would hide the name
    # and the first-party carve-out could never fire.
    (
        re.compile(r"^\s*generated-by:[ \t]*\S.*$", re.I | re.M),
        "'Generated-by:' trailer (machine authorship attribution)",
    ),
    (
        re.compile(
            r"^\s*(?:assisted-by|written-by|authored-by|ai-assisted(?:-by)?|agent):[ \t]*\S.*$",
            re.I | re.M,
        ),
        "machine-authorship trailer",
    ),
    (
        re.compile(r"generated\s+with\s+\[?\s*(?:claude|codex|gemini|copilot|chatgpt)", re.I),
        '"Generated with <agent>" tagline',
    ),
    (re.compile(r"🤖"), "robot emoji (agent-generated marker)"),
    (
        re.compile(
            r"(?:authored|written|created|generated|produced)\s+by\s+(?:claude|codex|gemini|copilot|chatgpt|anthropic|openai)",
            re.I,
        ),
        '"<verb> by <agent>" attribution',
    ),
    # Word-anchored on BOTH sides: without the trailing \b this fires on ordinary technical
    # prose in a repo that integrates these vendors — "Gemini codec", "Gemini code path",
    # "claude-3 codegen" — and blocking a legitimate commit teaches people to bypass the hook.
    (
        re.compile(r"\b(?:claude|gemini)\s+(?:opus|sonnet|haiku|code|pro|flash|\d+)\b", re.I),
        "agent model/product signature (e.g. 'Claude Opus', 'Claude Code')",
    ),
    (
        re.compile(
            r"noreply@anthropic\.com|noreply@openai\.com|@users\.noreply\.github\.com.*(?:claude|copilot)",
            re.I,
        ),
        "agent no-reply email trailer",
    ),
    (
        re.compile(r"\bco-authored-by:\s*(?:claude|copilot|codex)\b", re.I),
        "agent co-author trailer",
    ),
]


def scan(text: str, source: str) -> list[str]:
    """Return a list of violation descriptions for the given text.

    A match whose text names the first-party agent (``ALLOWED_PATTERNS``) is not
    a violation — we credit our own tooling; the rules exist to keep third-party
    agent attribution out.
    """
    violations: list[str] = []
    for rx, label in FORBIDDEN_PATTERNS:
        for m in rx.finditer(text):
            line = m.group(0).strip().replace("\n", " ")
            if any(allow.search(line) for allow, _ in ALLOWED_PATTERNS):
                continue
            violations.append(f"{source}: {label} -> '{line[:120]}'")
    return violations


def commit_messages_in_range(rng: str) -> list[tuple[str, str]]:
    """Return [(sha, message)] for each commit in the range."""
    out = subprocess.run(
        ["git", "rev-list", "--no-merges", rng],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    msgs = []
    for sha in out:
        msg = subprocess.run(
            ["git", "log", "-1", "--format=%B", sha],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        msgs.append((sha, msg))
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--message-file")
    g.add_argument("--range")
    g.add_argument("--stdin", action="store_true")
    args = ap.parse_args()

    violations: list[str] = []
    if args.message_file:
        with open(args.message_file, encoding="utf-8", errors="replace") as fh:
            violations += scan(fh.read(), "commit message")
    elif args.stdin:
        violations += scan(sys.stdin.read(), "PR title/body")
    else:
        try:
            for sha, msg in commit_messages_in_range(args.range):
                violations += scan(msg, f"commit {sha[:9]}")
        except subprocess.CalledProcessError as exc:
            print(f"error: could not read commit range '{args.range}': {exc}", file=sys.stderr)
            return 2

    if violations:
        print(
            "ERROR: AI-agent authorship attribution is not allowed in commit/PR text.",
            file=sys.stderr,
        )
        print("The human drives the code; remove the agent attribution below:\n", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        print(
            "\n(Mentions of CLAUDE.md/GEMINI.md or the Anthropic/OpenAI APIs are fine; "
            "this blocks authorship attribution only. victor-code-ai is the allowed "
            "first-party exception. See scripts/ci/check_no_agent_attribution.py.)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
