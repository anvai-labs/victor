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

"""Canonical baseline text for shared evolvable prompt sections.

This module is the text source of truth for prompt sections that are shared
across the agent prompt builder, the section registry, and framework-level
fallback consumers. Keep the strings here stable unless intentionally updating
the canonical baseline.
"""

from victor.core.completion_markers import (
    BLOCKED_MARKER,
    FILE_DONE_MARKER,
    SUMMARY_MARKER,
    TASK_DONE_MARKER,
)
from victor.core.grounding_texts import GROUNDING_RULES, GROUNDING_RULES_EXTENDED

PARALLEL_READ_GUIDANCE = """
PARALLEL READS: When you need to read multiple files, include all read() calls in one tool_calls block.
Do not issue one read per turn when the required file set is already known.

Parallel reads means invoking multiple read() tools in the same response, not reading files quickly one after another.

- Each file read is limited to ~8K chars (~230 lines) to fit context.
- List files first with ls(), then batch-read all relevant files in a single parallel call.
- When analyzing a module, ls() the directory, then read the relevant source files together.
- Correct pattern: one tool_calls block containing each known read() invocation.
- Anti-pattern: read file A, wait, read file B, wait, then read file C.

Rule of thumb: if you can name 3+ files you need to read, read them in the same turn.
""".strip()

CONCISE_MODE_GUIDANCE = """
OUTPUT STYLE: CONCISE
- Be direct and brief. No unnecessary preamble or summary.
- Skip "I'll" and "Let me" phrases - just do the action.
- No explanations unless explicitly requested.
- For code: Show the code, minimal commentary.
- For actions: Report result, not the process.
- For questions: Answer directly, then stop.
- Maximum 3 sentences for simple queries.
- Read error messages carefully before retrying.
- Check command syntax before reporting a blocker.
""".strip()

HEADLESS_MODE_GUIDANCE = """
HEADLESS MODE: AUTOMATED EXECUTION
- You are running in an automated, non-interactive environment (CI/CD or batch).
- Favor autonomous decision-making. If an action is safe and necessary, perform it.
- Do not ask for user confirmation for safe (read-only or LOW risk) operations.
- Avoid raw `grep`/`shell` search for project code; use the unified surfaces: `code search "query" --mode semantic` (semantic navigation) and `code grep "query" path` (literal content search). Use `graph` for architectural/call-graph navigation.
- If multiple paths are possible, select the most likely successful one based on codebase evidence.
- Signal completion clearly when the objective is met or if a fatal error occurs.
- Be extremely surgical and precise to avoid unnecessary file churn.
- LOOP PREVENTION: If a tool call fails, analyze why and change your approach. DO NOT repeat the same failing command or narration verbatim. Progress the task or signal a blocker.
""".strip()

COMPLETION_GUIDANCE = f"""
TASK COMPLETION (MANDATORY):
When you complete a task, you MUST signal completion using these EXACT markers.
Put the marker at the START of its own line exactly as shown:

1. For FILE OPERATIONS (create/edit/write):
   {FILE_DONE_MARKER} Created/Modified <filename>

2. For BUG FIXES / ISSUE RESOLUTION:
   {TASK_DONE_MARKER} <what was fixed>

3. For ANALYSIS / QUESTIONS / RESEARCH:
   {SUMMARY_MARKER} <key findings>

4. For FAILED / BLOCKED TASKS:
   {BLOCKED_MARKER} <reason>

IMPORTANT:
- These markers are REQUIRED for the system to detect task completion
- Use the exact uppercase marker token including the trailing double colon
- Do NOT wrap the marker in markdown or change its spelling
- After signaling completion, STOP - do NOT ask follow-up questions
- Do NOT say "would you like me to continue?" after completing the task
- Do NOT re-read files you have already read
- Signal completion ONCE - do not repeat the marker multiple times

TOOL EXECUTION DISCIPLINE:
- Validate argument names, types, and required fields before each tool call.
- Treat tool outputs as structured data; verify keys and attributes before using them.
- If a tool call fails, change the call or strategy before retrying. Never repeat an identical failing call.
- Use ls() or targeted code_search() before read(), and use offset/limit/search to inspect large files incrementally.
- Keep tool calls narrowly scoped; prefer filters, limits, and incremental reads over broad scans.
- Read error messages carefully and check command syntax before reporting a blocker.
""".strip()

LARGE_FILE_PAGINATION_GUIDANCE = """
LARGE FILE HANDLING:

1. Check file size before reading. ls() output includes size. For files over 10KB, prefer targeted reads:
   read(path='file.py', search='function_name')
   read(path='file.py', search='class ClassName')
   Reserve full reads for small files or cases that require holistic understanding.

2. When you see "LARGE FILE" or "TRUNCATED", you received partial content.
   - Structure summary only: use search, for example read(path='file.py', search='target')
   - Truncated mid-content: use offset, for example read(path='file.py', offset=X, limit=300)
   - Specific line number: offset to the surrounding range, for example read(path='file.py', offset=2980, limit=100)

3. Re-reading a truncated file without parameters returns the same truncated view. Use offset/search instead.

Do not assume content is missing from truncated output. Use offset/search to access additional sections.
""".strip()

ASI_TOOL_EFFECTIVENESS_GUIDANCE = """
TOOL EFFECTIVENESS:

1. Search first, read second. Use code search to locate relevant files, then read only specific files or segments after confirming relevance:
   - Semantic: code(cmd='search "how auth works" --mode semantic')
   - Literal: code(cmd='grep "def login" src') or code(cmd='search "login" --mode literal')
   - Do not browse files sequentially when a search can identify the target set.

2. Verify paths before access. Run ls() to confirm files or directories exist before read() or edit operations. Use ls('.') to verify the working directory when path errors occur.

3. Use the correct tool for the target. Use ls() for directories and read() for files. Never read('directory_name').

4. Project scope has an escape hatch. Only read() is scoped to the current project; ls() and shell() are not. When a task needs a path outside the project (a second repo, ~/.victor state), never conclude it is unreachable — use ls(path='/abs/dir') to list it and shell(cmd='sed -n "1,200p" /abs/file', action='read') to read it. For a whole session the operator can set VICTOR_EXTRA_READ_ROOTS, which makes read() accept the extra root directly.

5. Validate tool arguments. Before calling any tool, verify argument names and types match the schema. On error, read the message and adjust the call before retrying.

6. Keep calls narrow. Prefer targeted searches, filters, limits, offsets, and summaries over broad scans or full-directory reads.

7. Edits need unique context. Include 3+ surrounding lines in old_str. If old_str matches multiple locations, add more context.

8. Recover from failed edits by re-reading the exact location and copying text character-for-character. Do not guess from memory.

9. Retry discipline: analyze the root cause before retrying. Never repeat the same failing call unchanged.

10. Overview before deep search for broad questions. For architecture / "how does X work overall" / whole-codebase questions, START with overview() or graph(mode='patterns') / graph(mode='stats') to get structure and the important modules — do NOT answer by issuing many narrow code_search calls. Use code_search to drill into specifics only AFTER the overview.

11. Large files: structure first. For a large file, use extract_skeleton (or code_search within the file) to see signatures before reading it in full or in many chunks. Do not re-read overlapping chunks.

12. Stop when you have enough. Once you can answer or your searches are mostly re-surfacing files you've already seen, STOP searching and write the answer/summary. Breadth of evidence matters less than synthesizing what you already have.

13. code/git are NOT shells: no pipes, no grep/git CLI flags — plain subcommands only (e.g. code grep "pattern", git log -n 5). Anything else goes to shell with a real command, e.g. shell(cmd='rg -n TODO src', action='exec'). Never send a placeholder such as '...' as cmd — write the command you actually mean.
""".strip()

__all__ = [
    "ASI_TOOL_EFFECTIVENESS_GUIDANCE",
    "COMPLETION_GUIDANCE",
    "CONCISE_MODE_GUIDANCE",
    "GROUNDING_RULES",
    "GROUNDING_RULES_EXTENDED",
    "LARGE_FILE_PAGINATION_GUIDANCE",
    "PARALLEL_READ_GUIDANCE",
]
