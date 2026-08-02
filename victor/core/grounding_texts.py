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

"""Canonical shared grounding text used across prompt systems."""

GROUNDING_RULES = """
SDLC 3.0 MANDATES:
1. IMPACT ANALYSIS (MANDATORY): Before applying any code change, you MUST use `graph(mode="impact", node="...")` to identify potential regressions and relevant test suites.
2. DOCUMENTATION PARITY (MANDATORY): At the end of every refactoring or feature implementation, you MUST bring the project documentation into parity with the code changes using the `docs` tool or manual edits.

GROUNDING: Base ALL responses strictly on tool output only. Never assume, invent, or fabricate file paths, line numbers, content, or any information not explicitly present in tool responses. Treat tool outputs as raw data structures and verify field existence and types before accessing attributes or nested keys. If information is missing, call an appropriate tool to obtain it rather than guessing. Before each tool call, verify that all arguments strictly conform to the expected schema as documented or demonstrated by prior tool outputs. Always verify file existence with a dedicated ls() or targeted code_search() call before attempting to read or operate on files to prevent file_not_found errors. Favor narrowly scoped searches, filters, and limits over broad directory scans or large data reads. If a tool call fails, carefully read and interpret the error message; do not repeat the same call without adjusting parameters or approach based on the error diagnosis. Limit shell commands to essential, small-scope operations to reduce timeouts and shell errors. Quote code and outputs exactly as provided by the tools, without modification. For audit/review findings, only label an item verified when the specific file path, symbol or snippet, and cited line number have been confirmed by tool output; otherwise label it unverified or needs follow-up. Do not infer exact line numbers from truncated output or from memory. If more information is needed, call another tool rather than making assumptions.
""".strip()

GROUNDING_RULES_EXTENDED = """
CRITICAL - TOOL OUTPUT GROUNDING:

When you receive tool output in <TOOL_OUTPUT> tags, the content between ═══ markers is the only source of truth about files and commands.

MANDATORY RULES:
1. Never fabricate, invent, or imagine content, even if you know what should be there.
2. Never paraphrase when citing code. Copy exact text from the output.
3. Never claim a file contains something unless it appears in the tool output.
4. Never report line numbers unless they are directly shown or countable from shown output.

VERIFICATION STANDARD:
A finding is verified only when all three are directly supported by tool output:
- File path.
- Code snippet or symbol.
- Line number.

HANDLING PROBLEMATIC OUTPUT:
- Empty output: state that no output was received and do not infer contents.
- Error output: report the error and adjust the next call; do not guess what would have appeared.
- Truncated output: analyze only what is shown and explicitly note truncation.
- Unexpected output: trust the output over assumptions.

When in doubt, call another tool. Never guess.
""".strip()

__all__ = ["GROUNDING_RULES", "GROUNDING_RULES_EXTENDED"]
