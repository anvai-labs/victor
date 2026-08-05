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

"""Consolidated dangerous command detection.

Single source of truth for command safety validation, used by:
- victor/tools/subprocess_executor.py
- victor/tools/bash.py
- victor/tools/code_executor_tool.py
- victor/security/safety/code_patterns.py

Provides exact-match and pattern-based blocking for destructive
or irreversible shell commands.
"""

from __future__ import annotations

# Exact command strings that are always blocked.
DANGEROUS_COMMANDS: frozenset[str] = frozenset(
    {
        "rm -rf /",
        "rm -rf /*",
        "dd",
        "mkfs",
        ":(){ :|:& };:",  # Fork bomb
        "> /dev/sda",
    }
)

# Substring patterns — if any appears in the command, it is blocked.
DANGEROUS_PATTERNS: tuple[str, ...] = (
    "rm -rf /",
    "rm -rf /*",
    "rm -rf $HOME",
    "rm -rf ~",
    "dd if=/dev/",
    "dd of=/dev/",
    "mkfs.",
    "> /dev/sd",
    "wget | sh",
    "wget | bash",
    "curl | sh",
    "curl | bash",
    ":(){",  # Fork bomb variant
    "chmod 777 /",
    "chown root /",
)


def is_dangerous_command(command: str) -> bool:
    """Check if a command is potentially dangerous.

    Performs both exact-match and substring-pattern checks
    against the consolidated blocklists.

    Args:
        command: Shell command string to check.

    Returns:
        True if the command matches any dangerous pattern.
    """
    command_lower = command.lower().strip()

    if command_lower in DANGEROUS_COMMANDS:
        return True

    return any(pattern.lower() in command_lower for pattern in DANGEROUS_PATTERNS)
