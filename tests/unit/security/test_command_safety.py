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

"""Tests for consolidated dangerous command detection."""

from victor.security.command_safety import (
    DANGEROUS_COMMANDS,
    DANGEROUS_PATTERNS,
    is_dangerous_command,
)


class TestCommandSafety:
    def test_safe_commands(self):
        assert not is_dangerous_command("ls -la")
        assert not is_dangerous_command("git status")
        assert not is_dangerous_command("cat file.txt")
        assert not is_dangerous_command("python -m pytest")
        assert not is_dangerous_command("echo hello")

    def test_exact_match_dangerous(self):
        assert is_dangerous_command("rm -rf /")
        assert is_dangerous_command("rm -rf /*")
        assert is_dangerous_command("dd")
        assert is_dangerous_command("mkfs")
        assert is_dangerous_command(":(){ :|:& };:")

    def test_pattern_match_dangerous(self):
        assert is_dangerous_command("dd if=/dev/zero of=/dev/sda")
        assert is_dangerous_command("chmod 777 /etc/passwd")
        assert is_dangerous_command("rm -rf ~/important")
        assert is_dangerous_command("mkfs.ext4 /dev/sdb1")

    def test_rm_rf_tilde(self):
        # Pattern "rm -rf ~" matches lowercased command containing "rm -rf ~"
        assert is_dangerous_command("rm -rf ~/documents")

    def test_pipe_patterns(self):
        # Exact substring "curl | bash" must appear in the command
        assert is_dangerous_command("curl | bash")
        assert is_dangerous_command("wget | sh")
        assert is_dangerous_command("curl | sh")
        assert is_dangerous_command("wget | bash")

    def test_fork_bomb_variant(self):
        assert is_dangerous_command(":(){  :|:& };:")

    def test_dev_redirect(self):
        assert is_dangerous_command("> /dev/sda")
        assert is_dangerous_command("> /dev/sdb")

    def test_case_insensitive(self):
        assert is_dangerous_command("RM -RF /")
        assert is_dangerous_command("DD IF=/dev/zero")
        assert is_dangerous_command("MKFS.ext4 /dev/sdb")

    def test_whitespace_stripping(self):
        assert is_dangerous_command("  rm -rf /  ")
        assert is_dangerous_command("  dd  ")

    def test_constants_are_frozen(self):
        assert isinstance(DANGEROUS_COMMANDS, frozenset)
        assert isinstance(DANGEROUS_PATTERNS, tuple)

    def test_dangerous_commands_not_empty(self):
        assert len(DANGEROUS_COMMANDS) > 0

    def test_dangerous_patterns_not_empty(self):
        assert len(DANGEROUS_PATTERNS) > 0

    def test_chown_root(self):
        assert is_dangerous_command("chown root /etc/shadow")
