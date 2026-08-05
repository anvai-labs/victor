# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""``shell`` must name a copied documentation placeholder, not run it.

Live failure: prompt guidance read ``shell(cmd='...', action='exec')``. A model
emitted the literal ``...``; /bin/sh answered ``...: command not found`` with
exit 127, and the session concluded the shell tool did not exist and abandoned
the task. A shell error about a missing binary gives no hint that the *argument*
was the mistake.
"""

import pytest

from victor.tools.bash import _is_placeholder_cmd, shell


class TestIsPlaceholderCmd:
    @pytest.mark.parametrize(
        "cmd",
        ["...", "…", "'...'", '"..."', "  ...  ", "<cmd>", "<command>", "None", "null", "TODO"],
    )
    def test_placeholders_detected(self, cmd):
        assert _is_placeholder_cmd(cmd) is True

    @pytest.mark.parametrize(
        "cmd",
        [
            "ls -la",
            "git log -n 5",
            "python -c 'print(...)'",
            "echo ...",
            "sed -n '1,200p' file.txt",
            "rg -n TODO src",
        ],
    )
    def test_real_commands_pass_through(self, cmd):
        assert _is_placeholder_cmd(cmd) is False


class TestShellRejectsPlaceholder:
    @pytest.mark.asyncio
    async def test_placeholder_is_rejected_before_execution(self):
        result = await shell(cmd="...", action="exec")
        assert result["success"] is False
        assert "placeholder" in result["error"]
        # Must not have reached /bin/sh.
        assert result["return_code"] == -1
        assert "command not found" not in result.get("stderr", "")

    @pytest.mark.asyncio
    async def test_error_shows_a_usable_call(self):
        result = await shell(cmd="<cmd>", action="exec")
        assert "shell(cmd=" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_cmd_still_reports_missing_parameter(self):
        result = await shell(cmd="", action="exec")
        assert "Missing required parameter" in result["error"]
