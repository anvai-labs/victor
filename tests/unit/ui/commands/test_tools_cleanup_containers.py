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

"""`victor tools cleanup-containers` removes only orphaned sandboxes.

It used to remove every labelled sandbox container, including the live one of
a concurrent session. It now asks the coding vertical's owner-aware reaper.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from victor.ui.commands.tools import tools_app

runner = CliRunner()


def _container(short_id, labels=None):
    container = MagicMock()
    container.short_id = short_id
    container.name = f"name-{short_id}"
    container.labels = {"victor.sandbox": "code-executor"} if labels is None else labels
    return container


def _invoke(orphans, all_containers=(), args=("--dry-run",)):
    client = MagicMock()
    client.containers.list.return_value = list(all_containers)
    module = SimpleNamespace(
        DOCKER_AVAILABLE=True, find_orphaned_sandboxes=MagicMock(return_value=list(orphans))
    )
    with (
        patch("victor.core.utils.capability_loader.load_code_executor_module", return_value=module),
        patch("docker.from_env", return_value=client),
    ):
        result = runner.invoke(tools_app, ["cleanup-containers", *args])
    return result, module


def test_lists_only_what_the_owner_aware_reaper_selects():
    dead = _container("dead01")
    result, module = _invoke([dead])
    assert result.exit_code == 0, result.output
    module.find_orphaned_sandboxes.assert_called_once()
    assert "dead01" in result.output
    dead.remove.assert_not_called()  # dry run


def test_live_session_sandbox_is_never_listed():
    live = _container("live01")  # labelled, but its owner is alive: reaper excludes it
    result, _ = _invoke([], all_containers=[live])
    assert result.exit_code == 0, result.output
    assert "No orphaned containers found" in result.output
    live.remove.assert_not_called()


def test_unlabeled_scan_skips_labelled_sandboxes():
    live = _container("live01")
    legacy = _container("old001", labels={})
    legacy.image.tags = ["python:3.11-slim"]
    legacy.attrs = {"Config": {"Cmd": ["sleep", "infinity"]}}
    result, _ = _invoke([], all_containers=[live, legacy], args=("--dry-run", "-u"))
    assert result.exit_code == 0, result.output
    assert "old001" in result.output
    assert "live01" not in result.output


def test_confirmed_cleanup_removes_orphans():
    dead = _container("dead01")
    client = MagicMock()
    module = SimpleNamespace(DOCKER_AVAILABLE=True, find_orphaned_sandboxes=lambda c: [dead])
    with (
        patch("victor.core.utils.capability_loader.load_code_executor_module", return_value=module),
        patch("docker.from_env", return_value=client),
    ):
        result = runner.invoke(tools_app, ["cleanup-containers"], input="y\n")
    assert result.exit_code == 0, result.output
    dead.remove.assert_called_once_with(force=True)
