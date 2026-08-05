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

"""Focused sync-wrapper tests for legacy workflow context compatibility."""

from unittest.mock import MagicMock, patch

from victor.workflows import context as context_module


class TestExecutionContextWrapperSyncBridge:
    def test_get_uses_shared_sync_bridge_without_running_loop(self) -> None:
        wrapper = context_module.ExecutionContextWrapper(
            context_module.create_execution_context({"fallback": "local"})
        )
        wrapper._manager = MagicMock()
        coro = object()
        wrapper._manager.get = MagicMock(return_value=coro)

        with (
            patch.object(context_module.asyncio, "get_running_loop", side_effect=RuntimeError),
            patch.object(context_module, "run_sync", return_value="remote") as mock_run_sync,
        ):
            result = wrapper.get("answer", "default")

        assert result == "remote"
        wrapper._manager.get.assert_called_once_with("answer", "default")
        mock_run_sync.assert_called_once_with(coro)

    def test_get_falls_back_to_state_when_running_loop_exists(self) -> None:
        wrapper = context_module.ExecutionContextWrapper(
            context_module.create_execution_context({"fallback": "local"})
        )
        wrapper._manager = MagicMock()

        with (
            patch.object(context_module.asyncio, "get_running_loop", return_value=object()),
            patch.object(context_module, "run_sync") as mock_run_sync,
        ):
            result = wrapper.get("fallback", "default")

        assert result == "local"
        mock_run_sync.assert_not_called()

    def test_set_uses_shared_sync_bridge_without_running_loop(self) -> None:
        wrapper = context_module.ExecutionContextWrapper(context_module.create_execution_context())
        wrapper._manager = MagicMock()
        coro = object()
        wrapper._manager.set = MagicMock(return_value=coro)

        with (
            patch.object(context_module.asyncio, "get_running_loop", side_effect=RuntimeError),
            patch.object(context_module, "run_sync", return_value=None) as mock_run_sync,
        ):
            wrapper.set("answer", 42)

        wrapper._manager.set.assert_called_once_with("answer", 42)
        mock_run_sync.assert_called_once_with(coro)
        # Pydantic models use attribute access
        assert wrapper.state.data["answer"] == 42

    def test_update_uses_shared_sync_bridge_without_running_loop(self) -> None:
        wrapper = context_module.ExecutionContextWrapper(context_module.create_execution_context())
        wrapper._manager = MagicMock()
        coro = object()
        wrapper._manager.update = MagicMock(return_value=coro)
        values = {"answer": 42, "status": "ok"}

        with (
            patch.object(context_module.asyncio, "get_running_loop", side_effect=RuntimeError),
            patch.object(context_module, "run_sync", return_value=None) as mock_run_sync,
        ):
            wrapper.update(values)

        wrapper._manager.update.assert_called_once_with(values)
        mock_run_sync.assert_called_once_with(coro)
        # Pydantic models use attribute access
        assert wrapper.state.data == values
