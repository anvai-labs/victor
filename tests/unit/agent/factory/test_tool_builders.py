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

"""Completion-tool declaration wiring (FEP-0043 guided termination).

ToolBuildersMixin resolves ``tools.completion_tools`` from nested ToolSettings
(or legacy flat settings) and threads it into the pipeline config as
``completion_tool_names`` — the caller-declared set of terminal tools the
pipeline must never skip on budget/signature/spin paths.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from victor.agent.factory.tool_builders import ToolBuildersMixin


def _builders(settings):
    builders = ToolBuildersMixin.__new__(ToolBuildersMixin)
    builders.settings = settings
    return builders


def _pipeline_config(builders):
    """Run create_tool_pipeline against mocks and return the pipeline config."""
    with patch("victor.agent.tool_pipeline.ToolPipeline") as pipeline_cls:
        builders.create_tool_pipeline(
            tools=MagicMock(),
            tool_executor=MagicMock(),
            tool_budget=134,
            tool_cache=None,
            argument_normalizer=MagicMock(),
            on_tool_start=MagicMock(),
            on_tool_complete=MagicMock(),
        )
        assert pipeline_cls.call_count == 1
        return pipeline_cls.call_args.kwargs["config"]


def test_completion_tools_resolved_from_nested_tool_settings():
    builders = _builders(
        SimpleNamespace(tools=SimpleNamespace(completion_tools=("rr_propose_brief",)))
    )
    assert builders._tool_setting("completion_tools", ()) == ("rr_propose_brief",)


def test_completion_tools_resolved_from_legacy_flat_settings():
    builders = _builders(SimpleNamespace(completion_tools=("finish",)))
    assert builders._tool_setting("completion_tools", ()) == ("finish",)


def test_completion_tools_default_empty_when_unset():
    builders = _builders(SimpleNamespace())
    assert builders._tool_setting("completion_tools", ()) == ()


def test_pipeline_config_carries_completion_tool_names():
    builders = _builders(
        SimpleNamespace(tools=SimpleNamespace(completion_tools=("rr_propose_brief",)))
    )
    config = _pipeline_config(builders)
    assert config.completion_tool_names == ("rr_propose_brief",)


def test_pipeline_config_defaults_to_empty_completion_tool_names():
    builders = _builders(SimpleNamespace())
    config = _pipeline_config(builders)
    assert config.completion_tool_names == ()
