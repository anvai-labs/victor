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

import pytest
from victor.agent.prompt_builder import SystemPromptBuilder
from victor.agent.prompt_section_texts import HEADLESS_MODE_GUIDANCE


def test_headless_mode_prompt_inclusion():
    # Scenario 1: Headless mode OFF
    builder_off = SystemPromptBuilder(
        provider_name="anthropic", model="claude-3-5-sonnet", headless_mode=False
    )
    prompt_off = builder_off.build()
    assert "HEADLESS MODE" not in prompt_off

    # Scenario 2: Headless mode ON
    builder_on = SystemPromptBuilder(
        provider_name="anthropic", model="claude-3-5-sonnet", headless_mode=True
    )
    prompt_on = builder_on.build()
    assert "HEADLESS MODE" in prompt_on
    assert "AUTOMATED EXECUTION" in prompt_on
    assert "Do not ask for user confirmation" in prompt_on


def test_headless_mode_in_document():
    builder = SystemPromptBuilder(
        provider_name="anthropic", model="claude-3-5-sonnet", headless_mode=True
    )
    doc = builder.build_document()

    # Check if the block exists
    headless_block = doc.get_block("headless_mode")
    assert headless_block is not None
    assert HEADLESS_MODE_GUIDANCE in headless_block.content
    assert headless_block.priority == 25
