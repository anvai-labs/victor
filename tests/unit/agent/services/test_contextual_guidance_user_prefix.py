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

"""Per-turn contextual guidance must ride the user-prefix when the system prompt is frozen.

When KV optimization freezes the system prompt for cache stability, per-turn
contextual_guidance set after the freeze would otherwise be silently dropped. It must be
combined with task guidance and injected via the user-prefix instead (mirroring how the
non-frozen path carries both inside the rebuilt system prompt).
"""

from types import SimpleNamespace

from victor.agent.prompt_builder import SystemPromptBuilder
from victor.agent.services.prompt_builder_runtime import PromptBuilderRuntime


def test_builder_exposes_contextual_guidance_text():
    builder = SystemPromptBuilder(
        provider_name="ollama", model="qwen3", contextual_guidance="CTX-GUIDE"
    )
    assert builder.get_contextual_guidance_text() == "CTX-GUIDE"
    assert (
        SystemPromptBuilder(provider_name="ollama", model="q").get_contextual_guidance_text() == ""
    )


def test_combine_dynamic_guidance_merges_task_and_contextual():
    builder = SimpleNamespace(
        get_task_guidance_text=lambda: "TASK-G",
        get_contextual_guidance_text=lambda: "CTX-G",
    )
    assert PromptBuilderRuntime._combine_dynamic_guidance(builder) == "TASK-G\n\nCTX-G"


def test_combine_dynamic_guidance_contextual_only():
    builder = SimpleNamespace(
        get_task_guidance_text=lambda: "",
        get_contextual_guidance_text=lambda: "ONLY-CTX",
    )
    assert PromptBuilderRuntime._combine_dynamic_guidance(builder) == "ONLY-CTX"


def test_combine_dynamic_guidance_empty_returns_none():
    builder = SimpleNamespace(
        get_task_guidance_text=lambda: "",
        get_contextual_guidance_text=lambda: "",
    )
    assert PromptBuilderRuntime._combine_dynamic_guidance(builder) is None
