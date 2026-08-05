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

"""Per-provider model→context-window lookup tables.

Used by BaseProvider.context_window() and the tool broadcasting strategy
picker. Keep prefix entries ordered most-specific to least-specific; the
matcher iterates and returns on first prefix match.

Coverage target: top 20 models per provider, covering >95% of typical usage.
Unknown models fall back to a provider-level default (or to
BaseProvider.DEFAULT_CONTEXT_WINDOW if none).
"""

from __future__ import annotations

from typing import Optional

# Each provider table: list of (prefix, context_window) tuples. Order matters
# (longer/more-specific prefixes first).

ANTHROPIC: list[tuple[str, int]] = [
    # Claude 5 family / 4.8: 1M-token context (current as of 2026-07).
    ("claude-fable-5", 1_000_000),
    ("claude-opus-4-8", 1_000_000),
    ("claude-opus-4-7", 1_000_000),
    ("claude-sonnet-5", 1_000_000),
    ("claude-sonnet-4-6", 1_000_000),
    ("claude-sonnet-4-5", 200_000),
    ("claude-haiku-4-5", 200_000),
    ("claude-3-5-sonnet", 200_000),
    ("claude-3-5-haiku", 200_000),
    ("claude-3-opus", 200_000),
    ("claude-3-sonnet", 200_000),
    ("claude-3-haiku", 200_000),
    ("claude-2", 100_000),
    ("claude-instant", 100_000),
]
ANTHROPIC_DEFAULT = 200_000

OPENAI: list[tuple[str, int]] = [
    ("gpt-5", 200_000),
    ("gpt-4.5", 200_000),
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-4-1106", 128_000),
    ("gpt-4-0125", 128_000),
    ("gpt-4-32k", 32_768),
    ("gpt-4", 8_192),
    ("gpt-3.5-turbo-16k", 16_384),
    ("gpt-3.5-turbo", 16_385),
    ("o1-preview", 128_000),
    ("o1-mini", 128_000),
    ("o1", 200_000),
    ("o3", 200_000),
]
OPENAI_DEFAULT = 128_000

GOOGLE: list[tuple[str, int]] = [
    ("gemini-2.5-pro", 2_000_000),
    ("gemini-2.0-pro", 2_000_000),
    ("gemini-2.0-flash", 1_000_000),
    ("gemini-1.5-pro", 2_000_000),
    ("gemini-1.5-flash", 1_000_000),
    ("gemini-1.0-pro", 32_768),
    ("gemini-pro-vision", 16_384),
    ("gemini-pro", 32_768),
]
GOOGLE_DEFAULT = 1_000_000

GROQ: list[tuple[str, int]] = [
    ("llama-3.3-70b", 128_000),
    ("llama-3.1-70b", 128_000),
    ("llama-3.1-8b", 128_000),
    ("llama-3.2-90b", 128_000),
    ("llama-3.2-11b", 128_000),
    ("llama-3.2-3b", 128_000),
    ("llama-3.2-1b", 128_000),
    ("llama-3", 8_192),
    ("mixtral-8x7b", 32_768),
    ("gemma2-9b", 8_192),
    ("gemma-7b", 8_192),
    ("qwen", 32_768),
    ("deepseek", 64_000),
]
GROQ_DEFAULT = 32_768

DEEPSEEK: list[tuple[str, int]] = [
    ("deepseek-chat", 64_000),
    ("deepseek-coder", 16_384),
    ("deepseek-reasoner", 64_000),
]
DEEPSEEK_DEFAULT = 64_000

# Local providers: these usually expose context size via API metadata.
# These tables are the FALLBACK when API probing fails or model is unknown.
OLLAMA: list[tuple[str, int]] = [
    ("llama3.3:70b", 128_000),
    ("llama3.1:70b", 128_000),
    ("llama3.1:8b", 128_000),
    ("llama3.2:3b", 128_000),
    ("llama3:8b", 8_192),
    ("qwen2.5-coder:32b", 32_768),
    ("qwen2.5-coder:14b", 32_768),
    ("qwen2.5-coder:7b", 32_768),
    ("qwen2.5:72b", 32_768),
    ("qwen2.5:32b", 32_768),
    ("qwen2.5:14b", 32_768),
    ("qwen2.5:7b", 32_768),
    ("qwen2.5:3b", 32_768),
    ("deepseek-r1", 64_000),
    ("deepseek-coder-v2", 32_768),
    ("mistral-small", 32_768),
    ("mistral-nemo", 128_000),
    ("mistral", 8_192),
    ("phi3", 4_096),
    ("phi-2", 2_048),
    ("gemma2", 8_192),
    ("codellama:34b", 16_384),
    ("codellama", 4_096),
]
OLLAMA_DEFAULT = 8_192  # Conservative default for unknown local models

LMSTUDIO_DEFAULT = 8_192  # Same fallback as Ollama
MLX_DEFAULT = 32_768  # Apple Silicon usually loads larger-context models
LLAMACPP_DEFAULT = 8_192
VLLM_DEFAULT = 32_768  # Self-hosted vLLM commonly serves 32K+ models

XAI: list[tuple[str, int]] = [
    ("grok-4-1-fast", 2_000_000),
    ("grok-4", 256_000),
    ("grok-3", 131_072),
    ("grok-2", 131_072),
    ("grok-beta", 131_072),
]
XAI_DEFAULT = 131_072

CEREBRAS: list[tuple[str, int]] = [
    ("llama3.1-70b", 128_000),
    ("llama3.1-8b", 128_000),
    ("llama-3.3-70b", 128_000),
    ("qwen", 32_768),
]
CEREBRAS_DEFAULT = 128_000

AZURE_OPENAI: list[tuple[str, int]] = OPENAI  # Same models, same windows
AZURE_OPENAI_DEFAULT = OPENAI_DEFAULT

VERTEX: list[tuple[str, int]] = GOOGLE  # Vertex serves Gemini
VERTEX_DEFAULT = GOOGLE_DEFAULT


def lookup(table: list[tuple[str, int]], model: Optional[str], default: int) -> int:
    """Return context window from a (prefix, cw) table, falling back to default.

    Matches by prefix in iteration order; put more-specific entries first.
    """
    if not model:
        return default
    for prefix, cw in table:
        if model.startswith(prefix):
            return cw
    return default
