# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""ML/RL-friendly model metadata parsing.

Extracts structured features from model name strings for use in
prompt optimization, RL training data, and aggregation queries.

Usage:
    from victor.agent.ml_metadata import parse_model_metadata, ModelFamily

    meta = parse_model_metadata("llama-3.3-70b-versatile")
    # ModelMetadata(family=LLAMA, size=LARGE, params_b=70.0, ...)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ModelFamily(Enum):
    """Model architecture family for ML/RL feature extraction."""

    LLAMA = "llama"
    QWEN = "qwen"
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"
    CLAUDE = "claude"
    GPT = "gpt"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    PHI = "phi"
    CODELLAMA = "codellama"
    COMMAND = "command"  # Cohere
    GROK = "grok"
    UNKNOWN = "unknown"


class ModelSize(Enum):
    """Model parameter size category for coarse comparison."""

    TINY = "tiny"  # <1B params
    SMALL = "small"  # 1-8B
    MEDIUM = "medium"  # 8-32B
    LARGE = "large"  # 32-70B
    XLARGE = "xlarge"  # 70-175B
    XXLARGE = "xxlarge"  # >175B


class ContextSize(Enum):
    """Context window size category."""

    SMALL = "small"  # <8K tokens
    MEDIUM = "medium"  # 8K-32K
    LARGE = "large"  # 32K-128K
    XLARGE = "xlarge"  # 128K+


@dataclass
class ModelMetadata:
    """Parsed model metadata for ML/RL feature extraction."""

    model_family: ModelFamily
    model_size: ModelSize
    model_params_b: Optional[float]  # Parameters in billions
    context_size: ContextSize
    context_tokens: Optional[int]  # Actual context window
    is_moe: bool  # Mixture of Experts
    is_reasoning: bool  # Explicit reasoning model


# Model family detection patterns (order matters - more specific first)
_MODEL_FAMILY_PATTERNS = [
    (r"codellama|code-llama", ModelFamily.CODELLAMA),
    (r"deepseek[-_]?r1|deepseek[-_]?coder", ModelFamily.DEEPSEEK),
    (r"deepseek", ModelFamily.DEEPSEEK),
    (r"mixtral", ModelFamily.MIXTRAL),
    (r"mistral", ModelFamily.MISTRAL),
    (r"llama[-_]?3\.3|llama3\.3|llama[-_]?3\.1|llama3[-_]?[12]?", ModelFamily.LLAMA),
    (r"llama", ModelFamily.LLAMA),
    (r"qwen[-_]?2\.5|qwen2\.5|qwen[-_]?3|qwen3", ModelFamily.QWEN),
    (r"qwen", ModelFamily.QWEN),
    (
        r"claude[-_]?3|claude[-_]?opus|claude[-_]?sonnet|claude[-_]?haiku",
        ModelFamily.CLAUDE,
    ),
    (r"claude", ModelFamily.CLAUDE),
    (r"gpt[-_]?4|gpt[-_]?3\.5|chatgpt|openai", ModelFamily.GPT),
    (r"gemini|palm|bard", ModelFamily.GEMINI),
    (r"phi[-_]?[234]|phi[-_]?mini", ModelFamily.PHI),
    (r"command[-_]?r|cohere", ModelFamily.COMMAND),
    (r"grok", ModelFamily.GROK),
]

# MoE model patterns
_MOE_PATTERNS = [r"mixtral", r"8x7b", r"8x22b", r"moe", r"mixture"]

# MoE effective parameters (total active params)
_MOE_EFFECTIVE_PARAMS = {
    "8x7b": 46.7,  # Mixtral 8x7B
    "8x22b": 141.0,  # Mixtral 8x22B
}

# Reasoning model patterns
_REASONING_PATTERNS = [
    r"deepseek[-_]?r1",
    r"o1[-_]?",
    r"r1[-_]?",
    r"reasoning",
    r"gpt[-_]?oss",  # OpenAI gpt-oss uses the harmony analysis/commentary channels
    r"qwq",
]

# Context size patterns (extract from model name like "32k", "128k")
_CONTEXT_PATTERN = re.compile(r"(\d+)k", re.IGNORECASE)

# Parameter size patterns (extract from model name like "70b", "8b", "7b")
_PARAM_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*b(?:illion)?", re.IGNORECASE)
_PARAM_PATTERN_ALT = re.compile(r"[-_:](\d+(?:\.\d+)?)b(?!yte)", re.IGNORECASE)


def parse_model_metadata(
    model: str,
    provider: Optional[str] = None,
    known_context: Optional[int] = None,
    known_params_b: Optional[float] = None,
) -> ModelMetadata:
    """Parse model name to extract ML-friendly metadata.

    Args:
        model: Model name string (e.g., "llama-3.3-70b-versatile")
        provider: Optional provider name for disambiguation
        known_context: Optional known context window size
        known_params_b: Optional known parameter count in billions

    Returns:
        ModelMetadata with extracted features
    """
    model_lower = model.lower()

    # 1. Detect model family
    model_family = ModelFamily.UNKNOWN
    for pattern, family in _MODEL_FAMILY_PATTERNS:
        if re.search(pattern, model_lower):
            model_family = family
            break

    # Provider-based fallback
    if model_family == ModelFamily.UNKNOWN and provider:
        provider_lower = provider.lower()
        if "anthropic" in provider_lower or "claude" in provider_lower:
            model_family = ModelFamily.CLAUDE
        elif "openai" in provider_lower:
            model_family = ModelFamily.GPT
        elif "google" in provider_lower or "gemini" in provider_lower:
            model_family = ModelFamily.GEMINI
        elif "groq" in provider_lower:
            model_family = ModelFamily.LLAMA
        elif "xai" in provider_lower or "grok" in provider_lower:
            model_family = ModelFamily.GROK

    # 2. Extract parameter count
    params_b = known_params_b
    if params_b is None:
        for moe_pattern, moe_params in _MOE_EFFECTIVE_PARAMS.items():
            if moe_pattern in model_lower:
                params_b = moe_params
                break

        if params_b is None:
            match = _PARAM_PATTERN.search(model_lower)
            if not match:
                match = _PARAM_PATTERN_ALT.search(model_lower)
            if match:
                params_b = float(match.group(1))

        if params_b is None:
            if "gpt-4" in model_lower:
                params_b = 175.0
            elif "gpt-3.5" in model_lower:
                params_b = 175.0
            elif "claude-3-opus" in model_lower:
                params_b = 200.0
            elif "claude-3-sonnet" in model_lower or "claude-3.5-sonnet" in model_lower:
                params_b = 70.0
            elif "claude-3-haiku" in model_lower:
                params_b = 20.0

    # 3. Categorize model size
    if params_b is not None:
        if params_b < 1:
            model_size = ModelSize.TINY
        elif params_b < 8:
            model_size = ModelSize.SMALL
        elif params_b < 32:
            model_size = ModelSize.MEDIUM
        elif params_b < 70:
            model_size = ModelSize.LARGE
        elif params_b < 175:
            model_size = ModelSize.XLARGE
        else:
            model_size = ModelSize.XXLARGE
    else:
        model_size = ModelSize.MEDIUM

    # 4. Extract context window size
    context_tokens = known_context
    if context_tokens is None:
        match = _CONTEXT_PATTERN.search(model_lower)
        if match:
            context_tokens = int(match.group(1)) * 1024

        if context_tokens is None:
            if model_family == ModelFamily.CLAUDE:
                context_tokens = 200000
            elif model_family == ModelFamily.GPT:
                context_tokens = 128000 if "gpt-4" in model_lower else 16000
            elif model_family == ModelFamily.GEMINI:
                context_tokens = 1000000
            elif "128k" in model_lower or "128000" in model_lower:
                context_tokens = 128000

    if context_tokens is not None:
        if context_tokens < 8000:
            context_size = ContextSize.SMALL
        elif context_tokens < 32000:
            context_size = ContextSize.MEDIUM
        elif context_tokens < 128000:
            context_size = ContextSize.LARGE
        else:
            context_size = ContextSize.XLARGE
    else:
        context_size = ContextSize.MEDIUM

    # 5. Detect MoE architecture
    is_moe = any(re.search(pattern, model_lower) for pattern in _MOE_PATTERNS)

    # 6. Detect reasoning models
    is_reasoning = any(re.search(pattern, model_lower) for pattern in _REASONING_PATTERNS)

    return ModelMetadata(
        model_family=model_family,
        model_size=model_size,
        model_params_b=params_b,
        context_size=context_size,
        context_tokens=context_tokens,
        is_moe=is_moe,
        is_reasoning=is_reasoning,
    )


# Known model context windows (for accuracy)
_KNOWN_CONTEXT_WINDOWS = {
    "llama-3.3-70b-versatile": 128000,
    "llama-3.1-8b-instant": 128000,
    "mixtral-8x7b-32768": 32768,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3.5-sonnet": 200000,
    "claude-3-haiku": 200000,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "gemini-pro": 32768,
    "gemini-1.5-pro": 1000000,
    "deepseek-chat": 128000,
    "deepseek-coder": 128000,
    "qwen2.5-coder:32b": 128000,
    "qwen3:32b": 40960,
}

# Known model parameters (for accuracy)
_KNOWN_MODEL_PARAMS = {
    "llama-3.3-70b-versatile": 70.0,
    "llama-3.1-8b-instant": 8.0,
    "mixtral-8x7b-32768": 46.7,
    "deepseek-r1:32b": 32.0,
    "deepseek-r1:70b": 70.0,
    "qwen2.5-coder:32b": 32.0,
    "qwen3:32b": 32.0,
}


def get_known_model_context(model: str) -> Optional[int]:
    """Get known context window for a model."""
    model_lower = model.lower()
    for known_model, context in _KNOWN_CONTEXT_WINDOWS.items():
        if known_model in model_lower or model_lower in known_model:
            return context
    return None


def get_known_model_params(model: str) -> Optional[float]:
    """Get known parameter count for a model."""
    model_lower = model.lower()
    for known_model, params in _KNOWN_MODEL_PARAMS.items():
        if known_model in model_lower or model_lower in known_model:
            return params
    return None


__all__ = [
    "ContextSize",
    "ModelFamily",
    "ModelMetadata",
    "ModelSize",
    "get_known_model_context",
    "get_known_model_params",
    "parse_model_metadata",
]
