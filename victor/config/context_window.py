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

"""Single-source context window resolution.

Historically the "context window" for one session was resolved independently
by several components, each with its own denominator:

- profile ``context.max_context_tokens`` (settings, default 100000)
- ``provider_context_limits.yaml`` model patterns (e.g. ``"qwen3*"``)
- provider defaults from the same YAML (e.g. ``openai: 128000``)
- ``CONTEXT_LIMITS.max_context_chars`` (800000 chars char-space fallback)

The 2026-11-04 wedged-session incident (victor.log, session ed737a8b) showed
the compaction trigger dividing real usage by one denominator while the TUI
gauge divided by another: the model window was exceeded at a read utilization
of ~0.72, proactive compaction never fired, and only the late overflow +
session-prefill-budget stampede recovered the session.

This module resolves the window exactly once, with an explicit precedence and
a machine-readable ``source``, so every consumer (compaction trigger, TUI
gauge, context monitor, logs) divides by the same number:

1. ``explicit_tokens`` argument (caller-resolved override)
2. settings ``context.max_context_tokens`` — only when user/env/profile-set
   (``model_fields_set``), never the 100000 default
3. model pattern from ``provider_context_limits.yaml`` (``models:`` section)
4. provider default from ``provider_context_limits.yaml`` (``providers:``)
5. safe fallback (128000)
"""

import logging
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Effective chars-per-token used to derive the legacy char budget from a
#: token window. Matches the historical ``tokens * 3.5 * 0.8`` conversion in
#: ``orchestrator_utils.calculate_max_context_chars`` so char-space consumers
#: keep their long-standing threshold behavior.
CHARS_PER_TOKEN_BUDGET = 2.8

#: Ultimate fallback when nothing is known about the model (tokens).
DEFAULT_CONTEXT_WINDOW_TOKENS = 128000

#: Value of ``ContextSettings.max_context_tokens`` default; treated as
#: "not provided" when the field is not in ``model_fields_set``.
_DEFAULT_NESTED_CONTEXT_TOKENS = 100000


@dataclass(frozen=True)
class ContextWindowResolution:
    """Resolved context window with full provenance.

    Attributes:
        tokens: Context window in tokens (the canonical budget).
        source: Where the token number came from:
            ``explicit`` | ``settings`` | ``model-pattern`` | ``provider-default``
            | ``fallback``.
        pattern: Matching model pattern for source ``model-pattern``
            (e.g. ``"qwen3.7-plus*"``), else ``None``.
        provider: Lowercased provider name used for resolution.
        model: Model identifier used for resolution.
        chars_budget: Legacy char-space budget (``tokens * 2.8``) for
            components still operating in character space.
    """

    tokens: int
    source: str
    pattern: Optional[str]
    provider: str
    model: str
    chars_budget: int

    def describe(self) -> str:
        """One-line human-readable provenance for log lines."""
        origin = f"{self.source}:{self.pattern}" if self.pattern else self.source
        return (
            f"window_tokens={self.tokens:,} source={origin} "
            f"provider={self.provider or 'unknown'} model={self.model or 'unknown'} "
            f"chars_budget={self.chars_budget:,}"
        )


def _settings_context_tokens(settings: Any) -> Optional[int]:
    """Extract a user-provided ``context.max_context_tokens`` from settings.

    Only honors the value when it was explicitly set (profile, env, or code) —
    the 100000 nested default must never shrink a larger model window.
    """
    context = getattr(settings, "context", None)
    if context is None:
        return None
    value = getattr(context, "max_context_tokens", None)
    if not isinstance(value, int) or value <= 0:
        return None
    fields_set = getattr(context, "model_fields_set", None)
    if fields_set is not None:
        if "max_context_tokens" not in fields_set:
            return None
    elif value == _DEFAULT_NESTED_CONTEXT_TOKENS:
        return None
    return value


def _model_pattern_window(provider: str, model: str) -> tuple[Optional[int], Optional[str]]:
    """Look up the model pattern window from provider_context_limits.yaml.

    Mirrors ``config_loaders.get_provider_limits`` matching semantics
    (first ``fnmatch`` wins) but also reports which pattern matched.
    """
    if not model:
        return None, None
    try:
        from victor.config.config_loaders import _load_yaml_cached, PROVIDER_LIMITS_FILE

        data = _load_yaml_cached(PROVIDER_LIMITS_FILE, "provider_limits") or {}
        models = data.get("models", {}) or {}
        for pattern, overrides in models.items():
            if fnmatch(model, str(pattern)):
                window = overrides.get("context_window") if isinstance(overrides, dict) else None
                if isinstance(window, int) and window > 0:
                    return window, str(pattern)
                return None, None
    except Exception as exc:  # noqa: BLE001 - resolution must never raise
        logger.debug("Context window model-pattern lookup failed: %s", exc)
    return None, None


def _provider_default_window(provider: str) -> Optional[int]:
    """Provider-level default window from provider_context_limits.yaml."""
    if not provider:
        return None
    try:
        from victor.config.config_loaders import _load_yaml_cached, PROVIDER_LIMITS_FILE

        data = _load_yaml_cached(PROVIDER_LIMITS_FILE, "provider_limits") or {}
        providers = data.get("providers", {}) or {}
        window = (providers.get(provider) or {}).get("context_window")
        if isinstance(window, int) and window > 0:
            return window
    except Exception as exc:  # noqa: BLE001 - resolution must never raise
        logger.debug("Context window provider-default lookup failed: %s", exc)
    return None


def resolve_context_window(
    provider: Any,
    model: str,
    settings: Any = None,
    explicit_tokens: Optional[int] = None,
) -> ContextWindowResolution:
    """Resolve the session context window (tokens) with provenance.

    Args:
        provider: Provider instance or name (``str`` or object with ``.name``).
        model: Model identifier.
        settings: Settings object (may expose ``context.max_context_tokens``).
        explicit_tokens: Caller-resolved override; wins over everything.

    Returns:
        ContextWindowResolution with the winning token window and its source.
    """
    if isinstance(provider, str):
        provider_name = provider
    else:
        provider_name = str(getattr(provider, "name", "") or "").lower()

    tokens: Optional[int] = None
    source = ""
    pattern: Optional[str] = None

    if isinstance(explicit_tokens, int) and explicit_tokens > 0:
        tokens, source = explicit_tokens, "explicit"
    if tokens is None:
        settings_tokens = _settings_context_tokens(settings)
        if settings_tokens is not None:
            tokens, source = settings_tokens, "settings"
    if tokens is None:
        pattern_tokens, matched_pattern = _model_pattern_window(provider_name, model)
        if pattern_tokens is not None:
            tokens, source, pattern = pattern_tokens, "model-pattern", matched_pattern
    if tokens is None:
        provider_tokens = _provider_default_window(provider_name)
        if provider_tokens is not None:
            tokens, source = provider_tokens, "provider-default"
    if tokens is None:
        tokens, source = DEFAULT_CONTEXT_WINDOW_TOKENS, "fallback"

    return ContextWindowResolution(
        tokens=tokens,
        source=source,
        pattern=pattern,
        provider=provider_name,
        model=str(model or ""),
        chars_budget=int(tokens * CHARS_PER_TOKEN_BUDGET),
    )
