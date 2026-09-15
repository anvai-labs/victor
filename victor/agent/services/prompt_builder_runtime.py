"""Service-owned prompt-builder runtime helper."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


class PromptBuilderRuntime:
    """Bridge prompt-builder runtime state off the concrete orchestrator."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def check_cache_setting_enabled(self) -> bool:
        """Return whether prompt/cache optimization is enabled by runtime settings."""
        settings = getattr(self._runtime, "settings", None)
        if settings is not None:
            context = getattr(settings, "context", None)
            if context is not None and not getattr(context, "cache_optimization_enabled", True):
                return False
        return True

    def compute_cache_flags(self) -> None:
        """Compute and cache prompt/KV optimization flags on the runtime host."""
        runtime = self._runtime
        try:
            if not self.check_cache_setting_enabled():
                runtime._kv_opt_cached = False
                runtime._cache_opt_cached = False
                return

            provider = getattr(runtime, "provider", None)
            runtime._cache_opt_cached = (
                provider is not None
                and hasattr(provider, "supports_prompt_caching")
                and provider.supports_prompt_caching()
            )

            kv_setting = self._get_kv_setting_enabled()
            if not kv_setting:
                runtime._kv_opt_cached = False
            elif provider is not None and hasattr(provider, "supports_kv_prefix_caching"):
                runtime._kv_opt_cached = provider.supports_kv_prefix_caching()
            elif runtime._cache_opt_cached:
                runtime._kv_opt_cached = True
            else:
                runtime._kv_opt_cached = False
        except Exception:
            runtime._kv_opt_cached = False
            runtime._cache_opt_cached = False

    def is_cache_optimization_enabled(self) -> bool:
        """Return whether API-level prompt cache optimization is enabled."""
        runtime = self._runtime
        cached = getattr(runtime, "_cache_opt_cached", None)
        if cached is not None:
            return bool(cached)
        try:
            if not self.check_cache_setting_enabled():
                return False
            provider = getattr(runtime, "provider", None)
            if provider is not None and hasattr(provider, "supports_prompt_caching"):
                return bool(provider.supports_prompt_caching())
            return False
        except Exception:
            return False

    def is_kv_optimization_enabled(self) -> bool:
        """Return whether KV-prefix cache optimization is enabled."""
        runtime = self._runtime
        cached = getattr(runtime, "_kv_opt_cached", None)
        if cached is not None:
            return bool(cached)
        try:
            if not self.check_cache_setting_enabled():
                return False
            if not self._get_kv_setting_enabled():
                return False
            provider = getattr(runtime, "provider", None)
            if provider is not None:
                if hasattr(provider, "supports_kv_prefix_caching"):
                    return bool(provider.supports_kv_prefix_caching())
                if hasattr(provider, "supports_prompt_caching"):
                    return bool(provider.supports_prompt_caching())
            return False
        except Exception:
            return False

    async def warm_up_kv_cache(self) -> None:
        """Prime a KV-capable provider with the stable system prompt prefix."""
        if not self.is_kv_optimization_enabled():
            return
        try:
            from victor.providers.base import Message

            runtime = self._runtime
            messages = [
                Message(role="system", content=getattr(runtime, "_system_prompt", "") or "")
            ]
            await runtime.provider.chat(
                messages=messages,
                model=getattr(runtime, "model", ""),
                max_tokens=1,
            )
            logger.info("[kv-cache] Warm-up complete — KV prefix primed")
        except Exception as exc:
            logger.debug("[kv-cache] Warm-up failed (non-fatal): %s", exc)

    def kv_prefix_fingerprint(self) -> str:
        """Compute a short stable fingerprint of the current system prompt prefix."""
        prompt = getattr(self._runtime, "_system_prompt", "") or ""
        return hashlib.md5(prompt[:500].encode(), usedforsecurity=False).hexdigest()[:12]

    def _get_kv_setting_enabled(self) -> bool:
        """Return whether KV optimization is enabled independently of provider support."""
        settings = getattr(self._runtime, "settings", None)
        if settings is not None:
            context = getattr(settings, "context", None)
            if context is not None:
                return bool(getattr(context, "kv_optimization_enabled", True))
        return True

    def sync_prompt_builder_runtime_state(self) -> None:
        """Align prompt-builder state with current mode and enabled tools."""
        runtime = self._runtime
        builder = getattr(runtime, "prompt_builder", None)
        if builder is None:
            return

        cache_invalidated = False
        tool_state_changed = False
        try:
            enabled_tools = sorted(runtime.get_enabled_tools())
            if builder.available_tools != enabled_tools:
                builder.available_tools = enabled_tools
                tool_state_changed = True
        except Exception as exc:
            logger.debug("Failed to sync enabled tools into prompt builder: %s", exc)
            if builder.available_tools:
                builder.available_tools = []
                tool_state_changed = True

        stable_tools, dynamic_tools = self._split_prompt_tools(
            getattr(builder, "available_tools", [])
        )
        if (
            not hasattr(builder, "stable_prompt_tools")
            or builder.stable_prompt_tools != stable_tools
        ):
            builder.stable_prompt_tools = stable_tools
            cache_invalidated = True
        if (
            not hasattr(builder, "dynamic_prompt_tools")
            or builder.dynamic_prompt_tools != dynamic_tools
        ):
            builder.dynamic_prompt_tools = dynamic_tools

        try:
            mode_prompt = runtime.get_mode_system_prompt()
            if builder.mode_prompt_addition != mode_prompt:
                builder.mode_prompt_addition = mode_prompt
                cache_invalidated = True
        except Exception as exc:
            logger.debug("Failed to sync mode prompt into prompt builder: %s", exc)
            if builder.mode_prompt_addition:
                builder.mode_prompt_addition = ""
                cache_invalidated = True

        provider = getattr(runtime, "provider", None)
        provider_name = (getattr(runtime, "provider_name", "") or "").lower()
        if getattr(builder, "provider_name", "") != provider_name:
            builder.provider_name = provider_name
            cache_invalidated = True
            self._sync_tool_guidance_strategy(builder, provider_name)

        model = getattr(runtime, "model", "") or ""
        if getattr(builder, "model", "") != model:
            builder.model = model
            cache_invalidated = True

        provider_caches = bool(
            provider is not None
            and hasattr(provider, "supports_prompt_caching")
            and provider.supports_prompt_caching()
        )
        if getattr(builder, "provider_caches", False) != provider_caches:
            builder.provider_caches = provider_caches
            cache_invalidated = True

        provider_has_kv_cache = bool(
            provider is not None
            and hasattr(provider, "supports_kv_prefix_caching")
            and provider.supports_kv_prefix_caching()
        )
        if getattr(builder, "provider_has_kv_cache", False) != provider_has_kv_cache:
            builder.provider_has_kv_cache = provider_has_kv_cache
            cache_invalidated = True

        if tool_state_changed:
            runtime._session_tools = None
            runtime._session_semantic_tools = None

        if cache_invalidated:
            builder.invalidate_cache()

    def ensure_system_prompt_current(self) -> None:
        """Refresh the live system prompt when frozen-prefix inputs change."""
        runtime = self._runtime
        if getattr(runtime, "_prompt_refresh_in_progress", False):
            return

        self._reload_project_context_if_needed()
        self.sync_prompt_builder_runtime_state()

        current_signature = self._compute_prompt_signature()
        previous_signature = getattr(runtime, "_prompt_runtime_signature", None)
        if previous_signature is None:
            runtime._prompt_runtime_signature = current_signature
            return

        if previous_signature == current_signature:
            return

        logger.info("[cache] Prompt runtime signature changed; refreshing frozen prompt")
        self._force_reload_project_context()
        self.refresh_system_prompt()

    def build_system_prompt_fallback(self) -> str:
        """Build the non-pipeline system prompt path."""
        runtime = self._runtime
        builder = getattr(runtime, "prompt_builder", None)
        if builder is None:
            return ""

        prompt_orchestrator = getattr(runtime, "_prompt_orchestrator", None)
        if prompt_orchestrator is None:
            from victor.agent.prompt_orchestrator import get_prompt_orchestrator

            prompt_orchestrator = get_prompt_orchestrator()

        prompt_built_hook = getattr(runtime, "_emit_prompt_used_event", None)
        if not callable(prompt_built_hook):
            prompt_built_hook = None

        return prompt_orchestrator.build_system_prompt(
            builder_type="legacy",
            provider=getattr(runtime, "provider_name", ""),
            model=getattr(runtime, "model", ""),
            task_type=getattr(builder, "task_type", "default"),
            builder=builder,
            get_context_window=runtime._get_model_context_window,
            on_prompt_built=prompt_built_hook,
        )

    def compose_system_prompt(self, base_system_prompt: str) -> str:
        """Add project-context guidance to the built system prompt when present."""
        project_context = getattr(self._runtime, "project_context", None)
        if project_context and getattr(project_context, "content", None):
            return base_system_prompt + "\n\n" + project_context.get_system_prompt_addition()
        return base_system_prompt

    def update_system_prompt_for_query(self, query_classification=None) -> None:
        """Rebuild the runtime system prompt with query-aware classification."""
        runtime = self._runtime
        builder = getattr(runtime, "prompt_builder", None)
        if query_classification is not None and builder is not None:
            builder.query_classification = query_classification

        pipeline = getattr(runtime, "_prompt_pipeline", None)
        is_frozen = (
            pipeline.is_frozen if pipeline else getattr(runtime, "_system_prompt_frozen", False)
        )
        if getattr(runtime, "_kv_optimization_enabled", False) and is_frozen:
            # System prompt is frozen for KV stability, so per-turn task AND
            # contextual guidance must ride the user-prefix instead — otherwise
            # contextual_guidance (set after the freeze) is silently dropped.
            runtime._dynamic_task_guidance = self._combine_dynamic_guidance(builder)
            logger.debug("[cache] System prompt frozen - skipping rebuild for query classification")
            return

        if query_classification is not None and builder is not None:
            builder.invalidate_cache()

        base_system_prompt = runtime.build_system_prompt()
        runtime._system_prompt = self.compose_system_prompt(base_system_prompt)
        runtime._dynamic_task_guidance = None

        if getattr(runtime, "_kv_optimization_enabled", False):
            runtime._system_prompt_frozen = True

        runtime._prompt_runtime_signature = self._compute_prompt_signature()
        self.sync_conversation_system_prompt()

    def refresh_system_prompt(
        self,
        query_classification=None,
        *,
        preserve_existing_classification: bool = True,
    ) -> None:
        """Reset prompt runtime caches and rebuild the active system prompt."""
        runtime = self._runtime
        if getattr(runtime, "_prompt_refresh_in_progress", False):
            return

        runtime._prompt_refresh_in_progress = True
        try:
            pipeline = getattr(runtime, "_prompt_pipeline", None)
            if pipeline is not None:
                pipeline.unfreeze()
            runtime._system_prompt_frozen = False
            runtime._session_tools = None
            runtime._session_semantic_tools = None

            builder = getattr(runtime, "prompt_builder", None)
            if builder is not None:
                builder.invalidate_cache()

            if (
                query_classification is None
                and preserve_existing_classification
                and builder is not None
            ):
                query_classification = getattr(builder, "query_classification", None)

            self.update_system_prompt_for_query(query_classification=query_classification)
        finally:
            runtime._prompt_refresh_in_progress = False

    def build_dynamic_tool_guidance(
        self,
        user_message: Optional[str],
        *,
        goals: Optional[Iterable[str]] = None,
        planned_tools: Optional[Iterable[Any]] = None,
        selected_tools: Optional[Iterable[Any]] = None,
    ) -> str:
        """Build per-turn dynamic tool hints for long-tail tools."""
        builder = getattr(self._runtime, "prompt_builder", None)
        if builder is None or not hasattr(builder, "get_dynamic_tool_guidance_text"):
            return ""

        dynamic_tools = list(getattr(builder, "dynamic_prompt_tools", []) or [])
        if not dynamic_tools:
            return ""

        relevant_tools, selection_source = self._select_relevant_dynamic_tools(
            user_message or "",
            dynamic_tools,
            planned_tools=planned_tools,
            selected_tools=selected_tools,
        )
        if not relevant_tools:
            return ""

        guidance_kwargs: dict[str, Any] = {}

        normalized_goals = self._normalize_dynamic_tool_goals(goals)
        if normalized_goals:
            guidance_kwargs["goals"] = normalized_goals

        current_intent = self._normalize_dynamic_tool_intent(
            getattr(self._runtime, "_current_intent", None)
        )
        if current_intent:
            guidance_kwargs["current_intent"] = current_intent

        if selection_source:
            guidance_kwargs["selection_source"] = selection_source

        tool_rationale = self._build_dynamic_tool_rationale(
            relevant_tools,
            planned_tools=planned_tools,
            selected_tools=selected_tools,
        )
        if tool_rationale:
            guidance_kwargs["tool_rationale"] = tool_rationale

        return builder.get_dynamic_tool_guidance_text(relevant_tools, **guidance_kwargs)

    def get_system_prompt(self) -> str:
        """Return the current prompt-builder output for protocol consumers."""
        builder = getattr(self._runtime, "prompt_builder", None)
        if builder is None:
            return ""
        return builder.build()

    def set_system_prompt(self, prompt: str) -> None:
        """Set a custom prompt through the prompt-builder surface when supported."""
        builder = getattr(self._runtime, "prompt_builder", None)
        if builder is not None and hasattr(builder, "set_custom_prompt"):
            builder.set_custom_prompt(prompt)

    def append_to_system_prompt(self, content: str) -> None:
        """Append extra prompt content through the same custom-prompt surface."""
        current = self.get_system_prompt()
        self.set_system_prompt(current + "\n\n" + content)

    def sync_conversation_system_prompt(self) -> None:
        """Push the current runtime system prompt into the live conversation state."""
        runtime = self._runtime
        conversation = getattr(runtime, "conversation", None)
        if conversation is None:
            return

        prompt = getattr(runtime, "_system_prompt", "")
        conversation.system_prompt = prompt
        if getattr(conversation, "_system_added", False) and getattr(
            conversation, "_messages", None
        ):
            if conversation._messages[0].role == "system":
                from victor.providers.base import Message

                conversation._messages[0] = Message(role="system", content=prompt)

    def _split_prompt_tools(self, available_tools: Iterable[str]) -> tuple[list[str], list[str]]:
        """Split enabled tools into stable core tools and dynamic long-tail tools."""
        from victor.config.tool_tiers import (
            get_provider_category,
            get_provider_tool_tier,
        )
        from victor.tools.core_tool_aliases import canonicalize_core_tool_name

        runtime = self._runtime
        try:
            provider_category = get_provider_category(runtime._get_model_context_window())
        except Exception:
            provider_category = "large"

        normalized: list[str] = []
        seen: set[str] = set()
        for tool in available_tools:
            canonical = canonicalize_core_tool_name(tool)
            if canonical and canonical not in seen:
                normalized.append(canonical)
                seen.add(canonical)

        # P5 capability-first split: a prompt-caching provider caches the full tool set
        # nearly free (billing discount), so keep ALL tools in the stable system prefix —
        # per-turn dynamic injection would only churn the cache for no benefit. Only
        # non-caching / local providers keep the window-tiered CORE-stable + ADAPTIVE
        # split (where moving the long tail into per-turn user messages saves tokens).
        if self._provider_supports_prompt_caching():
            return normalized, []

        stable: list[str] = []
        dynamic: list[str] = []
        for tool_name in normalized:
            tier = get_provider_tool_tier(tool_name, provider_category)
            if tier in {"FULL", "COMPACT"}:
                stable.append(tool_name)
            else:
                dynamic.append(tool_name)
        return stable, dynamic

    def _provider_supports_prompt_caching(self) -> bool:
        """Best-effort: does the active provider offer API-level prompt caching?

        Used to decide the prompt-tool split (P5). Any failure -> False (fall back to
        the window-tiered split), so this never breaks prompt assembly.
        """
        try:
            provider = getattr(self._runtime, "provider", None)
            fn = getattr(provider, "supports_prompt_caching", None)
            return bool(fn()) if callable(fn) else False
        except Exception:
            return False

    def _reload_project_context_if_needed(self) -> None:
        """Refresh .victor/init.md content when the backing file changed."""
        project_context = getattr(self._runtime, "project_context", None)
        if project_context is None or not hasattr(project_context, "load"):
            return
        try:
            project_context.load(force_reload=False)
        except Exception as exc:
            logger.debug("Failed to refresh project context before prompt assembly: %s", exc)

    def _force_reload_project_context(self) -> None:
        """Bypass TTL caching when prompt invalidation already detected a context change."""
        project_context = getattr(self._runtime, "project_context", None)
        if project_context is None or not hasattr(project_context, "load"):
            return
        try:
            project_context.load(force_reload=True)
        except Exception as exc:
            logger.debug("Failed to force-reload project context before prompt refresh: %s", exc)

    def _compute_project_context_signature(self) -> tuple[Any, ...]:
        """Fingerprint project-context state for prompt invalidation."""
        project_context = getattr(self._runtime, "project_context", None)
        if project_context is None:
            return ("", 0.0, 0)

        signature_getter = getattr(project_context, "get_context_signature", None)
        if callable(signature_getter):
            try:
                signature = signature_getter()
                if signature:
                    return ("instruction_files", signature)
            except Exception as exc:
                logger.debug("Failed to read project context signature: %s", exc)

        context_file = getattr(project_context, "context_file", None)
        if isinstance(context_file, Path):
            try:
                stat = context_file.stat()
                return (str(context_file), stat.st_mtime, stat.st_size)
            except OSError:
                pass

        content = getattr(project_context, "content", "") or ""
        return ("", 0.0, len(content))

    def _compute_prompt_signature(self) -> tuple[Any, ...]:
        """Compute the frozen prompt invalidation signature."""
        builder = getattr(self._runtime, "prompt_builder", None)
        if builder is None:
            return ()
        return (
            getattr(builder, "provider_name", ""),
            getattr(builder, "model", ""),
            getattr(builder, "mode_prompt_addition", ""),
            tuple(getattr(builder, "stable_prompt_tools", []) or []),
            self._compute_project_context_signature(),
        )

    @staticmethod
    def _get_task_guidance_text(builder: Any) -> Optional[str]:
        """Read the current query guidance text from the prompt builder."""
        if builder is None:
            return None
        if hasattr(builder, "get_task_guidance_text"):
            return builder.get_task_guidance_text() or None
        if hasattr(builder, "_get_task_guidance_section"):
            return builder._get_task_guidance_section() or None
        return None

    @classmethod
    def _combine_dynamic_guidance(cls, builder: Any) -> Optional[str]:
        """Combine task + contextual guidance for frozen-prompt user-prefix injection.

        When the system prompt is frozen for KV stability, both the per-turn task
        guidance and the per-turn contextual guidance must be injected via the
        user-prefix. This mirrors how the non-frozen path carries both inside the
        rebuilt system prompt, keeping behavior consistent across providers.
        """
        parts = [cls._get_task_guidance_text(builder)]
        if builder is not None and hasattr(builder, "get_contextual_guidance_text"):
            parts.append(builder.get_contextual_guidance_text() or None)
        combined = "\n\n".join(p for p in parts if p)
        return combined or None

    @staticmethod
    def _sync_tool_guidance_strategy(builder: Any, provider_name: str) -> None:
        """Refresh provider-specific tool guidance when provider identity changes."""
        try:
            from victor.agent.provider_tool_guidance import get_tool_guidance_strategy

            builder._tool_guidance = get_tool_guidance_strategy(provider_name)
        except Exception:
            pass

    def _select_relevant_dynamic_tools(
        self,
        user_message: str,
        dynamic_tools: list[str],
        *,
        planned_tools: Optional[Iterable[Any]] = None,
        selected_tools: Optional[Iterable[Any]] = None,
    ) -> tuple[list[str], Optional[str]]:
        """Choose a compact dynamic tool subset for the current turn."""
        from victor.tools.core_tool_aliases import canonicalize_core_tool_name

        planned_dynamic = self._filter_dynamic_tool_names(
            planned_tools,
            dynamic_tools,
        )
        if planned_dynamic:
            return planned_dynamic[:6], "planned_tools"

        selector_dynamic = self._select_dynamic_tools_from_selector(
            user_message,
            dynamic_tools,
        )
        if selector_dynamic:
            return selector_dynamic[:6], "keyword_selector"

        tool_catalog = {}
        try:
            registry = getattr(self._runtime, "tools", None)
            if registry is not None and hasattr(registry, "list_tools"):
                for tool in registry.list_tools():
                    name = canonicalize_core_tool_name(getattr(tool, "name", ""))
                    if name:
                        tool_catalog[name] = tool
        except Exception:
            tool_catalog = {}

        selected_dynamic = self._filter_dynamic_tool_names(selected_tools, dynamic_tools)

        user_message_lower = user_message.lower().strip()
        keyword_matches = [
            tool_name
            for tool_name in dynamic_tools
            if self._tool_matches_message(
                tool_catalog.get(tool_name), tool_name, user_message_lower
            )
        ]

        if keyword_matches:
            relevant = keyword_matches
            selection_source = "message_keywords"
        else:
            relevant = selected_dynamic
            selection_source = "selected_tools" if selected_dynamic else None
        return relevant[:6], selection_source

    def _filter_dynamic_tool_names(
        self,
        tools: Optional[Iterable[Any]],
        dynamic_tools: list[str],
    ) -> list[str]:
        """Filter a tool iterable down to dynamic tool names in stable order."""
        from victor.tools.core_tool_aliases import canonicalize_core_tool_name

        if not tools:
            return []

        dynamic_set = set(dynamic_tools)
        relevant: list[str] = []
        for tool in tools:
            name = getattr(tool, "name", None)
            if name is None and isinstance(tool, dict):
                name = tool.get("name")
            canonical = canonicalize_core_tool_name(name or "")
            if canonical in dynamic_set and canonical not in relevant:
                relevant.append(canonical)
        return relevant

    def _select_dynamic_tools_from_selector(
        self,
        user_message: str,
        dynamic_tools: list[str],
    ) -> list[str]:
        """Reuse the canonical keyword/stage selector when available."""
        selector = getattr(self._runtime, "tool_selector", None)
        if selector is None or not hasattr(selector, "select_keywords"):
            return []
        try:
            selected_tools = selector.select_keywords(
                user_message,
                planned_tools=None,
                _record=False,
            )
        except Exception as exc:
            logger.debug("Tool selector unavailable for dynamic prompt guidance: %s", exc)
            return []

        from victor.tools.core_tool_aliases import canonicalize_core_tool_name

        selected_dynamic: list[str] = []
        dynamic_set = set(dynamic_tools)
        for tool in selected_tools or []:
            name = getattr(tool, "name", None)
            if name is None and isinstance(tool, dict):
                name = tool.get("name")
            canonical = canonicalize_core_tool_name(name or "")
            if canonical in dynamic_set and canonical not in selected_dynamic:
                selected_dynamic.append(canonical)
        return selected_dynamic

    def _build_dynamic_tool_rationale(
        self,
        relevant_tools: list[str],
        *,
        planned_tools: Optional[Iterable[Any]] = None,
        selected_tools: Optional[Iterable[Any]] = None,
    ) -> dict[str, str]:
        """Build compact per-tool rationale from existing metadata and descriptions."""
        tool_catalog: dict[str, Any] = {}
        self._extend_tool_catalog(tool_catalog, planned_tools)
        self._extend_tool_catalog(tool_catalog, selected_tools)

        registry = getattr(self._runtime, "tools", None)
        if registry is not None and hasattr(registry, "list_tools"):
            try:
                self._extend_tool_catalog(tool_catalog, registry.list_tools(), overwrite=False)
            except Exception as exc:
                logger.debug("Tool registry unavailable for dynamic prompt rationale: %s", exc)

        rationale: dict[str, str] = {}
        for tool_name in relevant_tools:
            summary = self._extract_tool_rationale(tool_catalog.get(tool_name))
            if summary:
                rationale[tool_name] = summary
        return rationale

    @staticmethod
    def _extend_tool_catalog(
        catalog: dict[str, Any],
        tools: Optional[Iterable[Any]],
        *,
        overwrite: bool = True,
    ) -> None:
        """Add canonicalized tool objects to a lookup catalog."""
        from victor.tools.core_tool_aliases import canonicalize_core_tool_name

        if not tools:
            return

        for tool in tools:
            name = getattr(tool, "name", None)
            if name is None and isinstance(tool, dict):
                name = tool.get("name")
            canonical = canonicalize_core_tool_name(name or "")
            if not canonical:
                continue
            if overwrite or canonical not in catalog:
                catalog[canonical] = tool

    def _extract_tool_rationale(self, tool: Any) -> Optional[str]:
        """Extract a compact rationale string from existing tool metadata."""
        if tool is None:
            return None

        metadata = getattr(tool, "metadata", None)
        if metadata is None and isinstance(tool, dict):
            metadata = tool.get("metadata")

        for field_name in ("use_cases", "priority_hints"):
            values = self._get_metadata_values(metadata, field_name)
            if values:
                summary = self._compact_tool_rationale(values[0])
                if summary:
                    return summary

        description = getattr(tool, "description", None)
        if description is None and isinstance(tool, dict):
            description = tool.get("description")
        return self._compact_tool_rationale(description)

    @staticmethod
    def _get_metadata_values(metadata: Any, field_name: str) -> list[str]:
        """Read list-like metadata values from dict or object metadata."""
        if metadata is None:
            return []

        values = getattr(metadata, field_name, None)
        if values is None and isinstance(metadata, dict):
            values = metadata.get(field_name)
        if not values:
            return []
        if isinstance(values, str):
            values = [values]
        return [str(value).strip() for value in values if str(value).strip()]

    @staticmethod
    def _compact_tool_rationale(text: Any) -> Optional[str]:
        """Reduce description-like text to a short single-clause rationale."""
        if text is None:
            return None
        normalized = " ".join(str(text).strip().split())
        if not normalized:
            return None

        sentence = normalized.split(".")[0].strip(" ;:-")
        if len(sentence) > 96:
            sentence = sentence[:93].rstrip() + "..."
        return sentence or None

    @staticmethod
    def _normalize_dynamic_tool_goals(goals: Optional[Iterable[str]]) -> list[str]:
        """Normalize planner goals for compact prompt rendering."""
        if not goals:
            return []

        normalized: list[str] = []
        for goal in goals:
            if goal is None:
                continue
            text = str(goal).strip()
            if text and text not in normalized:
                normalized.append(text)
        return normalized[:3]

    @staticmethod
    def _normalize_dynamic_tool_intent(current_intent: Any) -> Optional[str]:
        """Normalize current intent for dynamic tool guidance."""
        if current_intent is None:
            return None
        if hasattr(current_intent, "value"):
            current_intent = current_intent.value
        text = str(current_intent).strip()
        return text or None

    @staticmethod
    def _tool_matches_message(tool: Any, tool_name: str, user_message_lower: str) -> bool:
        """Heuristic keyword match for dynamic tool guidance."""
        if not user_message_lower:
            return False
        if tool_name.replace("_", " ") in user_message_lower:
            return True
        if tool_name in user_message_lower:
            return True
        metadata = getattr(tool, "metadata", None)
        keywords = getattr(metadata, "keywords", []) if metadata is not None else []
        if any(str(keyword).lower() in user_message_lower for keyword in keywords or []):
            return True
        description = getattr(tool, "description", "") or ""
        for word in description.lower().split()[:10]:
            if len(word) > 4 and word in user_message_lower:
                return True
        return False
