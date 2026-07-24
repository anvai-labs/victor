"""Service-owned skill auto-selection runtime helper."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SkillRuntime:
    """Bridge skill-selection orchestration off the concrete orchestrator."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def apply_skill_for_turn(self, user_message: str) -> None:
        """Apply skill auto-selection for a chat turn."""
        runtime = self._runtime
        self.clear_active_skills()

        matcher = getattr(runtime, "_skill_matcher", None)
        if (
            matcher is None
            or not getattr(matcher, "_initialized", False)
            or getattr(runtime, "_skill_auto_disabled", False)
            or getattr(runtime, "_manual_skill_active", False)
        ):
            return

        from victor.agent.action_authorizer import split_continuation_request

        is_continuation, continuation_payload = split_continuation_request(user_message)
        if is_continuation and not continuation_payload.strip():
            # A bare "continue" names no new task — matching skills against it
            # only injects noise into the very work being continued. Leave
            # skills cleared for this turn.
            logger.debug("Skipping skill auto-selection for bare continuation turn")
            return

        try:
            matches = matcher.match_multiple_sync(user_message)
            analytics = getattr(runtime, "_skill_analytics", None)

            if matches:
                if len(matches) == 1:
                    skill, score = matches[0]
                    logger.info("Auto-selected skill: %s (score=%.2f)", skill.name, score)
                    self.inject_skill(skill)
                    runtime._last_skill_match_info = {
                        "auto_skill": skill.name,
                        "auto_skill_score": round(score, 2),
                    }
                    if analytics:
                        analytics.record_selection(skill.name, score)
                    return

                names = [skill.name for skill, _ in matches]
                logger.info("Auto-selected %d skills: %s", len(matches), " → ".join(names))
                self.inject_skills(matches)
                runtime._last_skill_match_info = {
                    "auto_skills": [
                        {"name": skill.name, "score": round(score, 2)} for skill, score in matches
                    ],
                }
                if analytics:
                    analytics.record_multi_selection(
                        [(skill.name, score) for skill, score in matches]
                    )
                return

            if analytics:
                analytics.record_miss()
            runtime._last_skill_match_info = None
        except Exception:
            logger.debug("Skill auto-selection failed", exc_info=True)

    def get_last_skill_match_info(self) -> Optional[Dict[str, Any]]:
        """Return metadata about the last skill match for response attachment."""
        return getattr(self._runtime, "_last_skill_match_info", None)

    def clear_active_skills(self) -> None:
        """Remove any active skill injection from the current runtime state."""
        runtime = self._runtime
        runtime._active_skill_prompt = ""

        if getattr(runtime, "_kv_optimization_enabled", False):
            return

        base = getattr(runtime, "_base_system_prompt", None)
        if base is not None:
            runtime._system_prompt = base

        self._sync_conversation_system_prompt()

    def get_skill_user_prefix(self) -> str:
        """Return the cache-friendly active skill prefix for the next user turn."""
        return getattr(self._runtime, "_active_skill_prompt", "") or ""

    def inject_skill(self, skill: Any) -> None:
        """Inject a single skill prompt fragment into runtime state."""
        runtime = self._runtime
        skill_prompt = (
            f"ACTIVE SKILL: {skill.name}\n"
            f"Description: {skill.description}\n"
            f"{skill.prompt_fragment}\n\n"
        )
        runtime._active_skill_prompt = skill_prompt

        if getattr(runtime, "_kv_optimization_enabled", False):
            logger.info(
                "Skill '%s' stored for user message injection (cache-friendly)",
                skill.name,
            )
            return

        if not getattr(runtime, "_base_system_prompt", None):
            runtime._base_system_prompt = getattr(runtime, "_system_prompt", "") or ""

        runtime._system_prompt = skill_prompt + (runtime._base_system_prompt or "")
        self._sync_conversation_system_prompt()
        logger.info("Injected skill '%s' into system prompt", skill.name)

    def inject_skills(self, skills: List[Any]) -> None:
        """Inject multiple skill prompt fragments into runtime state."""
        runtime = self._runtime
        if not skills:
            return

        skills = skills[:3]
        skill_names = []
        fragments = []
        for item in skills:
            skill = item[0] if isinstance(item, tuple) else item
            skill_names.append(skill.name)
            fragments.append(
                f"ACTIVE SKILL: {skill.name}\n"
                f"Description: {skill.description}\n"
                f"{skill.prompt_fragment}\n"
            )

        composed = (
            f"ACTIVE SKILLS ({len(skill_names)}): {' → '.join(skill_names)}\n"
            f"Execute these skills in the listed order.\n\n" + "\n".join(fragments) + "\n"
        )
        runtime._active_skill_prompt = composed

        if getattr(runtime, "_kv_optimization_enabled", False):
            logger.info("Skills %s stored for user message injection", skill_names)
            return

        if not getattr(runtime, "_base_system_prompt", None):
            runtime._base_system_prompt = getattr(runtime, "_system_prompt", "") or ""

        runtime._system_prompt = composed + (runtime._base_system_prompt or "")
        self._sync_conversation_system_prompt()
        logger.info("Injected %d skills: %s", len(skill_names), " → ".join(skill_names))

    def _sync_conversation_system_prompt(self) -> None:
        """Push the current system prompt into the live conversation state."""
        runtime = self._runtime
        get_prompt_runtime = getattr(runtime, "_get_prompt_builder_runtime", None)
        if callable(get_prompt_runtime):
            get_prompt_runtime().sync_conversation_system_prompt()
            return

        prompt_runtime = getattr(runtime, "_prompt_builder_runtime", None)
        if prompt_runtime is not None and hasattr(
            prompt_runtime, "sync_conversation_system_prompt"
        ):
            prompt_runtime.sync_conversation_system_prompt()
            return

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
