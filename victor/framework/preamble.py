"""Preamble manager for system prompt section management.

Provides structured access to system prompt sections, enabling users to
inspect, toggle, and customize prompt sections at runtime. Integrates with
PromptBuilder to manage the full prompt lifecycle.

Features:
- **Section inspection**: List all active/inactive prompt sections
- **Toggle sections**: Enable/disable individual sections
- **Preamble injection**: Add custom preamble at configurable positions
- **Prompt optimization feedback**: Show GEPA/MIPROv2 optimization state
- **Reset**: Restore default prompt structure
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from victor.framework.prompt_builder import PromptBuilder, PromptSection

logger = logging.getLogger(__name__)


class PreamblePosition(str, Enum):
    """Where to inject a custom preamble in the prompt."""

    TOP = "top"  # Before all sections
    BOTTOM = "bottom"  # After all sections
    BEFORE_SECTION = "before_section"  # Before a specific section
    AFTER_SECTION = "after_section"  # After a specific section


@dataclass
class PreambleEntry:
    """A single preamble entry injected into the system prompt.

    Attributes:
        text: The preamble text to inject
        position: Where to inject the preamble
        target_section: Section name for BEFORE_SECTION/AFTER_SECTION positions
        enabled: Whether this preamble is active
    """

    text: str
    position: PreamblePosition = PreamblePosition.TOP
    target_section: Optional[str] = None
    enabled: bool = True


@dataclass
class SectionInfo:
    """Information about a prompt section for display.

    Attributes:
        name: Section name
        content: Section content (first 200 chars for preview)
        priority: Section priority
        enabled: Whether the section is active
        evolved: Whether this section has evolved content
        full_content: Complete section content
    """

    name: str
    content: str
    priority: int
    enabled: bool
    evolved: bool = False
    full_content: str = ""


@dataclass
class OptimizationStatus:
    """Status of prompt optimization for display.

    Attributes:
        enabled: Whether prompt optimization is enabled
        active_strategies: List of active optimization strategies
        evolved_sections: Number of sections with evolved content
        total_sections: Total number of evolvable sections
        last_evolution: Timestamp of last evolution cycle
        current_tier: Current GEPA tier (economic/balanced/performance)
    """

    enabled: bool = False
    active_strategies: List[str] = field(default_factory=list)
    evolved_sections: int = 0
    total_sections: int = 0
    last_evolution: Optional[str] = None
    current_tier: str = "balanced"


class PreambleManager:
    """Manages system prompt sections and user preambles.

    Provides a clean API for inspecting and modifying the system prompt
    at runtime. Integrates with PromptBuilder to apply changes.

    Usage:
        manager = PreambleManager(prompt_builder)
        sections = manager.get_active_sections()
        manager.set_preamble("Custom instruction", position=PreamblePosition.TOP)
        manager.toggle_section("tool_hints", enabled=False)
        manager.reset()
    """

    # Sections that can be toggled by users
    _TOGGLEABLE_SECTIONS = {
        "tool_hints": "Tool usage hints",
        "safety_rules": "Safety constraints",
        "grounding": "Grounding rules",
        "context": "Contextual information",
        "analysis_efficiency": "Analysis efficiency guidance",
    }

    # Sections that should never be toggled
    _PROTECTED_SECTIONS = {"identity", "capabilities"}

    def __init__(self, prompt_builder: Optional[PromptBuilder] = None):
        """Initialize PreambleManager.

        Args:
            prompt_builder: Optional PromptBuilder instance to manage
        """
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._preambles: List[PreambleEntry] = []
        self._section_overrides: Dict[str, bool] = {}  # name -> enabled

    # ── Section Inspection ─────────────────────────────────────────

    def get_active_sections(self) -> Dict[str, SectionInfo]:
        """Return all active prompt sections with metadata.

        Returns:
            Dict mapping section names to SectionInfo objects
        """
        sections: Dict[str, SectionInfo] = {}
        for name, section in self._prompt_builder.iter_named_sections():
            content = section.content or ""
            sections[name] = SectionInfo(
                name=name,
                content=content[:200] if content else "",
                priority=section.priority,
                enabled=self._is_section_enabled(name, section),
                evolved=self._is_section_evolved(name),
                full_content=content,
            )
        return sections

    def get_section(self, name: str) -> Optional[SectionInfo]:
        """Get information about a specific section.

        Args:
            name: Section name

        Returns:
            SectionInfo if found, None otherwise
        """
        section = self._prompt_builder.get_section(name)
        if not section:
            return None
        content = section.content or ""
        return SectionInfo(
            name=name,
            content=content[:200] if content else "",
            priority=section.priority,
            enabled=self._is_section_enabled(name, section),
            evolved=self._is_section_evolved(name),
            full_content=content,
        )

    def list_toggleable_sections(self) -> Dict[str, str]:
        """Return sections that can be toggled with descriptions.

        Returns:
            Dict mapping section names to human-readable descriptions
        """
        result: Dict[str, str] = {}
        for name, desc in self._TOGGLEABLE_SECTIONS.items():
            if self._prompt_builder.has_section(name):
                result[name] = desc
        return result

    def get_full_prompt(self) -> str:
        """Return the complete system prompt with all preambles applied.

        Returns:
            Complete system prompt string
        """
        return self._prompt_builder.build()

    # ── Section Toggling ───────────────────────────────────────────

    def toggle_section(self, name: str, enabled: bool) -> bool:
        """Enable or disable a prompt section.

        Args:
            name: Section name to toggle
            enabled: True to enable, False to disable

        Returns:
            True if the section was toggled, False if not found or protected
        """
        if name in self._PROTECTED_SECTIONS:
            logger.warning("Cannot toggle protected section: %s", name)
            return False

        if not self._prompt_builder.has_section(name):
            logger.warning("Section not found: %s", name)
            return False

        if enabled:
            self._prompt_builder.enable_section(name)
        else:
            self._prompt_builder.disable_section(name)

        self._section_overrides[name] = enabled
        logger.info("Section '%s' %s", name, "enabled" if enabled else "disabled")
        return True

    def is_section_enabled(self, name: str) -> bool:
        """Check if a section is currently enabled.

        Args:
            name: Section name

        Returns:
            True if the section is enabled
        """
        section = self._prompt_builder.get_section(name)
        if not section:
            return False
        return self._is_section_enabled(name, section)

    # ── Preamble Management ────────────────────────────────────────

    def set_preamble(
        self,
        text: str,
        position: PreamblePosition = PreamblePosition.TOP,
        target_section: Optional[str] = None,
    ) -> None:
        """Inject a custom preamble at the specified position.

        Args:
            text: Preamble text to inject
            position: Where to inject the preamble
            target_section: Required for BEFORE_SECTION/AFTER_SECTION positions

        Raises:
            ValueError: If target_section is required but not provided,
                       or if the target section doesn't exist
        """
        if position in (PreamblePosition.BEFORE_SECTION, PreamblePosition.AFTER_SECTION):
            if not target_section:
                raise ValueError(f"target_section required for position '{position.value}'")
            if not self._prompt_builder.has_section(target_section):
                raise ValueError(f"Target section '{target_section}' not found")

        entry = PreambleEntry(
            text=text,
            position=position,
            target_section=target_section,
            enabled=True,
        )
        self._preambles.append(entry)
        self._apply_preamble(entry)
        logger.info(
            "Preamble added at position '%s' (target=%s)",
            position.value,
            target_section or "none",
        )

    def remove_preamble(self, index: int) -> bool:
        """Remove a preamble by index.

        Args:
            index: Index of the preamble to remove (0-based)

        Returns:
            True if removed, False if index out of range
        """
        if index < 0 or index >= len(self._preambles):
            return False
        self._preambles.pop(index)
        self._rebuild_prompt()
        logger.info("Preamble at index %d removed", index)
        return True

    def list_preambles(self) -> List[PreambleEntry]:
        """Return all active preamble entries.

        Returns:
            List of PreambleEntry objects
        """
        return list(self._preambles)

    def clear_preambles(self) -> None:
        """Remove all preamble entries."""
        self._preambles.clear()
        self._rebuild_prompt()
        logger.info("All preambles cleared")

    # ── Reset ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset prompt to default state.

        Clears all preambles, re-enables all toggleable sections,
        and restores default prompt structure.
        """
        self._preambles.clear()
        self._section_overrides.clear()
        # Re-enable all toggleable sections
        for name in self._TOGGLEABLE_SECTIONS:
            if self._prompt_builder.has_section(name):
                self._prompt_builder.enable_section(name)
        # Re-enable any other disabled sections
        for name, section in self._prompt_builder.iter_named_sections():
            if not section.enabled:
                self._prompt_builder.enable_section(name)
        logger.info("Prompt reset to default state")

    # ── Optimization Status ────────────────────────────────────────

    def get_optimization_status(self) -> OptimizationStatus:
        """Get current prompt optimization status.

        Returns:
            OptimizationStatus with current optimization state
        """
        from victor.config import settings

        opt_settings = settings.prompt_optimization

        # Count evolved sections
        evolved = sum(
            1
            for name, _ in self._prompt_builder.iter_named_sections()
            if self._is_section_evolved(name)
        )

        total = len(self._prompt_builder.iter_named_sections())

        return OptimizationStatus(
            enabled=opt_settings.enabled,
            active_strategies=opt_settings.default_strategies.copy(),
            evolved_sections=evolved,
            total_sections=total,
            last_evolution=self._get_last_evolution_time(),
            current_tier=opt_settings.gepa.default_tier,
        )

    # ── Internal Helpers ───────────────────────────────────────────

    def _is_section_enabled(self, name: str, section: PromptSection) -> bool:
        """Check if a section is enabled, considering overrides."""
        if name in self._section_overrides:
            return self._section_overrides[name]
        return section.enabled

    def _is_section_evolved(self, name: str) -> bool:
        """Check if a section has evolved content.

        Checks for evolved content in the optimization system.
        """
        try:
            from victor.agent.evolved_content_resolver import EvolvedContentResolver
            from victor.agent.optimization_injector import OptimizationInjector

            resolver = EvolvedContentResolver(optimization_injector=OptimizationInjector())
            return resolver.has_evolved_content(name)
        except ImportError:
            return False
        except Exception:
            logger.debug("Failed to check evolved content for '%s'", name, exc_info=True)
            return False

    def _get_last_evolution_time(self) -> Optional[str]:
        """Get the timestamp of the last evolution cycle."""
        try:
            from victor.agent.optimization_injector import OptimizationInjector

            injector = OptimizationInjector()
            return injector.get_last_evolution_time()
        except Exception:
            return None

    def _apply_preamble(self, entry: PreambleEntry) -> None:
        """Apply a single preamble entry to the prompt builder."""
        if not entry.enabled:
            return

        section_name = f"_preamble_{len(self._preambles)}"

        if entry.position == PreamblePosition.TOP:
            self._prompt_builder.add_section(
                name=section_name,
                content=entry.text,
                priority=1,  # Very high priority (appears first)
                header="## User Preamble",
            )
        elif entry.position == PreamblePosition.BOTTOM:
            self._prompt_builder.add_section(
                name=section_name,
                content=entry.text,
                priority=999,  # Very low priority (appears last)
                header="## User Preamble",
            )
        elif entry.position == PreamblePosition.BEFORE_SECTION:
            # Add with priority just above the target section
            target = self._prompt_builder.get_section(entry.target_section)
            target_priority = target.priority if target else 50
            self._prompt_builder.add_section(
                name=section_name,
                content=entry.text,
                priority=target_priority - 1,
                header="## User Preamble",
            )
        elif entry.position == PreamblePosition.AFTER_SECTION:
            # Add with priority just below the target section
            target = self._prompt_builder.get_section(entry.target_section)
            target_priority = target.priority if target else 50
            self._prompt_builder.add_section(
                name=section_name,
                content=entry.text,
                priority=target_priority + 1,
                header="## User Preamble",
            )

    def _rebuild_prompt(self) -> None:
        """Rebuild the prompt after modifications.

        Re-applies all preambles to ensure consistent state.
        """
        # Remove existing preamble sections
        for entry in self._preambles:
            section_name = f"_preamble_{self._preambles.index(entry)}"
            self._prompt_builder.remove_section(section_name)

        # Re-apply all preambles
        for i, entry in enumerate(self._preambles):
            if entry.enabled:
                section_name = f"_preamble_{i}"
                priority = self._get_preamble_priority(entry)
                self._prompt_builder.add_section(
                    name=section_name,
                    content=entry.text,
                    priority=priority,
                    header="## User Preamble",
                )

    def _get_preamble_priority(self, entry: PreambleEntry) -> int:
        """Get the priority for a preamble based on its position."""
        if entry.position == PreamblePosition.TOP:
            return 1
        elif entry.position == PreamblePosition.BOTTOM:
            return 999
        elif entry.position in (PreamblePosition.BEFORE_SECTION, PreamblePosition.AFTER_SECTION):
            target = self._prompt_builder.get_section(entry.target_section)
            if target:
                base = target.priority
                return base - 1 if entry.position == PreamblePosition.BEFORE_SECTION else base + 1
        return 50
