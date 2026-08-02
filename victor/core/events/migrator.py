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

"""Migration utilities for consolidating event systems.

This module provided conversion functions from the legacy VictorEvent system
to the canonical Event system. NOTE: Migration is complete, these functions
are no longer needed but kept for reference.

Topic Mapping Reference (OLD -> NEW):
- Tool events: (EventCategory.TOOL, "tool.start") -> "tool.start"
- Model events: (EventCategory.MODEL, "llm.request") -> "model.request"
- State events: (EventCategory.STATE, "state.change") -> "state.transition"
- Lifecycle events: (EventCategory.LIFECYCLE, "session.start") -> "lifecycle.session.start"
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

from victor.core.events.protocols import MessagingEvent

# Migration complete - VictorEvent and EventCategory have been removed
# TOPIC_MAPPING kept for reference only
TOPIC_MAPPING: Dict[str, str] = {
    # Tool events
    "tool.start": "tool.start",
    "tool.complete": "tool.result",
    "tool.error": "tool.error",
    "tool.end": "tool.end",
    "tool.output": "tool.output",
    # Model events
    "llm.request": "model.request",
    "llm.response": "model.response",
    "llm.error": "model.error",
    "llm.start": "model.start",
    "llm.end": "model.end",
    # State events
    "state.change": "state.transition",
    "state.snapshot": "state.snapshot",
    "state.restore": "state.restore",
    # Lifecycle events
    "session.start": "lifecycle.session.start",
    "session.end": "lifecycle.session.end",
    "agent.init": "lifecycle.agent.init",
    "agent.shutdown": "lifecycle.agent.shutdown",
    # Error events
    "error.occurred": "error.raised",
    "error.recovered": "error.recovered",
}

logger = logging.getLogger(__name__)

# =============================================================================
# Topic Mapping (Reference Only)
# =============================================================================

# OLD TOPIC_MAPPING commented out - migration complete
# TOPIC_MAPPING: Dict[Tuple[EventCategory, str], str] = {
#     # Tool events
#     (EventCategory.TOOL, "tool.start"): "tool.start",
#     (EventCategory.TOOL, "tool.complete"): "tool.result",
#     (EventCategory.TOOL, "tool.error"): "tool.error",
#     (EventCategory.TOOL, "tool.end"): "tool.end",
#     (EventCategory.TOOL, "tool.output"): "tool.output",
#     # Model events (LLM)
#     (EventCategory.MODEL, "llm.request"): "model.request",
#     (EventCategory.MODEL, "llm.response"): "model.response",
#     (EventCategory.MODEL, "llm.error"): "model.error",
#     (EventCategory.MODEL, "llm.start"): "model.start",
#     (EventCategory.MODEL, "llm.end"): "model.end",
#     # State events
#     (EventCategory.STATE, "state.change"): "state.transition",
#     (EventCategory.STATE, "state.snapshot"): "state.snapshot",
#     (EventCategory.STATE, "state.restore"): "state.restore",
#     # Lifecycle events
#     (EventCategory.LIFECYCLE, "session.start"): "lifecycle.session.start",
#     (EventCategory.LIFECYCLE, "session.end"): "lifecycle.session.end",
#     (EventCategory.LIFECYCLE, "agent.init"): "lifecycle.agent.init",
#     (EventCategory.LIFECYCLE, "agent.shutdown"): "lifecycle.agent.shutdown",
#     # Error events
#     (EventCategory.ERROR, "error.occurred"): "error.raised",
#     (EventCategory.ERROR, "error.recovered"): "error.recovered",
#     # Metric events
#     (EventCategory.METRIC, "metric.counter"): "metric.counter",
#     (EventCategory.METRIC, "metric.gauge"): "metric.gauge",
#     (EventCategory.METRIC, "metric.histogram"): "metric.histogram",
#     # Audit events
#     (EventCategory.AUDIT, "audit.access"): "audit.access",
#     (EventCategory.AUDIT, "audit.modify"): "audit.modify",
#     # Vertical events
#     (EventCategory.VERTICAL, "vertical.init"): "vertical.init",
#     (EventCategory.VERTICAL, "vertical.execute"): "vertical.execute",
#     (EventCategory.VERTICAL, "vertical.result"): "vertical.result",
#     # Custom events (preserve as-is)
#     (EventCategory.CUSTOM, "*"): "custom",
# }


# =============================================================================
# Conversion Functions
# =============================================================================


# def victor_event_to_event(victor_event: VictorEvent) -> Event:
# """Convert VictorEvent to canonical Event.

# This function converts legacy VictorEvent instances to the canonical
# Event format, preserving all data in the migration.

# Mapping:
# - category + name → topic
# - category → data["category"]
# - name → data["name"]
# - priority → data["priority"]
# - All other fields → Event fields

# Args:
# victor_event: Legacy VictorEvent instance

# Returns:
# Canonical Event instance

# Example:
# >>> victor_event = VictorEvent(
# ...     category=EventCategory.TOOL,
# ...     name="tool.start",
# ...     data={"tool": "read_file", "path": "test.txt"}
# ... )
# >>> event = victor_event_to_event(victor_event)
# >>> event.topic
# 'tool.start'
# >>> event.data["category"]
# 'tool'
# >>> event.data["tool"]
# 'read_file'
# """
# # Determine topic from category and name
# topic = _get_topic_for_event(victor_event)

# # Build data dictionary with legacy fields preserved as metadata
# data = {
# **victor_event.data,
# "category": victor_event.category.value,
# "name": victor_event.name,
# }

# # Add priority if present
# if victor_event.priority is not None:
# data["priority"] = victor_event.priority.value

# # Convert timestamp
# timestamp = None
# if victor_event.timestamp:
# timestamp = victor_event.timestamp.timestamp()

# # Create canonical Event
# event = Event(
# topic=topic,
# data=data,
# id=victor_event.id,
# timestamp=timestamp,
# source=victor_event.source or "victor",
# correlation_id=victor_event.trace_id,
# headers={
# "session_id": victor_event.session_id or "",
# "event_category": victor_event.category.value,
# "event_name": victor_event.name,
# },
# )

# logger.debug(
# f"[Migrator] Converted VictorEvent: {victor_event.category}/{victor_event.name} "
# f"→ Event: {event.topic}"
# )

# return event


# def _get_topic_for_event(victor_event: VictorEvent) -> str:
# """Get canonical topic for a VictorEvent.

# Args:
# victor_event: Legacy VictorEvent instance

# Returns:
# Canonical topic string

# Example:
# >>> event = VictorEvent(category=EventCategory.TOOL, name="tool.start", data={})
# >>> _get_topic_for_event(event)
# 'tool.start'
# """
# # Check explicit mapping first
# mapping_key = (victor_event.category, victor_event.name)
# if mapping_key in TOPIC_MAPPING:
# return TOPIC_MAPPING[mapping_key]

# # Fall back to category.name pattern
# # Example: EventCategory.TOOL + "my_custom_event" → "tool.my_custom_event"
# category_value = victor_event.category.value
# event_name = victor_event.name

# # Remove redundant category prefix if present
# # Example: "tool.tool.start" → "tool.start"
# if event_name.startswith(f"{category_value}."):
# topic = event_name
# else:
# topic = f"{category_value}.{event_name}"

# logger.debug(f"[Migrator] No explicit mapping for {mapping_key}, using topic: {topic}")
# return topic


# def event_to_victor_event(event: Event) -> VictorEvent:
# """Convert canonical Event back to VictorEvent (for compatibility).

# This reverse conversion is provided for gradual migration scenarios
# where some components still use VictorEvent.

# Args:
# event: Canonical Event instance

# Returns:
# Legacy VictorEvent instance

# Example:
# >>> event = Event(topic="tool.start", data={"tool": "read_file", "category": "tool"})
# >>> victor_event = event_to_victor_event(event)
# >>> victor_event.category
# <EventCategory.TOOL: 'tool'>
# >>> victor_event.name
# 'tool.start'
# """
# from datetime import datetime, timezone

# # Extract category from metadata
# category_str = event.data.get("category", "custom")
# try:
# category = EventCategory(category_str.lower())
# except ValueError:
# logger.warning(f"[Migrator] Unknown category '{category_str}', using CUSTOM")
# category = EventCategory.CUSTOM

# # Extract name from metadata or topic
# name = event.data.get("name", event.topic)

# # Extract timestamp
# timestamp = None
# if event.timestamp:
# timestamp = datetime.fromtimestamp(event.timestamp, tz=timezone.utc)

# # Create VictorEvent
# victor_event = VictorEvent(
# id=event.id,
# timestamp=timestamp,
# category=category,
# name=name,
# data={k: v for k, v in event.data.items() if k not in ["category", "name", "priority"]},
# source=event.source,
# trace_id=event.correlation_id,
# session_id=event.headers.get("session_id"),
# )

# # Extract priority if present
# if "priority" in event.data:
# # EventPriority removed - use Event.data["priority"]
# try:
# priority_str = event.data["priority"]
# victor_event.priority = EventPriority(priority_str)
# except (ValueError, KeyError):
# pass

# return victor_event


# # =============================================================================
# # Batch Migration
# # =============================================================================


# def migrate_event_list(victor_events: list[VictorEvent]) -> list[Event]:
#     """Migrate a list of VictorEvents to canonical Events.
#
#     Args:
#         victor_events: List of VictorEvent instances
#
#     Returns:
#         List of canonical Event instances
#
#     Example:
#         >>> victor_events = [VictorEvent(...), VictorEvent(...)]
#         >>> events = migrate_event_list(victor_events)
#         >>> len(events)
#         2
#     """
#     return [victor_event_to_event(ve) for ve in victor_events]


# def get_topic_mapping() -> Dict[Tuple[EventCategory, str], str]:
#     """Get the complete topic mapping table.
#
#     Returns:
#         Dictionary mapping (category, name) tuples to topics
#
#     Example:
#         >>> mapping = get_topic_mapping()
#         >>> mapping[(EventCategory.TOOL, "tool.start")]
#         'tool.start'
#     """
#     return TOPIC_MAPPING.copy()
