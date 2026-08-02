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

"""State externalization service for vertical context persistence.

This module provides a service for persisting VerticalContext and related state
to the project database, enabling:
- State sharing across processes
- Session recovery and restoration
- State inspection and debugging
- Historical state analysis

Phase 4.3: State Externalization

Architecture:
    VerticalContext (in-memory)
            ↓
    StateService.save() / load()
            ↓
    project.db (SQLite) ← Reuses existing DatabaseManager

Benefits:
    - Process isolation: Vertical state survives process restarts
    - Debugging: Inspect state from external tools
    - Testing: Restore specific state for testing
    - Monitoring: Track state changes over time
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps, json_loads
import logging
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

from victor.agent.vertical_context import VerticalContext
from victor.core.database import get_project_database
from victor.core.schema import Tables, Schema

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=VerticalContext)


class StateService:
    """Service for persisting and loading vertical state.

    This service provides methods to save and load VerticalContext
    instances to/from the project database, enabling state
    externalization and cross-process sharing.

    Usage:
        from victor.agent.state_service import StateService

        # Save vertical state
        service = StateService()
        state_id = service.save_vertical_state(
            context=context,
            vertical_name="coding",
            vertical_version="1.5.0",
        )

        # Load vertical state
        context = service.load_vertical_state(state_id)

        # Query states
        states = service.list_vertical_states(vertical_name="coding")
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize state service.

        Args:
            db_path: Path to project database (default: .victor/project.db)
        """
        self._db = get_project_database(db_path)
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Ensure required tables exist."""
        try:
            # Create vertical state tables
            self._db.execute(Schema.VERTICAL_STATE)
            self._db.execute(Schema.VERTICAL_NEGOTIATION)
            self._db.execute(Schema.VERTICAL_SESSION)

            # Create indexes (split into individual statements)
            indexes = Schema.VERTICAL_INDEXES.strip().split(";")
            for index_sql in indexes:
                index_sql = index_sql.strip()
                if index_sql:
                    self._db.execute(index_sql)

            logger.debug("State service tables initialized")
        except Exception as e:
            logger.error(f"Failed to initialize state service tables: {e}")
            raise

    def save_vertical_state(
        self,
        context: VerticalContext,
        vertical_name: str,
        vertical_version: Optional[str] = None,
        project_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Save vertical context to database.

        Args:
            context: Vertical context to persist
            vertical_name: Name of the vertical
            vertical_version: Optional version string
            project_path: Optional project path
            session_id: Optional session ID

        Returns:
            State ID for later retrieval
        """
        state_id = str(uuid.uuid4())

        # Serialize context to JSON
        state_data = self._serialize_context(context)

        # Prepare row data
        row = {
            "id": state_id,
            "vertical_name": vertical_name,
            "vertical_version": vertical_version,
            "project_path": project_path,
            "session_id": session_id,
            "state_data": state_data,
            "config_json": json_dumps(context.config) if context.config else None,
            "stages_json": json_dumps(context.stages) if context.stages else None,
            "middleware_json": (
                json_dumps([self._serialize_middleware(m) for m in context.middleware])
                if context.middleware
                else None
            ),
            "safety_patterns_json": (
                json_dumps([self._serialize_safety_pattern(p) for p in context.safety_patterns])
                if context.safety_patterns
                else None
            ),
            "enabled_tools_json": (
                json_dumps(list(context.enabled_tools)) if context.enabled_tools else None
            ),
            "mode_configs_json": (
                json_dumps(context.mode_configs) if context.mode_configs else None
            ),
            "negotiation_results_json": (
                json_dumps(context.capability_negotiation_results)
                if context.capability_negotiation_results
                else None
            ),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        # Insert into database
        try:
            self._db.execute(
                f"INSERT INTO {Tables.VERTICAL_STATE} "
                f"(id, vertical_name, vertical_version, project_path, session_id, "
                f"state_data, config_json, stages_json, middleware_json, safety_patterns_json, "
                f"enabled_tools_json, mode_configs_json, negotiation_results_json, created_at, updated_at) "
                f"VALUES (:id, :vertical_name, :vertical_version, :project_path, :session_id, "
                f":state_data, :config_json, :stages_json, :middleware_json, :safety_patterns_json, "
                f":enabled_tools_json, :mode_configs_json, :negotiation_results_json, :created_at, :updated_at)",
                row,
            )

            # Save negotiation results separately if present
            if context.capability_negotiation_results:
                self._save_negotiation_results(state_id, context.capability_negotiation_results)

            logger.info(f"Saved vertical state: {state_id} ({vertical_name})")
            return state_id

        except Exception as e:
            logger.error(f"Failed to save vertical state: {e}")
            raise

    def load_vertical_state(
        self,
        state_id: str,
        context_class: Type[T] = VerticalContext,
    ) -> Optional[T]:
        """Load vertical context from database.

        Args:
            state_id: State ID from save_vertical_state()
            context_class: VerticalContext subclass to instantiate

        Returns:
            VerticalContext instance or None if not found
        """
        try:
            # Query state with explicit column names for clarity
            result = self._db.query(
                f"SELECT id, vertical_name, vertical_version, project_path, session_id, "
                f"state_data, config_json, stages_json, middleware_json, safety_patterns_json, "
                f"enabled_tools_json, mode_configs_json, negotiation_results_json, created_at, updated_at "
                f"FROM {Tables.VERTICAL_STATE} WHERE id = ?",
                (state_id,),
            )

            if not result:
                logger.warning(f"Vertical state not found: {state_id}")
                return None

            row = result[0]
            context = self._deserialize_context(row, context_class)

            # Load negotiation results
            negotiation_results = self._load_negotiation_results(state_id)
            if negotiation_results:
                context.capability_negotiation_results = negotiation_results

            logger.info(f"Loaded vertical state: {state_id}")
            return context

        except Exception as e:
            logger.error(f"Failed to load vertical state {state_id}: {e}")
            raise

    def update_vertical_state(
        self,
        state_id: str,
        context: VerticalContext,
    ) -> bool:
        """Update existing vertical state.

        Args:
            state_id: State ID to update
            context: New context data

        Returns:
            True if updated, False if not found
        """
        try:
            # Serialize context
            state_data = self._serialize_context(context)

            # Update row
            self._db.execute(
                f"UPDATE {Tables.VERTICAL_STATE} SET "
                f"state_data = ?, "
                f"config_json = ?, "
                f"stages_json = ?, "
                f"middleware_json = ?, "
                f"safety_patterns_json = ?, "
                f"enabled_tools_json = ?, "
                f"mode_configs_json = ?, "
                f"negotiation_results_json = ?, "
                f"updated_at = ? "
                f"WHERE id = ?",
                (
                    state_data,
                    json_dumps(context.config) if context.config else None,
                    json_dumps(context.stages) if context.stages else None,
                    (
                        json_dumps([self._serialize_middleware(m) for m in context.middleware])
                        if context.middleware
                        else None
                    ),
                    (
                        json_dumps(
                            [self._serialize_safety_pattern(p) for p in context.safety_patterns]
                        )
                        if context.safety_patterns
                        else None
                    ),
                    (json_dumps(list(context.enabled_tools)) if context.enabled_tools else None),
                    json_dumps(context.mode_configs) if context.mode_configs else None,
                    (
                        json_dumps(context.capability_negotiation_results)
                        if context.capability_negotiation_results
                        else None
                    ),
                    datetime.now().isoformat(),
                    state_id,
                ),
            )

            # Update negotiation results
            if context.capability_negotiation_results:
                # Delete old results
                self._db.execute(
                    f"DELETE FROM {Tables.VERTICAL_NEGOTIATION} WHERE vertical_state_id = ?",
                    (state_id,),
                )
                # Insert new results
                self._save_negotiation_results(state_id, context.capability_negotiation_results)

            logger.info(f"Updated vertical state: {state_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update vertical state {state_id}: {e}")
            return False

    def delete_vertical_state(self, state_id: str) -> bool:
        """Delete vertical state from database.

        Args:
            state_id: State ID to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            # Delete negotiation results first (foreign key)
            self._db.execute(
                f"DELETE FROM {Tables.VERTICAL_NEGOTIATION} WHERE vertical_state_id = ?",
                (state_id,),
            )

            # Delete state
            self._db.execute(
                f"DELETE FROM {Tables.VERTICAL_STATE} WHERE id = ?",
                (state_id,),
            )

            logger.info(f"Deleted vertical state: {state_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete vertical state {state_id}: {e}")
            return False

    def list_vertical_states(
        self,
        vertical_name: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List vertical states with optional filtering.

        Args:
            vertical_name: Filter by vertical name
            session_id: Filter by session ID
            limit: Maximum number of results

        Returns:
            List of state metadata dictionaries
        """
        try:
            query = f"SELECT id, vertical_name, vertical_version, project_path, session_id, created_at, updated_at FROM {Tables.VERTICAL_STATE}"
            params = []

            conditions = []
            if vertical_name:
                conditions.append("vertical_name = ?")
                params.append(vertical_name)
            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)

            result = self._db.query(query, tuple(params))

            return [
                {
                    "id": row[0],
                    "vertical_name": row[1],
                    "vertical_version": row[2],
                    "project_path": row[3],
                    "session_id": row[4],
                    "created_at": row[5],
                    "updated_at": row[6],
                }
                for row in result
            ]

        except Exception as e:
            logger.error(f"Failed to list vertical states: {e}")
            return []

    def _save_negotiation_results(
        self,
        state_id: str,
        results: Dict[str, Any],
    ) -> None:
        """Save capability negotiation results.

        Args:
            state_id: State ID
            results: Negotiation results dictionary
        """
        try:
            for capability_name, result in results.items():
                result_dict = result.to_dict() if hasattr(result, "to_dict") else result

                self._db.execute(
                    f"INSERT INTO {Tables.VERTICAL_NEGOTIATION} "
                    f"(vertical_state_id, capability_name, status, agreed_version, "
                    f"supported_features, unsupported_features, missing_required_features, "
                    f"fallback_version, error_message) "
                    f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        state_id,
                        capability_name,
                        result_dict.get("status"),
                        result_dict.get("agreed_version"),
                        json_dumps(result_dict.get("supported_features", [])),
                        json_dumps(result_dict.get("unsupported_features", [])),
                        json_dumps(result_dict.get("missing_required_features", [])),
                        result_dict.get("fallback_version"),
                        result_dict.get("error"),
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to save negotiation results: {e}")
            raise

    def _load_negotiation_results(
        self,
        state_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Load capability negotiation results.

        Args:
            state_id: State ID

        Returns:
            Negotiation results dictionary or None
        """
        try:
            result = self._db.query(
                f"SELECT capability_name, status, agreed_version, supported_features, "
                f"unsupported_features, missing_required_features, fallback_version, error_message "
                f"FROM {Tables.VERTICAL_NEGOTIATION} WHERE vertical_state_id = ?",
                (state_id,),
            )

            if not result:
                return None

            results = {}
            for row in result:
                results[row[0]] = {
                    "status": row[1],
                    "agreed_version": row[2],
                    "supported_features": json_loads(row[3]) if row[3] else [],
                    "unsupported_features": json_loads(row[4]) if row[4] else [],
                    "missing_required_features": json_loads(row[5]) if row[5] else [],
                    "fallback_version": row[6],
                    "error": row[7],
                }

            return results

        except Exception as e:
            logger.error(f"Failed to load negotiation results: {e}")
            return None

    def _serialize_context(self, context: VerticalContext) -> str:
        """Serialize context to JSON string.

        Args:
            context: Vertical context

        Returns:
            JSON string
        """
        # Convert to dict
        data = {
            "name": context.name,
            "stages": context.stages,
            "middleware": len(context.middleware),
            "safety_patterns": len(context.safety_patterns),
            "task_hints": context.task_hints,
            "system_prompt": context.system_prompt,
            "prompt_sections": context.prompt_sections,
            "mode_configs": context.mode_configs,
            "default_mode": context.default_mode,
            "default_budget": context.default_budget,
            "tool_dependencies": len(context.tool_dependencies),
            "tool_sequences": context.tool_sequences,
            "enabled_tools": list(context.enabled_tools),
        }
        return json_dumps(data)

    def _deserialize_context(
        self,
        row: tuple,
        context_class: Type[T],
    ) -> T:
        """Deserialize context from database row.

        Args:
            row: Database row (id, vertical_name, vertical_version, project_path, session_id,
                  state_data, config_json, stages_json, middleware_json, safety_patterns_json,
                  enabled_tools_json, mode_configs_json, negotiation_results_json, created_at, updated_at)
        Returns:
            VerticalContext instance
        """
        # Unpack row with explicit indices for clarity
        # 0:id, 1:vertical_name, 2:vertical_version, 3:project_path, 4:session_id,
        # 5:state_data, 6:config_json, 7:stages_json, 8:middleware_json, 9:safety_patterns_json,
        # 10:enabled_tools_json, 11:mode_configs_json, 12:negotiation_results_json, 13:created_at, 14:updated_at

        # Parse state_data
        state_data_str = row[5]
        if state_data_str:
            state_data = json_loads(state_data_str)
        else:
            state_data = {}

        # Parse stages
        stages_str = row[7]
        if stages_str:
            stages = json_loads(stages_str)
        else:
            stages = {}

        # Parse enabled_tools
        tools_str = row[10]
        if tools_str:
            enabled_tools = set(json_loads(tools_str))
        else:
            enabled_tools = set()

        # Parse mode_configs
        mode_str = row[11]
        if mode_str:
            mode_configs = json_loads(mode_str)
        else:
            mode_configs = {}

        # Create context instance
        context = context_class(
            name=row[1],  # vertical_name
            config=None,  # Config would need full deserialization
            stages=stages,
            middleware=[],  # Middleware objects would need reconstruction
            safety_patterns=[],  # Safety patterns would need reconstruction
            task_hints={},
            system_prompt=state_data.get("system_prompt"),
            prompt_sections=state_data.get("prompt_sections", []),
            mode_configs=mode_configs,
            default_mode=state_data.get("default_mode", "default"),
            default_budget=state_data.get("default_budget", 10),
            tool_dependencies=[],
            tool_sequences=state_data.get("tool_sequences", []),
            enabled_tools=enabled_tools,
        )

        return context

    def _serialize_middleware(self, middleware: Any) -> Dict[str, Any]:
        """Serialize middleware to dict.

        Args:
            middleware: Middleware instance

        Returns:
            Dictionary representation
        """
        return {
            "type": type(middleware).__name__,
            "module": type(middleware).__module__,
        }

    def _serialize_safety_pattern(self, pattern: Any) -> Dict[str, Any]:
        """Serialize safety pattern to dict.

        Args:
            pattern: Safety pattern instance

        Returns:
            Dictionary representation
        """
        return {
            "type": type(pattern).__name__,
            "module": type(pattern).__module__,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_state_service_instance: Optional[StateService] = None


def get_state_service() -> StateService:
    """Get singleton state service instance.

    Returns:
        StateService instance
    """
    global _state_service_instance
    if _state_service_instance is None:
        _state_service_instance = StateService()
    return _state_service_instance


def save_vertical_state(
    context: VerticalContext,
    vertical_name: str,
    **kwargs,
) -> str:
    """Save vertical context to database.

    Convenience function that uses the singleton StateService.

    Args:
        context: Vertical context to persist
        vertical_name: Name of the vertical
        **kwargs: Additional arguments for StateService.save_vertical_state()

    Returns:
        State ID for later retrieval
    """
    service = get_state_service()
    return service.save_vertical_state(context, vertical_name, **kwargs)


def load_vertical_state(
    state_id: str,
    context_class: Type[T] = VerticalContext,
) -> Optional[T]:
    """Load vertical context from database.

    Convenience function that uses the singleton StateService.

    Args:
        state_id: State ID from save_vertical_state()
        context_class: VerticalContext subclass to instantiate

    Returns:
        VerticalContext instance or None if not found
    """
    service = get_state_service()
    return service.load_vertical_state(state_id, context_class)


__all__ = [
    "StateService",
    "get_state_service",
    "save_vertical_state",
    "load_vertical_state",
]
