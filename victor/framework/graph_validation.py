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

"""State validation helpers for StateGraph."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type


class StateValidationError(Exception):
    """Raised when state validation fails."""

    def __init__(self, errors: List[str], state: Dict[str, Any]):
        self.errors = errors
        self.state = state
        message = f"State validation failed with {len(errors)} error(s):\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        super().__init__(message)


class StateValidator:
    """Validate state objects against a schema."""

    def __init__(self, schema: Optional[Type], strict: bool = False):
        self._schema = schema
        self._strict = strict
        self._is_pydantic = self._check_pydantic()
        self._is_typeddict = self._check_typeddict()

    def _check_pydantic(self) -> bool:
        if self._schema is None:
            return False
        try:
            from pydantic import BaseModel

            return isinstance(self._schema, type) and issubclass(self._schema, BaseModel)
        except (ImportError, TypeError):
            return False

    def _check_typeddict(self) -> bool:
        if self._schema is None:
            return False
        try:
            from typing_extensions import TypedDict

            if hasattr(self._schema, "__required_keys__"):
                return True
            return isinstance(self._schema, type) and issubclass(self._schema, TypedDict)
        except (ImportError, TypeError):
            return False

    def validate(self, state: Dict[str, Any]) -> List[str]:
        if self._schema is None:
            return []
        if self._is_pydantic:
            return self._validate_pydantic(state)
        if self._is_typeddict:
            return self._validate_typeddict(state)
        return []

    def _validate_pydantic(self, state: Dict[str, Any]) -> List[str]:
        try:
            from pydantic import ValidationError

            self._schema.model_validate(state)
            return []
        except ValidationError as error:
            errors = []
            for item in error.errors():
                loc = " -> ".join(str(x) for x in item["loc"])
                errors.append(f"{loc}: {item['msg']}")
            return errors
        except Exception as error:
            return [f"Validation error: {error}"]

    def _validate_typeddict(self, state: Dict[str, Any]) -> List[str]:
        try:
            from typing_extensions import TypedDict, get_type_hints

            errors = []
            hints = get_type_hints(self._schema)
            if hasattr(self._schema, "__required_keys__"):
                required = self._schema.__required_keys__
                optional = set(hints.keys()) - required
            else:
                required = set(hints.keys())
                optional = set()

            for key in required:
                if key not in state:
                    errors.append(f"Missing required field: '{key}'")

            for key, value in state.items():
                if key not in hints:
                    errors.append(f"Unexpected field: '{key}'")
                    continue
                expected_type = hints[key]
                if not self._check_type(value, expected_type):
                    errors.append(
                        f"Type mismatch for '{key}': expected {expected_type}, got {type(value).__name__}"
                    )

            for key in optional:
                if key in state:
                    expected_type = hints[key]
                    if not self._check_type(state[key], expected_type):
                        errors.append(
                            f"Type mismatch for optional field '{key}': expected {expected_type}, got {type(state[key]).__name__}"
                        )

            return errors
        except Exception as error:
            return [f"TypedDict validation error: {error}"]

    def _check_type(self, value: Any, expected_type: Type) -> bool:
        import typing
        from typing import get_args, get_origin

        origin = get_origin(expected_type)
        if origin is not None:
            if origin is list:
                if not isinstance(value, list):
                    return False
                if get_args(expected_type):
                    item_type = get_args(expected_type)[0]
                    return all(self._check_type(item, item_type) for item in value)
                return True

            if origin is dict:
                if not isinstance(value, dict):
                    return False
                if get_args(expected_type):
                    key_type, value_type = get_args(expected_type)
                    return all(
                        self._check_type(key, key_type) and self._check_type(item, value_type)
                        for key, item in value.items()
                    )
                return True

            if origin is typing.Union:
                args = get_args(expected_type)
                if len(args) == 2 and type(None) in args:
                    other_type = args[0] if args[1] is type(None) else args[1]
                    return value is None or self._check_type(value, other_type)
                return any(self._check_type(value, arg) for arg in args)

        try:
            return isinstance(value, expected_type)
        except TypeError:
            return True
