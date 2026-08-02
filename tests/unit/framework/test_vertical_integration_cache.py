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

"""Tests for JSON-based IntegrationResult caching (Phase 2.3: Pickle to JSON).

Phase 2.3: Switch from pickle to JSON serialization.
Uses existing to_dict() and from_dict() methods.
"""

import json

import pytest

from victor.framework.vertical_integration import IntegrationResult


class TestIntegrationResultJSONSerialization:
    """Test JSON serialization for IntegrationResult caching."""

    def test_to_dict_returns_dict(self):
        """to_dict should return a dictionary representation."""
        result = IntegrationResult(
            success=True,
            vertical_name="coding",
            tools_applied={"read", "edit"},
        )
        data = result.to_dict()
        assert isinstance(data, dict)

    def test_from_dict_recreates_result(self):
        """from_dict should recreate IntegrationResult from dict."""
        original = IntegrationResult(
            success=True,
            vertical_name="coding",
            tools_applied={"read", "edit"},
        )

        data = original.to_dict()
        recreated = IntegrationResult.from_dict(data)

        assert recreated.success == original.success
        assert recreated.vertical_name == original.vertical_name
        assert recreated.tools_applied == original.tools_applied

    def test_json_serialization_roundtrip(self):
        """IntegrationResult should survive JSON serialization roundtrip."""
        original = IntegrationResult(
            success=True,
            vertical_name="research",
            tools_applied={"web_search", "web_fetch"},
            middleware_count=2,
            safety_patterns_count=1,
        )

        # Serialize to JSON
        json_str = json.dumps(original.to_dict())

        # Deserialize from JSON
        data = json.loads(json_str)
        recreated = IntegrationResult.from_dict(data)

        assert recreated.success == original.success
        assert recreated.vertical_name == original.vertical_name
        assert recreated.tools_applied == original.tools_applied
        assert recreated.middleware_count == original.middleware_count

    def test_to_dict_handles_all_fields(self):
        """to_dict should handle all result fields."""
        result = IntegrationResult(
            success=False,
            vertical_name="devops",
            tools_applied={"shell", "docker"},
            middleware_count=1,
            safety_patterns_count=2,
            errors=["Test error"],
            warnings=["Test warning"],
        )
        data = result.to_dict()

        assert data["success"] is False
        assert data["vertical_name"] == "devops"
        assert "shell" in data["tools_applied"]
        assert "docker" in data["tools_applied"]
        assert data["middleware_count"] == 1
        assert data["safety_patterns_count"] == 2
        assert "Test error" in data["errors"]

    def test_from_dict_handles_all_fields(self):
        """from_dict should handle all result fields."""
        data = {
            "success": True,
            "validation_status": "success",
            "vertical_name": "dataanalysis",
            "tools_applied": ["shell", "read"],
            "middleware_count": 1,
            "safety_patterns_count": 0,
            "errors": [],
            "warnings": [],
            "info": [],
        }
        result = IntegrationResult.from_dict(data)

        assert result.success is True
        assert result.vertical_name == "dataanalysis"
        assert "shell" in result.tools_applied


class TestCacheUsesJSONNotPickle:
    """Test that caching uses JSON serialization, not pickle.

    This was the main issue in Phase 2.3 - pickle couldn't handle
    middleware objects with locks/file handles.
    """

    def test_to_dict_returns_json_serializable(self):
        """to_dict should return JSON-serializable data."""
        result = IntegrationResult(
            success=True,
            vertical_name="coding",
            tools_applied={"read", "grep", "shell"},
            middleware_count=2,
        )

        data = result.to_dict()

        # Should be JSON serializable (no error)
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

    def test_from_dict_handles_json_data(self):
        """from_dict should handle JSON data."""
        data = {
            "success": True,
            "validation_status": "success",
            "vertical_name": "research",
            "tools_applied": ["web_search"],
            "middleware_count": 0,
        }

        # Parse from JSON (simulate cache load)
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        # Should create valid IntegrationResult
        result = IntegrationResult.from_dict(parsed)
        assert result.vertical_name == "research"

    def test_roundtrip_preserves_essential_fields(self):
        """JSON roundtrip should preserve essential result data."""
        original = IntegrationResult(
            success=True,
            vertical_name="dataanalysis",
            tools_applied={"shell", "read", "write"},
            middleware_count=1,
            safety_patterns_count=2,
            rl_learners_count=3,
        )

        # Simulate cache roundtrip
        data = original.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        restored = IntegrationResult.from_dict(parsed)

        # Essential data should be preserved
        assert restored.success == original.success
        assert restored.vertical_name == original.vertical_name
        assert restored.tools_applied == original.tools_applied
        assert restored.middleware_count == original.middleware_count
        assert restored.safety_patterns_count == original.safety_patterns_count
        assert restored.rl_learners_count == original.rl_learners_count

    def test_complex_data_json_serializable(self):
        """Complex nested structures should be JSON serializable."""
        result = IntegrationResult(
            success=True,
            vertical_name="devops",
            step_status={
                "middleware": {"status": "success", "duration_ms": 100},
                "safety": {"status": "success", "patterns_applied": 3},
            },
        )

        data = result.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        # Step status should be preserved
        assert "middleware" in parsed["step_status"]
        assert parsed["step_status"]["middleware"]["status"] == "success"
