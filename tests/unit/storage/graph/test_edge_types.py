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

"""Unit tests for graph edge types."""

from __future__ import annotations

import pytest

from victor.storage.graph.edge_types import (
    EdgeType,
    CFG_EDGE_TYPES,
    CDG_EDGE_TYPES,
    DDG_EDGE_TYPES,
    CCG_EDGE_TYPES,
    LEGACY_EDGE_TYPES,
)


class TestEdgeType:
    """Tests for EdgeType enum and helper methods."""

    def test_legacy_edge_types_exist(self) -> None:
        """Test that legacy edge types are defined."""
        assert EdgeType.CALLS == "CALLS"
        assert EdgeType.REFERENCES == "REFERENCES"
        assert EdgeType.CONTAINS == "CONTAINS"
        assert EdgeType.INHERITS == "INHERITS"

    def test_cfg_edge_types_exist(self) -> None:
        """Test that CFG edge types are defined."""
        assert EdgeType.CFG_SUCCESSOR == "CFG_SUCCESSOR"
        assert EdgeType.CFG_TRUE_BRANCH == "CFG_TRUE"
        assert EdgeType.CFG_FALSE_BRANCH == "CFG_FALSE"
        assert EdgeType.CFG_CASE == "CFG_CASE"
        assert EdgeType.CFG_DEFAULT == "CFG_DEFAULT"
        assert EdgeType.CFG_LOOP_ENTRY == "CFG_LOOP_ENTRY"
        assert EdgeType.CFG_LOOP_EXIT == "CFG_LOOP_EXIT"
        assert EdgeType.CFG_EXCEPTION == "CFG_EXCEPTION"
        assert EdgeType.CFG_CATCH == "CFG_CATCH"
        assert EdgeType.CFG_FINALLY == "CFG_FINALLY"
        assert EdgeType.CFG_RETURN == "CFG_RETURN"
        assert EdgeType.CFG_BREAK == "CFG_BREAK"
        assert EdgeType.CFG_CONTINUE == "CFG_CONTINUE"

    def test_cdg_edge_types_exist(self) -> None:
        """Test that CDG edge types are defined."""
        assert EdgeType.CDG == "CDG"
        assert EdgeType.CDG_TRUE == "CDG_TRUE"
        assert EdgeType.CDG_FALSE == "CDG_FALSE"
        assert EdgeType.CDG_LOOP == "CDG_LOOP"
        assert EdgeType.CDG_CASE == "CDG_CASE"

    def test_ddg_edge_types_exist(self) -> None:
        """Test that DDG edge types are defined."""
        assert EdgeType.DDG_DEF_USE == "DDG_DEF_USE"
        assert EdgeType.DDG_RAW == "DDG_RAW"
        assert EdgeType.DDG_WAR == "DDG_WAR"
        assert EdgeType.DDG_WAW == "DDG_WAW"

    def test_requirement_edge_types_exist(self) -> None:
        """Test that requirement edge types are defined."""
        assert EdgeType.SATISFIES == "SATISFIES"
        assert EdgeType.TESTS == "TESTS"
        assert EdgeType.DERIVES_FROM == "DERIVES_FROM"
        assert EdgeType.REFINES == "REFINES"
        assert EdgeType.CONTRADICTS == "CONTRADICTS"
        assert EdgeType.COVERS == "COVERS"

    def test_semantic_edge_types_exist(self) -> None:
        """Test that semantic edge types are defined."""
        assert EdgeType.SEMANTIC_SIMILAR == "SEMANTIC_SIM"
        assert EdgeType.STRUCTURAL_SIMILAR == "STRUCTURAL_SIM"
        assert EdgeType.FUNCTIONAL_SIMILAR == "FUNCTIONAL_SIM"
        assert EdgeType.IS_A == "IS_A"
        assert EdgeType.HAS_A == "HAS_A"

    def test_is_cfg_edge(self) -> None:
        """Test CFG edge type detection."""
        assert EdgeType.is_cfg_edge("CFG_SUCCESSOR") is True
        assert EdgeType.is_cfg_edge("CFG_TRUE") is True
        assert EdgeType.is_cfg_edge("CFG_FALSE") is True
        assert EdgeType.is_cfg_edge("CFG_CASE") is True
        assert EdgeType.is_cfg_edge("CALLS") is False
        assert EdgeType.is_cfg_edge("DDG_DEF_USE") is False

    def test_is_cdg_edge(self) -> None:
        """Test CDG edge type detection."""
        assert EdgeType.is_cdg_edge("CDG") is True
        assert EdgeType.is_cdg_edge("CDG_TRUE") is True
        assert EdgeType.is_cdg_edge("CDG_LOOP") is True
        assert EdgeType.is_cdg_edge("CFG_SUCCESSOR") is False
        assert EdgeType.is_cdg_edge("CALLS") is False

    def test_is_ddg_edge(self) -> None:
        """Test DDG edge type detection."""
        assert EdgeType.is_ddg_edge("DDG_DEF_USE") is True
        assert EdgeType.is_ddg_edge("DDG_RAW") is True
        assert EdgeType.is_ddg_edge("DDG_WAR") is True
        assert EdgeType.is_ddg_edge("DDG_WAW") is True
        assert EdgeType.is_ddg_edge("CFG_SUCCESSOR") is False
        assert EdgeType.is_ddg_edge("CALLS") is False

    def test_is_ccg_edge(self) -> None:
        """Test CCG edge type detection (CFG, CDG, or DDG)."""
        assert EdgeType.is_ccg_edge("CFG_SUCCESSOR") is True
        assert EdgeType.is_ccg_edge("CDG") is True
        assert EdgeType.is_ccg_edge("DDG_DEF_USE") is True
        assert EdgeType.is_ccg_edge("CALLS") is False
        assert EdgeType.is_ccg_edge("REFERENCES") is False

    def test_is_requirement_edge(self) -> None:
        """Test requirement edge type detection."""
        assert EdgeType.is_requirement_edge("SATISFIES") is True
        assert EdgeType.is_requirement_edge("TESTS") is True
        assert EdgeType.is_requirement_edge("DERIVES_FROM") is True
        assert EdgeType.is_requirement_edge("REFINES") is True
        assert EdgeType.is_requirement_edge("CALLS") is False
        assert EdgeType.is_requirement_edge("CONTAINS") is False

    def test_is_semantic_edge(self) -> None:
        """Test semantic edge type detection."""
        assert EdgeType.is_semantic_edge("SEMANTIC_SIM") is True
        assert EdgeType.is_semantic_edge("STRUCTURAL_SIM") is True
        assert EdgeType.is_semantic_edge("FUNCTIONAL_SIM") is True
        assert EdgeType.is_semantic_edge("IS_A") is True
        assert EdgeType.is_semantic_edge("HAS_A") is True
        assert EdgeType.is_semantic_edge("CALLS") is False
        assert EdgeType.is_semantic_edge("CFG_SUCCESSOR") is False

    def test_get_cfg_edge_types(self) -> None:
        """Test getting all CFG edge types."""
        cfg_types = EdgeType.get_cfg_edge_types()
        assert isinstance(cfg_types, set)
        assert "CFG_SUCCESSOR" in cfg_types
        assert "CFG_TRUE" in cfg_types
        assert "CFG_FALSE" in cfg_types
        assert len(cfg_types) >= 12  # At least 12 CFG edge types

    def test_get_cdg_edge_types(self) -> None:
        """Test getting all CDG edge types."""
        cdg_types = EdgeType.get_cdg_edge_types()
        assert isinstance(cdg_types, set)
        assert "CDG" in cdg_types
        assert "CDG_TRUE" in cdg_types
        assert "CDG_LOOP" in cdg_types

    def test_get_ddg_edge_types(self) -> None:
        """Test getting all DDG edge types."""
        ddg_types = EdgeType.get_ddg_edge_types()
        assert isinstance(ddg_types, set)
        assert "DDG_DEF_USE" in ddg_types
        assert "DDG_RAW" in ddg_types
        assert "DDG_WAR" in ddg_types
        assert "DDG_WAW" in ddg_types

    def test_get_ccg_edge_types(self) -> None:
        """Test getting all CCG edge types (CFG + CDG + DDG)."""
        ccg_types = EdgeType.get_ccg_edge_types()
        assert isinstance(ccg_types, set)
        assert "CFG_SUCCESSOR" in ccg_types
        assert "CDG" in ccg_types
        assert "DDG_DEF_USE" in ccg_types

    def test_get_legacy_edge_types(self) -> None:
        """Test getting legacy edge types."""
        legacy_types = EdgeType.get_legacy_edge_types()
        assert isinstance(legacy_types, set)
        assert "CALLS" in legacy_types
        assert "REFERENCES" in legacy_types
        assert "CONTAINS" in legacy_types
        assert "INHERITS" in legacy_types

    def test_cfg_edge_types_constant(self) -> None:
        """Test CFG_EDGE_TYPES constant matches enum."""
        assert CFG_EDGE_TYPES == EdgeType.get_cfg_edge_types()

    def test_cdg_edge_types_constant(self) -> None:
        """Test CDG_EDGE_TYPES constant matches enum."""
        assert CDG_EDGE_TYPES == EdgeType.get_cdg_edge_types()

    def test_ddg_edge_types_constant(self) -> None:
        """Test DDG_EDGE_TYPES constant matches enum."""
        assert DDG_EDGE_TYPES == EdgeType.get_ddg_edge_types()

    def test_ccg_edge_types_constant(self) -> None:
        """Test CCG_EDGE_TYPES constant matches enum."""
        assert CCG_EDGE_TYPES == EdgeType.get_ccg_edge_types()

    def test_legacy_edge_types_constant(self) -> None:
        """Test LEGACY_EDGE_TYPES constant matches enum."""
        assert LEGACY_EDGE_TYPES == EdgeType.get_legacy_edge_types()

    def test_no_duplicate_edge_type_names(self) -> None:
        """Test that edge type names are unique."""
        all_values = [e.value for e in EdgeType]
        assert len(all_values) == len(set(all_values)), "Edge type names must be unique"

    def test_env_var_name_format(self) -> None:
        """Test that env var names follow the expected format."""
        for edge_type in EdgeType:
            env_var = edge_type.get_env_var_name()
            assert env_var.startswith("VICTOR_")
            assert "_" in env_var

    def test_yaml_key_format(self) -> None:
        """Test that YAML keys follow the expected format."""
        for edge_type in EdgeType:
            yaml_key = edge_type.get_yaml_key()
            assert yaml_key == yaml_key.lower()  # Should be lowercase
            assert " " not in yaml_key  # No spaces
