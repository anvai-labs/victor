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

"""Edge type constants for unified code graph.

This module defines all edge types used in Victor's graph-based code intelligence
system. Edge types are organized by category:

- **Existing edges**: CALLS, REFERENCES, CONTAINS, INHERITS (for backward compatibility)
- **CFG edges**: Control Flow Graph edges from GraphCoder methodology
- **CDG edges**: Control Dependence Graph edges
- **DDG edges**: Data Dependence Graph edges
- **Requirement edges**: Mapping requirements to code (GraphCodeAgent pattern)
- **Semantic edges**: Similarity and semantic relationships

Usage:
    from victor.storage.graph.edge_types import EdgeType

    # Create a CFG edge
    edge = GraphEdge(src="node1", dst="node2", type=EdgeType.CFG_SUCCESSOR)

    # Check edge type category
    if EdgeType.is_cfg_edge(edge.type):
        # Handle CFG edge
"""

from __future__ import annotations

from enum import Enum
from typing import Set


class EdgeType(str, Enum):
    """Unified edge type constants for code graph.

    Categories:
    - LEGACY: Existing edges (v4 and earlier)
    - CFG: Control Flow Graph edges (GraphCoder)
    - CDG: Control Dependence Graph edges (GraphCoder)
    - DDG: Data Dependence Graph edges (GraphCoder)
    - REQUIREMENT: Requirement-to-code mapping (GraphCodeAgent)
    - SEMANTIC: Semantic similarity and relationships
    """

    # ===========================================
    # LEGACY EDGES (v4 compatibility)
    # ===========================================
    CALLS = "CALLS"
    REFERENCES = "REFERENCES"
    CONTAINS = "CONTAINS"
    INHERITS = "INHERITS"
    IMPLEMENTS = "IMPLEMENTS"
    IMPORTS = "IMPORTS"
    INSTANTIATES = "INSTANTIATES"

    # ===========================================
    # CONTROL FLOW GRAPH (CFG) - GraphCoder
    # ===========================================
    CFG_SUCCESSOR = "CFG_SUCCESSOR"  # Sequential control flow
    CFG_TRUE_BRANCH = "CFG_TRUE"  # True branch of conditional
    CFG_FALSE_BRANCH = "CFG_FALSE"  # False branch of conditional
    CFG_CASE = "CFG_CASE"  # Switch/match case
    CFG_DEFAULT = "CFG_DEFAULT"  # Default case in switch
    CFG_LOOP_ENTRY = "CFG_LOOP_ENTRY"  # Entry to loop body
    CFG_LOOP_EXIT = "CFG_LOOP_EXIT"  # Exit from loop
    CFG_LOOP_BACK = "CFG_LOOP_BACK"  # Back-edge: last body statement -> loop header
    CFG_EXCEPTION = "CFG_EXCEPTION"  # Exception flow
    CFG_CATCH = "CFG_CATCH"  # Catch block entry
    CFG_FINALLY = "CFG_FINALLY"  # Finally block entry
    CFG_RETURN = "CFG_RETURN"  # Return statement flow
    CFG_BREAK = "CFG_BREAK"  # Break statement flow
    CFG_CONTINUE = "CFG_CONTINUE"  # Continue statement flow

    # ===========================================
    # CONTROL DEPENDENCE GRAPH (CDG) - GraphCoder
    # ===========================================
    CDG = "CDG"  # General control dependence
    CDG_TRUE = "CDG_TRUE"  # True branch control dependence
    CDG_FALSE = "CDG_FALSE"  # False branch control dependence
    CDG_LOOP = "CDG_LOOP"  # Loop body control dependence
    CDG_CASE = "CDG_CASE"  # Case control dependence

    # ===========================================
    # DATA DEPENDENCE GRAPH (DDG) - GraphCoder
    # ===========================================
    DDG_DEF_USE = "DDG_DEF_USE"  # Definition-use chain
    DDG_RAW = "DDG_RAW"  # Read-after-write (true dependence)
    DDG_WAR = "DDG_WAR"  # Write-after-read (anti-dependence)
    DDG_WAW = "DDG_WAW"  # Write-after-write (output dependence)

    # ===========================================
    # REQUIREMENT MAPPING - GraphCodeAgent
    # ===========================================
    SATISFIES = "SATISFIES"  # Code satisfies requirement
    TESTS = "TESTS"  # Test code validates requirement
    DERIVES_FROM = "DERIVES_FROM"  # Requirement derived from another
    REFINES = "REFINES"  # Code refines requirement
    CONTRADICTS = "CONTRADICTS"  # Code contradicts requirement
    COVERS = "COVERS"  # Test covers code

    # ===========================================
    # SEMANTIC RELATIONSHIPS
    # ===========================================
    SEMANTIC_SIMILAR = "SEMANTIC_SIM"  # Semantic similarity (thresholded)
    STRUCTURAL_SIMILAR = "STRUCTURAL_SIM"  # Structural similarity
    FUNCTIONAL_SIMILAR = "FUNCTIONAL_SIM"  # Functional similarity
    IS_A = "IS_A"  # Is-a relationship (type hierarchy)
    HAS_A = "HAS_A"  # Has-a relationship (composition)

    # ===========================================
    # VERSIONING AND TEMPORAL
    # ===========================================
    VERSION_OF = "VERSION_OF"  # Version relationship
    DERIVED_FROM = "DERIVED_FROM"  # Derived from another node
    REPLACES = "REPLACES"  # Replaces deprecated code

    # ===========================================
    # ANNOTATION AND METADATA
    # ===========================================
    ANNOTATES = "ANNOTATES"  # Comment/type annotation
    DOCUMENTS = "DOCUMENTS"  # Docstring documents code
    ASSERTS = "ASSERTS"  # Assert statement documents assumption

    @classmethod
    def is_cfg_edge(cls, edge_type: str) -> bool:
        """Check if edge type is a Control Flow Graph edge.

        Args:
            edge_type: Edge type string to check

        Returns:
            True if edge type is a CFG edge
        """
        return edge_type.startswith("CFG_")

    @classmethod
    def is_cdg_edge(cls, edge_type: str) -> bool:
        """Check if edge type is a Control Dependence Graph edge.

        Args:
            edge_type: Edge type string to check

        Returns:
            True if edge type is a CDG edge
        """
        return edge_type.startswith("CDG")

    @classmethod
    def is_ddg_edge(cls, edge_type: str) -> bool:
        """Check if edge type is a Data Dependence Graph edge.

        Args:
            edge_type: Edge type string to check

        Returns:
            True if edge type is a DDG edge
        """
        return edge_type.startswith("DDG_")

    @classmethod
    def is_ccg_edge(cls, edge_type: str) -> bool:
        """Check if edge type is any Code Context Graph edge (CFG, CDG, or DDG).

        Args:
            edge_type: Edge type string to check

        Returns:
            True if edge type is a CCG edge
        """
        return (
            cls.is_cfg_edge(edge_type) or cls.is_cdg_edge(edge_type) or cls.is_ddg_edge(edge_type)
        )

    @classmethod
    def is_requirement_edge(cls, edge_type: str) -> bool:
        """Check if edge type is a requirement mapping edge.

        Args:
            edge_type: Edge type string to check

        Returns:
            True if edge type is a requirement edge
        """
        return edge_type in {
            cls.SATISFIES,
            cls.TESTS,
            cls.DERIVES_FROM,
            cls.REFINES,
            cls.CONTRADICTS,
            cls.COVERS,
        }

    @classmethod
    def is_semantic_edge(cls, edge_type: str) -> bool:
        """Check if edge type is a semantic relationship edge.

        Args:
            edge_type: Edge type string to check

        Returns:
            True if edge type is a semantic edge
        """
        return edge_type.endswith("_SIM") or edge_type in {
            cls.IS_A,
            cls.HAS_A,
        }

    @classmethod
    def get_cfg_edge_types(cls) -> Set[str]:
        """Get all CFG edge type strings.

        Returns:
            Set of CFG edge type strings
        """
        return {e.value for e in cls if e.is_cfg_edge(e.value)}

    @classmethod
    def get_cdg_edge_types(cls) -> Set[str]:
        """Get all CDG edge type strings.

        Returns:
            Set of CDG edge type strings
        """
        return {e.value for e in cls if e.is_cdg_edge(e.value)}

    @classmethod
    def get_ddg_edge_types(cls) -> Set[str]:
        """Get all DDG edge type strings.

        Returns:
            Set of DDG edge type strings
        """
        return {e.value for e in cls if e.is_ddg_edge(e.value)}

    @classmethod
    def get_ccg_edge_types(cls) -> Set[str]:
        """Get all Code Context Graph edge type strings (CFG + CDG + DDG).

        Returns:
            Set of CCG edge type strings
        """
        return cls.get_cfg_edge_types() | cls.get_cdg_edge_types() | cls.get_ddg_edge_types()

    @classmethod
    def get_legacy_edge_types(cls) -> Set[str]:
        """Get all legacy edge type strings (v4 and earlier).

        Returns:
            Set of legacy edge type strings
        """
        return {
            cls.CALLS,
            cls.REFERENCES,
            cls.CONTAINS,
            cls.INHERITS,
            cls.IMPLEMENTS,
            cls.IMPORTS,
            cls.INSTANTIATES,
        }

    def get_env_var_name(self) -> str:
        """Get environment variable name for this edge type.

        Returns:
            Environment variable name (e.g., "VICTOR_CFG_SUCCESSOR" for CFG_SUCCESSOR)
        """
        return f"VICTOR_{self.value}"

    def get_yaml_key(self) -> str:
        """Get YAML configuration key for this edge type.

        Returns:
            YAML key (lowercase, e.g., "cfg_successor" for CFG_SUCCESSOR)
        """
        return self.value.lower()


# Precompute sets for fast lookup
CFG_EDGE_TYPES = EdgeType.get_cfg_edge_types()
CDG_EDGE_TYPES = EdgeType.get_cdg_edge_types()
DDG_EDGE_TYPES = EdgeType.get_ddg_edge_types()
CCG_EDGE_TYPES = EdgeType.get_ccg_edge_types()
LEGACY_EDGE_TYPES = EdgeType.get_legacy_edge_types()


__all__ = [
    "EdgeType",
    "CFG_EDGE_TYPES",
    "CDG_EDGE_TYPES",
    "DDG_EDGE_TYPES",
    "CCG_EDGE_TYPES",
    "LEGACY_EDGE_TYPES",
]
