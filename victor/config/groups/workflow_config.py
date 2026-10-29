"""Workflow execution and orchestration configuration.

This module contains settings for:
- Workflow definition caching
- StateGraph copy-on-write behavior
- Workflow orchestration parameters
"""

from pydantic import BaseModel, field_validator


class WorkflowSettings(BaseModel):
    """Workflow execution and StateGraph settings.

    Controls how workflows are defined, cached, and executed.
    Includes StateGraph copy-on-write optimization for read-heavy workflows.
    """

    # ==========================================================================
    # Workflow Definition Cache (P1 Scalability)
    # ==========================================================================
    # Caches parsed YAML workflow definitions to avoid redundant parsing.
    # Uses TTL + file mtime invalidation for freshness.
    workflow_definition_cache_enabled: bool = True
    workflow_definition_cache_ttl: int = 3600  # seconds (1 hour)
    workflow_definition_cache_max_entries: int = 100

    # ==========================================================================
    # StateGraph Copy-on-Write (P2 Scalability)
    # ==========================================================================
    # Enables copy-on-write state management for StateGraph workflows.
    # Delays deep copy of state until the first mutation, reducing overhead
    # for read-heavy workflows where nodes often only read state.
    stategraph_copy_on_write_enabled: bool = True

    @field_validator("workflow_definition_cache_ttl")
    @classmethod
    def validate_cache_ttl(cls, v: int) -> int:
        """Validate workflow cache TTL is non-negative.

        Args:
            v: Cache TTL in seconds

        Returns:
            Validated TTL

        Raises:
            ValueError: If TTL is negative
        """
        if v < 0:
            raise ValueError("workflow_definition_cache_ttl must be >= 0")
        return v

    @field_validator("workflow_definition_cache_max_entries")
    @classmethod
    def validate_max_entries(cls, v: int) -> int:
        """Validate max entries is positive.

        Args:
            v: Maximum number of cached entries

        Returns:
            Validated max entries

        Raises:
            ValueError: If max entries is not positive
        """
        if v < 1:
            raise ValueError("workflow_definition_cache_max_entries must be >= 1")
        return v
