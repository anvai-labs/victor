"""Utility functions for dependency analysis."""

from typing import List, Dict

from .protocol import PackageDependency


def count_by_type(dependencies: List[PackageDependency]) -> Dict[str, int]:
    """Return a dict mapping DependencyType names to counts."""
    counts: Dict[str, int] = {}
    for dep in dependencies:
        key = dep.dependency_type.value
        counts[key] = counts.get(key, 0) + 1
    return counts
