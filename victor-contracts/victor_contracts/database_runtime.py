"""Runtime bridge: project database access."""

# ruff: noqa: F822

from __future__ import annotations


from victor_contracts._lazy import make_lazy_getattr

__all__ = ["get_project_database", "resolve_project_db_root"]

_LAZY_IMPORTS = {
    "get_project_database": "victor.core.database",
    "resolve_project_db_root": "victor.core.database",
}


__getattr__ = make_lazy_getattr(__name__, _LAZY_IMPORTS)
