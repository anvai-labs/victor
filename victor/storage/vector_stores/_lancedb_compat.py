"""Backward/forward compatibility layer for LanceDB API changes.

LanceDB's table-listing API has shifted across versions:

- ``list_tables()`` is the canonical method on current versions (returns a
  plain list of names). It is tried first so call-sites never trigger the
  ``table_names() is deprecated`` warning emitted by recent releases.
- ``table_names()`` is the legacy listing method (also returns a plain list
  on newer versions but is deprecated there); kept as a fallback for older
  installs that predate ``list_tables()``.
- On the oldest versions ``list_tables()`` returned an object with a
  ``.tables`` attribute rather than a plain list; that shape is normalized.

This helper transparently supports all three so call-sites don't break when
the installed version changes.
"""

from __future__ import annotations

import warnings
import math
from collections.abc import Collection, Mapping
from typing import Any, List

from victor.core.sql_utils import validate_identifier


def _string_literal(value: str) -> str:
    if type(value) is not str or "\0" in value:
        raise ValueError("LanceDB string predicates require strings without NUL")
    return "'" + value.replace("'", "''") + "'"


def lance_string_predicate(column: str, value: str) -> str:
    """Exact string equality for internal columns; never accept SQL expressions.

    Use SQL strings rather than newer SDK expression objects. Do not use
    SQLite's parameter syntax here. Engine qualification is recorded in the
    DATA-01 tracker; this helper does not establish a version support matrix.
    """
    return f"`{validate_identifier(column)}` = {_string_literal(value)}"


def lance_filter_predicate(
    filters: Mapping[str, object], *, allowed_columns: Collection[str]
) -> str:
    """Compile flat scalar equalities once, preserving every eligibility filter.

    Older LanceDB SDKs replace (rather than conjoin) successive ``where`` calls.
    Identifiers must be simple existing columns; values are JSON scalars with
    finite floats and signed 64-bit integers. None means SQL IS NULL. Compound
    values and embedded SQL are not a predicate language supported by this API.
    """
    if not isinstance(filters, Mapping):
        raise ValueError("LanceDB metadata filters must be a mapping")
    predicates: list[str] = []
    for key, value in filters.items():
        column = validate_identifier(key)
        if column not in allowed_columns:
            raise ValueError("LanceDB metadata filter column is not in the table schema")
        quoted = f"`{column}`"
        if value is None:
            predicates.append(f"{quoted} IS NULL")
            continue
        if type(value) is str:
            literal = _string_literal(value)
        elif type(value) is bool:
            literal = "TRUE" if value else "FALSE"
        elif type(value) is int and -(2**63) <= value < 2**63:
            literal = str(value)
        elif type(value) is float and math.isfinite(value):
            literal = repr(value)
        else:
            raise ValueError("LanceDB metadata filters require finite bounded scalar values")
        predicates.append(f"{quoted} = {literal}")
    return " AND ".join(f"({predicate})" for predicate in predicates)


def get_table_names(db: Any) -> List[str]:
    """Get table names with backward/forward LanceDB API compatibility."""
    if hasattr(db, "list_tables"):
        response = db.list_tables()
        if hasattr(response, "tables"):
            return list(response.tables)
        return list(response) if response else []
    if hasattr(db, "table_names"):
        # Legacy LanceDB: list_tables() did not exist yet. table_names() is
        # deprecated on current versions but is the only option here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = db.table_names()
        return list(result) if result else []
    return []
