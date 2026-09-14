"""Identifier handling for generated SQLite statements; values still use parameters."""

import re


def quote_identifier(name: str) -> str:
    """Quote one identifier, including embedded quotes, never a SQL expression."""
    if not isinstance(name, str) or not name or "\0" in name:
        raise ValueError("SQL identifier must be a nonempty string without NUL")
    return '"' + name.replace('"', '""') + '"'


def validate_identifier(name: str) -> str:
    """Validate an identifier used as a prefix in legacy generated table names."""
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
        raise ValueError("SQL identifier must contain only letters, digits and underscores")
    return name
