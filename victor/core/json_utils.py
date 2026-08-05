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

"""Fast JSON serialization facade.

Uses ``orjson`` (Rust-backed, ~5-10x faster than stdlib json).

Usage::

    from victor.core.json_utils import json_dumps, json_loads, json_dumps_bytes

    text = json_dumps({"key": "value"})            # -> str
    raw  = json_dumps_bytes({"key": "value"})       # -> bytes
    obj  = json_loads(text)                        # -> Any
    obj  = json_loads(raw)                         # -> Any  (bytes accepted)

All public functions are API-compatible with stdlib ``json`` for the
common case; callers should not need to change signatures.
"""

from __future__ import annotations

import orjson
from typing import Any, Optional, Union

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def json_dumps(
    obj: Any,
    *,
    sort_keys: bool = False,
    indent: bool = False,
    ensure_ascii: bool = False,
    default: Optional[Any] = None,
    separators: Any = None,
    **_extra: Any,
) -> str:
    """Serialize *obj* to a JSON ``str``.

    Drop-in replacement for ``json.dumps()`` using orjson.

    ``separators`` and other stdlib-only kwargs are accepted but ignored —
    orjson always produces compact, non-ASCII-safe output (equivalent to
    ``separators=(",", ":")`` + ``ensure_ascii=False`` in stdlib json).
    """
    opts = 0
    if sort_keys:
        opts |= orjson.OPT_SORT_KEYS
    if indent:
        opts |= orjson.OPT_INDENT_2
    if not ensure_ascii:
        opts |= orjson.OPT_NON_STR_KEYS
    return orjson.dumps(obj, option=opts or None, default=default).decode("utf-8")


def json_dumps_bytes(
    obj: Any,
    *,
    sort_keys: bool = False,
    default: Optional[Any] = None,
) -> bytes:
    """Serialize *obj* to JSON ``bytes`` (zero-copy with orjson)."""
    opts = 0
    if sort_keys:
        opts |= orjson.OPT_SORT_KEYS
    return orjson.dumps(obj, option=opts or None, default=default)


def json_loads(data: Union[str, bytes, bytearray, memoryview]) -> Any:
    """Deserialize JSON ``str`` or ``bytes`` to a Python object."""
    return orjson.loads(data)
