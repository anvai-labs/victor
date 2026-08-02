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

"""Fast YAML loading facade.

Uses the C-accelerated ``CSafeLoader`` / ``CSafeDumper`` when the
libyaml C extension is compiled into PyYAML, with automatic fallback
to the pure-Python ``SafeLoader`` / ``SafeDumper``.

The C loader is typically **5-10x faster** for large YAML documents.

Usage::

    from victor.core.yaml_utils import safe_load, safe_load_all, safe_dump

    data = safe_load(yaml_string)
    docs = list(safe_load_all(multi_doc_string))
    output = safe_dump(data)
"""

from __future__ import annotations

from typing import Any, IO, Iterator, Optional, Union

import yaml

# Prefer C-accelerated loader/dumper when available
try:
    _SafeLoader = yaml.CSafeLoader  # type: ignore[attr-defined]
except AttributeError:
    _SafeLoader = yaml.SafeLoader

try:
    _SafeDumper = yaml.CSafeDumper  # type: ignore[attr-defined]
except AttributeError:
    _SafeDumper = yaml.SafeDumper


def safe_load(stream: Union[str, bytes, IO[str], IO[bytes]]) -> Any:
    """Load a single YAML document (safe, no arbitrary Python objects)."""
    return yaml.load(stream, Loader=_SafeLoader)


def safe_load_all(stream: Union[str, bytes, IO[str], IO[bytes]]) -> Iterator[Any]:
    """Load all YAML documents from a multi-document stream."""
    return yaml.load_all(stream, Loader=_SafeLoader)


def safe_dump(
    data: Any,
    stream: Optional[IO[str]] = None,
    *,
    default_flow_style: bool = False,
    sort_keys: bool = True,
) -> Optional[str]:
    """Dump data to YAML string or stream."""
    return yaml.dump(
        data,
        stream=stream,
        Dumper=_SafeDumper,
        default_flow_style=default_flow_style,
        sort_keys=sort_keys,
    )
