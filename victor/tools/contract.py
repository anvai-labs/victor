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

"""Single resolution authority for a tool's canonical metadata/contract.

Tool identity + traits were resolved ad hoc in several places: ``BaseTool.get_metadata``
(explicit-or-autogen), the ``@tool`` decorator (which stashes decorator-supplied traits
onto the tool's ``metadata``), and assorted call sites. ``resolve_contract`` makes that
one function (SRP/DIP) so every consumer derives a tool's metadata the same way and the
auto-generated result can be cached.

Framework-internal by design (tool-supply P6, minimal cut): this returns the EXISTING
:class:`~victor.tools.metadata.ToolMetadata` — it does NOT introduce a parallel contract
type. Precedence below is exactly the prior ``get_metadata`` behavior, so results are
byte-stable.
"""

from __future__ import annotations

import weakref
from typing import Any

from victor.tools.metadata import ToolMetadata

# Cache the auto-generated result per tool instance (weak — entries vanish on GC).
# Only the *derived* result is cached: explicit metadata is returned directly (cheap and
# avoids pinning a possibly-dynamic ``metadata`` property).
_autogen_cache: "weakref.WeakKeyDictionary[Any, ToolMetadata]" = weakref.WeakKeyDictionary()


def resolve_contract(tool: Any) -> ToolMetadata:
    """Resolve a tool's canonical :class:`ToolMetadata` (the one fusion authority).

    Precedence (extends the historical two-tier ``get_metadata`` strategy with the
    SDK contract tier from FEP-0009):

    1. **Explicit metadata** — ``tool.metadata`` when set. This is also where the
       ``@tool`` decorator records decorator-supplied traits, so decorator params are
       honored through this tier.
    2. **SDK contract** — ``tool.contract`` (a ``victor_contracts.tools.ToolContract``)
       when set, bridged via :meth:`ToolMetadata.from_contract`. Duck-typed: no
       ``victor_contracts`` import here, so this works regardless of the installed
       contracts version.
    3. **Auto-generated** — derived from the tool's own ``name`` / ``description`` /
       ``parameters`` / ``cost_tier`` via :meth:`ToolMetadata.generate_from_tool`.

    Args:
        tool: A tool exposing ``name`` / ``description`` / ``parameters`` (and optionally
            ``metadata`` / ``contract`` / ``cost_tier``).

    Returns:
        The tool's canonical :class:`ToolMetadata`.
    """
    explicit = getattr(tool, "metadata", None)
    if explicit is not None:
        return explicit

    try:
        cached = _autogen_cache.get(tool)
        if cached is not None:
            return cached
    except TypeError:  # tool not weak-referenceable / unhashable
        cached = None

    contract = getattr(tool, "contract", None)
    if contract is not None:
        result = ToolMetadata.from_contract(contract, tool)
    else:
        result = ToolMetadata.generate_from_tool(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            cost_tier=getattr(tool, "cost_tier", None),
        )

    try:
        _autogen_cache[tool] = result
    except TypeError:
        pass
    return result
