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

"""Sandhi usage-gateway bridge (FEP-0020 Phase 2).

Victor computes local USD cost per session but historically had **no per-user /
per-team attribution and no shared metering mechanism**. Rather than reimplement
metering, Victor adopts the standalone open-core gateway ``sandhi`` (Apache-2.0,
``anvai-labs/sandhi``; see AnvaiOps ADR-0047 / Victor ADR-018): the OSS core owns
the neutral usage-event *mechanism*, Victor supplies the *attribution* (which user,
which team) drawn from its identity/auth seam.

This module is the thin in-process bridge — the Phase 2 "metering middleware"
adoption. Each completed provider call is metered against a per-subject virtual
key, producing exactly one neutral usage event (with the full prompt-cache split)
that lands in ``sandhi``'s local sink.

**Principal & billing-group rule** (Codex wave-2 finding 23; D5 in
``docs/architecture/fou-a4-rbac-sso-design.md``): the STABLE metering identity is
the principal (the subject label — the legacy key label for key-only principals,
the issuer-qualified principal ID once A4c lands). The billing GROUP is recorded
PER EVENT, realized as the **versioned metering key**: the gateway's meter path is
key-authoritative (ADR-0004 D4 — ``meter_tokens`` resolves the event's
``subject_id``/``group_id`` from the virtual key and admits no per-event group
parameter), so the group travels in the key id. A principal whose group changes
meters through a NEW virtual key carrying the new group — subsequent events
reflect the change without disturbing history, and events already emitted keep
the group they were recorded with. Versioned keys are also race-free under
overlapping HTTP/SSE/WS calls: each event's key encodes its own event-time group
rather than a shared mutable binding. Key-only principals (no verified group)
keep the unversioned ``victor:{subject}`` key with no group — the pre-finding-23
shape, pinned behavior-neutral.

The dependency is **optional and default-off**: ``sandhi-gateway`` lives behind the
``victor[sandhi]`` extra. When it is not installed, :func:`sandhi_available` returns
``False`` and callers skip the bridge — the base install is byte-for-byte unaffected
(FEP-0020 "Migration Path": ships default-off, zero behavior change).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:  # optional dependency — victor[sandhi]
    import sandhi_gateway as _sg

    _AVAILABLE = True
except Exception:  # pragma: no cover - exercised only where the extra is absent
    _sg = None  # type: ignore[assignment]
    _AVAILABLE = False


def sandhi_available() -> bool:
    """Whether the ``sandhi-gateway`` optional dependency is importable."""
    return _AVAILABLE


class SandhiMeter:
    """Bridge Victor provider usage → the ``sandhi`` usage gateway.

    A Victor *principal* (a ``client_id`` / user) meters through one virtual key
    per (subject, billing group) pair — the versioned metering key of the
    finding-23 billing-group rule (see the module docstring). The gateway emits
    one neutral usage event per call to its local sink, carrying the
    ``subject_id`` / ``group_id`` attribution that Victor's own cost records lacked.

    Emission is **best-effort and off the critical path**: any failure is logged at
    debug and swallowed, never raised into the caller's request path (FEP-0020 /
    ADR-0047 D7 — metering must not fail the LLM call).
    """

    def __init__(self, *, sink_path: Optional[str] = None) -> None:
        if not _AVAILABLE:
            raise RuntimeError(
                "sandhi-gateway is not installed; run `pip install victor[sandhi]` to "
                "enable usage-gateway attribution (FEP-0020 Phase 2)."
            )
        # Gateway(sink_path) persists events to JSONL at sink_path; Gateway() keeps
        # them in memory (queryable via .events()).
        self._gw = _sg.Gateway(sink_path) if sink_path else _sg.Gateway()
        self._known_keys: set[str] = set()

    @property
    def wire_contract_version(self) -> str:
        """The neutral usage-event schema version the linked ``sandhi`` core speaks."""
        return str(_sg.wire_contract_version())

    @staticmethod
    def _metering_key(subject: str, group_id: Optional[str]) -> str:
        """Versioned metering key: ``victor:{subject}`` (no verified group) or
        ``victor:{subject}:{group}``.

        The id is opaque — it is only ever computed forward and never parsed
        back, so the plain concatenation is safe; both components come from
        verified registry vocabularies (``principal_id`` / verified groups).
        """
        return f"victor:{subject}" if group_id is None else f"victor:{subject}:{group_id}"

    def _ensure_key(self, subject_id: Optional[str], group_id: Optional[str]) -> str:
        subject = subject_id or "anonymous"
        virtual_key = self._metering_key(subject, group_id)
        if virtual_key not in self._known_keys:
            # upstream="" — Victor holds the real provider key; the bridge only meters.
            self._gw.add_virtual_key(virtual_key, subject, group_id, "")
            self._known_keys.add(virtual_key)
        return virtual_key

    def record(
        self,
        *,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        subject_id: Optional[str] = None,
        group_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Emit one neutral usage event for a completed provider call.

        Maps Victor's cache-write/read counts onto the gateway's
        ``cache_creation``/``cache_read`` fields (the ADR-0047 D4 prompt-cache
        split). Never raises.
        """
        try:
            virtual_key = self._ensure_key(subject_id, group_id)
            self._gw.meter_tokens(
                virtual_key,
                provider,
                model,
                int(prompt_tokens),
                int(completion_tokens),
                int(cache_write_tokens),  # cache_creation_tokens
                int(cache_read_tokens),  # cache_read_tokens
                session_id,
            )
        except Exception as exc:  # pragma: no cover - defensive, off critical path
            logger.debug("sandhi usage-gateway emit failed (ignored): %s", exc)

    def events(self) -> List[Dict[str, Any]]:
        """Neutral usage events accumulated in the in-memory sink.

        Backs the FEP-0020 "per-subject display". Returns an empty list when a
        persistent (JSONL) sink was configured, since those events are not held
        in memory.
        """
        try:
            return list(self._gw.events())
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("sandhi events() read failed (ignored): %s", exc)
            return []
