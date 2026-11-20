# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Single-trusted-operator posture enforcement for shared-state surfaces.

The HTTP server resolves ONE server-wide conversation client and ONE shared
memory store: chat conversation addressing, export, reset and /memory carry
no owner or session in their lookup, so a second authenticated credential
can read or destroy the first user's state (Codex review findings 3/4).
Full ownership (per-principal clients, stores and authorization) is the
paused A4a+ design (VAS-06) — this module enforces the honest posture until
it lands:

- Default (SAFE): when MORE THAN ONE distinct credential is configured
  (``api_keys`` / ``VICTOR_API_KEYS``) the shared-state surfaces REFUSE with
  the typed ``MultiPrincipalRefused`` payload (HTTP 409) naming the
  limitation.
- Operator escape: an explicit ``single_operator_mode`` declaration (server
  kwarg or ``VICTOR_SINGLE_OPERATOR_MODE`` env alias) attests that every
  configured key belongs to one trusted operator and restores the shared
  multi-key posture.

The refusal is a posture statement, not an authorization decision: no
ownership is evaluated (that requires the A4a verified-principal seam).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

#: Structured-refusal type name surfaced in every refusal payload.
MULTI_PRINCIPAL_REFUSAL_TYPE = "MultiPrincipalRefused"

_REFUSAL_MESSAGE = (
    "{surface} is refused while more than one API credential is configured: "
    "this server resolves one shared state and cannot attribute it per "
    "credential until owner isolation lands (VAS-06 / FOU-A4a). Keep a "
    "single credential, or declare single-trusted-operator mode explicitly "
    "(single_operator_mode / VICTOR_SINGLE_OPERATOR_MODE) to attest that "
    "every configured key belongs to one trusted operator."
)


class MultiPrincipalRefused(RuntimeError):
    """A shared-state surface was addressed under the multi-principal posture.

    Raised by programmatic callers; HTTP surfaces translate this into the
    structured 409 payload (``refusal: MultiPrincipalRefused``).
    """

    def __init__(self, surface: str) -> None:
        super().__init__(_REFUSAL_MESSAGE.format(surface=surface))
        self.surface = surface


def is_multi_principal(
    api_keys: Optional[Mapping[str, str]],
    single_operator_mode: bool = False,
) -> bool:
    """True when the posture must refuse: >1 credential, no declaration.

    Distinct *credentials* are counted (not client_id labels): two keys are
    two credentials even when they map to the same label, because either can
    address the same shared conversation/store.
    """
    if single_operator_mode:
        return False
    return len(api_keys or {}) > 1


def refusal_payload(surface: str) -> Dict[str, Any]:
    """Structured 409 body naming the limitation and the operator escape."""
    return {
        "success": False,
        "error": _REFUSAL_MESSAGE.format(surface=surface),
        "refusal": MULTI_PRINCIPAL_REFUSAL_TYPE,
        "surface": surface,
    }


__all__ = [
    "MULTI_PRINCIPAL_REFUSAL_TYPE",
    "MultiPrincipalRefused",
    "is_multi_principal",
    "refusal_payload",
]
