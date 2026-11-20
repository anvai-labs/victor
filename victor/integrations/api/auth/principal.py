# Copyright 2026 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""Verified principal resolution at the API-server auth seam (FOU-A4a).

The auth dependency used to reduce every credential to an opaque
``client_id`` string. This module introduces the :class:`VerifiedPrincipal`
shape of record (docs/architecture/fou-a4-rbac-sso-design.md §A4a, itself
derived from the VAS-05a auth-policy contract) and the synthesis of the
legacy principal for registry-less API keys:

- ``principal_id`` of a legacy principal IS today's key label, so
  ``request.state.client_id`` consumers (broadcast owner scoping, route
  owner checks, FEP-0020 subjects) keep working untouched — zero migration
  for existing operators.
- ``credential_id`` is a public digest of the presented key, never the
  secret itself.
- OIDC (``AuthMethod.OIDC``) is A4c and waits on decisions D1/D2; the
  enum value is reserved here so the principal shape is stable.

Attribution (FOU-A4b / FEP-0020) binds subject+group through
:func:`bind_principal_attribution`, which is **best-effort**: an
attribution failure is logged and swallowed — it must never fail the
request path.
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Iterator, Optional

from victor.core.context import bind_attribution

logger = logging.getLogger(__name__)

__all__ = [
    "AuthMethod",
    "PrincipalKind",
    "VerifiedPrincipal",
    "bind_principal_attribution",
    "legacy_api_key_principal",
]


class AuthMethod(str, Enum):
    """How the credential was verified. ``oidc`` is A4c (decisions D1/D2)."""

    API_KEY = "api_key"
    OIDC = "oidc"


class PrincipalKind(str, Enum):
    """Whether the principal denotes a human or a non-human caller."""

    HUMAN = "human"
    SERVICE = "service"


@dataclass(frozen=True)
class VerifiedPrincipal:
    """A credential whose verification succeeded, with stable identity.

    ``principal_id`` is the stable, immutable Victor-side identifier. For
    registry-less API keys it is the legacy key label (D7: ``client_id``
    compatibility is kept indefinitely); OIDC principals (A4c) derive it
    from the verified ``(issuer, subject)`` pair — never email/username.
    """

    principal_id: str
    kind: PrincipalKind
    auth_method: AuthMethod
    credential_id: Optional[str] = None
    issuer: Optional[str] = None
    subject: Optional[str] = None
    groups: tuple[str, ...] = ()
    expires_at: Optional[datetime] = None
    scopes: frozenset[str] = field(default=frozenset())

    @property
    def client_id(self) -> str:
        """VAS-05j/05k compatibility label (today's ``client_id`` value).

        Every existing consumer — ``request.state.client_id``, broadcast
        owner scoping, route owner checks — keeps its exact semantics.
        """
        return self.principal_id

    @property
    def subject_label(self) -> str:
        """FEP-0020 attribution subject (Decision D7).

        Key-only mode keeps the legacy client_id label; OIDC mode (A4c)
        will use the immutable principal ID instead.
        """
        return self.subject or self.principal_id

    @property
    def primary_group(self) -> Optional[str]:
        """First verified group, or ``None`` (group provenance is verified
        only — a registry-less key has no groups and binds no group)."""
        return self.groups[0] if self.groups else None


def legacy_api_key_principal(*, api_key: str, client_id: str) -> VerifiedPrincipal:
    """Synthesize the registry-less (legacy) principal for a matched key.

    Zero-migration path (Decision D4 pending): every key already in
    ``VICTOR_API_KEYS`` resolves to this principal. ``principal_id`` is the
    key-map label — exactly the ``client_id`` the seam produced before A4a
    — and ``credential_id`` is a short SHA-256 digest so a public key ID
    can exist without ever storing the secret.
    """
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return VerifiedPrincipal(
        principal_id=client_id,
        kind=PrincipalKind.SERVICE,
        auth_method=AuthMethod.API_KEY,
        credential_id=f"sha256:{digest}",
    )


@contextmanager
def bind_principal_attribution(
    principal: Optional[VerifiedPrincipal],
    *,
    fallback_subject_id: Optional[str] = None,
) -> Iterator[None]:
    """Bind the principal's subject/group to the execution context (FEP-0020).

    Used by the A4a auth dependency and the ``/ws`` turn handler so every
    downstream cost/usage record carries the authenticated identity.

    Best-effort per FEP-0020: an attribution failure is logged and the
    context left unbound — the request continues with the operator-level
    config defaults, never an error. Only the binding operations are
    guarded; exceptions raised by the wrapped body propagate unchanged.
    """
    if principal is not None:
        subject_id: Optional[str] = principal.subject_label
        group_id: Optional[str] = principal.primary_group
    else:
        subject_id = fallback_subject_id
        group_id = None

    if subject_id is None and group_id is None:
        yield
        return

    try:
        cm = bind_attribution(subject_id=subject_id, group_id=group_id)
        cm.__enter__()
    except Exception as exc:
        logger.debug("Attribution binding failed (best-effort, continuing): %s", exc)
        yield
        return

    try:
        yield
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception as exc:
            logger.debug("Attribution unbinding failed (best-effort): %s", exc)
