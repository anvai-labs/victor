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

"""Operator-managed API principals registry (FOU-A4, Decision D4).

Owner-blessed decision D4 (modified recommendation): a versioned,
operator-managed YAML registry of API-key principals that MUST live
outside every sandbox-writable surface — default
``/etc/victor/api_principals.yaml`` for system installs, the
``server.auth_principals_path`` settings override otherwise — because the
registry is the source of principal identity: if it sat inside an
agent-writable surface (the exact Codex finding: ``~/.victor``), the agent
could rewrite its own principals.

Schema (strict — unknown fields are rejected, VAS-05a discipline):

.. code-block:: yaml

    revision: 3            # int, monotonic; workers re-read on change
    principals:
      - key: sk-alice      # unique, non-empty
        principal_id: alice # unique, non-empty (becomes client_id)
        kind: api_key      # only "api_key" in the A4a window (A4c adds oidc)
        subject_label: alice@corp  # FEP-0020 attribution subject (non-empty)
        groups:            # list of unique non-empty strings (optional)
          - eng
        expiry: 2027-01-01T00:00:00Z  # optional ISO-8601 instant

Placement invariant (enforced at startup): the resolved registry path must
not fall inside ANY sandbox-writable surface enumerated by
:func:`victor.tools.sandbox.writable_surface_roots` — workspace root,
Victor state root, temp dirs, or configured allowed mounts. Violation is a
typed :class:`PrincipalsRegistryPathError` naming the colliding surface,
and the server refuses to start.

Activation semantics: loading is read-only; activation is process restart
or the explicit ``VictorFastAPIServer.reload_principals_registry()``.
Operators publish updates via atomic rename (write a temp file, then
``os.replace`` onto the registry path) so multi-worker deployments
converge without partial reads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from victor.integrations.api.auth.principal import (
    AuthMethod,
    PrincipalKind,
    VerifiedPrincipal,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ApiPrincipalEntry",
    "PrincipalsRegistry",
    "PrincipalsRegistryError",
    "PrincipalsRegistryPathError",
    "DEFAULT_REGISTRY_PATH",
    "ensure_registry_path_outside_writable_surfaces",
    "load_principals_registry",
    "resolve_principals_registry_path",
]

#: System-install default (owner-blessed D4): outside every writable surface.
DEFAULT_REGISTRY_PATH = Path("/etc/victor/api_principals.yaml")

_ALLOWED_TOP_FIELDS = frozenset({"revision", "principals"})
_ALLOWED_ENTRY_FIELDS = frozenset(
    {"key", "principal_id", "kind", "subject_label", "groups", "expiry"}
)
_ALLOWED_KINDS = frozenset({"api_key"})


class PrincipalsRegistryError(RuntimeError):
    """The principals registry is malformed or violates a registry rule."""


class PrincipalsRegistryPathError(PrincipalsRegistryError):
    """The registry path falls inside a sandbox-writable surface.

    Startup fail-closed (D4 invariant): the agent must never be able to
    rewrite the file that defines its own principal identity.
    """

    def __init__(self, path: Path, surface: Dict[str, str]) -> None:
        self.path = path
        self.surface = surface
        super().__init__(
            f"Principals registry {path} must not live inside the "
            f"sandbox-writable surface '{surface['kind']}' "
            f"({surface['root']}). Move the registry outside every "
            f"agent-writable root (e.g. {DEFAULT_REGISTRY_PATH}) — otherwise "
            "the agent could rewrite its own principals."
        )


def resolve_principals_registry_path(explicit: Optional[str]) -> Optional[Path]:
    """Resolve the registry path: operator override, else the system default
    when it exists, else ``None`` (registry disabled — legacy principals only)."""
    if explicit:
        return Path(explicit).expanduser()
    if DEFAULT_REGISTRY_PATH.is_file():
        return DEFAULT_REGISTRY_PATH
    return None


def ensure_registry_path_outside_writable_surfaces(
    path: Path, surfaces: Iterable[Dict[str, str]]
) -> None:
    """Raise :class:`PrincipalsRegistryPathError` if ``path`` is inside any
    listed writable surface."""
    resolved = Path(path).expanduser().resolve()
    for surface in surfaces:
        root = Path(surface["root"])
        if resolved == root or root in resolved.parents:
            raise PrincipalsRegistryPathError(resolved, surface)


@dataclass(frozen=True)
class ApiPrincipalEntry:
    """One operator-declared API-key principal."""

    key: str
    principal_id: str
    kind: str
    subject_label: str
    groups: tuple[str, ...] = ()
    expires_at: Optional[datetime] = None

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.expires_at is None:
            return False
        moment = now or datetime.now(timezone.utc)
        return moment >= self.expires_at

    def to_verified_principal(self) -> VerifiedPrincipal:
        """Map the entry onto the A4a principal shape.

        ``principal_id`` becomes ``client_id`` (owner-scoping identity);
        ``subject_label`` is the FEP-0020 attribution subject (D5/D7:
        the operator-assigned stable label, never an email guess);
        ``credential_id`` is the same public SHA-256 digest convention the
        legacy principal uses — never the raw key.
        """
        import hashlib

        digest = hashlib.sha256(self.key.encode("utf-8")).hexdigest()[:16]
        return VerifiedPrincipal(
            principal_id=self.principal_id,
            kind=PrincipalKind.SERVICE,
            auth_method=AuthMethod.API_KEY,
            credential_id=f"sha256:{digest}",
            subject=self.subject_label,
            groups=self.groups,
            expires_at=self.expires_at,
        )


@dataclass(frozen=True)
class PrincipalsRegistry:
    """A validated registry snapshot (``revision`` + entries)."""

    revision: int
    entries: tuple[ApiPrincipalEntry, ...]
    _by_key: Dict[str, ApiPrincipalEntry] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_key", {entry.key: entry for entry in self.entries})

    def find(self, raw_key: str) -> Optional[ApiPrincipalEntry]:
        return self._by_key.get(raw_key)


def _parse_expiry(raw: Any, where: str) -> datetime:
    if isinstance(raw, datetime):
        parsed = raw
    elif isinstance(raw, str) and raw.strip():
        text = raw.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise PrincipalsRegistryError(
                f"{where}: 'expiry' is not a valid ISO-8601 instant: {raw!r}"
            ) from exc
    else:
        raise PrincipalsRegistryError(f"{where}: 'expiry' must be a non-empty ISO-8601 string")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _require_text(entry: Dict[str, Any], field: str, where: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise PrincipalsRegistryError(f"{where}: '{field}' must be a non-empty string")
    return value


def parse_principals_registry(data: Any) -> PrincipalsRegistry:
    """Strictly parse the registry document (unknown fields rejected)."""
    if not isinstance(data, dict):
        raise PrincipalsRegistryError(
            "Principals registry must be a mapping with 'revision' and 'principals'"
        )
    unknown_top = sorted(set(data) - _ALLOWED_TOP_FIELDS)
    if unknown_top:
        raise PrincipalsRegistryError(
            f"Principals registry has unknown top-level fields: {unknown_top}"
        )
    revision = data.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise PrincipalsRegistryError("'revision' must be a monotonic integer >= 1")
    raw_entries = data.get("principals")
    if not isinstance(raw_entries, list):
        raise PrincipalsRegistryError("'principals' must be a list of entries")

    entries: List[ApiPrincipalEntry] = []
    seen_keys: Dict[str, int] = {}
    seen_ids: Dict[str, int] = {}
    for index, raw in enumerate(raw_entries):
        where = f"principals[{index}]"
        if not isinstance(raw, dict):
            raise PrincipalsRegistryError(f"{where} must be a mapping")
        unknown = sorted(set(raw) - _ALLOWED_ENTRY_FIELDS)
        if unknown:
            raise PrincipalsRegistryError(f"{where} has unknown fields: {unknown}")
        key = _require_text(raw, "key", where)
        principal_id = _require_text(raw, "principal_id", where)
        kind = _require_text(raw, "kind", where)
        subject_label = _require_text(raw, "subject_label", where)
        if kind not in _ALLOWED_KINDS:
            raise PrincipalsRegistryError(
                f"{where}: unsupported 'kind' {kind!r} "
                f"(allowed in this release: {sorted(_ALLOWED_KINDS)})"
            )
        if key in seen_keys:
            raise PrincipalsRegistryError(
                f"{where}: duplicate key {key!r} (already used by " f"principals[{seen_keys[key]}])"
            )
        if principal_id in seen_ids:
            raise PrincipalsRegistryError(
                f"{where}: duplicate principal_id {principal_id!r} (already used "
                f"by principals[{seen_ids[principal_id]}])"
            )
        seen_keys[key] = index
        seen_ids[principal_id] = index

        raw_groups = raw.get("groups", [])
        if not isinstance(raw_groups, list):
            raise PrincipalsRegistryError(f"{where}: 'groups' must be a list")
        groups: List[str] = []
        for group_index, group in enumerate(raw_groups):
            if not isinstance(group, str) or not group.strip():
                raise PrincipalsRegistryError(
                    f"{where}: groups[{group_index}] must be a non-empty string"
                )
            if group in groups:
                raise PrincipalsRegistryError(f"{where}: duplicate group {group!r} in 'groups'")
            groups.append(group)

        expires_at = None
        if "expiry" in raw:
            expires_at = _parse_expiry(raw["expiry"], where)

        entries.append(
            ApiPrincipalEntry(
                key=key,
                principal_id=principal_id,
                kind=kind,
                subject_label=subject_label,
                groups=tuple(groups),
                expires_at=expires_at,
            )
        )

    return PrincipalsRegistry(revision=revision, entries=tuple(entries))


def load_principals_registry(path: Path) -> PrincipalsRegistry:
    """Read and validate the registry file (read-only; never written by us)."""
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        # An explicitly configured registry that does not exist is a loud,
        # typed startup failure — never a silent legacy downgrade.
        raise PrincipalsRegistryError(
            f"Principals registry {path} is configured but missing"
        ) from exc
    except yaml.YAMLError as exc:
        raise PrincipalsRegistryError(
            f"Principals registry {path} is not valid YAML: {exc}"
        ) from exc
    registry = parse_principals_registry(data)
    logger.info(
        "Principals registry loaded from %s (revision %s, %d principals)",
        path,
        registry.revision,
        len(registry.entries),
    )
    return registry
