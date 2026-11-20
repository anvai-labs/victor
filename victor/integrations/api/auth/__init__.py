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

"""Client-layer auth contracts for the FastAPI server (FOU-A4).

Principal resolution lives in the Client layer alongside the auth seam it
serves; no ``victor/framework`` public API changes and no FEP is triggered
(fou-a4-rbac-sso-design.md, layering note). A4c (OIDC) and A4d (policy)
extend this package behind their pending owner decisions (D1/D2, D3/D4).
"""

from victor.integrations.api.auth.principal import (
    AuthMethod,
    PrincipalKind,
    VerifiedPrincipal,
    bind_principal_attribution,
    legacy_api_key_principal,
)
from victor.integrations.api.auth.principals_registry import (
    ApiPrincipalEntry,
    DEFAULT_REGISTRY_PATH,
    PrincipalsRegistry,
    PrincipalsRegistryError,
    PrincipalsRegistryPathError,
    ensure_registry_path_outside_writable_surfaces,
    load_principals_registry,
    resolve_principals_registry_path,
)

__all__ = [
    "AuthMethod",
    "PrincipalKind",
    "VerifiedPrincipal",
    "bind_principal_attribution",
    "legacy_api_key_principal",
    "ApiPrincipalEntry",
    "DEFAULT_REGISTRY_PATH",
    "PrincipalsRegistry",
    "PrincipalsRegistryError",
    "PrincipalsRegistryPathError",
    "ensure_registry_path_outside_writable_surfaces",
    "load_principals_registry",
    "resolve_principals_registry_path",
]
