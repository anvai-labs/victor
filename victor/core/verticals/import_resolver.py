"""Helpers for resolving vertical module imports across package layouts.

This module centralizes import fallback behavior used during the external
vertical extraction migration:

1. External package namespace (preferred): ``victor_<vertical>``
2. Legacy in-package namespace: ``victor.<vertical>``
3. Contrib fallback namespace: ``victor.verticals.contrib.<vertical>``

It also supports the temporary mixed-mode package layout used during
definition/runtime extraction, where runtime-owned modules may move under a
``runtime/`` subpackage before the package root shims are removed.
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_RUNTIME_MODULE_PREFIXES = frozenset(
    {
        "capabilities",
        "composed_chains",
        "conversation_enhanced",
        "enrichment",
        "escape_hatches",
        "handlers",
        "middleware",
        "mode_config",
        "rl",
        "safety",
        "safety_enhanced",
        "service_provider",
        "teams",
        "tool_dependencies",
        "workflows",
    }
)
_WARNED_LEGACY_RESOLUTIONS: set[str] = set()


def normalize_vertical_name(vertical_name: str) -> str:
    """Normalize a vertical name for import path construction."""
    normalized = vertical_name.strip().lower().replace("-", "_")
    if normalized in {"data_analysis"}:
        return "dataanalysis"
    return normalized


def _dedupe(candidates: List[str]) -> List[str]:
    """Deduplicate candidate paths while preserving order."""
    seen = set()
    ordered: List[str] = []
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        ordered.append(candidate)
        seen.add(candidate)
    return ordered


def _join_module_path(package: str, module_suffix: str) -> str:
    """Join package + optional module suffix into a module path."""
    suffix = module_suffix.strip(".")
    if not suffix:
        return package
    return f"{package}.{suffix}"


def _external_package_candidates(vertical_name: str) -> List[str]:
    """Return external package candidates for a vertical."""
    normalized = normalize_vertical_name(vertical_name)

    candidates: List[str] = [f"victor_{normalized}"]

    # Support historical spellings like data_analysis vs dataanalysis.
    if "_" in normalized:
        candidates.append(f"victor_{normalized.replace('_', '')}")

    return _dedupe(candidates)


def _package_namespace_candidates(vertical_name: str) -> List[str]:
    """Return ordered package namespaces for a vertical."""
    normalized = normalize_vertical_name(vertical_name)
    return _dedupe(
        _external_package_candidates(normalized)
        + [
            f"victor.{normalized}",
            f"victor.verticals.contrib.{normalized}",
        ]
    )


def _is_runtime_owned_suffix(module_suffix: str) -> bool:
    """Return True when a module suffix belongs to the runtime layer."""
    suffix = module_suffix.strip(".")
    if not suffix:
        return False
    if suffix.startswith("runtime."):
        suffix = suffix[len("runtime.") :]
    head = suffix.split(".", 1)[0]
    return head in _RUNTIME_MODULE_PREFIXES


def vertical_module_candidates(vertical_name: str, module_suffix: str) -> List[str]:
    """Build ordered import candidates for a vertical module suffix."""
    suffix = module_suffix.strip(".")
    candidates: List[str] = []

    for package in _external_package_candidates(vertical_name):
        candidates.append(_join_module_path(package, suffix))

    normalized = normalize_vertical_name(vertical_name)
    candidates.append(_join_module_path(f"victor.{normalized}", suffix))
    candidates.append(_join_module_path(f"victor.verticals.contrib.{normalized}", suffix))
    return _dedupe(candidates)


def vertical_runtime_module_candidates(vertical_name: str, module_suffix: str) -> List[str]:
    """Build mixed-mode import candidates for runtime-owned vertical modules.

    Runtime modules are resolved with this order per namespace:
    1. ``runtime.<suffix>`` in the preferred package
    2. package-root ``<suffix>`` shim in the same package

    Definition-layer modules such as ``prompts`` continue to use the
    root-only resolution order from :func:`vertical_module_candidates`.
    """
    suffix = module_suffix.strip(".")
    if not _is_runtime_owned_suffix(suffix):
        return vertical_module_candidates(vertical_name, suffix)

    runtime_suffix = suffix
    if not runtime_suffix.startswith("runtime."):
        runtime_suffix = f"runtime.{runtime_suffix}"

    candidates: List[str] = []
    for package in _package_namespace_candidates(vertical_name):
        candidates.append(_join_module_path(package, runtime_suffix))
        candidates.append(_join_module_path(package, suffix))
    return _dedupe(candidates)


def module_import_candidates(module_path: str) -> List[str]:
    """Expand a module path into ordered compatibility candidates."""
    path = module_path.strip()
    if not path:
        return []

    parts = path.split(".")

    # Contrib path: victor.verticals.contrib.<vertical>[.<suffix>...]
    if len(parts) >= 4 and parts[:3] == ["victor", "verticals", "contrib"]:
        vertical = parts[3]
        suffix = ".".join(parts[4:])
        return _dedupe(vertical_module_candidates(vertical, suffix) + [path])

    # Legacy path: victor.<vertical>[.<suffix>...]
    if len(parts) >= 2 and parts[0] == "victor":
        vertical = parts[1]
        suffix = ".".join(parts[2:])
        return _dedupe(vertical_module_candidates(vertical, suffix) + [path])

    # External path: victor_<vertical>[.<suffix>...]
    root = parts[0]
    if root.startswith("victor_"):
        vertical = root[len("victor_") :]
        suffix = ".".join(parts[1:])
        return _dedupe([path] + vertical_module_candidates(vertical, suffix))

    return [path]


def import_module_with_fallback(
    module_path: str,
) -> Tuple[Optional[ModuleType], Optional[str]]:
    """Import first available module from compatibility candidates."""
    for candidate in module_import_candidates(module_path):
        try:
            module = importlib.import_module(candidate)
            _warn_legacy_resolution(candidate)
            return module, candidate
        except Exception as exc:
            logger.debug("Failed importing module candidate '%s': %s", candidate, exc)
            continue
    return None, None


def _warn_legacy_resolution(resolved_module_path: str) -> None:
    """Emit a one-time warning when runtime falls back to legacy module paths."""

    if resolved_module_path.startswith("victor_"):
        return
    if resolved_module_path not in _WARNED_LEGACY_RESOLUTIONS and (
        resolved_module_path.startswith("victor.verticals.contrib.")
        or resolved_module_path.startswith("victor.")
    ):
        if resolved_module_path.startswith("victor.verticals.contrib."):
            message = (
                "Resolved vertical module via deprecated contrib fallback "
                f"'{resolved_module_path}'. Migrate to the extracted 'victor_<vertical>' package."
            )
        else:
            message = (
                "Resolved vertical module via deprecated legacy package path "
                f"'{resolved_module_path}'. Migrate to the extracted 'victor_<vertical>' package."
            )

        logger.warning(message)
        _WARNED_LEGACY_RESOLUTIONS.add(resolved_module_path)
