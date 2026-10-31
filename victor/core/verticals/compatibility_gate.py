"""Shared compatibility gate for vertical runtime activation.

This module centralizes framework-version, manifest negotiation, and
version-matrix checks so multiple runtime paths rely on one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Callable

from victor.core.verticals.capability_negotiator import CapabilityNegotiator
from victor.core.verticals.framework_version import get_framework_version
from victor.core.verticals.version_matrix import (
    CompatibilityResult,
    get_compatibility_matrix,
)
from victor_contracts.verticals.manifest import ExtensionManifest

logger = logging.getLogger(__name__)

# Oldest first-party vertical releases this framework supports. Older ones call
# SDK hooks at import that victor-contracts removes (FEP-0042 phase 3), so they
# are refused before their modules are imported, with an upgrade hint.
FIRST_PARTY_MIN_VERTICAL_VERSIONS: dict[str, str] = {
    "victor-coding": "0.14.0",
    "victor-dataanalysis": "0.14.0",
    "victor-devops": "0.14.0",
    "victor-rag": "0.14.0",
    "victor-research": "0.14.0",
}


class VerticalCompatibilityError(ValueError):
    """Raised when a vertical is incompatible with the running framework.

    Provides a clear, actionable error message instead of confusing
    ModuleNotFoundError or silent degradation.
    """

    def __init__(self, report: "VerticalCompatibilityReport"):
        self.report = report
        errors = "; ".join(report.errors)
        super().__init__(
            f"Vertical '{report.vertical_name}' v{report.vertical_version} "
            f"is incompatible with framework v{report.framework_version}: {errors}"
        )


@dataclass
class VerticalCompatibilityReport:
    """Structured result of checking a vertical against the running framework."""

    vertical_name: str
    vertical_version: str
    framework_version: str
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    required_features: set[str] = field(default_factory=set)
    matrix_result: CompatibilityResult | None = None

    @property
    def compatible(self) -> bool:
        """Return True when no hard compatibility errors were detected."""

        return not self.errors

    def raise_if_incompatible(self) -> None:
        """Raise ``VerticalCompatibilityError`` when the report contains hard errors."""

        if not self.errors:
            return
        raise VerticalCompatibilityError(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "vertical_name": self.vertical_name,
            "vertical_version": self.vertical_version,
            "framework_version": self.framework_version,
            "compatible": self.compatible,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "required_features": sorted(self.required_features),
            "matrix_status": (
                self.matrix_result.status.value if self.matrix_result is not None else None
            ),
            "matrix_message": (
                self.matrix_result.message if self.matrix_result is not None else ""
            ),
        }


class VerticalCompatibilityGate:
    """Single authoritative compatibility checker for vertical runtime paths."""

    def __init__(
        self,
        *,
        framework_version_provider: Callable[[], str] | None = None,
        negotiator_factory: Callable[[], CapabilityNegotiator] | None = None,
        matrix_provider: Callable[[], Any] | None = None,
    ) -> None:
        self._get_framework_version = framework_version_provider or get_framework_version
        self._negotiator_factory = negotiator_factory or CapabilityNegotiator
        self._matrix_provider = matrix_provider or get_compatibility_matrix

    def assess_manifest(self, manifest: ExtensionManifest) -> VerticalCompatibilityReport:
        """Assess compatibility for a normalized manifest."""

        framework_version = self._get_framework_version()
        report = VerticalCompatibilityReport(
            vertical_name=manifest.name,
            vertical_version=getattr(manifest, "version", "1.0.0"),
            framework_version=framework_version,
        )

        self._check_framework_version_requirement(manifest, framework_version, report)
        if not report.compatible:
            return report

        self._check_manifest_negotiation(manifest, report)
        if not report.compatible:
            return report

        self._check_version_matrix(
            manifest.name,
            report.vertical_version,
            framework_version,
            report,
        )
        return report

    def assess_distribution(self, distribution: str, version: str) -> VerticalCompatibilityReport:
        """Check an installed vertical distribution against the supported minimum.

        Only first-party distributions listed in
        ``FIRST_PARTY_MIN_VERTICAL_VERSIONS`` are checked; anything else (or an
        unparseable version) passes, since the manifest checks still apply.
        """

        report = VerticalCompatibilityReport(
            vertical_name=distribution,
            vertical_version=version,
            framework_version=self._get_framework_version(),
        )
        from packaging.utils import canonicalize_name
        from packaging.version import InvalidVersion, Version

        minimum = FIRST_PARTY_MIN_VERTICAL_VERSIONS.get(canonicalize_name(distribution))
        if minimum is None:
            return report
        try:

            too_old = Version(version) < Version(minimum)
        except InvalidVersion:
            return report
        if too_old:
            report.errors.append(
                f"{distribution} {version} is older than the minimum supported "
                f"{minimum}; upgrade with: pip install -U '{distribution}>={minimum}'"
            )
        return report

    def assess_vertical(
        self,
        *,
        vertical_name: str,
        vertical_version: str,
        manifest: ExtensionManifest | None = None,
    ) -> VerticalCompatibilityReport:
        """Assess compatibility for a vertical with or without a manifest."""

        if manifest is not None:
            return self.assess_manifest(manifest)

        framework_version = self._get_framework_version()
        report = VerticalCompatibilityReport(
            vertical_name=vertical_name,
            vertical_version=vertical_version,
            framework_version=framework_version,
        )
        self._check_version_matrix(vertical_name, vertical_version, framework_version, report)
        return report

    def _check_framework_version_requirement(
        self,
        manifest: ExtensionManifest,
        framework_version: str,
        report: VerticalCompatibilityReport,
    ) -> None:
        """Validate ``min_framework_version`` exactly once in one place."""

        required = getattr(manifest, "min_framework_version", None)
        if not required:
            return

        try:
            from packaging import version
            from packaging.specifiers import SpecifierSet

            spec = SpecifierSet(required)
            if version.parse(framework_version) not in spec:
                report.errors.append(
                    f"Incompatible framework version: {framework_version} "
                    f"does not meet requirement {required} for vertical {manifest.name}"
                )
        except ImportError as exc:
            logger.debug("Core version negotiation skipped: %s", exc)

    def _check_manifest_negotiation(
        self,
        manifest: ExtensionManifest,
        report: VerticalCompatibilityReport,
    ) -> None:
        """Run manifest capability negotiation via the shared negotiator."""

        negotiator = self._negotiator_factory()
        result = negotiator.negotiate(manifest)
        report.warnings.extend(result.warnings)
        if not result.compatible:
            report.errors.append(
                f"Vertical '{manifest.name}' manifest negotiation failed: "
                f"{'; '.join(result.errors)}"
            )

    def _check_version_matrix(
        self,
        vertical_name: str,
        vertical_version: str,
        framework_version: str,
        report: VerticalCompatibilityReport,
    ) -> None:
        """Check version compatibility matrix status for the vertical."""

        matrix = self._matrix_provider()
        if not matrix.is_loaded():
            matrix.load_default_rules()

        result = matrix.check_compatibility(
            vertical_name=vertical_name,
            vertical_version=vertical_version,
            framework_version=framework_version,
        )
        report.matrix_result = result

        if result.is_incompatible:
            report.errors.append(
                f"Vertical '{vertical_name}' is incompatible with framework "
                f"{framework_version}: {result.message}"
            )
            return

        status_val = result.status.value if hasattr(result.status, "value") else str(result.status)
        if status_val == "degraded":
            report.warnings.append(
                f"Vertical '{vertical_name}' is running in degraded mode: {result.message}"
            )

        if result.required_features:
            report.required_features.update(result.required_features)
            report.warnings.append(
                f"Vertical '{vertical_name}' requires features that are not available: "
                f"{', '.join(sorted(result.required_features))}"
            )
