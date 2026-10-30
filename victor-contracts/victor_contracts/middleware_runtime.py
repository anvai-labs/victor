"""SDK host adapters for framework middleware helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from victor_contracts._lazy import make_lazy_getattr

if TYPE_CHECKING:
    from victor.agent.code_correction_middleware import (
        CodeCorrectionConfig,
        CodeCorrectionMiddleware,
        CorrectionResult,
    )
    from victor.evaluation.correction.types import CodeValidationResult, Language
    from victor.framework.middleware import GitSafetyMiddleware, MiddlewareComposer

__all__ = [
    "CodeCorrectionConfig",
    "CodeCorrectionMiddleware",
    "CodeValidationResult",
    "CorrectionResult",
    "GitSafetyMiddleware",
    "Language",
    "MiddlewareComposer",
]

_LAZY_IMPORTS = {
    "CodeCorrectionConfig": "victor.agent.code_correction_middleware",
    "CodeCorrectionMiddleware": "victor.agent.code_correction_middleware",
    "CorrectionResult": "victor.agent.code_correction_middleware",
    "CodeValidationResult": "victor.evaluation.correction.types",
    "Language": "victor.evaluation.correction.types",
    "GitSafetyMiddleware": "victor.framework.middleware",
    "MiddlewareComposer": "victor.framework.middleware",
}


__getattr__ = make_lazy_getattr(__name__, _LAZY_IMPORTS)
