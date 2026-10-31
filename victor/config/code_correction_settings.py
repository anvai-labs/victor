"""Automatic code correction configuration."""

from __future__ import annotations

from pydantic import BaseModel


class CodeCorrectionSettings(BaseModel):
    """Automatic code correction configuration."""

    code_correction_enabled: bool = True
    code_correction_auto_fix: bool = True
    code_correction_max_iterations: int = 15
