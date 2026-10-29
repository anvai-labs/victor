"""Runtime bridge: feature flag access."""

# ruff: noqa: F822

from __future__ import annotations


from victor_contracts._lazy import make_lazy_getattr

__all__ = ["get_feature_flag_manager", "FeatureFlag", "is_feature_enabled"]

_LAZY_IMPORTS = {
    "get_feature_flag_manager": "victor.core.feature_flags",
    "FeatureFlag": "victor.core.feature_flags",
    "is_feature_enabled": "victor.core.feature_flags",
}


__getattr__ = make_lazy_getattr(__name__, _LAZY_IMPORTS)
