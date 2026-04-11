from __future__ import annotations

from dataclasses import fields, replace
from typing import Iterable

from lada.restorationpipeline.runtime_options import (
    RestorationRuntimeFeatures,
    resolve_restoration_runtime_features,
)


_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_FALSY_VALUES = {"0", "false", "no", "off"}
_FEATURE_FIELD_NAMES = {
    field.name for field in fields(RestorationRuntimeFeatures)
}


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUTHY_VALUES:
        return True
    if normalized in _FALSY_VALUES:
        return False
    raise RuntimeError(
        f"Invalid runtime feature override value '{value}'. "
        "Use one of: 1/0, true/false, yes/no, on/off."
    )


def parse_runtime_feature_overrides(
    override_specs: Iterable[str] | None,
) -> dict[str, bool]:
    """Parse repeated CLI override specs into one validated feature map."""
    if override_specs is None:
        return {}

    overrides: dict[str, bool] = {}
    for raw_spec in override_specs:
        spec = raw_spec.strip()
        if not spec:
            continue
        if "=" in spec:
            feature_name, raw_value = spec.split("=", 1)
            feature_name = feature_name.strip()
            feature_value = _parse_bool(raw_value)
        else:
            feature_name = spec
            feature_value = True
        if feature_name not in _FEATURE_FIELD_NAMES:
            supported = ", ".join(sorted(_FEATURE_FIELD_NAMES))
            raise RuntimeError(
                f"Unknown runtime feature override '{feature_name}'. "
                f"Supported features: {supported}"
            )
        overrides[feature_name] = feature_value
    return overrides


def apply_runtime_feature_overrides(
    model: object,
    overrides: dict[str, bool] | None,
) -> RestorationRuntimeFeatures:
    """Apply validated restoration runtime overrides to one loaded model."""
    normalized_overrides = overrides or {}
    current_features = getattr(model, "runtime_features", None)
    if not isinstance(current_features, RestorationRuntimeFeatures):
        current_features = resolve_restoration_runtime_features(model)
    if not normalized_overrides:
        return current_features

    if not hasattr(model, "runtime_features"):
        raise RuntimeError(
            "The active restoration model does not expose mutable runtime features."
        )

    updated_features = replace(current_features, **normalized_overrides)
    setattr(model, "runtime_features", updated_features)
    return updated_features
