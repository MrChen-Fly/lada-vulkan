from __future__ import annotations

from pathlib import Path

from lada.compute_targets import UnsupportedComputeTargetError, get_compute_target


def _require_available_target(compute_target_id: str) -> None:
    target = get_compute_target(compute_target_id, include_experimental=True)
    if target is None:
        raise UnsupportedComputeTargetError(
            f"Unknown compute target '{compute_target_id}'."
        )
    if not target.available:
        raise UnsupportedComputeTargetError(
            target.notes or f"Compute target '{compute_target_id}' is not available."
        )


def build_vulkan_iree_restoration_model(
    *,
    model_name: str,
    model_path: str,
    config_path: str | dict | None,
    compute_target_id: str,
    torch_device,
    fp16: bool,
    artifacts_dir: str | Path | None = None,
):
    del model_path
    del config_path
    del torch_device
    del fp16
    del artifacts_dir
    _require_available_target(compute_target_id)
    if not model_name.startswith("basicvsrpp"):
        raise UnsupportedComputeTargetError(
            "IREE Vulkan restoration currently targets BasicVSR++ planning only."
        )
    raise UnsupportedComputeTargetError(
        "IREE Vulkan restoration is now isolated under 'lada/extensions/iree/', "
        "but the stage-by-stage BasicVSR++ audit is still pending."
    )
