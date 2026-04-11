from __future__ import annotations

from pathlib import Path

from lada.compute_targets import UnsupportedComputeTargetError


def build_vulkan_restoration_model(
    *,
    model_name: str,
    model_path: str,
    config_path: str | dict | None,
    compute_target_id: str,
    torch_device,
    fp16: bool,
    artifacts_dir: str | Path | None = None,
):
    del compute_target_id
    del torch_device

    if not model_name.startswith("basicvsrpp"):
        raise UnsupportedComputeTargetError(
            "Vulkan restoration backend currently supports BasicVSR++ models only."
        )

    from lada.extensions.vulkan.basicvsrpp_common import (
        _MODULAR_FRAME_COUNT,
    )
    from lada.extensions.vulkan.basicvsrpp_restorer import (
        NcnnVulkanBasicvsrppMosaicRestorer,
    )

    return (
        NcnnVulkanBasicvsrppMosaicRestorer(
            model_path,
            config_path=config_path,
            fp16=fp16,
            frame_count=_MODULAR_FRAME_COUNT,
            artifacts_dir=artifacts_dir,
        ),
        "zero",
    )
