# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def _format_ncnn_layer(
    layer_type: str,
    layer_name: str,
    inputs: list[str],
    outputs: list[str],
    params: list[str],
) -> str:
    blobs = " ".join([*inputs, *outputs, *params]).strip()
    prefix = f"{layer_type:<24}{layer_name:<24}{len(inputs)} {len(outputs)}"
    return f"{prefix} {blobs}".rstrip()


def _parse_ncnn_layer(
    line: str,
) -> tuple[str, str, list[str], list[str], list[str]] | None:
    parts = line.split()
    if len(parts) < 4:
        return None

    input_count = int(parts[2])
    output_count = int(parts[3])
    blobs = parts[4 : 4 + input_count + output_count]
    if len(blobs) != input_count + output_count:
        return None

    inputs = blobs[:input_count]
    outputs = blobs[input_count:]
    params = parts[4 + input_count + output_count :]
    return parts[0], parts[1], inputs, outputs, params


def _rebuild_ncnn_param_header(layer_lines: list[str]) -> tuple[int, int]:
    blob_names: set[str] = set()
    for line in layer_lines:
        parsed = _parse_ncnn_layer(line)
        if parsed is None:
            continue
        _, _, inputs, outputs, _ = parsed
        blob_names.update(inputs)
        blob_names.update(outputs)

    return len(layer_lines), len(blob_names)


@lru_cache(maxsize=1)
def _has_lada_gridsample_runtime() -> bool:
    try:
        from .ncnn_vulkan import (
            import_ncnn_module,
            ncnn_has_lada_gridsample_layer,
        )
    except Exception:
        return False

    try:
        ncnn_module = import_ncnn_module()
    except Exception:
        return False
    return ncnn_has_lada_gridsample_layer(ncnn_module)


def _is_scalar_mul_layer(
    parsed: tuple[str, str, list[str], list[str], list[str]],
) -> bool:
    layer_type, _, inputs, outputs, params = parsed
    return (
        layer_type == "BinaryOp"
        and len(inputs) == 1
        and len(outputs) == 1
        and len(params) >= 3
        and params[0] == "0=2"
        and params[1] == "1=1"
    )


def _patch_spynet_resize_back_tail(
    layer_lines: list[str],
) -> tuple[list[str], bool]:
    if not layer_lines:
        return layer_lines, False

    last_layer = _parse_ncnn_layer(layer_lines[-1])
    if last_layer is None:
        return layer_lines, False
    last_type, _, _, final_outputs, _ = last_layer
    if last_type != "CopyTo" or "out0" not in final_outputs:
        return layer_lines, False

    interp_index: int | None = None
    for index in range(len(layer_lines) - 1, -1, -1):
        parsed = _parse_ncnn_layer(layer_lines[index])
        if parsed is None:
            return layer_lines, False
        if parsed[0] == "Interp":
            interp_index = index
            break

    if interp_index is None or interp_index >= len(layer_lines) - 1:
        return layer_lines, False

    tail_layers = [
        _parse_ncnn_layer(line)
        for line in layer_lines[interp_index + 1 :]
    ]
    if any(parsed is None for parsed in tail_layers):
        return layer_lines, False

    crop_layers = [
        parsed
        for parsed in tail_layers
        if parsed is not None and parsed[0] == "Crop"
    ]
    scale_layers = [
        parsed
        for parsed in tail_layers
        if parsed is not None and _is_scalar_mul_layer(parsed)
    ]
    if len(crop_layers) < 2 or len(scale_layers) < 2:
        return layer_lines, False

    interp_layer = _parse_ncnn_layer(layer_lines[interp_index])
    if interp_layer is None or len(interp_layer[3]) != 1:
        return layer_lines, False
    resized_blob = interp_layer[3][0]

    x_crop_params = crop_layers[0][4]
    y_crop_params = crop_layers[1][4]
    x_scale_params = scale_layers[0][4]
    y_scale_params = scale_layers[1][4]
    patched_lines = [
        *layer_lines[: interp_index + 1],
        _format_ncnn_layer(
            "Split",
            "spynet_output_split",
            [resized_blob],
            ["spynet_output_split_x", "spynet_output_split_y"],
            [],
        ),
        _format_ncnn_layer(
            "Crop",
            "spynet_output_x",
            ["spynet_output_split_x"],
            ["spynet_output_x_crop"],
            x_crop_params,
        ),
        _format_ncnn_layer(
            "BinaryOp",
            "spynet_output_x_scale",
            ["spynet_output_x_crop"],
            ["spynet_output_x_scaled"],
            x_scale_params,
        ),
        _format_ncnn_layer(
            "Crop",
            "spynet_output_y",
            ["spynet_output_split_y"],
            ["spynet_output_y_crop"],
            y_crop_params,
        ),
        _format_ncnn_layer(
            "BinaryOp",
            "spynet_output_y_scale",
            ["spynet_output_y_crop"],
            ["spynet_output_y_scaled"],
            y_scale_params,
        ),
        _format_ncnn_layer(
            "Concat",
            "spynet_output_concat",
            ["spynet_output_x_scaled", "spynet_output_y_scaled"],
            ["out0"],
            ["0=0"],
        ),
    ]
    return patched_lines, True


def _patch_spynet_output_tail(
    param_path: Path,
    layer_lines: list[str],
) -> tuple[list[str], bool]:
    if not param_path.name.startswith("spynet"):
        return layer_lines, False

    patched_lines, changed = _patch_spynet_resize_back_tail(layer_lines)
    if changed:
        return patched_lines, True
    return layer_lines, False


def patch_ncnn_param_for_vulkan_runtime(
    param_path: str | Path,
    *,
    enable_lada_gridsample: bool | None = None,
) -> bool:
    """Patch pnnx-generated layers that ncnn cannot load for Vulkan runtime use."""
    param_path = Path(param_path)
    if enable_lada_gridsample is None:
        enable_lada_gridsample = _has_lada_gridsample_runtime()

    lines = param_path.read_text(encoding="utf-8").splitlines()
    patched_lines: list[str] = []
    changed = False
    supported_lada_gridsample_params = {
        ("0=1", "1=1", "2=1", "3=0"),
        ("0=1", "1=2", "2=1", "3=0"),
    }
    for index, line in enumerate(lines):
        if index < 2:
            continue

        parts = line.split()
        if len(parts) < 5:
            patched_lines.append(line)
            continue

        layer_type = parts[0]
        layer_name = parts[1]
        input_count = int(parts[2])
        output_count = int(parts[3])
        inputs = parts[4 : 4 + input_count]
        outputs = parts[4 + input_count : 4 + input_count + output_count]
        params = parts[4 + input_count + output_count :]

        if layer_type == "pnnx.Expression" and input_count == 0 and output_count == 1:
            patched_lines.append(_format_ncnn_layer("Input", layer_name, [], outputs, []))
            changed = True
            continue

        if layer_type == "aten::to" and input_count == 4 and output_count == 1:
            patched_lines.append(_format_ncnn_layer("Noop", layer_name, [inputs[0]], outputs, []))
            changed = True
            continue

        if (
            layer_type == "GridSample"
            and enable_lada_gridsample
            and tuple(params) in supported_lada_gridsample_params
        ):
            patched_lines.append(
                _format_ncnn_layer("lada.GridSample", layer_name, inputs, outputs, params)
            )
            changed = True
            continue

        patched_lines.append(line)

    patched_lines, spynet_tail_changed = _patch_spynet_output_tail(param_path, patched_lines)
    changed = changed or spynet_tail_changed

    if changed:
        layer_count, blob_count = _rebuild_ncnn_param_header(patched_lines)
        param_path.write_text(
            "\n".join([lines[0], f"{layer_count} {blob_count}", *patched_lines]) + "\n",
            encoding="utf-8",
        )
    return changed
