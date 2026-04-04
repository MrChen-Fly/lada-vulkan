# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from torch.nn.modules.utils import _pair, _single


def _compute_output_shape(
    input_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    """Return the deform-conv output spatial shape."""
    input_h, input_w = input_shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    output_h = (input_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    output_w = (input_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    return output_h, output_w


def _modulated_deform_conv2d_pure_torch(
    x: torch.Tensor,
    offset: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    mask: torch.Tensor,
    deform_groups: int,
) -> torch.Tensor:
    """Implement modulated deform conv with grid_sample and a 1x1 conv."""
    batch_size, channels, input_h, input_w = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    output_h, output_w = _compute_output_shape(
        (input_h, input_w),
        (kernel_h, kernel_w),
        stride,
        padding,
        dilation,
    )
    channels_per_deform_group = channels // deform_groups
    kernel_size = kernel_h * kernel_w

    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation

    base_y = (
        torch.arange(output_h, dtype=x.dtype, device=x.device).view(1, 1, output_h, 1)
        * stride_h
        - padding_h
    )
    base_x = (
        torch.arange(output_w, dtype=x.dtype, device=x.device).view(1, 1, 1, output_w)
        * stride_w
        - padding_w
    )

    sampled_groups = []
    for group_index in range(deform_groups):
        x_group = x[
            :,
            group_index * channels_per_deform_group : (group_index + 1) * channels_per_deform_group,
        ]
        offset_group = offset[
            :,
            group_index * 2 * kernel_size : (group_index + 1) * 2 * kernel_size,
        ]
        mask_group = mask[:, group_index * kernel_size : (group_index + 1) * kernel_size]
        sampled_kernels = []
        for kernel_index in range(kernel_size):
            kernel_y = (kernel_index // kernel_w) * dilation_h
            kernel_x = (kernel_index % kernel_w) * dilation_w

            # torchvision packs offsets as (y, x) pairs.
            offset_y = offset_group[:, 2 * kernel_index : 2 * kernel_index + 1]
            offset_x = offset_group[:, 2 * kernel_index + 1 : 2 * kernel_index + 2]
            mask_value = mask_group[:, kernel_index : kernel_index + 1]

            sample_y = base_y + kernel_y + offset_y
            sample_x = base_x + kernel_x + offset_x
            norm_y = (2.0 * sample_y + 1.0) / input_h - 1.0
            norm_x = (2.0 * sample_x + 1.0) / input_w - 1.0
            grid = torch.stack((norm_x[:, 0], norm_y[:, 0]), dim=-1)
            sampled = F.grid_sample(
                x_group,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled_kernels.append(sampled * mask_value)
        sampled_groups.append(torch.cat(sampled_kernels, dim=1))

    sampled = torch.cat(sampled_groups, dim=1)
    weight_1x1 = weight.view(out_channels, deform_groups, channels_per_deform_group, kernel_size)
    weight_1x1 = weight_1x1.permute(0, 1, 3, 2).reshape(
        out_channels,
        channels * kernel_size,
        1,
        1,
    )
    return F.conv2d(sampled, weight_1x1, bias)

class ModulatedDeformConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 bias=True):
        super(ModulatedDeformConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        self.with_bias = bias
        self.export_mode = False
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x, offset, mask):
        if self.groups != 1:
            raise NotImplementedError("Export-time deform conv only supports groups=1.")
        if x.shape[1] % self.deform_groups != 0:
            raise ValueError("Input channels must be divisible by deform_groups.")
        return _modulated_deform_conv2d_pure_torch(
            x,
            offset,
            self.weight,
            self.bias,
            _pair(self.stride),
            _pair(self.padding),
            _pair(self.dilation),
            mask,
            self.deform_groups,
        )
