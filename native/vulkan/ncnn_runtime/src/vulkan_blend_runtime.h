// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

#ifndef LADA_VULKAN_BLEND_RUNTIME_H
#define LADA_VULKAN_BLEND_RUNTIME_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace lada {

void bind_vulkan_blend_runtime(py::module_& m);

} // namespace lada

#endif // LADA_VULKAN_BLEND_RUNTIME_H
