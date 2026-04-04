// SPDX-FileCopyrightText: Lada Authors
// SPDX-License-Identifier: AGPL-3.0

#ifndef LADA_BASICVSRPP_CLIP_RUNNER_H
#define LADA_BASICVSRPP_CLIP_RUNNER_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace lada {

void bind_basicvsrpp_clip_runner(py::module_& m);

} // namespace lada

#endif // LADA_BASICVSRPP_CLIP_RUNNER_H
