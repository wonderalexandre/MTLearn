#pragma once

// Pybind wrapper for the CPU tree-traversal CFP backend.
//
// The C++ traversal logic lives in mtlearn/src/mtlearn/cfp/tree_traversal.hpp.
// This wrapper only maps method names to Python and keeps compatibility with
// the Python autograd layer that calls these static methods.

#include "mtlearn/cfp/tree_traversal.hpp"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace mtlearn {

namespace py = pybind11;

// Expose the traversal helper as a static-method class rather than free
// functions to keep the Python API grouped with the tensor helper class.
inline void init_ConnectedFilterPreprocessingTreeTraversal(py::module& m)
{
    using TreeTraversal = cfp::ConnectedFilterPreprocessingTreeTraversal;

    py::class_<TreeTraversal>(
        m,
        "ConnectedFilterPreprocessingTreeTraversal",
        R"pbdoc(CPU tree-traversal backend used by reference CFP implementations.

The primary CFP layer uses implicit tensor metadata, but this class remains
available for compatibility and validation of the CPU traversal path.
)pbdoc")
        .def_static(
            "filtering",
            &TreeTraversal::filtering,
            py::arg("tree"),
            py::arg("sigmoid"),
            "Reconstruct a filtered image from node-wise sigmoid decisions.")
        // Legacy 4-argument gradient API for weights and bias. Keep this
        // signature stable because the CPU traversal autograd Function calls
        // it directly.
        .def_static(
            "gradients",
            &TreeTraversal::gradients,
            py::arg("tree"),
            py::arg("attributes"),
            py::arg("sigmoid"),
            py::arg("grad_output"),
            "Compute CFP weight and bias gradients from CPU tree traversal.");
}

} // namespace mtlearn
