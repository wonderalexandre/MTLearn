#pragma once

// Pybind wrapper for CFP tensor extraction.
//
// The wrapped implementation lives in mtlearn/src/mtlearn/cfp/tree_tensors.hpp.
// This file should only describe Python names, argument names, docstrings, and
// pybind call policy.

#include "mtlearn/cfp/tree_tensors.hpp"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace mtlearn {

namespace py = pybind11;

// Expose stateless tensor helpers as a Python class to preserve the historical
// mtlearn API shape used by notebooks and layer implementations.
inline void init_ConnectedFilterPreprocessingTreeTensors(py::module& m)
{
    using TreeTensors = cfp::ConnectedFilterPreprocessingTreeTensors;

    py::class_<TreeTensors>(
        m,
        "ConnectedFilterPreprocessingTreeTensors",
        R"pbdoc(Stateless tensor extractors used by CFP layer implementations.

These helpers expose residues, sparse Jacobians, and traversal metadata for a
``WeightedMorphologicalTree``. They are primarily implementation support for
``mtlearn.layers``.
)pbdoc")
        .def_static(
            "get_residues",
            &TreeTensors::getResidues,
            py::arg("tree"),
            "Return node residues as a 1D torch tensor indexed by node slot.")
        .def_static(
            "get_jacobian",
            &TreeTensors::getJacobian,
            py::arg("tree"),
            "Return the sparse node-to-pixel tree Jacobian as a torch tensor.")
        .def_static(
            "get_info_for_jacobian",
            &TreeTensors::getInfoForJacobian,
            py::arg("tree"),
            py::call_guard<py::gil_scoped_release>(),
            R"pbdoc(Return residues and implicit-Jacobian helper tensors.

The returned list contains residues, preorder times, postorder times, parent
ids, and the node owner of each flattened pixel.
)pbdoc");
}

} // namespace mtlearn
