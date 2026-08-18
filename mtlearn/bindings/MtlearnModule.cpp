#include <pybind11/pybind11.h>

// Native Python extension entry point for mtlearn.
//
// This translation unit should stay small: it only composes binding
// initializers from the morphology facade and CFP helpers. Algorithmic code
// belongs under mtlearn/src or mtlearn/include, not in the module definition.

#include "cfp/TreeTensorsBinding.hpp"
#include "cfp/TreeTraversalBinding.hpp"
#include "morphology/AttributeBinding.hpp"
#include "morphology/AttributeFiltersBinding.hpp"
#include "morphology/WeightedTreeBinding.hpp"

namespace mtlearn {

// Register morphology-related symbols before CFP symbols because CFP Python
// classes accept ValuedMorphologicalTree handles produced by this binding.
void init_Morphology(py::module& m)
{
    using namespace morphology_pybind;

    bindCoreMorphologyEnums(m);
    bindWeightedTree(m);
    bindAttribute(m);
    bindAttributeFilters(m);
}

} // namespace mtlearn

PYBIND11_MODULE(_mtlearn, m)
{
    // WITH_TORCH is intentionally exposed from the native module so Python can
    // skip CFP tests cleanly when a future build disables LibTorch support.
    m.doc() = "Python bindings for MTLearn morphology and connected filters";
    m.attr("WITH_TORCH") = true;

    mtlearn::init_Morphology(m);
    mtlearn::init_ConnectedFilterPreprocessingTreeTensors(m);
    mtlearn::init_ConnectedFilterPreprocessingTreeTraversal(m);
}
