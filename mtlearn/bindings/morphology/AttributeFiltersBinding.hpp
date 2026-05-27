#pragma once

// Pybind exposure for backend attribute filters.
//
// AttributeFiltersPybind keeps the mtlearn WeightedTree alive while wrapping
// mmcfilters::AttributeFilters. It validates Python inputs against the backend
// node-slot count before delegating to pruning/direct/subtractive rules.

#include "BindingSupport.hpp"

#include <mmcfilters/filters/AttributeFilters.hpp>

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace mtlearn {
namespace morphology_pybind {

// Stateful wrapper because mmcfilters::AttributeFilters is constructed from a
// specific backend tree. The shared tree handle guarantees the topology remains
// alive for the filter object's lifetime.
class AttributeFiltersPybind {
public:
    explicit AttributeFiltersPybind(morphology::WeightedTreePtr tree)
        : tree_(requireTree(std::move(tree))), filter_(morphology::detail::backend(*tree_))
    {
    }

    // Threshold-based filters consume one float value per internal node slot.
    py::array_t<uint8_t> filteringMin(FloatArray attribute, float threshold)
    {
        requireNodeAttributeArray(attribute, "attr");
        return imageToNumpy(filter_.filteringByPruningMin(floatArrayView(attribute), threshold));
    }

    // Criterion-based filters consume one boolean decision per internal node
    // slot, preserving the backend convention used by AttributeFilters.
    py::array_t<uint8_t> filteringMin(std::vector<bool> criterion)
    {
        requireNodeCriterion(criterion, "criterion");
        return imageToNumpy(filter_.filteringByPruningMin(criterion));
    }

    py::array_t<uint8_t> filteringMax(FloatArray attribute, float threshold)
    {
        requireNodeAttributeArray(attribute, "attr");
        return imageToNumpy(filter_.filteringByPruningMax(floatArrayView(attribute), threshold));
    }

    py::array_t<uint8_t> filteringMax(std::vector<bool> criterion)
    {
        requireNodeCriterion(criterion, "criterion");
        return imageToNumpy(filter_.filteringByPruningMax(criterion));
    }

    py::array_t<uint8_t> filteringDirectRule(std::vector<bool> criterion)
    {
        requireNodeCriterion(criterion, "criterion");
        return imageToNumpy(filter_.filteringByDirectRule(criterion));
    }

    py::array_t<uint8_t> filteringSubtractiveRule(std::vector<bool> criterion)
    {
        requireNodeCriterion(criterion, "criterion");
        return imageToNumpy(filter_.filteringBySubtractiveRule(criterion));
    }

    py::array_t<float> filteringSubtractiveScoreRule(std::vector<float> scores)
    {
        requireNodeScores(scores, "scores");
        return imageToNumpy(filter_.filteringBySubtractiveScoreRule(scores));
    }

    // Adaptive criteria are returned as std::vector<bool> because the backend
    // naturally operates on node-level boolean masks.
    std::vector<bool> getAdaptiveCriterion(std::vector<bool> criterion, int delta)
    {
        requireNodeCriterion(criterion, "criterion");
        return filter_.getAdaptiveCriterion(criterion, delta);
    }

private:
    // Convert a null shared_ptr into a Python ValueError before backend
    // construction can dereference it.
    static morphology::WeightedTreePtr requireTree(morphology::WeightedTreePtr tree)
    {
        if (!tree) {
            throw py::value_error("invalid WeightedMorphologicalTree");
        }
        return tree;
    }

    // Keep all shape checks tied to the exact backend topology used by the
    // filter object.
    const morphology::detail::TreeTopology& topology() const noexcept
    {
        return morphology::detail::topology(*tree_);
    }

    void requireNodeAttributeArray(const FloatArray& attribute, std::string_view argumentName) const
    {
        require1DArray(attribute.request(), topology().getNumInternalNodeSlots(), argumentName);
    }

    void requireNodeCriterion(const std::vector<bool>& criterion, std::string_view argumentName) const
    {
        requireVectorSize(criterion, static_cast<std::size_t>(topology().getNumInternalNodeSlots()), argumentName);
    }

    void requireNodeScores(const std::vector<float>& scores, std::string_view argumentName) const
    {
        requireVectorSize(scores, static_cast<std::size_t>(topology().getNumInternalNodeSlots()), argumentName);
    }

    morphology::WeightedTreePtr tree_;
    mmcfilters::AttributeFilters<std::uint8_t> filter_;
};

inline void bindAttributeFilters(py::module& m)
{
    py::class_<AttributeFiltersPybind>(
        m,
        "AttributeFilters",
        py::module_local(),
        R"pbdoc(Attribute-filter helper bound to one weighted morphology tree.

Filtering methods consume node-slot-sized attribute arrays, boolean criteria,
or node scores and return reconstructed NumPy images. Create instances through
``mtlearn.morphology.create_attribute_filter(tree)`` when possible.
)pbdoc")
        .def(py::init<morphology::WeightedTreePtr>(), "tree"_a, "Create filters bound to ``tree``.")
        .def("filteringMin",
            py::overload_cast<FloatArray, float>(&AttributeFiltersPybind::filteringMin),
            "attr"_a,
            "threshold"_a,
            "Prune by a minimum-threshold rule over one node attribute array.")
        .def("filteringMin",
            py::overload_cast<std::vector<bool>>(&AttributeFiltersPybind::filteringMin),
            "criterion"_a,
            "Prune by a boolean minimum criterion with one value per node slot.")
        .def("filteringMax",
            py::overload_cast<FloatArray, float>(&AttributeFiltersPybind::filteringMax),
            "attr"_a,
            "threshold"_a,
            "Prune by a maximum-threshold rule over one node attribute array.")
        .def("filteringMax",
            py::overload_cast<std::vector<bool>>(&AttributeFiltersPybind::filteringMax),
            "criterion"_a,
            "Prune by a boolean maximum criterion with one value per node slot.")
        .def("filteringDirectRule",
            &AttributeFiltersPybind::filteringDirectRule,
            "criterion"_a,
            "Apply the backend direct-rule attribute filter.")
        .def("filteringSubtractiveRule",
            &AttributeFiltersPybind::filteringSubtractiveRule,
            "criterion"_a,
            "Apply the backend subtractive-rule attribute filter.")
        .def("filteringSubtractiveScoreRule",
            &AttributeFiltersPybind::filteringSubtractiveScoreRule,
            "scores"_a,
            "Apply the backend subtractive score rule and return a float image.")
        .def("getAdaptiveCriterion",
            &AttributeFiltersPybind::getAdaptiveCriterion,
            "criterion"_a,
            "delta"_a,
            "Expand a node criterion with the backend adaptive-criterion rule.");
}

} // namespace morphology_pybind
} // namespace mtlearn
