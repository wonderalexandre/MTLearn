#pragma once

// Pybind exposure for backend attribute filters.
//
// AttributeFiltersPybind keeps the mtlearn WeightedTree alive while wrapping
// mmcfilters::AttributeFilters. It validates Python inputs against the backend
// node-slot count before delegating to pruning/direct/subtractive rules.

#include "BindingSupport.hpp"

#include <mmcfilters/filters/AttributeFilters.hpp>
#include <mmcfilters/filters/ExtinctionValues.hpp>

#include <cmath>
#include <concepts>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace mtlearn {
namespace morphology_pybind {

// Stateful wrapper because mmcfilters::AttributeFilters is constructed from a
// specific backend tree. The shared tree handle guarantees the topology remains
// alive for the filter object's lifetime.
class AttributeFiltersPybind {
private:
    enum class ExtinctionSelectionMode {
        ExtremaToKeep,
        MinExtinction,
    };

    struct ExtinctionSelection {
        ExtinctionSelectionMode mode;
        int extremaToKeep{0};
        double minExtinction{0.0};
    };

public:
    explicit AttributeFiltersPybind(morphology::WeightedTreePtr tree)
        : tree_(requireTree(std::move(tree))), filter_(morphology::detail::backend(*tree_))
    {
    }

    // Threshold-based filters consume one floating-point value per internal
    // node slot. Float32 remains the default output dtype of attribute
    // computation, but float64 buffers are accepted without downcasting.
    py::array_t<uint8_t> filteringMin(py::array attribute, double threshold)
    {
        if (parseFloatingArrayDType(attribute, "attr") == FloatingDType::Float64) {
            return filteringMinTyped<double>(std::move(attribute), threshold);
        }
        return filteringMinTyped<float>(std::move(attribute), static_cast<float>(threshold));
    }

    // Criterion-based filters consume one boolean decision per internal node
    // slot, preserving the backend convention used by AttributeFilters.
    py::array_t<uint8_t> filteringMin(std::vector<bool> criterion)
    {
        requireNodeCriterion(criterion, "criterion");
        return imageToNumpy(filter_.filteringByPruningMin(criterion));
    }

    py::array_t<uint8_t> filteringMax(py::array attribute, double threshold)
    {
        if (parseFloatingArrayDType(attribute, "attr") == FloatingDType::Float64) {
            return filteringMaxTyped<double>(std::move(attribute), threshold);
        }
        return filteringMaxTyped<float>(std::move(attribute), static_cast<float>(threshold));
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

    py::array_t<uint8_t> filteringByExtinctionValue(
        py::array attribute,
        py::object minExtinction,
        py::object extremaToKeep)
    {
        const ExtinctionSelection selection = parseExtinctionSelection(std::move(minExtinction), std::move(extremaToKeep));
        if (parseFloatingArrayDType(attribute, "attr") == FloatingDType::Float64) {
            return filteringByExtinctionValueTyped<double>(std::move(attribute), selection);
        }
        return filteringByExtinctionValueTyped<float>(std::move(attribute), selection);
    }

    py::array saliencyMapByExtinctionValue(
        py::array attribute,
        py::object minExtinction,
        bool unweighted,
        py::object extremaToKeep)
    {
        const ExtinctionSelection selection = parseExtinctionSelection(std::move(minExtinction), std::move(extremaToKeep));
        if (parseFloatingArrayDType(attribute, "attr") == FloatingDType::Float64) {
            return saliencyMapByExtinctionValueTyped<double>(std::move(attribute), selection, unweighted);
        }
        return saliencyMapByExtinctionValueTyped<float>(std::move(attribute), selection, unweighted);
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

    template <std::floating_point Real>
    py::array_t<Real, py::array::c_style> requireNodeAttributeArray(
        py::array attribute,
        std::string_view argumentName) const
    {
        return require1DFloatingArray<Real>(
            std::move(attribute),
            topology().getNumInternalNodeSlots(),
            argumentName);
    }

    template <std::floating_point Real>
    py::array_t<uint8_t> filteringMinTyped(py::array attribute, Real threshold)
    {
        auto typed = requireNodeAttributeArray<Real>(std::move(attribute), "attr");
        return imageToNumpy(filter_.filteringByPruningMin(floatingArrayView(typed), threshold));
    }

    template <std::floating_point Real>
    py::array_t<uint8_t> filteringMaxTyped(py::array attribute, Real threshold)
    {
        auto typed = requireNodeAttributeArray<Real>(std::move(attribute), "attr");
        return imageToNumpy(filter_.filteringByPruningMax(floatingArrayView(typed), threshold));
    }

    template <std::floating_point Real>
    mmcfilters::ExtinctionValues<std::uint8_t, Real> makeExtinctionValues(py::array attribute)
    {
        auto typed = requireNodeAttributeArray<Real>(std::move(attribute), "attr");
        requireFiniteNodeAttributeValues(typed);
        return mmcfilters::ExtinctionValues<std::uint8_t, Real>(
            morphology::detail::backend(*tree_),
            floatingArrayView(typed));
    }

    static void requireFiniteExtinctionThreshold(double minExtinction)
    {
        if (!std::isfinite(minExtinction)) {
            throw std::invalid_argument("min_extinction must be finite");
        }
    }

    template <std::floating_point Real>
    static void requireFiniteNodeAttributeValues(const py::array_t<Real, py::array::c_style>& attribute)
    {
        const py::buffer_info buffer = attribute.request();
        const auto* values = static_cast<const Real*>(buffer.ptr);
        for (py::ssize_t i = 0; i < buffer.shape[0]; ++i) {
            if (!std::isfinite(static_cast<double>(values[i]))) {
                throw std::invalid_argument("attr must contain only finite values");
            }
        }
    }

    template <std::floating_point Real>
    static int countExtremaAtLeast(mmcfilters::ExtinctionValues<std::uint8_t, Real>& extinction, double minExtinction)
    {
        int extremaToKeep = 0;
        for (const auto& record : extinction.getExtinctionValues()) {
            if (static_cast<double>(record.extinction) < minExtinction) {
                break;
            }
            ++extremaToKeep;
        }
        return extremaToKeep;
    }

    static ExtinctionSelection parseExtinctionSelection(py::object minExtinction, py::object extremaToKeep)
    {
        const bool hasMinExtinction = !minExtinction.is_none();
        const bool hasExtremaToKeep = !extremaToKeep.is_none();
        if (hasMinExtinction == hasExtremaToKeep) {
            throw std::invalid_argument("pass exactly one of min_extinction or extrema_to_keep");
        }

        if (hasExtremaToKeep) {
            return ExtinctionSelection{
                ExtinctionSelectionMode::ExtremaToKeep,
                extremaToKeep.cast<int>(),
                0.0};
        }

        const double threshold = minExtinction.cast<double>();
        requireFiniteExtinctionThreshold(threshold);
        return ExtinctionSelection{
            ExtinctionSelectionMode::MinExtinction,
            0,
            threshold};
    }

    template <std::floating_point Real>
    static int resolveExtremaToKeep(
        mmcfilters::ExtinctionValues<std::uint8_t, Real>& extinction,
        const ExtinctionSelection& selection)
    {
        if (selection.mode == ExtinctionSelectionMode::ExtremaToKeep) {
            return selection.extremaToKeep;
        }
        return countExtremaAtLeast(extinction, selection.minExtinction);
    }

    template <std::floating_point Real>
    py::array_t<uint8_t> filteringByExtinctionValueTyped(py::array attribute, const ExtinctionSelection& selection)
    {
        auto extinction = makeExtinctionValues<Real>(std::move(attribute));
        return imageToNumpy(extinction.filtering(resolveExtremaToKeep(extinction, selection)));
    }

    template <std::floating_point Real>
    py::array saliencyMapByExtinctionValueTyped(
        py::array attribute,
        const ExtinctionSelection& selection,
        bool unweighted)
    {
        auto extinction = makeExtinctionValues<Real>(std::move(attribute));
        return imageToNumpy(extinction.saliencyMap(resolveExtremaToKeep(extinction, selection), unweighted));
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
            py::overload_cast<py::array, double>(&AttributeFiltersPybind::filteringMin),
            "attr"_a,
            "threshold"_a,
            "Prune by a minimum-threshold rule over one node attribute array.")
        .def("filteringMin",
            py::overload_cast<std::vector<bool>>(&AttributeFiltersPybind::filteringMin),
            "criterion"_a,
            "Prune by a boolean minimum criterion with one value per node slot.")
        .def("filteringMax",
            py::overload_cast<py::array, double>(&AttributeFiltersPybind::filteringMax),
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
            "Expand a node criterion with the backend adaptive-criterion rule.")
        .def("filteringByExtinctionValue",
            &AttributeFiltersPybind::filteringByExtinctionValue,
            "attr"_a,
            "min_extinction"_a = py::none(),
            "extrema_to_keep"_a = py::none(),
            "Filter by keeping extrema selected either by ``min_extinction`` or ``extrema_to_keep``.")
        .def("saliencyMapByExtinctionValue",
            &AttributeFiltersPybind::saliencyMapByExtinctionValue,
            "attr"_a,
            "min_extinction"_a = py::none(),
            "unweighted"_a = false,
            "extrema_to_keep"_a = py::none(),
            "Build a contour saliency map from extrema selected either by ``min_extinction`` or ``extrema_to_keep``.");
}

} // namespace morphology_pybind
} // namespace mtlearn
