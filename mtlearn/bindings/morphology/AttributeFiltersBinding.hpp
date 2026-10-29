#pragma once

// Pybind exposure for backend attribute filters.
//
// AttributeFiltersPybind keeps the mtlearn WeightedTree alive while wrapping
// mmcfilters::AttributeFilters. It validates Python inputs against the backend
// node-slot count before delegating to pruning/direct/subtractive rules.

#include "BindingSupport.hpp"

#include <mmcfilters/filters/AttributeFilters.hpp>
#include <mmcfilters/filters/AttributeReconstructionFilters.hpp>
#include <mmcfilters/filters/ExtinctionValues.hpp>
#include <mmcfilters/filters/NodeDecisionMasks.hpp>

#include <cmath>
#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
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
public:
    struct ExtinctionSelectionPolicy {
        enum class Kind { TopK, MinimumExtinction };
        Kind kind;
        int extremaToKeep{0};
        double threshold{0.0};

        static ExtinctionSelectionPolicy byTopK(int extremaToKeep)
        {
            return {Kind::TopK, extremaToKeep, 0.0};
        }

        static ExtinctionSelectionPolicy byThreshold(double threshold)
        {
            return {Kind::MinimumExtinction, 0, threshold};
        }
    };

    explicit AttributeFiltersPybind(morphology::WeightedTreePtr tree)
        : tree_(requireTree(std::move(tree))),
          treeMutationVersion_(morphology::detail::topology(*tree_).getMutationVersion()),
          filter_(morphology::detail::backend(*tree_))
    {
    }

    // Threshold-based filters consume one floating-point value per internal
    // node slot. Float32 remains the default output dtype of attribute
    // computation, but float64 buffers are accepted without downcasting.
    py::array_t<uint8_t> filteringMin(py::array attribute, double threshold)
    {
        if (parseFloatingArrayDType(attribute, "attribute") == FloatingDType::Float64) {
            return filteringMinTyped<double>(std::move(attribute), threshold);
        }
        return filteringMinTyped<float>(std::move(attribute), static_cast<float>(threshold));
    }

    // Preservation masks contain one Boolean decision per internal node slot.
    py::array_t<uint8_t> filteringMin(py::array nodePreservationMask)
    {
        auto decisions = requireNodePreservationMask(nodePreservationMask);
        return imageToNumpy(filter_.filteringByPruningMin(std::move(decisions)));
    }

    py::array_t<uint8_t> filteringMax(py::array attribute, double threshold)
    {
        if (parseFloatingArrayDType(attribute, "attribute") == FloatingDType::Float64) {
            return filteringMaxTyped<double>(std::move(attribute), threshold);
        }
        return filteringMaxTyped<float>(std::move(attribute), static_cast<float>(threshold));
    }

    py::array_t<uint8_t> filteringMax(py::array nodePreservationMask)
    {
        auto decisions = requireNodePreservationMask(nodePreservationMask);
        return imageToNumpy(filter_.filteringByPruningMax(std::move(decisions)));
    }

    py::array_t<uint8_t> filteringDirectRule(py::array nodePreservationMask)
    {
        auto decisions = requireNodePreservationMask(nodePreservationMask);
        return imageToNumpy(mmcfilters::applyDirectAttributeFilter(
            morphology::detail::backend(*tree_),
            std::move(decisions)));
    }

    // The subtractive rule can produce residues outside the source range,
    // so the backend publishes them in the wider signed altitude-difference type
    // and mtlearn exposes that type unchanged rather than saturating it.
    py::array_t<std::int64_t> filteringSubtractiveRule(py::array nodePreservationMask)
    {
        auto decisions = requireNodePreservationMask(nodePreservationMask);
        return imageToNumpy(mmcfilters::applySubtractiveAttributeFilter(
            morphology::detail::backend(*tree_),
            std::move(decisions)));
    }

    py::array filteringSubtractiveScoreRule(py::array nodePreservationScores)
    {
        if (parseFloatingArrayDType(nodePreservationScores, "node_preservation_scores") == FloatingDType::Float64) {
            return filteringSubtractiveScoreRuleTyped<double>(std::move(nodePreservationScores));
        }
        return filteringSubtractiveScoreRuleTyped<float>(std::move(nodePreservationScores));
    }

    py::array_t<uint8_t> filteringByExtinctionValue(
        py::array attribute,
        const ExtinctionSelectionPolicy& selection)
    {
        if (parseFloatingArrayDType(attribute, "attribute") == FloatingDType::Float64) {
            return filteringByExtinctionValueTyped<double>(std::move(attribute), selection);
        }
        return filteringByExtinctionValueTyped<float>(std::move(attribute), selection);
    }

private:
    // Convert a null shared_ptr into a Python ValueError before backend
    // construction can dereference it.
    static morphology::WeightedTreePtr requireTree(morphology::WeightedTreePtr tree)
    {
        if (!tree) {
            throw py::value_error("invalid ValuedMorphologicalTree");
        }
        return tree;
    }

    // Keep all shape checks tied to the exact backend topology used by the
    // filter object.
    const morphology::detail::TreeTopology& topology() const
    {
        const auto& topology = morphology::detail::topology(*tree_);
        if (topology.getMutationVersion() != treeMutationVersion_) {
            throw std::invalid_argument("AttributeFilters topology changed; create a new filter after tree mutation.");
        }
        return topology;
    }

    template <std::floating_point Real>
    py::array_t<Real, py::array::c_style> requireNodeAttributeArray(
        py::array attribute,
        std::string_view argumentName) const
    {
        return require1DFloatingArray<Real>(
            std::move(attribute),
            topology().numInternalNodeSlots(),
            argumentName);
    }

    template <std::floating_point Real>
    py::array_t<uint8_t> filteringMinTyped(py::array attribute, Real threshold)
    {
        auto typed = requireNodeAttributeArray<Real>(std::move(attribute), "attribute");
        return imageToNumpy(filter_.filteringByPruningMin(floatingArrayView(typed), threshold));
    }

    template <std::floating_point Real>
    py::array_t<uint8_t> filteringMaxTyped(py::array attribute, Real threshold)
    {
        auto typed = requireNodeAttributeArray<Real>(std::move(attribute), "attribute");
        return imageToNumpy(filter_.filteringByPruningMax(floatingArrayView(typed), threshold));
    }

    template <std::floating_point Real>
    mmcfilters::ExtinctionValues<std::uint8_t, Real> makeExtinctionValues(py::array attribute)
    {
        auto typed = requireNodeAttributeArray<Real>(std::move(attribute), "attribute");
        requireFiniteNodeAttributeValues(typed);
        return mmcfilters::ExtinctionValues<std::uint8_t, Real>(
            morphology::detail::backend(*tree_),
            floatingArrayView(typed));
    }

    template <std::floating_point Real>
    static void requireFiniteNodeAttributeValues(const py::array_t<Real, py::array::c_style>& attribute)
    {
        const py::buffer_info buffer = attribute.request();
        const auto* values = static_cast<const Real*>(buffer.ptr);
        for (py::ssize_t i = 0; i < buffer.shape[0]; ++i) {
            if (!std::isfinite(static_cast<double>(values[i]))) {
                throw std::invalid_argument("attribute must contain only finite values");
            }
        }
    }

    // The backend selection policy carries both modes natively, so the facade no
    // longer walks the extrema to turn a minimum extinction into a count.
    template <std::floating_point Real>
    static mmcfilters::ExtinctionSelectionPolicy<Real> toSelectionPolicy(const ExtinctionSelectionPolicy& selection)
    {
        if (selection.kind == ExtinctionSelectionPolicy::Kind::TopK) {
            return mmcfilters::ExtinctionSelectionPolicy<Real>::byTopK(selection.extremaToKeep);
        }
        return mmcfilters::ExtinctionSelectionPolicy<Real>::byThreshold(static_cast<Real>(selection.threshold));
    }

    template <std::floating_point Real>
    py::array_t<uint8_t> filteringByExtinctionValueTyped(py::array attribute, const ExtinctionSelectionPolicy& selection)
    {
        auto extinction = makeExtinctionValues<Real>(std::move(attribute));
        return imageToNumpy(extinction.filtering(toSelectionPolicy<Real>(selection)));
    }

    mmcfilters::NodePreservationMask requireNodePreservationMask(const py::array& nodePreservationMask) const
    {
        if (!nodePreservationMask.dtype().is(py::dtype::of<bool>())) {
            throw py::type_error("node_preservation_mask must have dtype np.bool_");
        }
        const auto buffer = nodePreservationMask.request();
        require1DArray(buffer, topology().numInternalNodeSlots(), "node_preservation_mask");
        if (!(nodePreservationMask.flags() & py::array::c_style)) {
            throw py::value_error("node_preservation_mask must be C-contiguous");
        }
        const auto* values = static_cast<const unsigned char*>(buffer.ptr);
        std::vector<bool> decisions(static_cast<std::size_t>(buffer.size));
        for (py::ssize_t index = 0; index < buffer.size; ++index) {
            decisions[static_cast<std::size_t>(index)] = values[index] != 0;
        }
        return mmcfilters::NodePreservationMask(std::move(decisions));
    }

    template <std::floating_point Real>
    py::array filteringSubtractiveScoreRuleTyped(py::array nodePreservationScores)
    {
        auto typed = requireNodeAttributeArray<Real>(std::move(nodePreservationScores), "node_preservation_scores");
        const auto buffer = typed.request();
        const auto* values = static_cast<const Real*>(buffer.ptr);
        return imageToNumpy(mmcfilters::applySoftSubtractiveAttributeFilter<std::uint8_t, Real>(
            morphology::detail::backend(*tree_),
            std::span<const Real>(values, static_cast<std::size_t>(buffer.size))));
    }

    morphology::WeightedTreePtr tree_;
    std::uint64_t treeMutationVersion_;
    mmcfilters::AttributeFilters<std::uint8_t> filter_;
};

inline void bindAttributeFilters(py::module& m)
{
    using Selection = AttributeFiltersPybind::ExtinctionSelectionPolicy;
    py::class_<Selection>(m, "ExtinctionSelectionPolicy", py::module_local())
        .def_static("by_top_k", &Selection::byTopK, "extrema_to_keep"_a)
        .def_static("by_threshold", &Selection::byThreshold, "threshold"_a)
        .def_readonly("extrema_to_keep", &Selection::extremaToKeep)
        .def_readonly("threshold", &Selection::threshold);

    py::class_<AttributeFiltersPybind>(m, "AttributeFilters", py::module_local(),
        "Filters over a valued morphological tree. Masks are contiguous 1D Boolean NumPy arrays; true preserves a node.")
        .def(py::init<morphology::WeightedTreePtr>(), "tree"_a.none(false))
        .def("filtering_by_pruning_min_rule",
            py::overload_cast<py::array, double>(&AttributeFiltersPybind::filteringMin),
            "attribute"_a.noconvert(), "threshold"_a,
            "Preserve nodes whose attribute is greater than or equal to threshold.")
        .def("filtering_by_pruning_min_rule",
            py::overload_cast<py::array>(&AttributeFiltersPybind::filteringMin),
            "node_preservation_mask"_a.noconvert())
        .def("filtering_by_pruning_max_rule",
            py::overload_cast<py::array, double>(&AttributeFiltersPybind::filteringMax),
            "attribute"_a.noconvert(), "threshold"_a,
            "Preserve nodes whose attribute is greater than or equal to threshold.")
        .def("filtering_by_pruning_max_rule",
            py::overload_cast<py::array>(&AttributeFiltersPybind::filteringMax),
            "node_preservation_mask"_a.noconvert())
        .def("filtering_by_direct_rule", &AttributeFiltersPybind::filteringDirectRule,
            "node_preservation_mask"_a.noconvert())
        .def("filtering_by_subtractive_rule", &AttributeFiltersPybind::filteringSubtractiveRule,
            "node_preservation_mask"_a.noconvert())
        .def("filtering_by_soft_subtractive_rule", &AttributeFiltersPybind::filteringSubtractiveScoreRule,
            "node_preservation_scores"_a.noconvert())
        .def("filtering_by_extinction_rule", &AttributeFiltersPybind::filteringByExtinctionValue,
            "attribute"_a.noconvert(), "selection"_a);
}

} // namespace morphology_pybind
} // namespace mtlearn
