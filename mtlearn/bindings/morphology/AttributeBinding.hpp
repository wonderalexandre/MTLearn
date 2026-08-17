#pragma once

// Pybind exposure for morphology attributes.
//
// Python keeps the historical `mtlearn.Attribute` namespace-style class while
// C++ owns the actual enum types in mtlearn::morphology. This file translates
// attribute requests into backend calls and returns NumPy-owned outputs with a
// stable column-index dictionary.

#include "BindingSupport.hpp"

#include <mmcfilters/attributes/AttributeComputation.hpp>
#include <mmcfilters/attributes/AttributeNames.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace mtlearn {
namespace morphology_pybind {

// Empty tag type used only to create the Python `Attribute` namespace class.
// All behavior is attached as static methods or nested enums.
class AttributeApi {};

// Authoritative ordered list of public attributes exposed to Python. The order
// is used only for describeAll(); numerical output order still comes from the
// backend AttributeNames mapping returned by attribute computation.
inline const std::vector<morphology::Attribute>& allAttributes()
{
    static const std::vector<morphology::Attribute> attributes = {
        morphology::Attribute::AREA,
        morphology::Attribute::VOLUME,
        morphology::Attribute::RELATIVE_VOLUME,
        morphology::Attribute::LEVEL,
        morphology::Attribute::GRAY_HEIGHT,
        morphology::Attribute::MEAN_LEVEL,
        morphology::Attribute::VARIANCE_LEVEL,
        morphology::Attribute::BOX_WIDTH,
        morphology::Attribute::BOX_HEIGHT,
        morphology::Attribute::DIAGONAL_LENGTH,
        morphology::Attribute::RECTANGULARITY,
        morphology::Attribute::RATIO_WH,
        morphology::Attribute::BOX_COL_MIN,
        morphology::Attribute::BOX_COL_MAX,
        morphology::Attribute::BOX_ROW_MIN,
        morphology::Attribute::BOX_ROW_MAX,
        morphology::Attribute::CENTRAL_MOMENT_20,
        morphology::Attribute::CENTRAL_MOMENT_02,
        morphology::Attribute::CENTRAL_MOMENT_11,
        morphology::Attribute::CENTRAL_MOMENT_30,
        morphology::Attribute::CENTRAL_MOMENT_03,
        morphology::Attribute::CENTRAL_MOMENT_21,
        morphology::Attribute::CENTRAL_MOMENT_12,
        morphology::Attribute::HU_MOMENT_1,
        morphology::Attribute::HU_MOMENT_2,
        morphology::Attribute::HU_MOMENT_3,
        morphology::Attribute::HU_MOMENT_4,
        morphology::Attribute::HU_MOMENT_5,
        morphology::Attribute::HU_MOMENT_6,
        morphology::Attribute::HU_MOMENT_7,
        morphology::Attribute::INERTIA,
        morphology::Attribute::COMPACTNESS,
        morphology::Attribute::ECCENTRICITY,
        morphology::Attribute::LENGTH_MAJOR_AXIS,
        morphology::Attribute::LENGTH_MINOR_AXIS,
        morphology::Attribute::AXIS_ORIENTATION,
        morphology::Attribute::CIRCULARITY,
        morphology::Attribute::BITQUADS_AREA,
        morphology::Attribute::BITQUADS_NUMBER_EULER,
        morphology::Attribute::BITQUADS_NUMBER_HOLES,
        morphology::Attribute::BITQUADS_PERIMETER,
        morphology::Attribute::BITQUADS_PERIMETER_CONTINUOUS,
        morphology::Attribute::BITQUADS_CIRCULARITY,
        morphology::Attribute::BITQUADS_PERIMETER_AVERAGE,
        morphology::Attribute::BITQUADS_LENGTH_AVERAGE,
        morphology::Attribute::BITQUADS_WIDTH_AVERAGE,
        morphology::Attribute::HEIGHT_NODE,
        morphology::Attribute::DEPTH_NODE,
        morphology::Attribute::IS_LEAF_NODE,
        morphology::Attribute::IS_ROOT_NODE,
        morphology::Attribute::NUM_CHILDREN_NODE,
        morphology::Attribute::NUM_SIBLINGS_NODE,
        morphology::Attribute::NUM_DESCENDANTS_NODE,
        morphology::Attribute::NUM_LEAF_DESCENDANTS_NODE,
        morphology::Attribute::LEAF_RATIO_NODE,
        morphology::Attribute::BALANCE_NODE,
        morphology::Attribute::MAX_DIST,
        morphology::Attribute::AVG_CHILD_HEIGHT_NODE,
        morphology::Attribute::CONTOUR_PIXELS,
        morphology::Attribute::CONTOUR_PERIMETER,
        morphology::Attribute::CONTOUR_SIDE_NORTH,
        morphology::Attribute::CONTOUR_SIDE_WEST,
        morphology::Attribute::CONTOUR_SIDE_EAST,
        morphology::Attribute::CONTOUR_SIDE_SOUTH,
    };
    return attributes;
}

// Forward user-facing text from mmcfilters through the mtlearn facade. This is
// intentionally kept at the binding layer until a fuller C++ attribute metadata
// facade is introduced.
inline std::string describeAttribute(morphology::Attribute attribute)
{
    return mmcfilters::AttributeNames::describe(toBackend(attribute));
}

// Return all descriptions keyed by backend/public attribute name. Keeping the
// names aligned with enum values makes the Python API easy to inspect.
inline py::dict describeAllAttributes()
{
    py::dict descriptions;
    for (const auto attribute : allAttributes()) {
        const auto backendAttribute = toBackend(attribute);
        descriptions[py::str(mmcfilters::AttributeNames::toString(backendAttribute))] =
            py::str(mmcfilters::AttributeNames::describe(backendAttribute));
    }
    return descriptions;
}

inline std::vector<morphology::Attribute> expandAttributeGroup(morphology::AttributeGroup group)
{
    const auto backendGroup = toBackend(group);
    const auto& backendGroups = mmcfilters::ATTRIBUTE_GROUPS;
    const auto it = backendGroups.find(backendGroup);
    if (it == backendGroups.end()) {
        throw std::invalid_argument("unknown AttributeGroup");
    }

    std::vector<morphology::Attribute> attributes;
    attributes.reserve(it->second.size());
    for (const auto attribute : it->second) {
        attributes.push_back(fromBackend(attribute));
    }
    return attributes;
}

// Attribute outputs may be indexed by morphological-tree node slots or by
// Higra-exported nodes. This helper asks the backend topology for the correct
// row count for the selected node-id space.
inline int outputSize(const morphology::WeightedTree& tree, morphology::NodeIdSpace outputSpace)
{
    return morphology::detail::topology(tree).getNodeIdSpaceSize(toBackend(outputSpace));
}

// mmcfilters returns an attribute-name map that is not guaranteed to iterate in
// column order. Sort by output column so Python callers can inspect the mapping
// deterministically.
inline py::dict sortedAttributeIndex(const mmcfilters::AttributeNames& attributeNames)
{
    std::vector<std::string> keys;
    std::vector<int> values;
    keys.reserve(attributeNames.indexMap.size());
    values.reserve(attributeNames.indexMap.size());

    for (const auto& item : attributeNames.indexMap) {
        keys.push_back(attributeNames.toString(item.first));
        values.push_back(item.second);
    }

    std::vector<std::size_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&values](std::size_t lhs, std::size_t rhs) {
        return values[lhs] < values[rhs];
    });

    py::dict result;
    for (std::size_t index : indices) {
        result[py::str(keys[index])] = values[index];
    }
    return result;
}

template <std::floating_point Real>
std::pair<py::dict, py::array> computeAttributesTyped(
    morphology::WeightedTreePtr tree,
    const std::vector<morphology::AttributeOrGroup>& attributes,
    morphology::NodeIdSpace outputSpace)
{
    if (!tree) {
        throw py::value_error("invalid ValuedMorphologicalTree");
    }

    const auto backendAttributes = toBackend(attributes);
    auto [attributeNames, buffer] =
        mmcfilters::AttributeComputation::computeAttributes<Real>(
            morphology::detail::backend(*tree),
            backendAttributes,
            toBackend(outputSpace));

    return {
        sortedAttributeIndex(attributeNames),
        vectorToNumpyOwned(std::move(buffer), outputSize(*tree, outputSpace), attributeNames.NUM_ATTRIBUTES)};
}

// Compute several attributes or attribute groups and return the pair used by
// Python: {attribute_name: column_index}, values. The values array is owned by
// NumPy through vectorToNumpyOwned.
inline std::pair<py::dict, py::array> computeAttributes(
    morphology::WeightedTreePtr tree,
    const std::vector<morphology::AttributeOrGroup>& attributes,
    morphology::NodeIdSpace outputSpace = morphology::NodeIdSpace::MORPHOLOGICAL_TREE,
    py::object dtype = py::none())
{
    if (parseFloatingDType(std::move(dtype)) == FloatingDType::Float64) {
        return computeAttributesTyped<double>(std::move(tree), attributes, outputSpace);
    }
    return computeAttributesTyped<float>(std::move(tree), attributes, outputSpace);
}

template <std::floating_point Real>
py::array computeSingleAttributeTyped(
    morphology::WeightedTreePtr tree,
    morphology::Attribute attribute,
    morphology::NodeIdSpace outputSpace)
{
    if (!tree) {
        throw py::value_error("invalid ValuedMorphologicalTree");
    }

    auto [attributeNames, buffer] =
        mmcfilters::AttributeComputation::computeSingleAttribute<Real>(
            morphology::detail::backend(*tree),
            toBackend(attribute),
            toBackend(outputSpace));
    (void)attributeNames;

    return vectorToNumpyOwned(std::move(buffer), outputSize(*tree, outputSpace));
}

// Single-attribute convenience wrapper. The backend still returns an
// AttributeNames object, but only the value vector is part of this Python API.
inline py::array computeSingleAttribute(
    morphology::WeightedTreePtr tree,
    morphology::Attribute attribute,
    morphology::NodeIdSpace outputSpace = morphology::NodeIdSpace::MORPHOLOGICAL_TREE,
    py::object dtype = py::none())
{
    if (parseFloatingDType(std::move(dtype)) == FloatingDType::Float64) {
        return computeSingleAttributeTyped<double>(std::move(tree), attribute, outputSpace);
    }
    return computeSingleAttributeTyped<float>(std::move(tree), attribute, outputSpace);
}

// Register Attribute static methods plus nested Group and Type enums. The
// nested shape keeps attribute operations grouped under one Python namespace.
inline void bindAttribute(py::module& m)
{
    auto attribute = py::class_<AttributeApi>(
        m,
        "Attribute",
        py::module_local(),
        R"pbdoc(Namespace-style class for morphology attribute operations.

Use ``Attribute.Type`` for scalar attributes and ``Attribute.Group`` for
predefined groups. The high-level aliases in ``mtlearn.morphology`` expose the
same objects as ``AttributeType`` and ``AttributeGroup``.
)pbdoc")
        .def_static(
            "computeAttributes",
            &computeAttributes,
            "tree"_a,
            "attributes"_a,
            "outputSpace"_a = morphology::NodeIdSpace::MORPHOLOGICAL_TREE,
            "dtype"_a = py::none(),
            R"pbdoc(Compute several attributes or attribute groups for a tree.

Returns:
    ``(attribute_index, values)`` where ``attribute_index`` maps attribute names
    to columns and ``values`` is a NumPy array. ``dtype`` accepts ``np.float32``
    and ``np.float64``; the default is ``np.float32``. By default rows are
    indexed by morphology-tree node slots.
)pbdoc")
        .def_static(
            "computeSingleAttribute",
            &computeSingleAttribute,
            "tree"_a,
            "attribute"_a,
            "outputSpace"_a = morphology::NodeIdSpace::MORPHOLOGICAL_TREE,
            "dtype"_a = py::none(),
            "Compute one scalar attribute and return a 1D NumPy array.")
        .def_static(
            "describe",
            &describeAttribute,
            "attribute"_a,
            "Return a user-facing description of one attribute.")
        .def_static(
            "describeAll",
            &describeAllAttributes,
            "Return descriptions for all public attributes keyed by attribute name.")
        .def_static(
            "expandGroup",
            &expandAttributeGroup,
            "group"_a,
            "Return the scalar attributes represented by an attribute group.");

    py::enum_<morphology::AttributeGroup>(
        attribute,
        "Group",
        py::module_local(),
        "Predefined attribute groups accepted by attribute computation.")
        .value("ALL", morphology::AttributeGroup::ALL)
        .value("GRAY_LEVEL", morphology::AttributeGroup::GRAY_LEVEL)
        .value("SHAPE", morphology::AttributeGroup::SHAPE)
        .value("MOMENTS", morphology::AttributeGroup::MOMENTS)
        .value("BOUNDARY", morphology::AttributeGroup::BOUNDARY)
        .value("TREE_TOPOLOGY", morphology::AttributeGroup::TREE_TOPOLOGY)
        .export_values();

    py::enum_<morphology::Attribute>(
        attribute,
        "Type",
        py::module_local(),
        "Scalar morphology attributes supported by the current backend.")
        .value("AREA", morphology::Attribute::AREA)
        .value("VOLUME", morphology::Attribute::VOLUME)
        .value("RELATIVE_VOLUME", morphology::Attribute::RELATIVE_VOLUME)
        .value("LEVEL", morphology::Attribute::LEVEL)
        .value("ALTITUDE", morphology::Attribute::LEVEL)
        .value("GRAY_HEIGHT", morphology::Attribute::GRAY_HEIGHT)
        .value("MEAN_LEVEL", morphology::Attribute::MEAN_LEVEL)
        .value("VARIANCE_LEVEL", morphology::Attribute::VARIANCE_LEVEL)
        .value("BOX_WIDTH", morphology::Attribute::BOX_WIDTH)
        .value("BOX_HEIGHT", morphology::Attribute::BOX_HEIGHT)
        .value("RECTANGULARITY", morphology::Attribute::RECTANGULARITY)
        .value("DIAGONAL_LENGTH", morphology::Attribute::DIAGONAL_LENGTH)
        .value("BOX_COL_MIN", morphology::Attribute::BOX_COL_MIN)
        .value("BOX_COL_MAX", morphology::Attribute::BOX_COL_MAX)
        .value("BOX_ROW_MIN", morphology::Attribute::BOX_ROW_MIN)
        .value("BOX_ROW_MAX", morphology::Attribute::BOX_ROW_MAX)
        .value("RATIO_WH", morphology::Attribute::RATIO_WH)
        .value("CENTRAL_MOMENT_20", morphology::Attribute::CENTRAL_MOMENT_20)
        .value("CENTRAL_MOMENT_02", morphology::Attribute::CENTRAL_MOMENT_02)
        .value("CENTRAL_MOMENT_11", morphology::Attribute::CENTRAL_MOMENT_11)
        .value("CENTRAL_MOMENT_30", morphology::Attribute::CENTRAL_MOMENT_30)
        .value("CENTRAL_MOMENT_03", morphology::Attribute::CENTRAL_MOMENT_03)
        .value("CENTRAL_MOMENT_21", morphology::Attribute::CENTRAL_MOMENT_21)
        .value("CENTRAL_MOMENT_12", morphology::Attribute::CENTRAL_MOMENT_12)
        .value("AXIS_ORIENTATION", morphology::Attribute::AXIS_ORIENTATION)
        .value("LENGTH_MAJOR_AXIS", morphology::Attribute::LENGTH_MAJOR_AXIS)
        .value("LENGTH_MINOR_AXIS", morphology::Attribute::LENGTH_MINOR_AXIS)
        .value("ECCENTRICITY", morphology::Attribute::ECCENTRICITY)
        .value("CIRCULARITY", morphology::Attribute::CIRCULARITY)
        .value("COMPACTNESS", morphology::Attribute::COMPACTNESS)
        .value("INERTIA", morphology::Attribute::INERTIA)
        .value("HU_MOMENT_1", morphology::Attribute::HU_MOMENT_1)
        .value("HU_MOMENT_2", morphology::Attribute::HU_MOMENT_2)
        .value("HU_MOMENT_3", morphology::Attribute::HU_MOMENT_3)
        .value("HU_MOMENT_4", morphology::Attribute::HU_MOMENT_4)
        .value("HU_MOMENT_5", morphology::Attribute::HU_MOMENT_5)
        .value("HU_MOMENT_6", morphology::Attribute::HU_MOMENT_6)
        .value("HU_MOMENT_7", morphology::Attribute::HU_MOMENT_7)
        .value("HEIGHT_NODE", morphology::Attribute::HEIGHT_NODE)
        .value("DEPTH_NODE", morphology::Attribute::DEPTH_NODE)
        .value("IS_LEAF_NODE", morphology::Attribute::IS_LEAF_NODE)
        .value("IS_ROOT_NODE", morphology::Attribute::IS_ROOT_NODE)
        .value("NUM_CHILDREN_NODE", morphology::Attribute::NUM_CHILDREN_NODE)
        .value("NUM_SIBLINGS_NODE", morphology::Attribute::NUM_SIBLINGS_NODE)
        .value("NUM_DESCENDANTS_NODE", morphology::Attribute::NUM_DESCENDANTS_NODE)
        .value("NUM_LEAF_DESCENDANTS_NODE", morphology::Attribute::NUM_LEAF_DESCENDANTS_NODE)
        .value("LEAF_RATIO_NODE", morphology::Attribute::LEAF_RATIO_NODE)
        .value("BALANCE_NODE", morphology::Attribute::BALANCE_NODE)
        .value("AVG_CHILD_HEIGHT_NODE", morphology::Attribute::AVG_CHILD_HEIGHT_NODE)
        .value("BITQUADS_AREA", morphology::Attribute::BITQUADS_AREA)
        .value("BITQUADS_NUMBER_EULER", morphology::Attribute::BITQUADS_NUMBER_EULER)
        .value("BITQUADS_NUMBER_HOLES", morphology::Attribute::BITQUADS_NUMBER_HOLES)
        .value("BITQUADS_PERIMETER", morphology::Attribute::BITQUADS_PERIMETER)
        .value("BITQUADS_PERIMETER_CONTINUOUS", morphology::Attribute::BITQUADS_PERIMETER_CONTINUOUS)
        .value("BITQUADS_CIRCULARITY", morphology::Attribute::BITQUADS_CIRCULARITY)
        .value("BITQUADS_PERIMETER_AVERAGE", morphology::Attribute::BITQUADS_PERIMETER_AVERAGE)
        .value("BITQUADS_LENGTH_AVERAGE", morphology::Attribute::BITQUADS_LENGTH_AVERAGE)
        .value("BITQUADS_WIDTH_AVERAGE", morphology::Attribute::BITQUADS_WIDTH_AVERAGE)
        .value("MAX_DIST", morphology::Attribute::MAX_DIST)
        .value("CONTOUR_PIXELS", morphology::Attribute::CONTOUR_PIXELS)
        .value("CONTOUR_PERIMETER", morphology::Attribute::CONTOUR_PERIMETER)
        .value("CONTOUR_SIDE_NORTH", morphology::Attribute::CONTOUR_SIDE_NORTH)
        .value("CONTOUR_SIDE_WEST", morphology::Attribute::CONTOUR_SIDE_WEST)
        .value("CONTOUR_SIDE_EAST", morphology::Attribute::CONTOUR_SIDE_EAST)
        .value("CONTOUR_SIDE_SOUTH", morphology::Attribute::CONTOUR_SIDE_SOUTH)
        .export_values();
}

} // namespace morphology_pybind
} // namespace mtlearn
