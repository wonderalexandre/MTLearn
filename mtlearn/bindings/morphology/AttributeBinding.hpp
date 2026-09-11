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
        morphology::Attribute::Area,
        morphology::Attribute::Volume,
        morphology::Attribute::RelativeVolume,
        morphology::Attribute::GrayLevelHeight,
        morphology::Attribute::MeanGrayLevel,
        morphology::Attribute::GrayLevelVariance,
        morphology::Attribute::BoxWidth,
        morphology::Attribute::BoundingBoxHeight,
        morphology::Attribute::DiagonalLength,
        morphology::Attribute::Rectangularity,
        morphology::Attribute::RatioWh,
        morphology::Attribute::BoxColumnMin,
        morphology::Attribute::BoxColumnMax,
        morphology::Attribute::BoxRowMin,
        morphology::Attribute::BoxRowMax,
        morphology::Attribute::CentralMoment20,
        morphology::Attribute::CentralMoment02,
        morphology::Attribute::CentralMoment11,
        morphology::Attribute::CentralMoment30,
        morphology::Attribute::CentralMoment03,
        morphology::Attribute::CentralMoment21,
        morphology::Attribute::CentralMoment12,
        morphology::Attribute::HuMoment1,
        morphology::Attribute::HuMoment2,
        morphology::Attribute::HuMoment3,
        morphology::Attribute::HuMoment4,
        morphology::Attribute::HuMoment5,
        morphology::Attribute::HuMoment6,
        morphology::Attribute::HuMoment7,
        morphology::Attribute::Inertia,
        morphology::Attribute::Compactness,
        morphology::Attribute::Eccentricity,
        morphology::Attribute::LengthMajorAxis,
        morphology::Attribute::LengthMinorAxis,
        morphology::Attribute::AxisOrientation,
        morphology::Attribute::Circularity,
        morphology::Attribute::BitquadArea,
        morphology::Attribute::BitquadNumberEuler,
        morphology::Attribute::BitquadNumberHoles,
        morphology::Attribute::BitquadPerimeter,
        morphology::Attribute::BitquadPerimeterContinuous,
        morphology::Attribute::BitquadCircularity,
        morphology::Attribute::BitquadPerimeterAverage,
        morphology::Attribute::BitquadLengthAverage,
        morphology::Attribute::BitquadWidthAverage,
        morphology::Attribute::SubtreeHeight,
        morphology::Attribute::DepthNode,
        morphology::Attribute::IsLeafNode,
        morphology::Attribute::IsRootNode,
        morphology::Attribute::NumChildrenNode,
        morphology::Attribute::NumSiblingsNode,
        morphology::Attribute::NumDescendantsNode,
        morphology::Attribute::NumLeafDescendantsNode,
        morphology::Attribute::LeafRatioNode,
        morphology::Attribute::BalanceNode,
        morphology::Attribute::MaxDist,
        morphology::Attribute::AvgChildHeightNode,
        morphology::Attribute::ContourPixels,
        morphology::Attribute::ContourPerimeter,
        morphology::Attribute::ContourSideNorth,
        morphology::Attribute::ContourSideWest,
        morphology::Attribute::ContourSideEast,
        morphology::Attribute::ContourSideSouth,
        morphology::Attribute::MaxDistExact,
        morphology::Attribute::DistSquaredSumExact,
        morphology::Attribute::DistSquaredMeanExact,
        morphology::Attribute::DistRmsExact,
        morphology::Attribute::DistSquaredVarianceExact,
        morphology::Attribute::DistSquaredSum,
        morphology::Attribute::DistSquaredMean,
        morphology::Attribute::DistRms,
        morphology::Attribute::DistSquaredVariance,
        morphology::Attribute::MaxDistCenterRowExact,
        morphology::Attribute::MaxDistCenterColumnExact,
        morphology::Attribute::MaxDistCenterRow,
        morphology::Attribute::MaxDistCenterColumn,
        morphology::Attribute::MaxDistPlateauAreaExact,
        morphology::Attribute::MaxDistPlateauCentroidRowExact,
        morphology::Attribute::MaxDistPlateauCentroidColumnExact,
        morphology::Attribute::MaxDistPlateauArea,
        morphology::Attribute::MaxDistPlateauCentroidRow,
        morphology::Attribute::MaxDistPlateauCentroidColumn,
        morphology::Attribute::DistSum,
        morphology::Attribute::DistMean,
        morphology::Attribute::DistVariance,
        morphology::Attribute::DistMedian,
        morphology::Attribute::DistMode,
        morphology::Attribute::DistQ25,
        morphology::Attribute::DistQ75,
        morphology::Attribute::DistQ90,
        morphology::Attribute::DistEntropy,
        morphology::Attribute::DistPositiveArea,
        morphology::Attribute::DistLevelCount,
        morphology::Attribute::DistWeightedCentroidRow,
        morphology::Attribute::DistWeightedCentroidColumn,
        morphology::Attribute::DistWeightedCentralMoment20,
        morphology::Attribute::DistWeightedCentralMoment02,
        morphology::Attribute::DistWeightedCentralMoment11,
        morphology::Attribute::DistWeightedAxisOrientation,
        morphology::Attribute::DistWeightedEccentricity,
        morphology::Attribute::DistSumExact,
        morphology::Attribute::DistMeanExact,
        morphology::Attribute::DistVarianceExact,
        morphology::Attribute::DistMedianExact,
        morphology::Attribute::DistModeExact,
        morphology::Attribute::DistQ25Exact,
        morphology::Attribute::DistQ75Exact,
        morphology::Attribute::DistQ90Exact,
        morphology::Attribute::DistEntropyExact,
        morphology::Attribute::DistPositiveAreaExact,
        morphology::Attribute::DistLevelCountExact,
        morphology::Attribute::DistWeightedCentroidRowExact,
        morphology::Attribute::DistWeightedCentroidColumnExact,
        morphology::Attribute::DistWeightedCentralMoment20Exact,
        morphology::Attribute::DistWeightedCentralMoment02Exact,
        morphology::Attribute::DistWeightedCentralMoment11Exact,
        morphology::Attribute::DistWeightedAxisOrientationExact,
        morphology::Attribute::DistWeightedEccentricityExact,
        morphology::Attribute::MaxSquaredDist,
        morphology::Attribute::MaxSquaredDistExact,
        morphology::Attribute::FilledArea,
        morphology::Attribute::FilledCentroidRow,
        morphology::Attribute::FilledCentroidColumn,
        morphology::Attribute::FilledLengthMajorAxis,
        morphology::Attribute::FilledLengthMinorAxis,
        morphology::Attribute::FilledAxisOrientation,
        morphology::Attribute::FilledEccentricity,
        morphology::Attribute::FilledInertia,
        morphology::Attribute::HoleAreaFraction,
        morphology::Attribute::FilledCentroidDisplacementNormalized,
        morphology::Attribute::FilledCompactness,
        morphology::Attribute::FilledCircularity,
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
    morphology::NodeIdSpace outputSpace = morphology::NodeIdSpace::MorphologicalTree,
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
    morphology::NodeIdSpace outputSpace = morphology::NodeIdSpace::MorphologicalTree,
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
            "compute_attributes",
            &computeAttributes,
            "tree"_a,
            "attributes"_a,
            "output_space"_a = morphology::NodeIdSpace::MorphologicalTree,
            "dtype"_a = py::none(),
            R"pbdoc(Compute several attributes or attribute groups for a tree.

Returns:
    ``(attribute_index, values)`` where ``attribute_index`` maps attribute names
    to columns and ``values`` is a NumPy array. ``dtype`` accepts ``np.float32``
    and ``np.float64``; the default is ``np.float32``. By default rows are
    indexed by morphology-tree node slots.
)pbdoc")
        .def_static(
            "compute_single_attribute",
            &computeSingleAttribute,
            "tree"_a,
            "attribute"_a,
            "output_space"_a = morphology::NodeIdSpace::MorphologicalTree,
            "dtype"_a = py::none(),
            "Compute one scalar attribute and return a 1D NumPy array.")
        .def_static(
            "describe",
            &describeAttribute,
            "attribute"_a,
            "Return a user-facing description of one attribute.")
        .def_static(
            "describe_all",
            &describeAllAttributes,
            "Return descriptions for all public attributes keyed by attribute name.")
        .def_static(
            "expand_group",
            &expandAttributeGroup,
            "group"_a,
            "Return the scalar attributes represented by an attribute group.");

    py::enum_<morphology::AttributeGroup>(
        attribute,
        "Group",
        py::module_local(),
        "Predefined attribute groups accepted by attribute computation.")
        .value("ALL", morphology::AttributeGroup::All)
        .value("GRAY_LEVEL", morphology::AttributeGroup::GrayLevel)
        .value("SHAPE", morphology::AttributeGroup::Shape)
        .value("MOMENTS", morphology::AttributeGroup::Moments)
        .value("BOUNDARY", morphology::AttributeGroup::Boundary)
        .value("TREE_TOPOLOGY", morphology::AttributeGroup::TreeTopology)
        .value("DIST_TRANSF", morphology::AttributeGroup::DistTransf)
        .value("DIST_TRANSF_EXACT", morphology::AttributeGroup::DistTransfExact)
        .value("FILLED_SHAPE", morphology::AttributeGroup::FilledShape)
        .export_values();

    py::enum_<morphology::Attribute>(
        attribute,
        "Type",
        py::module_local(),
        "Scalar morphology attributes supported by the current backend.")
        .value("AREA", morphology::Attribute::Area)
        .value("VOLUME", morphology::Attribute::Volume)
        .value("RELATIVE_VOLUME", morphology::Attribute::RelativeVolume)
        .value("GRAY_LEVEL_HEIGHT", morphology::Attribute::GrayLevelHeight)
        .value("MEAN_GRAY_LEVEL", morphology::Attribute::MeanGrayLevel)
        .value("GRAY_LEVEL_VARIANCE", morphology::Attribute::GrayLevelVariance)
        .value("BOX_WIDTH", morphology::Attribute::BoxWidth)
        .value("BOUNDING_BOX_HEIGHT", morphology::Attribute::BoundingBoxHeight)
        .value("RECTANGULARITY", morphology::Attribute::Rectangularity)
        .value("DIAGONAL_LENGTH", morphology::Attribute::DiagonalLength)
        .value("BOX_COLUMN_MIN", morphology::Attribute::BoxColumnMin)
        .value("BOX_COLUMN_MAX", morphology::Attribute::BoxColumnMax)
        .value("BOX_ROW_MIN", morphology::Attribute::BoxRowMin)
        .value("BOX_ROW_MAX", morphology::Attribute::BoxRowMax)
        .value("RATIO_WH", morphology::Attribute::RatioWh)
        .value("CENTRAL_MOMENT_20", morphology::Attribute::CentralMoment20)
        .value("CENTRAL_MOMENT_02", morphology::Attribute::CentralMoment02)
        .value("CENTRAL_MOMENT_11", morphology::Attribute::CentralMoment11)
        .value("CENTRAL_MOMENT_30", morphology::Attribute::CentralMoment30)
        .value("CENTRAL_MOMENT_03", morphology::Attribute::CentralMoment03)
        .value("CENTRAL_MOMENT_21", morphology::Attribute::CentralMoment21)
        .value("CENTRAL_MOMENT_12", morphology::Attribute::CentralMoment12)
        .value("AXIS_ORIENTATION", morphology::Attribute::AxisOrientation)
        .value("LENGTH_MAJOR_AXIS", morphology::Attribute::LengthMajorAxis)
        .value("LENGTH_MINOR_AXIS", morphology::Attribute::LengthMinorAxis)
        .value("ECCENTRICITY", morphology::Attribute::Eccentricity)
        .value("CIRCULARITY", morphology::Attribute::Circularity)
        .value("COMPACTNESS", morphology::Attribute::Compactness)
        .value("INERTIA", morphology::Attribute::Inertia)
        .value("HU_MOMENT_1", morphology::Attribute::HuMoment1)
        .value("HU_MOMENT_2", morphology::Attribute::HuMoment2)
        .value("HU_MOMENT_3", morphology::Attribute::HuMoment3)
        .value("HU_MOMENT_4", morphology::Attribute::HuMoment4)
        .value("HU_MOMENT_5", morphology::Attribute::HuMoment5)
        .value("HU_MOMENT_6", morphology::Attribute::HuMoment6)
        .value("HU_MOMENT_7", morphology::Attribute::HuMoment7)
        .value("SUBTREE_HEIGHT", morphology::Attribute::SubtreeHeight)
        .value("DEPTH_NODE", morphology::Attribute::DepthNode)
        .value("IS_LEAF_NODE", morphology::Attribute::IsLeafNode)
        .value("IS_ROOT_NODE", morphology::Attribute::IsRootNode)
        .value("NUM_CHILDREN_NODE", morphology::Attribute::NumChildrenNode)
        .value("NUM_SIBLINGS_NODE", morphology::Attribute::NumSiblingsNode)
        .value("NUM_DESCENDANTS_NODE", morphology::Attribute::NumDescendantsNode)
        .value("NUM_LEAF_DESCENDANTS_NODE", morphology::Attribute::NumLeafDescendantsNode)
        .value("LEAF_RATIO_NODE", morphology::Attribute::LeafRatioNode)
        .value("BALANCE_NODE", morphology::Attribute::BalanceNode)
        .value("AVG_CHILD_HEIGHT_NODE", morphology::Attribute::AvgChildHeightNode)
        .value("BITQUAD_AREA", morphology::Attribute::BitquadArea)
        .value("BITQUAD_NUMBER_EULER", morphology::Attribute::BitquadNumberEuler)
        .value("BITQUAD_NUMBER_HOLES", morphology::Attribute::BitquadNumberHoles)
        .value("BITQUAD_PERIMETER", morphology::Attribute::BitquadPerimeter)
        .value("BITQUAD_PERIMETER_CONTINUOUS", morphology::Attribute::BitquadPerimeterContinuous)
        .value("BITQUAD_CIRCULARITY", morphology::Attribute::BitquadCircularity)
        .value("BITQUAD_PERIMETER_AVERAGE", morphology::Attribute::BitquadPerimeterAverage)
        .value("BITQUAD_LENGTH_AVERAGE", morphology::Attribute::BitquadLengthAverage)
        .value("BITQUAD_WIDTH_AVERAGE", morphology::Attribute::BitquadWidthAverage)
        .value("MAX_DIST", morphology::Attribute::MaxDist)
        .value("CONTOUR_PIXELS", morphology::Attribute::ContourPixels)
        .value("CONTOUR_PERIMETER", morphology::Attribute::ContourPerimeter)
        .value("CONTOUR_SIDE_NORTH", morphology::Attribute::ContourSideNorth)
        .value("CONTOUR_SIDE_WEST", morphology::Attribute::ContourSideWest)
        .value("CONTOUR_SIDE_EAST", morphology::Attribute::ContourSideEast)
        .value("CONTOUR_SIDE_SOUTH", morphology::Attribute::ContourSideSouth)
        .value("MAX_DIST_EXACT", morphology::Attribute::MaxDistExact)
        .value("DIST_SQUARED_SUM_EXACT", morphology::Attribute::DistSquaredSumExact)
        .value("DIST_SQUARED_MEAN_EXACT", morphology::Attribute::DistSquaredMeanExact)
        .value("DIST_RMS_EXACT", morphology::Attribute::DistRmsExact)
        .value("DIST_SQUARED_VARIANCE_EXACT", morphology::Attribute::DistSquaredVarianceExact)
        .value("DIST_SQUARED_SUM", morphology::Attribute::DistSquaredSum)
        .value("DIST_SQUARED_MEAN", morphology::Attribute::DistSquaredMean)
        .value("DIST_RMS", morphology::Attribute::DistRms)
        .value("DIST_SQUARED_VARIANCE", morphology::Attribute::DistSquaredVariance)
        .value("MAX_DIST_CENTER_ROW_EXACT", morphology::Attribute::MaxDistCenterRowExact)
        .value("MAX_DIST_CENTER_COLUMN_EXACT", morphology::Attribute::MaxDistCenterColumnExact)
        .value("MAX_DIST_CENTER_ROW", morphology::Attribute::MaxDistCenterRow)
        .value("MAX_DIST_CENTER_COLUMN", morphology::Attribute::MaxDistCenterColumn)
        .value("MAX_DIST_PLATEAU_AREA_EXACT", morphology::Attribute::MaxDistPlateauAreaExact)
        .value("MAX_DIST_PLATEAU_CENTROID_ROW_EXACT", morphology::Attribute::MaxDistPlateauCentroidRowExact)
        .value("MAX_DIST_PLATEAU_CENTROID_COLUMN_EXACT", morphology::Attribute::MaxDistPlateauCentroidColumnExact)
        .value("MAX_DIST_PLATEAU_AREA", morphology::Attribute::MaxDistPlateauArea)
        .value("MAX_DIST_PLATEAU_CENTROID_ROW", morphology::Attribute::MaxDistPlateauCentroidRow)
        .value("MAX_DIST_PLATEAU_CENTROID_COLUMN", morphology::Attribute::MaxDistPlateauCentroidColumn)
        .value("DIST_SUM", morphology::Attribute::DistSum)
        .value("DIST_MEAN", morphology::Attribute::DistMean)
        .value("DIST_VARIANCE", morphology::Attribute::DistVariance)
        .value("DIST_MEDIAN", morphology::Attribute::DistMedian)
        .value("DIST_MODE", morphology::Attribute::DistMode)
        .value("DIST_Q25", morphology::Attribute::DistQ25)
        .value("DIST_Q75", morphology::Attribute::DistQ75)
        .value("DIST_Q90", morphology::Attribute::DistQ90)
        .value("DIST_ENTROPY", morphology::Attribute::DistEntropy)
        .value("DIST_POSITIVE_AREA", morphology::Attribute::DistPositiveArea)
        .value("DIST_LEVEL_COUNT", morphology::Attribute::DistLevelCount)
        .value("DIST_WEIGHTED_CENTROID_ROW", morphology::Attribute::DistWeightedCentroidRow)
        .value("DIST_WEIGHTED_CENTROID_COLUMN", morphology::Attribute::DistWeightedCentroidColumn)
        .value("DIST_WEIGHTED_CENTRAL_MOMENT_20", morphology::Attribute::DistWeightedCentralMoment20)
        .value("DIST_WEIGHTED_CENTRAL_MOMENT_02", morphology::Attribute::DistWeightedCentralMoment02)
        .value("DIST_WEIGHTED_CENTRAL_MOMENT_11", morphology::Attribute::DistWeightedCentralMoment11)
        .value("DIST_WEIGHTED_AXIS_ORIENTATION", morphology::Attribute::DistWeightedAxisOrientation)
        .value("DIST_WEIGHTED_ECCENTRICITY", morphology::Attribute::DistWeightedEccentricity)
        .value("DIST_SUM_EXACT", morphology::Attribute::DistSumExact)
        .value("DIST_MEAN_EXACT", morphology::Attribute::DistMeanExact)
        .value("DIST_VARIANCE_EXACT", morphology::Attribute::DistVarianceExact)
        .value("DIST_MEDIAN_EXACT", morphology::Attribute::DistMedianExact)
        .value("DIST_MODE_EXACT", morphology::Attribute::DistModeExact)
        .value("DIST_Q25_EXACT", morphology::Attribute::DistQ25Exact)
        .value("DIST_Q75_EXACT", morphology::Attribute::DistQ75Exact)
        .value("DIST_Q90_EXACT", morphology::Attribute::DistQ90Exact)
        .value("DIST_ENTROPY_EXACT", morphology::Attribute::DistEntropyExact)
        .value("DIST_POSITIVE_AREA_EXACT", morphology::Attribute::DistPositiveAreaExact)
        .value("DIST_LEVEL_COUNT_EXACT", morphology::Attribute::DistLevelCountExact)
        .value("DIST_WEIGHTED_CENTROID_ROW_EXACT", morphology::Attribute::DistWeightedCentroidRowExact)
        .value("DIST_WEIGHTED_CENTROID_COLUMN_EXACT", morphology::Attribute::DistWeightedCentroidColumnExact)
        .value("DIST_WEIGHTED_CENTRAL_MOMENT_20_EXACT", morphology::Attribute::DistWeightedCentralMoment20Exact)
        .value("DIST_WEIGHTED_CENTRAL_MOMENT_02_EXACT", morphology::Attribute::DistWeightedCentralMoment02Exact)
        .value("DIST_WEIGHTED_CENTRAL_MOMENT_11_EXACT", morphology::Attribute::DistWeightedCentralMoment11Exact)
        .value("DIST_WEIGHTED_AXIS_ORIENTATION_EXACT", morphology::Attribute::DistWeightedAxisOrientationExact)
        .value("DIST_WEIGHTED_ECCENTRICITY_EXACT", morphology::Attribute::DistWeightedEccentricityExact)
        .value("MAX_SQUARED_DIST", morphology::Attribute::MaxSquaredDist)
        .value("MAX_SQUARED_DIST_EXACT", morphology::Attribute::MaxSquaredDistExact)
        .value("FILLED_AREA", morphology::Attribute::FilledArea)
        .value("FILLED_CENTROID_ROW", morphology::Attribute::FilledCentroidRow)
        .value("FILLED_CENTROID_COLUMN", morphology::Attribute::FilledCentroidColumn)
        .value("FILLED_LENGTH_MAJOR_AXIS", morphology::Attribute::FilledLengthMajorAxis)
        .value("FILLED_LENGTH_MINOR_AXIS", morphology::Attribute::FilledLengthMinorAxis)
        .value("FILLED_AXIS_ORIENTATION", morphology::Attribute::FilledAxisOrientation)
        .value("FILLED_ECCENTRICITY", morphology::Attribute::FilledEccentricity)
        .value("FILLED_INERTIA", morphology::Attribute::FilledInertia)
        .value("HOLE_AREA_FRACTION", morphology::Attribute::HoleAreaFraction)
        .value("FILLED_CENTROID_DISPLACEMENT_NORMALIZED", morphology::Attribute::FilledCentroidDisplacementNormalized)
        .value("FILLED_COMPACTNESS", morphology::Attribute::FilledCompactness)
        .value("FILLED_CIRCULARITY", morphology::Attribute::FilledCircularity)
        .export_values();
}

} // namespace morphology_pybind
} // namespace mtlearn
