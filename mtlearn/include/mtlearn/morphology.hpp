#pragma once

/// @file
/// @brief Public C++ morphology facade for mtlearn.
///
/// This header is the only morphology header installed as part of the public
/// C++ API. It intentionally exposes mtlearn-owned types and enums instead of
/// backend types from mmcfilters. The implementation may continue to use a
/// backend internally, but downstream C++ consumers should only need this
/// facade to create trees, inspect topology, reconstruct images, and request
/// attribute identifiers compatible with the Python layer.

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

namespace mtlearn::morphology {

/// Integer node identifier used by the public morphology facade.
///
/// mtlearn uses the same integer node-id domain as the current morphology
/// backend. A node id may refer either to a live tree node or to an internal
/// backend slot, depending on the method being called. Prefer querying
/// `numInternalNodeSlots()` when allocating node-indexed vectors for backend
/// attributes, criteria, or CFP tensors.
using NodeId = int;

/// Sentinel value used when a node id is absent or invalid.
inline constexpr NodeId InvalidNode = -1;

namespace detail {
struct BackendAccess;
} // namespace detail

/// Default tree-of-shapes infinity seed row.
inline constexpr int TreeOfShapesDefaultInfinityRow = 0;

/// Default tree-of-shapes infinity seed column.
inline constexpr int TreeOfShapesDefaultInfinityCol = 0;

/// Interpolation policy used when constructing a tree of shapes.
///
/// The names match the current backend concepts, but the enum is owned by
/// mtlearn so the backend can be replaced without changing the public
/// C++/Python API.
enum class TreeOfShapesInterpolation {
    SelfDual,   ///< Self-dual interpolation.
    Min4cMax8c, ///< 4-connected min-tree and 8-connected max-tree interpolation.
    Min8cMax4c, ///< 8-connected min-tree and 4-connected max-tree interpolation.
};

/// Node-id space used by attribute computation outputs.
///
/// Attribute computation can return values indexed either by mtlearn's
/// morphological-tree node ids or by the Higra-compatible hierarchy exported by
/// the backend.
enum class NodeIdSpace {
    MorphologicalTree, ///< Rows are indexed by morphology-tree node ids.
    Higra,              ///< Rows are indexed by exported Higra hierarchy ids.
};

/// Public attribute identifiers supported by the current morphology backend.
///
/// Keep this enum synchronized with the conversion table in
/// `bindings/morphology/BindingSupport.hpp` and the Python exposure in
/// `bindings/morphology/AttributeBinding.hpp`.
enum class Attribute {
    Area,                         ///< Component area.
    Volume,                       ///< Component volume.
    RelativeVolume,              ///< Recursive contrast volume over the node subtree.
    GrayLevelHeight,                  ///< Gray-level height.
    MeanGrayLevel,                   ///< Mean gray-level value.
    GrayLevelVariance,               ///< Gray-level variance.
    BoxWidth,                    ///< Bounding-box width.
    BoundingBoxHeight,                   ///< Bounding-box height.
    DiagonalLength,              ///< Bounding-box diagonal length.
    Rectangularity,               ///< Component rectangularity.
    RatioWh,                     ///< Ratio of the longer to the shorter bounding-box side.
    BoxColumnMin,                  ///< Minimum bounding-box column.
    BoxColumnMax,                  ///< Maximum bounding-box column.
    BoxRowMin,                  ///< Minimum bounding-box row.
    BoxRowMax,                  ///< Maximum bounding-box row.
    CentralMoment20,            ///< Central moment mu20.
    CentralMoment02,            ///< Central moment mu02.
    CentralMoment11,            ///< Central moment mu11.
    CentralMoment30,            ///< Central moment mu30.
    CentralMoment03,            ///< Central moment mu03.
    CentralMoment21,            ///< Central moment mu21.
    CentralMoment12,            ///< Central moment mu12.
    HuMoment1,                  ///< First Hu invariant moment.
    HuMoment2,                  ///< Second Hu invariant moment.
    HuMoment3,                  ///< Third Hu invariant moment.
    HuMoment4,                  ///< Fourth Hu invariant moment.
    HuMoment5,                  ///< Fifth Hu invariant moment.
    HuMoment6,                  ///< Sixth Hu invariant moment.
    HuMoment7,                  ///< Seventh Hu invariant moment.
    Inertia,                      ///< Moment-based inertia.
    Compactness,                  ///< Shape compactness.
    Eccentricity,                 ///< Moment-based eccentricity.
    LengthMajorAxis,            ///< Major-axis length.
    LengthMinorAxis,            ///< Minor-axis length.
    AxisOrientation,             ///< Principal-axis orientation.
    Circularity,                  ///< Shape circularity.
    BitquadArea,                ///< Bitquad area estimate.
    BitquadNumberEuler,        ///< Bitquad Euler number.
    BitquadNumberHoles,        ///< Bitquad hole count.
    BitquadPerimeter,           ///< Bitquad perimeter estimate.
    BitquadPerimeterContinuous, ///< Continuous bitquad perimeter estimate.
    BitquadCircularity,         ///< Bitquad circularity.
    BitquadPerimeterAverage,   ///< Average bitquad perimeter.
    BitquadLengthAverage,      ///< Average bitquad length.
    BitquadWidthAverage,       ///< Average bitquad width.
    SubtreeHeight,                  ///< Node height in the tree.
    DepthNode,                   ///< Node depth in the tree.
    IsLeafNode,                 ///< Nonzero when the node is a leaf.
    IsRootNode,                 ///< Nonzero when the node is the root.
    NumChildrenNode,            ///< Number of direct children.
    NumSiblingsNode,            ///< Number of siblings.
    NumDescendantsNode,         ///< Number of descendants.
    NumLeafDescendantsNode,    ///< Number of leaf descendants.
    LeafRatioNode,              ///< Leaf-descendant ratio.
    BalanceNode,                 ///< Tree-balance descriptor.
    MaxDist,                     ///< Maximum distance descriptor.
    AvgChildHeightNode,        ///< Average child height.
    ContourPixels,               ///< Number of contour pixels.
    ContourPerimeter,            ///< Contour perimeter.
    ContourSideNorth,           ///< North-side contour contribution.
    ContourSideWest,            ///< West-side contour contribution.
    ContourSideEast,            ///< East-side contour contribution.
    ContourSideSouth,           ///< South-side contour contribution.
    MaxDistExact,
    DistSquaredSumExact,
    DistSquaredMeanExact,
    DistRmsExact,
    DistSquaredVarianceExact,
    DistSquaredSum,
    DistSquaredMean,
    DistRms,
    DistSquaredVariance,
    MaxDistCenterRowExact,
    MaxDistCenterColumnExact,
    MaxDistCenterRow,
    MaxDistCenterColumn,
    MaxDistPlateauAreaExact,
    MaxDistPlateauCentroidRowExact,
    MaxDistPlateauCentroidColumnExact,
    MaxDistPlateauArea,
    MaxDistPlateauCentroidRow,
    MaxDistPlateauCentroidColumn,
    DistSum,
    DistMean,
    DistVariance,
    DistMedian,
    DistMode,
    DistQ25,
    DistQ75,
    DistQ90,
    DistEntropy,
    DistPositiveArea,
    DistLevelCount,
    DistWeightedCentroidRow,
    DistWeightedCentroidColumn,
    DistWeightedCentralMoment20,
    DistWeightedCentralMoment02,
    DistWeightedCentralMoment11,
    DistWeightedAxisOrientation,
    DistWeightedEccentricity,
    DistSumExact,
    DistMeanExact,
    DistVarianceExact,
    DistMedianExact,
    DistModeExact,
    DistQ25Exact,
    DistQ75Exact,
    DistQ90Exact,
    DistEntropyExact,
    DistPositiveAreaExact,
    DistLevelCountExact,
    DistWeightedCentroidRowExact,
    DistWeightedCentroidColumnExact,
    DistWeightedCentralMoment20Exact,
    DistWeightedCentralMoment02Exact,
    DistWeightedCentralMoment11Exact,
    DistWeightedAxisOrientationExact,
    DistWeightedEccentricityExact,
    MaxSquaredDist,
    MaxSquaredDistExact,
    FilledArea,
    FilledCentroidRow,
    FilledCentroidColumn,
    FilledLengthMajorAxis,
    FilledLengthMinorAxis,
    FilledAxisOrientation,
    FilledEccentricity,
    FilledInertia,
    HoleAreaFraction,
    FilledCentroidDisplacementNormalized,
    FilledCompactness,
    FilledCircularity,
};

/// Attribute groups expanded by the backend attribute computer.
///
/// Groups are part of the facade because Python notebooks and future C++
/// consumers should not depend on backend attribute-group types directly.
enum class AttributeGroup {
    All,           ///< All public scalar attributes supported by the backend.
    GrayLevel,    ///< Gray-level attributes.
    Shape,         ///< Shape attributes.
    Moments,       ///< Moment-based attributes.
    Boundary,      ///< Boundary and contour attributes.
    TreeTopology, ///< Tree-topology attributes.
    DistTransf,
    DistTransfExact,
    FilledShape,
};

/// Attribute request accepted by APIs that can consume a scalar attribute or a group.
using AttributeOrGroup = std::variant<Attribute, AttributeGroup>;

/// Non-owning row-major 2D uint8 image view used at the C++ API boundary.
///
/// Callers must keep `data` alive for the duration of the tree-construction
/// call. The constructed tree owns its backend representation after
/// construction, so the input buffer may be released after the factory method
/// returns.
struct ImageViewUInt8 {
    /// Pointer to the first pixel in row-major order.
    const std::uint8_t* data{nullptr};

    /// Number of rows. Must be positive for tree construction.
    int rows{0};

    /// Number of columns. Must be positive for tree construction.
    int cols{0};
};

/// Owning row-major uint8 image returned by reconstruction routines.
///
/// This type keeps the public API independent from the backend image
/// container.
struct UInt8Image {
    /// Number of image rows.
    int rows{0};

    /// Number of image columns.
    int cols{0};

    /// Row-major pixel buffer with `rows * cols` values.
    std::vector<std::uint8_t> pixels;
};

/// Copyable handle around a backend weighted morphological tree.
///
/// `WeightedTree` is deliberately a small shared handle instead of exposing the
/// backend object by value. This keeps Python bindings, CFP helpers, and C++
/// consumers aligned around one stable facade while preserving cheap copies.
class WeightedTree {
public:
    /// Copy the shared tree handle.
    WeightedTree(const WeightedTree&) noexcept = default;

    /// Assign another shared tree handle.
    WeightedTree& operator=(const WeightedTree&) noexcept = default;

    /// Move the shared tree handle.
    WeightedTree(WeightedTree&&) noexcept = default;

    /// Move-assign the shared tree handle.
    WeightedTree& operator=(WeightedTree&&) noexcept = default;

    /// Destroy the shared tree handle.
    ~WeightedTree();

    /// Build a max-tree or min-tree from a 2D uint8 image.
    ///
    /// @param image Non-owning image view. `image.data` must not be null and
    /// `image.rows` and `image.cols` must be positive.
    /// @param isMaxTree Selects max-tree construction when true and min-tree
    /// construction when false.
    /// @param radius Adjacency radius forwarded to component-tree construction.
    /// @return A weighted morphology tree facade owning its backend tree.
    /// @throws std::invalid_argument if the image view is null or has invalid
    /// dimensions.
    static WeightedTree createComponentTree(ImageViewUInt8 image, bool isMaxTree, double radius = 1.5);

    /// Build a tree of shapes from a 2D uint8 image.
    ///
    /// The infinity seed is the pixel the backend floods from, expressed in the
    /// coordinates of `image` itself; mtlearn converts it to the interpolated
    /// domain. It should normally remain at the default unless the caller needs
    /// exact compatibility with a previous experiment.
    ///
    /// The tree is always built without an exterior ring, which keeps the
    /// published altitudes on the source 8-bit lattice.
    ///
    /// @param image Non-owning image view. `image.data` must not be null and
    /// `image.rows` and `image.cols` must be positive.
    /// @param interpolation Tree-of-shapes interpolation policy.
    /// @param infinitySeedRow Infinity seed row, in `image` coordinates.
    /// @param infinitySeedCol Infinity seed column, in `image` coordinates.
    /// @return A weighted tree-of-shapes facade owning its backend tree.
    /// @throws std::invalid_argument if the image view is null or has invalid
    /// dimensions, or if an unknown interpolation value is supplied.
    static WeightedTree createTreeOfShapes(
        ImageViewUInt8 image,
        TreeOfShapesInterpolation interpolation = TreeOfShapesInterpolation::SelfDual,
        int infinitySeedRow = TreeOfShapesDefaultInfinityRow,
        int infinitySeedCol = TreeOfShapesDefaultInfinityCol);

    /// Return the number of rows in the original image.
    int numRows() const;

    /// Return the number of columns in the original image.
    int numCols() const;

    /// Return the number of live topology nodes.
    int numNodes() const;

    /// Return the backend node-slot count used by node-indexed arrays.
    int numInternalNodeSlots() const;

    /// Return the altitude value for a morphology-tree node id.
    ///
    /// @param nodeId Node id in the morphology-tree node-id space.
    /// @return Node altitude converted to `float`.
    float getAltitude(NodeId nodeId) const;

    /// Return the residue value for a morphology-tree node id.
    ///
    /// @param nodeId Node id in the morphology-tree node-id space.
    /// @return Node residue converted to `float`.
    float getNodeResidue(NodeId nodeId) const;

    /// Prune a node from the current tree.
    ///
    /// This mutating operation preserves the backend pruning semantics. The
    /// current tree handle remains valid after pruning, but topology-dependent
    /// values should be queried again.
    ///
    /// @param nodeId Node id to prune.
    void pruneNode(NodeId nodeId);

    /// Merge a node into its parent.
    ///
    /// This mutating operation preserves the backend merge semantics. The
    /// current tree handle remains valid after merging, but topology-dependent
    /// values should be queried again.
    ///
    /// @param nodeId Node id to merge into its parent.
    void mergeNodeIntoParent(NodeId nodeId);

    /// Reconstruct the current image represented by the tree.
    ///
    /// @return Row-major image after any pruning or merging operations already
    /// applied to this tree.
    UInt8Image reconstructionImage() const;

    /// Export parent and altitude vectors compatible with Higra-style hierarchies.
    ///
    /// @return Pair `(parents, altitudes)`, where both vectors have the same
    /// length and describe the exported hierarchy.
    std::pair<std::vector<NodeId>, std::vector<float>> exportHigraHierarchy() const;

private:
    struct Impl;

    explicit WeightedTree(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> impl_;

    friend struct detail::BackendAccess;
};

/// Shared pointer type used by bindings and internal CFP helpers.
using WeightedTreePtr = std::shared_ptr<WeightedTree>;

/// Return the altitude value for `nodeId`.
///
/// This short free function is used by internal CFP code to keep formulas
/// readable.
inline float altitude(const WeightedTree& tree, NodeId nodeId)
{
    return tree.getAltitude(nodeId);
}

/// Return the residue value for `nodeId`.
///
/// This short free function is used by internal CFP code to keep formulas
/// readable.
inline float residue(const WeightedTree& tree, NodeId nodeId)
{
    return tree.getNodeResidue(nodeId);
}

} // namespace mtlearn::morphology
