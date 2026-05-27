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
///
/// @defgroup mtlearn_morphology Morphology API
/// @brief Public morphology-tree construction and query facade.
/// @{

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
    MORPHOLOGICAL_TREE, ///< Rows are indexed by morphology-tree node ids.
    HIGRA,              ///< Rows are indexed by exported Higra hierarchy ids.
};

/// Public attribute identifiers supported by the current morphology backend.
///
/// Keep this enum synchronized with the conversion table in
/// `bindings/morphology/BindingSupport.hpp` and the Python exposure in
/// `bindings/morphology/AttributeBinding.hpp`.
enum class Attribute {
    AREA,                         ///< Component area.
    VOLUME,                       ///< Component volume.
    RELATIVE_VOLUME,              ///< Volume relative to its parent context.
    LEVEL,                        ///< Node gray-level altitude.
    GRAY_HEIGHT,                  ///< Gray-level height.
    MEAN_LEVEL,                   ///< Mean gray-level value.
    VARIANCE_LEVEL,               ///< Gray-level variance.
    BOX_WIDTH,                    ///< Bounding-box width.
    BOX_HEIGHT,                   ///< Bounding-box height.
    DIAGONAL_LENGTH,              ///< Bounding-box diagonal length.
    RECTANGULARITY,               ///< Component rectangularity.
    RATIO_WH,                     ///< Width-to-height ratio.
    BOX_COL_MIN,                  ///< Minimum bounding-box column.
    BOX_COL_MAX,                  ///< Maximum bounding-box column.
    BOX_ROW_MIN,                  ///< Minimum bounding-box row.
    BOX_ROW_MAX,                  ///< Maximum bounding-box row.
    CENTRAL_MOMENT_20,            ///< Central moment mu20.
    CENTRAL_MOMENT_02,            ///< Central moment mu02.
    CENTRAL_MOMENT_11,            ///< Central moment mu11.
    CENTRAL_MOMENT_30,            ///< Central moment mu30.
    CENTRAL_MOMENT_03,            ///< Central moment mu03.
    CENTRAL_MOMENT_21,            ///< Central moment mu21.
    CENTRAL_MOMENT_12,            ///< Central moment mu12.
    HU_MOMENT_1,                  ///< First Hu invariant moment.
    HU_MOMENT_2,                  ///< Second Hu invariant moment.
    HU_MOMENT_3,                  ///< Third Hu invariant moment.
    HU_MOMENT_4,                  ///< Fourth Hu invariant moment.
    HU_MOMENT_5,                  ///< Fifth Hu invariant moment.
    HU_MOMENT_6,                  ///< Sixth Hu invariant moment.
    HU_MOMENT_7,                  ///< Seventh Hu invariant moment.
    INERTIA,                      ///< Moment-based inertia.
    COMPACTNESS,                  ///< Shape compactness.
    ECCENTRICITY,                 ///< Moment-based eccentricity.
    LENGTH_MAJOR_AXIS,            ///< Major-axis length.
    LENGTH_MINOR_AXIS,            ///< Minor-axis length.
    AXIS_ORIENTATION,             ///< Principal-axis orientation.
    CIRCULARITY,                  ///< Shape circularity.
    BITQUADS_AREA,                ///< Bitquad area estimate.
    BITQUADS_NUMBER_EULER,        ///< Bitquad Euler number.
    BITQUADS_NUMBER_HOLES,        ///< Bitquad hole count.
    BITQUADS_PERIMETER,           ///< Bitquad perimeter estimate.
    BITQUADS_PERIMETER_CONTINUOUS, ///< Continuous bitquad perimeter estimate.
    BITQUADS_CIRCULARITY,         ///< Bitquad circularity.
    BITQUADS_PERIMETER_AVERAGE,   ///< Average bitquad perimeter.
    BITQUADS_LENGTH_AVERAGE,      ///< Average bitquad length.
    BITQUADS_WIDTH_AVERAGE,       ///< Average bitquad width.
    HEIGHT_NODE,                  ///< Node height in the tree.
    DEPTH_NODE,                   ///< Node depth in the tree.
    IS_LEAF_NODE,                 ///< Nonzero when the node is a leaf.
    IS_ROOT_NODE,                 ///< Nonzero when the node is the root.
    NUM_CHILDREN_NODE,            ///< Number of direct children.
    NUM_SIBLINGS_NODE,            ///< Number of siblings.
    NUM_DESCENDANTS_NODE,         ///< Number of descendants.
    NUM_LEAF_DESCENDANTS_NODE,    ///< Number of leaf descendants.
    LEAF_RATIO_NODE,              ///< Leaf-descendant ratio.
    BALANCE_NODE,                 ///< Tree-balance descriptor.
    MAX_DIST,                     ///< Maximum distance descriptor.
    AVG_CHILD_HEIGHT_NODE,        ///< Average child height.
    CONTOUR_PIXELS,               ///< Number of contour pixels.
    CONTOUR_PERIMETER,            ///< Contour perimeter.
    CONTOUR_SIDE_NORTH,           ///< North-side contour contribution.
    CONTOUR_SIDE_WEST,            ///< West-side contour contribution.
    CONTOUR_SIDE_EAST,            ///< East-side contour contribution.
    CONTOUR_SIDE_SOUTH,           ///< South-side contour contribution.
};

/// Attribute groups expanded by the backend attribute computer.
///
/// Groups are part of the facade because Python notebooks and future C++
/// consumers should not depend on backend attribute-group types directly.
enum class AttributeGroup {
    ALL,           ///< All public scalar attributes supported by the backend.
    GRAY_LEVEL,    ///< Gray-level attributes.
    SHAPE,         ///< Shape attributes.
    MOMENTS,       ///< Moment-based attributes.
    BOUNDARY,      ///< Boundary and contour attributes.
    TREE_TOPOLOGY, ///< Tree-topology attributes.
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
    /// The infinity seed controls backend boundary handling and should normally
    /// remain at the default unless the caller needs exact compatibility with a
    /// previous experiment.
    ///
    /// @param image Non-owning image view. `image.data` must not be null and
    /// `image.rows` and `image.cols` must be positive.
    /// @param interpolation Tree-of-shapes interpolation policy.
    /// @param infinitySeedRow Boundary infinity seed row.
    /// @param infinitySeedCol Boundary infinity seed column.
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

/// @}
