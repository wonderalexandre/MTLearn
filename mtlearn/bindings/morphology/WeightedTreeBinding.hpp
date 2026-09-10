#pragma once

// Pybind exposure for the mtlearn WeightedTree facade.
//
// This file intentionally exposes a rich query surface because existing
// notebooks inspect component-tree topology directly. The C++ public facade
// remains smaller; topology-heavy methods are routed through internal backend
// accessors here so Python can keep its current behavior without making those
// backend details part of the installed C++ API.

#include "BindingSupport.hpp"

#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

namespace mtlearn {
namespace morphology_pybind {

// The tree kind used to be published as the backend enum's raw integer, which
// silently changed meaning when the backend renumbered it. Publishing the name
// instead keeps mtlearn's own TreeType vocabulary and fails loudly rather than
// quietly if the backend set ever changes again.
inline const char* treeKindName(mmcfilters::MorphologicalTreeKind kind)
{
    switch (kind) {
    case mmcfilters::MorphologicalTreeKind::MaxTree:
        return "max-tree";
    case mmcfilters::MorphologicalTreeKind::MinTree:
        return "min-tree";
    case mmcfilters::MorphologicalTreeKind::TreeOfShapes:
        return "tree-of-shapes";
    case mmcfilters::MorphologicalTreeKind::UnrestrictedResidualTree:
        return "unrestricted-residual-tree";
    case mmcfilters::MorphologicalTreeKind::SaturatedResidualTree:
        return "saturated-residual-tree";
    case mmcfilters::MorphologicalTreeKind::Generic:
        return "generic";
    }
    throw std::invalid_argument("unknown morphological tree kind");
}

// The tree-of-shapes adjacency radii used to be plain accessors on the tree.
// They now live inside the retained topographic convention, which keeps the
// resolved complementary adjacencies whenever the immersion is a complementary
// grid. A self-dual span immersion carries no adjacency pair at all.
inline double treeOfShapesAdjacencyRadius(morphology::WeightedTree& tree, bool minimum)
{
    const auto* convention = morphology::detail::topology(tree).topographicConvention();
    if (convention == nullptr) {
        throw std::invalid_argument("tree was not built as a tree of shapes");
    }

    const auto* grid = std::get_if<mmcfilters::ComplementaryGridImmersion>(&convention->immersion);
    if (grid == nullptr) {
        throw std::invalid_argument("tree-of-shapes immersion carries no complementary adjacencies");
    }

    return minimum ? grid->complementaryAdjacencies.minAdjacency.getRadius()
                   : grid->complementaryAdjacencies.maxAdjacency.getRadius();
}

// Convert backend traversal ranges into Python-friendly vectors. Backend
// iterators are often lightweight views, so bindings materialize them before
// returning to Python.
template <class Range>
std::vector<morphology::NodeId> collectNodeIds(const Range& range)
{
    std::vector<morphology::NodeId> ids;
    for (morphology::NodeId id : range) {
        ids.push_back(id);
    }
    return ids;
}

// A connected component is represented by a node plus all proper parts in its
// subtree. This helper is used by reconstructNode to produce a binary mask for
// inspection/debugging in notebooks.
inline std::vector<int> collectPixelsOfConnectedComponent(
    const morphology::detail::TreeTopology& tree,
    morphology::NodeId nodeId)
{
    std::vector<int> pixels;
    for (morphology::NodeId subtreeNodeId : tree.subtreeNodes(nodeId)) {
        for (int pixel : tree.properPart(subtreeNodeId)) {
            pixels.push_back(pixel);
        }
    }
    return pixels;
}

// Build a uint8 mask for one connected component. The output is not the tree
// reconstruction image; it is an inspection aid that marks pixels covered by a
// selected node.
inline py::array_t<uint8_t> reconstructNode(const morphology::detail::TreeTopology& tree, morphology::NodeId nodeId)
{
    if (!tree.isNode(nodeId) || !tree.isAlive(nodeId)) {
        throw std::invalid_argument("invalid NodeId for reconstruction");
    }

    auto image = mmcfilters::ImageUInt8::create(tree.numRows(), tree.numColumns());
    image->fill(0);
    for (int pixel : collectPixelsOfConnectedComponent(tree, nodeId)) {
        (*image)[pixel] = 255;
    }
    return imageToNumpy(image);
}

// Register shared morphology enums at module level. Attribute-specific enums
// are nested under Attribute in AttributeBinding.hpp.
inline void bindCoreMorphologyEnums(py::module& m)
{
    py::enum_<morphology::TreeOfShapesInterpolation>(
        m,
        "ToSInterpolation",
        py::module_local(),
        "Interpolation policy used by tree-of-shapes construction.")
        .value("SELF_DUAL", morphology::TreeOfShapesInterpolation::SelfDual)
        .value("MIN4_MAX8", morphology::TreeOfShapesInterpolation::Min4cMax8c)
        .value("MIN8_MAX4", morphology::TreeOfShapesInterpolation::Min8cMax4c)
        .export_values();

    py::enum_<morphology::NodeIdSpace>(
        m,
        "NodeIdSpace",
        py::module_local(),
        "Node-id space used by attribute-computation outputs.")
        .value("MORPHOLOGICAL_TREE", morphology::NodeIdSpace::MorphologicalTree)
        .value("HIGRA", morphology::NodeIdSpace::Higra)
        .export_values();
}

// Attach topology, traversal, and mutation queries to the Python tree class.
template <class PyClass>
void bindWeightedTreeQueries(PyClass& cls)
{
    cls.def_property_readonly("num_internal_node_slots", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).numInternalNodeSlots();
        }, "Number of backend node slots used by node-indexed arrays.")
        .def_property_readonly("num_pixels", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).numPixels();
        }, "Number of pixels in the finite tree domain.")
        .def_property_readonly("num_higra_nodes", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getNumHigraNodes();
        }, "Number of nodes in the exported Higra-compatible hierarchy.")
        .def_property_readonly("root", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).root();
        }, "Root node id.")
        .def_property_readonly("num_free_node_slots", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getNumFreeNodeSlots();
        }, "Number of inactive node slots currently held by the backend.")
        .def_property_readonly("num_leaf_nodes", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).numLeafNodes();
        }, "Number of live leaf nodes.")
        .def_property_readonly("alive_node_ids", [](morphology::WeightedTree& self) {
            return collectNodeIds(morphology::detail::topology(self).aliveNodeIds());
        }, "Live node ids in the morphology-tree node-id space.")
        .def_property_readonly("leaves", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).leaves();
        }, "Live leaf-node ids.")
        .def("children", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).children(nodeId));
        }, "node_id"_a, "Return direct children of ``node_id``.")
        .def("num_descendants", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).numDescendants(nodeId);
        }, "node_id"_a)
        .def("num_siblings", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).numSiblings(nodeId);
        }, "node_id"_a)
        .def("proper_part_cardinality", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).properPartCardinality(nodeId);
        }, "node_id"_a)
        .def("dfs_entry_index", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).dfsEntryIndex(nodeId);
        }, "node_id"_a)
        .def("dfs_exit_index", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).dfsExitIndex(nodeId);
        }, "node_id"_a)
        .def("proper_part", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).properPart(nodeId));
        }, "node_id"_a, "Return the pixels in the proper part of ``node_id``.")
        .def("reconstruct_node", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return reconstructNode(morphology::detail::topology(self), nodeId);
        }, "node_id"_a, "Return a uint8 mask for the connected component represented by ``node_id``.")
        .def("post_order", [](morphology::WeightedTree& self, std::optional<morphology::NodeId> rootNodeId) {
            return rootNodeId.has_value()
                ? collectNodeIds(morphology::detail::topology(self).postOrder(*rootNodeId))
                : collectNodeIds(morphology::detail::topology(self).postOrder());
        }, "root_node_id"_a = std::nullopt)
        .def("breadth_first_traversal", [](morphology::WeightedTree& self, std::optional<morphology::NodeId> rootNodeId) {
            return rootNodeId.has_value()
                ? collectNodeIds(morphology::detail::topology(self).breadthFirstTraversal(*rootNodeId))
                : collectNodeIds(morphology::detail::topology(self).breadthFirstTraversal());
        }, "root_node_id"_a = std::nullopt)
        .def("ancestors", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).ancestors(nodeId));
        }, "node_id"_a, "Return ``node_id`` and its ancestors up to and including the root.")
        .def("path_between_nodes", [](morphology::WeightedTree& self, morphology::NodeId sourceNodeId, morphology::NodeId targetNodeId) {
            return collectNodeIds(morphology::detail::topology(self).getPathBetweenNodes(sourceNodeId, targetNodeId));
        }, "source_node_id"_a, "target_node_id"_a, "Return the tree path between two nodes.")
        .def("subtree_nodes", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).subtreeNodes(nodeId));
        }, "node_id"_a, "Return ``node_id`` and all its descendants.")
        .def("descendants", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).descendants(nodeId));
        }, "node_id"_a, "Return the strict descendants of ``node_id``.")
        .def("parent", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).parent(nodeId);
        }, "node_id"_a, "Return the parent node id for ``node_id``.")
        .def("smallest_node", [](morphology::WeightedTree& self, int pixelId) {
            return morphology::detail::topology(self).smallestNode(pixelId);
        }, "pixel_id"_a, "Return the smallest node whose support contains ``pixel_id``.")
        .def("higra_node_id", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).getHigraNodeId(nodeId);
        }, "node_id"_a)
        .def("num_children", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).numChildren(nodeId);
        }, "node_id"_a)
        .def("first_child", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).getFirstChild(nodeId);
        }, "node_id"_a)
        .def("next_sibling", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).getNextSibling(nodeId);
        }, "node_id"_a)
        .def("is_node", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).isNode(nodeId);
        }, "node_id"_a, "Return whether ``node_id`` is a topology node slot.")
        .def("is_pixel", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).isPixel(nodeId);
        }, "node_id"_a, "Return whether the id is a pixel in the finite tree domain.")
        .def("is_alive", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).isAlive(nodeId);
        }, "node_id"_a, "Return whether ``node_id`` currently belongs to the live tree.")
        .def("is_root", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).isRoot(nodeId);
        }, "node_id"_a, "Return whether ``node_id`` is the root.")
        .def("is_leaf", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).isLeaf(nodeId);
        }, "node_id"_a, "Return whether ``node_id`` is a leaf.")
        .def("has_child", [](morphology::WeightedTree& self, morphology::NodeId parentId, morphology::NodeId childId) {
            return morphology::detail::topology(self).hasChild(parentId, childId);
        }, "parent_id"_a, "child_id"_a, "Return whether ``child_id`` is a direct child of ``parent_id``.")
        .def("prune_node", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            self.pruneNode(nodeId);
        }, "node_id"_a, "Prune ``node_id`` from the tree in place.")
        .def("merge_node_into_parent", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            self.mergeNodeIntoParent(nodeId);
        }, "node_id"_a, "Merge ``node_id`` into its parent in place.")
        // Read declared tree semantics and topographic adjacency.
        .def_property_readonly("kind", [](morphology::WeightedTree& self) {
            return treeKindName(morphology::detail::topology(self).semantics().kind);
        }, "Declared tree kind, using the same vocabulary as morphology.TreeType.")
        .def_property_readonly("has_adjacency_relation", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).sharedAdjacencyContext() != nullptr;
        })
        .def_property_readonly("has_tree_of_shapes_adjacency_policy", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).topographicConvention() != nullptr;
        })
        .def("tree_of_shapes_min_adjacency_radius", [](morphology::WeightedTree& self) {
            return treeOfShapesAdjacencyRadius(self, /*minimum=*/true);
        })
        .def("tree_of_shapes_max_adjacency_radius", [](morphology::WeightedTree& self) {
            return treeOfShapesAdjacencyRadius(self, /*minimum=*/false);
        })
        .def_property_readonly("num_rows", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).numRows();
        }, "Number of rows in the source image.")
        .def_property_readonly("num_columns", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).numColumns();
        }, "Number of columns in the source image.")
        .def_property_readonly("num_nodes", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).numNodes();
        }, "Number of live morphology-tree nodes.")
        .def("node_altitude", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return self.getAltitude(nodeId);
        }, "node_id"_a, "Return the altitude value for ``node_id``.")
        .def("node_residue", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return self.getNodeResidue(nodeId);
        }, "node_id"_a, "Return the residue value for ``node_id``.")
        .def("reconstruct_from_node_altitudes", [](morphology::WeightedTree& self) {
            return imageToNumpy(self.reconstructionImage());
        }, "Reconstruct the current image represented by the tree.")
        .def("export_higra_hierarchy", [](morphology::WeightedTree& self) {
            return self.exportHigraHierarchy();
        }, "Return ``(parents, altitudes)`` vectors for a Higra-compatible hierarchy.");
}

// Register constructors for max-tree, min-tree, and tree of shapes. The Python
// class stores shared_ptr<WeightedTree> so CFP tensors, filters, and attributes
// can safely share the same tree handle.
inline void bindWeightedTree(py::module& m)
{
    auto weightedTree = py::class_<morphology::WeightedTree, morphology::WeightedTreePtr>(
        m,
        // Python-facing name: this is mtlearn's own API surface and stays put
        // even though the backend type it wraps was renamed.
        "WeightedMorphologicalTree",
        py::module_local(),
        R"pbdoc(Native weighted morphology-tree handle returned by ``mtlearn.morphology``.

Instances expose topology queries, node altitude/residue values, mutation
operations, reconstruction, and hierarchy export. Prefer high-level factories
such as ``mtlearn.morphology.create_max_tree`` instead of constructing this
class directly.
)pbdoc");

    weightedTree
        .def_static("create_component_tree", [](const UInt8InputArray& input, bool isMaxTree, double radius) {
            return std::make_shared<morphology::WeightedTree>(
                morphology::WeightedTree::createComponentTree(imageViewFromArray(input), isMaxTree, radius));
        }, "input"_a, "is_max_tree"_a, "radius"_a = 1.5, "Build a max-tree or min-tree from a 2D uint8 image.")
        .def_static("create_max_tree", [](const UInt8InputArray& input, double radius) {
            return std::make_shared<morphology::WeightedTree>(
                morphology::WeightedTree::createComponentTree(imageViewFromArray(input), true, radius));
        }, "input"_a, "radius"_a = 1.5, "Build a max-tree from a 2D uint8 image.")
        .def_static("create_min_tree", [](const UInt8InputArray& input, double radius) {
            return std::make_shared<morphology::WeightedTree>(
                morphology::WeightedTree::createComponentTree(imageViewFromArray(input), false, radius));
        }, "input"_a, "radius"_a = 1.5, "Build a min-tree from a 2D uint8 image.")
        .def_static("create_tree_of_shapes", [](const UInt8InputArray& input, morphology::TreeOfShapesInterpolation interpolation, int infinitySeedRow, int infinitySeedCol) {
            return std::make_shared<morphology::WeightedTree>(
                morphology::WeightedTree::createTreeOfShapes(imageViewFromArray(input), interpolation, infinitySeedRow, infinitySeedCol));
        },
            "input"_a,
            "interpolation"_a = morphology::TreeOfShapesInterpolation::SelfDual,
            "infinity_seed_row"_a = morphology::TreeOfShapesDefaultInfinityRow,
            "infinity_seed_col"_a = morphology::TreeOfShapesDefaultInfinityCol,
            "Build a tree of shapes from a 2D uint8 image.");

    bindWeightedTreeQueries(weightedTree);
}

} // namespace morphology_pybind
} // namespace mtlearn
