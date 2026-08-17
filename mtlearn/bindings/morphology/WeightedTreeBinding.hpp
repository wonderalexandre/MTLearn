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
        for (int properPart : tree.properPart(subtreeNodeId)) {
            pixels.push_back(properPart);
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
        .value("SelfDual", morphology::TreeOfShapesInterpolation::SelfDual)
        .value("Min4cMax8c", morphology::TreeOfShapesInterpolation::Min4cMax8c)
        .value("Min8cMax4c", morphology::TreeOfShapesInterpolation::Min8cMax4c)
        .export_values();

    py::enum_<morphology::NodeIdSpace>(
        m,
        "NodeIdSpace",
        py::module_local(),
        "Node-id space used by attribute-computation outputs.")
        .value("MORPHOLOGICAL_TREE", morphology::NodeIdSpace::MORPHOLOGICAL_TREE)
        .value("HIGRA", morphology::NodeIdSpace::HIGRA)
        .export_values();
}

// Attach topology, traversal, and mutation methods to the Python
// ValuedMorphologicalTree class. Multiple naming styles are intentionally
// preserved because notebooks historically used camelCase and snake_case.
template <class PyClass>
void bindWeightedTreeQueries(PyClass& cls)
{
    cls.def_property_readonly("numInternalNodeSlots", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).numInternalNodeSlots();
        }, "Number of backend node slots used by node-indexed arrays.")
        .def_property_readonly("numTotalProperParts", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).numPixels();
        }, "Number of proper parts, normally matching the number of image pixels.")
        .def_property_readonly("numHigraNodes", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getNumHigraNodes();
        }, "Number of nodes in the exported Higra-compatible hierarchy.")
        .def("getRoot", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).root();
        }, "Return the root node id.")
        .def_property_readonly("root", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).root();
        }, "Root node id.")
        .def_property_readonly("numFreeNodeSlots", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).getNumFreeNodeSlots();
        }, "Number of inactive node slots currently held by the backend.")
        .def_property_readonly("numLeafNodes", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).numLeafNodes();
        }, "Number of live leaf nodes.")
        .def("getAliveNodeIds", [](morphology::WeightedTree& self) {
            return collectNodeIds(morphology::detail::topology(self).aliveNodeIds());
        }, "Return live node ids in the morphology-tree node-id space.")
        .def_property_readonly("aliveNodeIds", [](morphology::WeightedTree& self) {
            return collectNodeIds(morphology::detail::topology(self).aliveNodeIds());
        }, "Live node ids in the morphology-tree node-id space.")
        .def_property_readonly("alive_node_ids", [](morphology::WeightedTree& self) {
            return collectNodeIds(morphology::detail::topology(self).aliveNodeIds());
        }, "Live node ids in the morphology-tree node-id space.")
        .def("getLeafNodeIds", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).leaves();
        }, "Return live leaf-node ids.")
        .def_property_readonly("leafNodeIds", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).leaves();
        }, "Live leaf-node ids.")
        .def_property_readonly("leaf_node_ids", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).leaves();
        }, "Live leaf-node ids.")
        .def("getChildren", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).children(nodeId));
        }, "nodeId"_a, "Return direct children of ``nodeId``.")
        .def("childrenOf", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).children(nodeId));
        }, "nodeId"_a, "Alias for ``getChildren``.")
        .def("children_of", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).children(nodeId));
        }, "nodeId"_a, "Alias for ``getChildren``.")
        .def("getNodeNumDescendants", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).numDescendants(nodeId);
        }, "nodeId"_a)
        .def("getNodeNumSiblings", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).numSiblings(nodeId);
        }, "nodeId"_a)
        .def("getNumProperParts", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).properPartCardinality(nodeId);
        }, "nodeId"_a)
        .def("getNodeTimePreOrder", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).dfsEntryIndex(nodeId);
        }, "nodeId"_a)
        .def("getNodeTimePostOrder", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).dfsExitIndex(nodeId);
        }, "nodeId"_a)
        .def("getProperParts", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).properPart(nodeId));
        }, "nodeId"_a, "Return proper parts owned directly by ``nodeId``.")
        .def("properPartsOf", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).properPart(nodeId));
        }, "nodeId"_a, "Alias for ``getProperParts``.")
        .def("proper_parts_of", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).properPart(nodeId));
        }, "nodeId"_a, "Alias for ``getProperParts``.")
        .def("reconstructNode", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return reconstructNode(morphology::detail::topology(self), nodeId);
        }, "nodeId"_a, "Return a uint8 mask for the connected component represented by ``nodeId``.")
        .def("getPostOrderNodes", [](morphology::WeightedTree& self, std::optional<morphology::NodeId> rootNodeId) {
            return rootNodeId.has_value()
                ? collectNodeIds(morphology::detail::topology(self).postOrder(*rootNodeId))
                : collectNodeIds(morphology::detail::topology(self).postOrder());
        }, "rootNodeId"_a = std::nullopt)
        .def("getIteratorBreadthFirstTraversal", [](morphology::WeightedTree& self, std::optional<morphology::NodeId> rootNodeId) {
            return rootNodeId.has_value()
                ? collectNodeIds(morphology::detail::topology(self).breadthFirstTraversal(*rootNodeId))
                : collectNodeIds(morphology::detail::topology(self).breadthFirstTraversal());
        }, "rootNodeId"_a = std::nullopt)
        .def("getPathToRootNodes", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).ancestors(nodeId));
        }, "nodeId"_a, "Return the path from ``nodeId`` to the root.")
        .def("getPathBetweenNodes", [](morphology::WeightedTree& self, morphology::NodeId sourceNodeId, morphology::NodeId targetNodeId) {
            return collectNodeIds(morphology::detail::topology(self).getPathBetweenNodes(sourceNodeId, targetNodeId));
        }, "sourceNodeId"_a, "targetNodeId"_a, "Return the tree path between two nodes.")
        .def("getNodeSubtree", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).subtreeNodes(nodeId));
        }, "nodeId"_a, "Return nodes in the subtree rooted at ``nodeId``.")
        .def("nodeSubtreeOf", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).subtreeNodes(nodeId));
        }, "nodeId"_a, "Alias for ``getNodeSubtree``.")
        .def("node_subtree_of", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).subtreeNodes(nodeId));
        }, "nodeId"_a, "Alias for ``getNodeSubtree``.")
        .def("getDescendants", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).descendants(nodeId));
        }, "nodeId"_a, "Return descendants of ``nodeId``.")
        .def("descendantsOf", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).descendants(nodeId));
        }, "nodeId"_a, "Alias for ``getDescendants``.")
        .def("descendants_of", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return collectNodeIds(morphology::detail::topology(self).descendants(nodeId));
        }, "nodeId"_a, "Alias for ``getDescendants``.")
        .def("getNodeParent", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).parent(nodeId);
        }, "nodeId"_a, "Return the parent node id for ``nodeId``.")
        .def("parentOf", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).parent(nodeId);
        }, "nodeId"_a, "Alias for ``getNodeParent``.")
        .def("parent_of", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).parent(nodeId);
        }, "nodeId"_a, "Alias for ``getNodeParent``.")
        .def("getProperPartOwner", [](morphology::WeightedTree& self, int pixelId) {
            return morphology::detail::topology(self).smallestNode(pixelId);
        }, "pixelId"_a, "Return the node that owns a flattened image pixel/proper part.")
        .def("properPartOwnerOf", [](morphology::WeightedTree& self, int pixelId) {
            return morphology::detail::topology(self).smallestNode(pixelId);
        }, "pixelId"_a, "Alias for ``getProperPartOwner``.")
        .def("proper_part_owner_of", [](morphology::WeightedTree& self, int pixelId) {
            return morphology::detail::topology(self).smallestNode(pixelId);
        }, "pixelId"_a, "Alias for ``getProperPartOwner``.")
        .def("getHigraNodeId", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).getHigraNodeId(nodeId);
        }, "nodeId"_a)
        .def("getNumChildren", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).numChildren(nodeId);
        }, "nodeId"_a)
        .def("getFirstChild", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).getFirstChild(nodeId);
        }, "nodeId"_a)
        .def("getNextSibling", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).getNextSibling(nodeId);
        }, "nodeId"_a)
        .def("isNode", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).isNode(nodeId);
        }, "nodeId"_a, "Return whether ``nodeId`` is a topology node slot.")
        .def("isProperPart", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).isPixel(nodeId);
        }, "nodeId"_a, "Return whether the id is a proper-part/pixel slot.")
        .def("isAlive", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).isAlive(nodeId);
        }, "nodeId"_a, "Return whether ``nodeId`` currently belongs to the live tree.")
        .def("isRoot", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).isRoot(nodeId);
        }, "nodeId"_a, "Return whether ``nodeId`` is the root.")
        .def("isLeaf", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return morphology::detail::topology(self).isLeaf(nodeId);
        }, "nodeId"_a, "Return whether ``nodeId`` is a leaf.")
        .def("hasChild", [](morphology::WeightedTree& self, morphology::NodeId parentId, morphology::NodeId childId) {
            return morphology::detail::topology(self).hasChild(parentId, childId);
        }, "parentId"_a, "childId"_a, "Return whether ``childId`` is a direct child of ``parentId``.")
        .def("pruneNode", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            self.pruneNode(nodeId);
        }, "nodeId"_a, "Prune ``nodeId`` from the tree in place.")
        .def("mergeNodeIntoParent", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            self.mergeNodeIntoParent(nodeId);
        }, "nodeId"_a, "Merge ``nodeId`` into its parent in place.")
        // The backend replaced the flat tree-type enum and the two loose radii
        // with a declared semantics record and a typed topographic convention.
        // These accessors keep the previous Python shape by reading the new
        // models here.
        .def_property_readonly("treeType", [](morphology::WeightedTree& self) {
            return static_cast<int>(morphology::detail::topology(self).semantics().kind);
        })
        .def_property_readonly("hasAdjacencyRelation", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).sharedAdjacencyContext() != nullptr;
        })
        .def_property_readonly("hasTreeOfShapesAdjacencyPolicy", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).topographicConvention() != nullptr;
        })
        .def("getTreeOfShapesMinTreeAdjacencyRadius", [](morphology::WeightedTree& self) {
            return treeOfShapesAdjacencyRadius(self, /*minimum=*/true);
        })
        .def("getTreeOfShapesMaxTreeAdjacencyRadius", [](morphology::WeightedTree& self) {
            return treeOfShapesAdjacencyRadius(self, /*minimum=*/false);
        })
        .def_property_readonly("numRows", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).numRows();
        }, "Number of rows in the source image.")
        .def_property_readonly("numCols", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).numColumns();
        }, "Number of columns in the source image.")
        .def_property_readonly("numNodes", [](morphology::WeightedTree& self) {
            return morphology::detail::topology(self).numNodes();
        }, "Number of live morphology-tree nodes.")
        .def("getAltitude", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return self.getAltitude(nodeId);
        }, "nodeId"_a, "Return the altitude value for ``nodeId``.")
        .def("altitudeOf", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return self.getAltitude(nodeId);
        }, "nodeId"_a)
        .def("altitude_of", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return self.getAltitude(nodeId);
        }, "nodeId"_a)
        .def("getNodeResidue", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return self.getNodeResidue(nodeId);
        }, "nodeId"_a, "Return the residue value for ``nodeId``.")
        .def("residueOf", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return self.getNodeResidue(nodeId);
        }, "nodeId"_a)
        .def("residue_of", [](morphology::WeightedTree& self, morphology::NodeId nodeId) {
            return self.getNodeResidue(nodeId);
        }, "nodeId"_a)
        .def("reconstructionImage", [](morphology::WeightedTree& self) {
            return imageToNumpy(self.reconstructionImage());
        }, "Reconstruct the current image represented by the tree.")
        .def("exportHigraHierarchy", [](morphology::WeightedTree& self) {
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
        .def_static("createComponentTree", [](const UInt8InputArray& input, bool isMaxTree, double radius) {
            return std::make_shared<morphology::WeightedTree>(
                morphology::WeightedTree::createComponentTree(imageViewFromArray(input), isMaxTree, radius));
        }, "input"_a, "isMaxtree"_a, "radius"_a = 1.5, "Build a max-tree or min-tree from a 2D uint8 image.")
        .def_static("createMaxTree", [](const UInt8InputArray& input, double radius) {
            return std::make_shared<morphology::WeightedTree>(
                morphology::WeightedTree::createComponentTree(imageViewFromArray(input), true, radius));
        }, "input"_a, "radius"_a = 1.5, "Build a max-tree from a 2D uint8 image.")
        .def_static("createMinTree", [](const UInt8InputArray& input, double radius) {
            return std::make_shared<morphology::WeightedTree>(
                morphology::WeightedTree::createComponentTree(imageViewFromArray(input), false, radius));
        }, "input"_a, "radius"_a = 1.5, "Build a min-tree from a 2D uint8 image.")
        .def_static("createTreeOfShapes", [](const UInt8InputArray& input, morphology::TreeOfShapesInterpolation interpolation, int infinitySeedRow, int infinitySeedCol) {
            return std::make_shared<morphology::WeightedTree>(
                morphology::WeightedTree::createTreeOfShapes(imageViewFromArray(input), interpolation, infinitySeedRow, infinitySeedCol));
        },
            "input"_a,
            "interpolation"_a = morphology::TreeOfShapesInterpolation::SelfDual,
            "infinitySeedRow"_a = morphology::TreeOfShapesDefaultInfinityRow,
            "infinitySeedCol"_a = morphology::TreeOfShapesDefaultInfinityCol,
            "Build a tree of shapes from a 2D uint8 image.");

    bindWeightedTreeQueries(weightedTree);
}

} // namespace morphology_pybind
} // namespace mtlearn
