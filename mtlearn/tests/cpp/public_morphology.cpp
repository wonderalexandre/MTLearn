// Smoke/regression test for the installed C++ morphology facade.
//
// This test intentionally uses only <mtlearn/morphology.hpp> and standard
// library headers. It protects the public C++ contract that downstream
// consumers can create trees, inspect basic topology, reconstruct images, and
// export Higra-compatible hierarchy vectors without including backend headers.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <variant>

#include <mtlearn/morphology.hpp>

static_assert(std::is_copy_constructible_v<mtlearn::morphology::WeightedTree>);
static_assert(!std::is_default_constructible_v<mtlearn::morphology::WeightedTree>);

int main()
{
    namespace morphology = mtlearn::morphology;

    // Attribute requests are variant-based so callers can pass either a single
    // attribute or a backend-expanded attribute group through one API.
    morphology::AttributeOrGroup attribute = morphology::Attribute::AREA;
    assert(std::holds_alternative<morphology::Attribute>(attribute));

    attribute = morphology::AttributeGroup::SHAPE;
    assert(std::holds_alternative<morphology::AttributeGroup>(attribute));

    const auto outputSpace = morphology::NodeIdSpace::MORPHOLOGICAL_TREE;
    assert(outputSpace == morphology::NodeIdSpace::MORPHOLOGICAL_TREE);

    const auto interpolation = morphology::TreeOfShapesInterpolation::SelfDual;
    assert(interpolation == morphology::TreeOfShapesInterpolation::SelfDual);

    bool rejectedNullData = false;
    try {
        (void)morphology::WeightedTree::createComponentTree({nullptr, 2, 2}, true);
    } catch (const std::invalid_argument&) {
        rejectedNullData = true;
    }
    assert(rejectedNullData);

    bool rejectedInvalidDimensions = false;
    try {
        std::array<std::uint8_t, 1> invalidPixels{0};
        (void)morphology::WeightedTree::createComponentTree({invalidPixels.data(), 0, 1}, true);
    } catch (const std::invalid_argument&) {
        rejectedInvalidDimensions = true;
    }
    assert(rejectedInvalidDimensions);

    std::array<std::uint8_t, 4> pixels{1, 2, 3, 4};
    const morphology::ImageViewUInt8 imageView{pixels.data(), 2, 2};

    // The max-tree path is checked most thoroughly because it exercises
    // reconstruction and hierarchy export used by both C++ and Python layers.
    auto maxTree = morphology::WeightedTree::createComponentTree(imageView, true);
    assert(maxTree.numRows() == 2);
    assert(maxTree.numCols() == 2);
    assert(maxTree.numNodes() > 0);
    assert(maxTree.numInternalNodeSlots() >= maxTree.numNodes());

    const auto reconstructed = maxTree.reconstructionImage();
    assert(reconstructed.rows == 2);
    assert(reconstructed.cols == 2);
    assert(reconstructed.pixels.size() == pixels.size());
    assert(std::equal(reconstructed.pixels.begin(), reconstructed.pixels.end(), pixels.begin()));

    const auto [parent, altitude] = maxTree.exportHigraHierarchy();
    assert(!parent.empty());
    assert(parent.size() == altitude.size());
    assert(parent.size() >= pixels.size());

    // Keep construction smoke tests for the other supported tree families so
    // facade/backend enum and factory mappings fail fast if they drift.
    auto minTree = morphology::WeightedTree::createComponentTree(imageView, false);
    assert(minTree.numRows() == 2);
    assert(minTree.numCols() == 2);
    assert(minTree.numNodes() > 0);

    auto treeOfShapes = morphology::WeightedTree::createTreeOfShapes(imageView);
    assert(treeOfShapes.numRows() == 2);
    assert(treeOfShapes.numCols() == 2);
    assert(treeOfShapes.numNodes() > 0);

    auto treeOfShapesMin4Max8 = morphology::WeightedTree::createTreeOfShapes(
        imageView,
        morphology::TreeOfShapesInterpolation::Min4cMax8c);
    assert(treeOfShapesMin4Max8.numRows() == 2);
    assert(treeOfShapesMin4Max8.numCols() == 2);
    assert(treeOfShapesMin4Max8.numNodes() > 0);

    auto treeOfShapesMin8Max4 = morphology::WeightedTree::createTreeOfShapes(
        imageView,
        morphology::TreeOfShapesInterpolation::Min8cMax4c);
    assert(treeOfShapesMin8Max4.numRows() == 2);
    assert(treeOfShapesMin8Max4.numCols() == 2);
    assert(treeOfShapesMin8Max4.numNodes() > 0);

    return 0;
}
