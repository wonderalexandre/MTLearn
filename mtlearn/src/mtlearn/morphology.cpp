#include "mtlearn/morphology.hpp"

// Implementation of the public C++ morphology facade.
//
// All backend-specific construction and conversion code is kept here so the
// installed header can remain independent from mmcfilters. The Python bindings
// and CFP helpers reach the backend through detail::BackendAccess, but regular
// C++ consumers should interact with the WeightedTree facade only.

#include "mtlearn/detail/morphology_backend.hpp"

#include <mmcfilters/trees/MorphologicalTreeFactory.hpp>

#include <cstddef>
#include <stdexcept>

namespace mtlearn::morphology {
namespace {

// Convert the public non-owning image view into the backend image view. The
// backend API expects a mutable pointer even though tree construction does not
// conceptually mutate the caller's image; the cast is contained at this narrow
// boundary so the public facade can expose a const-correct view.
mmcfilters::ImageUInt8Ptr backendImageFromView(ImageViewUInt8 image)
{
    if (image.data == nullptr) {
        throw std::invalid_argument("ImageViewUInt8 data must not be null");
    }
    if (image.rows <= 0 || image.cols <= 0) {
        throw std::invalid_argument("ImageViewUInt8 dimensions must be positive");
    }

    return mmcfilters::ImageUInt8::fromExternal(
        const_cast<std::uint8_t*>(image.data),
        image.rows,
        image.cols);
}

// Translate the public infinity seed into the backend's infinity pixel.
//
// The facade parameter is a pixel of the source image domain. The backend
// expects a row-major index into the interpolated domain, which without an
// exterior ring spans 2n-1 entries on each axis and places source pixel
// (row, col) at (2*row, 2*col).
int interpolatedInfinityPixel(int numRows, int numCols, int infinitySeedRow, int infinitySeedCol)
{
    if (infinitySeedRow < 0 || infinitySeedRow >= numRows || infinitySeedCol < 0 || infinitySeedCol >= numCols) {
        throw std::invalid_argument("tree-of-shapes infinity seed must be inside the image domain");
    }

    const int interpolatedCols = 2 * numCols - 1;
    return (2 * infinitySeedRow) * interpolatedCols + (2 * infinitySeedCol);
}

// Translate the facade interpolation enum into a complete backend topographic
// convention. Keeping this conversion local avoids exposing mmcfilters types in
// the public header.
//
// mtlearn always builds the tree of shapes without the exterior ring, and that
// is what keeps the published altitudes on the source 8-bit lattice: the
// boundary reference level is the only construction level that can fall between
// two source levels, and without the ring it is cropped away before any
// interior cell reads it. The 8-bit encoding is therefore exact here, not a
// quantization.
mmcfilters::TopographicConvention toBackendConvention(
    TreeOfShapesInterpolation interpolation,
    int numRows,
    int numCols,
    int infinitySeedRow,
    int infinitySeedCol)
{
    mmcfilters::TopographicConvention convention;
    convention.domainExtension = mmcfilters::TopographicDomainExtension::None;
    convention.altitudeEncoding = mmcfilters::TopographicAltitudeEncoding::UInt8;

    switch (interpolation) {
    case TreeOfShapesInterpolation::SelfDual:
        convention.immersion = mmcfilters::SelfDualSpanImmersion{};
        break;
    case TreeOfShapesInterpolation::Min4cMax8c:
        convention.immersion =
            mmcfilters::CanonicalComplementaryGridImmersion{mmcfilters::ComplementaryPairing::Min4Max8};
        break;
    case TreeOfShapesInterpolation::Min8cMax4c:
        convention.immersion =
            mmcfilters::CanonicalComplementaryGridImmersion{mmcfilters::ComplementaryPairing::Min8Max4};
        break;
    default:
        throw std::invalid_argument("unknown tree-of-shapes interpolation");
    }

    convention.infinityPixel = interpolatedInfinityPixel(numRows, numCols, infinitySeedRow, infinitySeedCol);
    return convention;
}

} // namespace

// Private implementation object for the shared-handle facade. The backend tree
// remains movable/copyable according to its own semantics while public
// WeightedTree instances stay cheap to copy across C++ and pybind boundaries.
struct WeightedTree::Impl {
    explicit Impl(detail::BackendWeightedTree backend) : backend(std::move(backend)) {}

    detail::BackendWeightedTree backend;
};

WeightedTree::WeightedTree(std::shared_ptr<Impl> impl) : impl_(std::move(impl))
{
    if (!impl_) {
        throw std::invalid_argument("mtlearn::morphology::WeightedTree requires a backend tree");
    }
}

WeightedTree::~WeightedTree() = default;

// Component-tree and tree-of-shapes construction are the only public creation
// points. They validate the image at the facade boundary, then delegate the
// actual morphology algorithm to the current backend.
WeightedTree WeightedTree::createComponentTree(ImageViewUInt8 image, bool isMaxTree, double radius)
{
    return WeightedTree(std::make_shared<Impl>(
        isMaxTree
            ? mmcfilters::MorphologicalTreeFactory::createMaxTree(backendImageFromView(image), radius)
            : mmcfilters::MorphologicalTreeFactory::createMinTree(backendImageFromView(image), radius)));
}

WeightedTree WeightedTree::createTreeOfShapes(
    ImageViewUInt8 image,
    TreeOfShapesInterpolation interpolation,
    int infinitySeedRow,
    int infinitySeedCol)
{
    return WeightedTree(std::make_shared<Impl>(
        mmcfilters::MorphologicalTreeFactory::createTreeOfShapes<std::uint8_t>(
            backendImageFromView(image),
            toBackendConvention(interpolation, image.rows, image.cols, infinitySeedRow, infinitySeedCol))));
}

int WeightedTree::numRows() const
{
    return impl_->backend.topology().numRows();
}

int WeightedTree::numCols() const
{
    return impl_->backend.topology().numColumns();
}

int WeightedTree::numNodes() const
{
    return impl_->backend.topology().numNodes();
}

int WeightedTree::numInternalNodeSlots() const
{
    return impl_->backend.topology().numInternalNodeSlots();
}

// The following accessors intentionally forward directly to the backend. They
// keep the public API small while preserving the exact backend node-id behavior
// expected by the Python notebooks and CFP preprocessing code.
float WeightedTree::getAltitude(NodeId nodeId) const
{
    return static_cast<float>(impl_->backend.nodeAltitude(nodeId));
}

float WeightedTree::getNodeResidue(NodeId nodeId) const
{
    return static_cast<float>(impl_->backend.nodeResidue(nodeId));
}

void WeightedTree::pruneNode(NodeId nodeId)
{
    impl_->backend.pruneNode(nodeId);
}

void WeightedTree::mergeNodeIntoParent(NodeId nodeId)
{
    impl_->backend.mergeNodeIntoParent(nodeId);
}

UInt8Image WeightedTree::reconstructionImage() const
{
    auto backendImage = impl_->backend.reconstructFromNodeAltitudes();
    UInt8Image image;
    image.rows = backendImage->getNumRows();
    image.cols = backendImage->getNumColumns();

    const auto buffer = backendImage->rawDataPtr();
    const auto size = static_cast<std::size_t>(image.rows) * static_cast<std::size_t>(image.cols);
    image.pixels.assign(buffer.get(), buffer.get() + size);
    return image;
}

// Convert backend export vectors into mtlearn-owned containers. This avoids
// leaking backend vector aliases or scalar types through the public API.
std::pair<std::vector<NodeId>, std::vector<float>> WeightedTree::exportHigraHierarchy() const
{
    auto [backendParent, backendAltitude] = impl_->backend.exportHigraHierarchy();
    std::vector<NodeId> parent(backendParent.begin(), backendParent.end());
    std::vector<float> altitude(backendAltitude.begin(), backendAltitude.end());
    return {std::move(parent), std::move(altitude)};
}

// BackendAccess is the single authorized unwrap point for internal code. Any
// future backend replacement should keep this section small and localized.
detail::BackendWeightedTree& detail::BackendAccess::backend(WeightedTree& tree) noexcept
{
    return tree.impl_->backend;
}

const detail::BackendWeightedTree& detail::BackendAccess::backend(const WeightedTree& tree) noexcept
{
    return tree.impl_->backend;
}

} // namespace mtlearn::morphology
